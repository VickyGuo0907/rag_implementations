"""
GraphRAG: Knowledge-graph augmented retrieval — LangChain Implementation
==========================================================================================
Builds a knowledge graph over the corpus at index time and answers queries by
graph traversal instead of (or before) vector similarity:

  1. Extract    — one LLM call per chunk pulls out entities (typed) and
                  relationships between them
  2. Build      — entities become graph nodes, relationships become edges
                  weighted by repetition; Louvain communities are detected
  3. Query      — extract entities from the question, match them to graph
                  nodes, expand the neighborhood up to `max_hops`, and gather
                  every chunk those nodes/edges reference
  4. Backfill   — if the graph yields fewer chunks than `top_k`, supplement
                  with dense vector retrieval
  5. Generate   — answer from the assembled context

This makes multi-hop questions ("Who works with X and what did they co-author?")
first-class: the answer path is a walk through explicit relations, not hope that
one embedding happens to land near both hops.

Reference: https://github.com/NirDiamant/RAG_Techniques
Paper: "From Local to Global: A Graph RAG Approach to Query-Focused
Summarization" (Edge et al., Microsoft, 2024) — https://arxiv.org/abs/2404.16130
"""

import logging
import re
from collections import Counter

import networkx as nx
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.base_rag import BaseRAG, Document, RAGResult
from core.config_loader import ConfigLoader
from core.document_loader import get_text_splitter, load_texts
from core.embeddings import get_langchain_embeddings
from core.llm_client import get_langchain_llm
from core.vector_store import build_langchain_vector_store

logger = logging.getLogger(__name__)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning traces some LMStudio models emit."""
    return _THINK_TAG_RE.sub("", text).strip()


def _parse_entities(text: str) -> list[tuple[str, str]]:
    """Parse 'ENTITIES:' section lines of '- name | type' into tuples."""
    section = _extract_section(_strip_thinking(text), "ENTITIES", "RELATIONSHIPS")
    return [tuple(p[:2]) for line in _section_items(section) if (p := _split_pipe(line, 2))]


def _parse_relationships(text: str) -> list[tuple[str, str, str]]:
    """Parse 'RELATIONSHIPS:' section lines of '- src | rel | tgt' into tuples."""
    section = _extract_section(_strip_thinking(text), "RELATIONSHIPS", None)
    return [tuple(p[:3]) for line in _section_items(section) if (p := _split_pipe(line, 3))]


def _parse_question_entities(text: str) -> list[str]:
    """Parse a bare 'ENTITIES:' response into plain entity names."""
    section = _extract_section(_strip_thinking(text), "ENTITIES", None)
    return [p[0] for line in _section_items(section) if (p := _split_pipe(line, 1))]


def _extract_section(text: str, start: str, stop: str | None) -> str:
    """Return lines between a `start:` marker and an optional `stop:` marker."""
    lines, inside, out = text.split("\n"), False, []
    for line in lines:
        stripped = line.strip().upper()
        if stop and stripped.startswith(f"{stop}:"):
            break
        if stripped.startswith(f"{start}:"):
            inside = True
            continue
        if inside:
            out.append(line)
    return "\n".join(out)


def _section_items(section: str) -> list[str]:
    """Extract bullet items ('- ...') from a section body."""
    return [
        raw.strip()[2:].strip()
        for raw in section.split("\n")
        if raw.strip().startswith("-")
    ]


def _split_pipe(line: str, min_parts: int) -> list[str]:
    """Split on '|', pad missing parts with ''; returns [] when malformed."""
    parts = [p.strip() for p in line.split("|")]
    if len(parts) < min_parts or not parts[0]:
        return []
    while len(parts) < 3:
        parts.append("")
    return parts


def _normalize_entity(name: str) -> str:
    """Normalize an entity name for matching (lowercase, collapse spaces)."""
    return " ".join(name.lower().split())


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = ChatPromptTemplate.from_template(
    """Extract entities and relationships from the text below.

Allowed entity types: {entity_types}

Respond in EXACTLY this format:
ENTITIES:
- EntityName | TYPE
RELATIONSHIPS:
- SourceEntity | relation_phrase | TargetEntity

Rules:
- Use the exact surface form of each entity as it appears in the text.
- Only use the allowed entity types; skip anything else.
- A relationship must connect two extracted entities.

Text:
{text}"""
)

QUESTION_ENTITIES_PROMPT = ChatPromptTemplate.from_template(
    """List every concrete entity mentioned in this question (people,
organizations, locations, concepts, events, technologies).

Respond in EXACTLY this format:
ENTITIES:
- EntityName

Question: {question}
Entities:"""
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Answer the question based ONLY on the provided context.
The context was assembled from a knowledge graph, so it may contain facts about
several related entities — synthesize across all of it.
If the context doesn't contain sufficient information, acknowledge the limitation.

Context:
{context}"""),
    ("human", "{question}"),
])


class GraphRAGLangChain(BaseRAG):
    """
    GraphRAG using LangChain: LLM entity/relation extraction builds a NetworkX
    knowledge graph; queries traverse it to assemble context, with dense
    retrieval as backfill.

    Best for: Entity-rich corpora (reports, papers, meeting notes) where
    questions ask about relationships and chains of connections that similarity
    search alone keeps missing.
    """

    TECHNIQUE_NAME = "graph_rag"
    FRAMEWORK = "langchain"

    def _build_pipeline(self) -> None:
        """Initialize LLM, embeddings, extraction prompts, and graph settings."""
        cfg = ConfigLoader.get()
        tech_cfg = cfg.get_technique_config("graph_rag")

        self.llm = get_langchain_llm()
        self.embeddings = get_langchain_embeddings()
        self.text_splitter = get_text_splitter()
        self.top_k = cfg.retrieval.get("top_k", 5)

        self.entity_types = tech_cfg.get(
            "entity_types",
            ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "EVENT", "TECHNOLOGY"],
        )
        self.max_hops = tech_cfg.get("max_hops", 2)
        self.community_detection = tech_cfg.get("community_detection", True)
        self.max_context_chunks = tech_cfg.get("max_context_chunks", 12)

        if tech_cfg.get("graph_store", "networkx") != "networkx":
            logger.warning(
                "[GraphRAG/LC] Only the 'networkx' graph store is implemented — "
                f"ignoring configured store '{tech_cfg.get('graph_store')}'."
            )

        self.graph = nx.Graph()
        self.chunk_texts: dict[int, str] = {}
        self.communities: list[set[str]] = []
        self.vector_store = None

        logger.info(
            f"[GraphRAG/LC] entity_types={self.entity_types}, max_hops={self.max_hops}"
        )

    def index(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        """Chunk documents, extract entities/relations per chunk, build the graph."""
        logger.info(f"[GraphRAG/LC] Indexing {len(documents)} documents...")

        lc_docs = load_texts(documents, metadatas)
        chunks = self.text_splitter.split_documents(lc_docs)
        chain = EXTRACT_PROMPT | self.llm | StrOutputParser()

        for idx, chunk in enumerate(chunks):
            self.chunk_texts[idx] = chunk.page_content
            result = chain.invoke({
                "text": chunk.page_content,
                "entity_types": ", ".join(self.entity_types),
            })
            self._ingest_extraction(result, idx)
            if (idx + 1) % 10 == 0:
                logger.info(
                    f"[GraphRAG/LC] Chunk {idx + 1}/{len(chunks)} — "
                    f"nodes={self.graph.number_of_nodes()}, "
                    f"edges={self.graph.number_of_edges()}"
                )

        self._detect_communities()
        self.vector_store = build_langchain_vector_store(chunks, collection_name="graph_rag")
        self._is_indexed = True
        logger.info(
            f"[GraphRAG/LC] Indexed {len(chunks)} chunks → "
            f"{self.graph.number_of_nodes()} entities, "
            f"{self.graph.number_of_edges()} relations ✓"
        )

    def _ingest_extraction(self, raw: str, chunk_idx: int) -> None:
        """Merge one chunk's extracted entities and relationships into the graph."""
        for name, etype in _parse_entities(raw):
            node = _normalize_entity(name)
            if not node:
                continue
            attrs = self.graph.nodes.get(node, {"type": "UNKNOWN"})
            attrs.setdefault("labels", set()).add(name)
            if etype:
                attrs["type"] = etype.upper()
            attrs.setdefault("chunk_ids", set()).add(chunk_idx)
            self.graph.add_node(node, **attrs)

        for src, rel, tgt in _parse_relationships(raw):
            s_node, t_node = _normalize_entity(src), _normalize_entity(tgt)
            if not s_node or not t_node or s_node == t_node:
                continue
            for endpoint in (s_node, t_node):
                if not self.graph.has_node(endpoint):
                    self.graph.add_node(endpoint, type="UNKNOWN", labels={endpoint},
                                        chunk_ids={chunk_idx})
            if self.graph.has_edge(s_node, t_node):
                edge_attrs = self.graph.edges[s_node, t_node]
                edge_attrs["weight"] += 1
                edge_attrs["relations"].add(rel)
                edge_attrs["chunk_ids"].add(chunk_idx)
            else:
                self.graph.add_edge(
                    s_node, t_node, weight=1, relations={rel}, chunk_ids={chunk_idx}
                )

    def _detect_communities(self) -> None:
        """Run Louvain community detection and tag each node with its group id."""
        if not self.community_detection or self.graph.number_of_nodes() == 0:
            return
        try:
            self.communities = nx.community.louvain_communities(self.graph, seed=42)
        except Exception as exc:  # noqa: BLE001 - community layout must never break indexing
            logger.warning(f"[GraphRAG/LC] Community detection failed: {exc}")
            return
        for cid, members in enumerate(self.communities):
            for node in members:
                self.graph.nodes[node]["community"] = cid
        sizes = sorted((len(c) for c in self.communities), reverse=True)
        logger.info(
            "[GraphRAG/LC] Detected %d communities, sizes=%s", len(self.communities), sizes[:8]
        )

    # ------------------------------------------------------------------
    # Query-time helpers
    # ------------------------------------------------------------------

    def _question_entities(self, question: str) -> list[str]:
        """Extract candidate entity names from the question via one LLM call."""
        chain = QUESTION_ENTITIES_PROMPT | self.llm | StrOutputParser()
        return _parse_question_entities(chain.invoke({"question": question}))

    def _match_nodes(self, names: list[str]) -> list[str]:
        """Match question entity names to graph nodes (exact then substring)."""
        graph_nodes = set(self.graph.nodes())
        matched: list[str] = []
        for name in names:
            norm = _normalize_entity(name)
            if not norm:
                continue
            if norm in graph_nodes:
                matched.append(norm)
                continue
            partials = [n for n in graph_nodes if norm in n or n in norm]
            matched.extend(partials[:2])
        return list(dict.fromkeys(matched))

    def _subgraph_chunk_refs(self, seed_nodes: list[str]) -> Counter:
        """Count chunk references from the ego-subgraph around the matched seeds."""
        refs: Counter = Counter()
        visited: set[str] = set()
        for seed in seed_nodes:
            ego = nx.ego_graph(self.graph, seed, radius=self.max_hops)
            for node in ego.nodes():
                if node in visited:
                    continue
                visited.add(node)
                refs.update(self.graph.nodes[node].get("chunk_ids", set()))
            for _, _, data in ego.edges(data=True):
                refs.update(data.get("chunk_ids", set()))
        return refs

    def _assemble_context(self, refs: Counter) -> list[int]:
        """Pick the most-referenced chunks, capped at max_context_chunks."""
        ranked = [cid for cid, _count in refs.most_common(self.max_context_chunks)]
        return sorted(ranked)

    def _dense_backfill(self, question: str, have: set[int]) -> list[int]:
        """Retrieve top_k chunks via vectors, skipping ones already selected."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
        extra: list[int] = []
        for doc in retriever.invoke(question):
            for cid, text in self.chunk_texts.items():
                if cid not in have and text[:100] == doc.page_content[:100]:
                    extra.append(cid)
                    have.add(cid)
                    break
        return extra[: self.top_k]

    def _generate_with_context(self, question: str, contexts: list[str]) -> str:
        context = "\n\n---\n\n".join(contexts)
        chain = ANSWER_PROMPT | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})

    def _query(self, question: str) -> RAGResult:
        """Full GraphRAG pipeline: match entities → walk graph → backfill → generate."""
        if not self.vector_store:
            raise RuntimeError("Call index() before querying.")

        q_entities = self._question_entities(question)
        matched = self._match_nodes(q_entities)

        if matched:
            refs = self._subgraph_chunk_refs(matched)
            chunk_ids = set(self._assemble_context(refs))
            context_source = "graph"
        else:
            logger.warning("[GraphRAG/LC] No question entities matched graph nodes.")
            refs, chunk_ids, context_source = Counter(), set(), "dense_only"

        backfilled = self._dense_backfill(question, chunk_ids)
        ordered = sorted(chunk_ids) + sorted(backfilled)
        contexts = [self.chunk_texts[cid] for cid in ordered]
        answer = self._generate_with_context(question, contexts)

        return RAGResult(
            query=question,
            answer=answer,
            source_documents=[
                Document(content=self.chunk_texts[cid], metadata={"chunk_id": cid})
                for cid in ordered
            ],
            intermediate_steps=[{
                "step": "graph_traversal",
                "question_entities": q_entities,
                "matched_entities": matched,
                "context_source": context_source,
                "graph_chunks": len(ordered) - len(backfilled),
                "dense_backfilled": len(backfilled),
            }],
            metadata={
                "matched_entities": matched,
                "context_source": context_source,
                "graph_chunks": len(ordered) - len(backfilled),
                "dense_backfilled": len(backfilled),
            },
        )
