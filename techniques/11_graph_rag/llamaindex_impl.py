"""
GraphRAG: Knowledge-graph augmented retrieval — LlamaIndex Implementation
==========================================================================================
Same pipeline as the LangChain version: per-chunk LLM extraction of typed
entities and relationships builds a NetworkX knowledge graph; queries extract
question entities, traverse up to `max_hops`, assemble the referenced chunks,
backfill from dense retrieval when the graph comes up short -- implemented
with direct LLM `.complete()` calls for parity with the LangChain version.
"""

import logging
import re
from collections import Counter

import networkx as nx

from core.base_rag import BaseRAG, Document, RAGResult
from core.config_loader import ConfigLoader
from core.embeddings import get_llamaindex_embeddings
from core.llm_client import get_llamaindex_llm

logger = logging.getLogger(__name__)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning traces some LMStudio models emit in raw completions."""
    return _THINK_TAG_RE.sub("", text).strip()


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


def _normalize_entity(name: str) -> str:
    """Normalize an entity name for matching (lowercase, collapse spaces)."""
    return " ".join(name.lower().split())


EXTRACT_PROMPT = """Extract entities and relationships from the text below.

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

QUESTION_ENTITIES_PROMPT = """List every concrete entity mentioned in this question \
(people, organizations, locations, concepts, events, technologies).

Respond in EXACTLY this format:
ENTITIES:
- EntityName

Question: {question}
Entities:"""

ANSWER_PROMPT = """Answer the question based ONLY on the provided context. \
The context was assembled from a knowledge graph, so it may contain facts about \
several related entities — synthesize across all of it. If the context doesn't \
contain sufficient information, acknowledge the limitation.

Context:
{context}

Question: {question}
Answer:"""


class GraphRAGLlamaIndex(BaseRAG):
    """
    GraphRAG using LlamaIndex: LLM entity/relation extraction builds a NetworkX
    knowledge graph; queries traverse it to assemble context, with dense
    retrieval as backfill.

    Best for: Entity-rich corpora (reports, papers, meeting notes) where
    questions ask about relationships and chains of connections that similarity
    search alone keeps missing.
    """

    TECHNIQUE_NAME = "graph_rag"
    FRAMEWORK = "llamaindex"

    def _build_pipeline(self) -> None:
        """Initialize LLM, embeddings, node parser, and graph settings."""
        from llama_index.core import Settings
        from llama_index.core.node_parser import SentenceSplitter

        cfg = ConfigLoader.get()
        tech_cfg = cfg.get_technique_config("graph_rag")
        doc_cfg = cfg.document

        self.llm = get_llamaindex_llm()
        Settings.llm = self.llm
        Settings.embed_model = get_llamaindex_embeddings()
        Settings.node_parser = SentenceSplitter(
            chunk_size=doc_cfg.get("chunk_size", 512),
            chunk_overlap=doc_cfg.get("chunk_overlap", 50),
        )

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
                "[GraphRAG/LI] Only the 'networkx' graph store is implemented — "
                f"ignoring configured store '{tech_cfg.get('graph_store')}'."
            )

        self.graph = nx.Graph()
        self.chunk_texts: dict[int, str] = {}
        self.communities: list[set[str]] = []
        self.vector_index = None
        self.retriever = None

        logger.info(
            f"[GraphRAG/LI] entity_types={self.entity_types}, max_hops={self.max_hops}"
        )

    def index(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        """Chunk into nodes, extract entities/relations per node, build the graph."""
        from llama_index.core import Settings, VectorStoreIndex
        from llama_index.core.schema import Document as LIDocument

        logger.info(f"[GraphRAG/LI] Indexing {len(documents)} documents...")

        metas = metadatas or [{}] * len(documents)
        li_docs = [LIDocument(text=t, metadata=m) for t, m in zip(documents, metas, strict=True)]
        nodes = Settings.node_parser.get_nodes_from_documents(li_docs)
        entity_types = ", ".join(self.entity_types)

        for idx, node in enumerate(nodes):
            self.chunk_texts[idx] = node.get_content()
            raw = self.llm.complete(
                EXTRACT_PROMPT.format(text=node.get_content(), entity_types=entity_types)
            ).text
            self._ingest_extraction(raw, idx)
            if (idx + 1) % 10 == 0:
                logger.info(
                    f"[GraphRAG/LI] Node {idx + 1}/{len(nodes)} — "
                    f"nodes={self.graph.number_of_nodes()}, "
                    f"edges={self.graph.number_of_edges()}"
                )

        self._detect_communities()
        self.vector_index = VectorStoreIndex(nodes, show_progress=True)
        self.retriever = self.vector_index.as_retriever(similarity_top_k=self.top_k)

        self._is_indexed = True
        logger.info(
            f"[GraphRAG/LI] Indexed {len(nodes)} nodes → "
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
            logger.warning(f"[GraphRAG/LI] Community detection failed: {exc}")
            return
        for cid, members in enumerate(self.communities):
            for node in members:
                self.graph.nodes[node]["community"] = cid
        sizes = sorted((len(c) for c in self.communities), reverse=True)
        logger.info(
            "[GraphRAG/LI] Detected %d communities, sizes=%s", len(self.communities), sizes[:8]
        )

    # ------------------------------------------------------------------
    # Query-time helpers
    # ------------------------------------------------------------------

    def _question_entities(self, question: str) -> list[str]:
        """Extract candidate entity names from the question via one LLM call."""
        response = self.llm.complete(QUESTION_ENTITIES_PROMPT.format(question=question))
        return _parse_question_entities(response.text)

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
        """Retrieve top_k nodes via vectors, skipping ones already selected."""
        extra: list[int] = []
        for scored in self.retriever.retrieve(question):
            content = scored.get_content()
            for cid, text in self.chunk_texts.items():
                if cid not in have and text[:100] == content[:100]:
                    extra.append(cid)
                    have.add(cid)
                    break
        return extra[: self.top_k]

    def _generate_with_context(self, question: str, contexts: list[str]) -> str:
        context = "\n\n---\n\n".join(contexts)
        response = self.llm.complete(ANSWER_PROMPT.format(context=context, question=question))
        return _strip_thinking(response.text)

    def _query(self, question: str) -> RAGResult:
        """Full GraphRAG pipeline: match entities → walk graph → backfill → generate."""
        if not self.retriever:
            raise RuntimeError("Call index() before querying.")

        q_entities = self._question_entities(question)
        matched = self._match_nodes(q_entities)

        if matched:
            refs = self._subgraph_chunk_refs(matched)
            chunk_ids = set(self._assemble_context(refs))
            context_source = "graph"
        else:
            logger.warning("[GraphRAG/LI] No question entities matched graph nodes.")
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
