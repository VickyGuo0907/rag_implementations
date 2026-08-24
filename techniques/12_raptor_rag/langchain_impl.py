"""
RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval — LangChain
==========================================================================================
Builds a summary tree over the corpus instead of a flat chunk pile:

  1. Leaf level   — ordinary document chunks
  2. Cluster      — embed the level's texts, group them (GMM w/ BIC model
                    selection, or deterministic agglomerative fallback)
  3. Summarize    — one LLM call per sufficiently large cluster produces the
                    next level's nodes
  4. Recurse      — repeat until max_depth, no valid clusters, or summaries
                    stop shrinking the frontier
  5. Collapse     — every node from every level goes into ONE vector store

At query time, dense retrieval runs across all abstraction levels at once, so
a question naturally pulls detail-level chunks, mid-level syntheses, or
top-level summaries depending on what it actually needs.

Reference: https://github.com/NirDiamant/RAG_Techniques
Paper: "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
(Sarthi et al., 2024) — https://arxiv.org/abs/2401.18059
"""

import logging
import re

from langchain_core.documents import Document as LCDocument
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


def _truncate(text: str, limit: int) -> str:
    """Hard-cap text length for summarization prompts."""
    return text if len(text) <= limit else text[:limit] + "..."


def _select_best_gmm(
    vectors: list[list[float]],
    max_clusters: int,
    random_state: int = 42,
) -> list[int]:
    """
    Fit Gaussian Mixtures over k=1..max_clusters, keep the lowest-BIC fit.

    Falls back to a single cluster when there are too few samples to compare.
    """
    import numpy as np
    from sklearn.mixture import GaussianMixture

    matrix = np.asarray(vectors, dtype=float)
    n = len(matrix)
    if n < 2:
        return [0] * n

    best_labels, best_bic = None, float("inf")
    upper_k = min(max_clusters, n - 1)
    for k in range(1, upper_k + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=random_state,
        )
        labels = gmm.fit_predict(matrix)
        bic = gmm.bic(matrix)
        if bic < best_bic:
            best_labels, best_bic = labels, bic
    return [int(label) for label in best_labels]


def _cluster_texts(
    vectors: list[list[float]],
    algorithm: str = "gmm",
    max_clusters: int = 6,
) -> list[int]:
    """Dispatch clustering by algorithm name; agglomerative is the fallback."""
    if algorithm == "gmm":
        return _select_best_gmm(vectors, max_clusters=max_clusters)

    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    if len(vectors) < 2:
        return [0] * len(vectors)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.9,
        metric="cosine",
        linkage="average",
    )
    return [int(label) for label in clustering.fit_predict(np.asarray(vectors, dtype=float))]


def _group_by_label(labels: list[int]) -> list[list[int]]:
    """Turn a flat label array into index groups ordered by cluster id."""
    groups: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        groups.setdefault(label, []).append(idx)
    return [groups[label] for label in sorted(groups)]


def _should_stop(level: int, parent_count: int, prev_count: int, max_depth: int) -> bool:
    """True when recursion must end: depth reached, empty level, or no shrinkage."""
    return (
        level > max_depth
        or parent_count == 0
        or parent_count >= prev_count
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """Write a dense, self-contained summary of the following related passages.
Capture every distinct fact, entity, and claim — the summary will be retrieved
INSTEAD of these passages for big-picture questions.

Passages:
{text}

Summary:"""
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Answer the question based ONLY on the provided context.
The context mixes original chunks with multi-level summaries — trust detail
over summary when they conflict.
If the context doesn't contain sufficient information, acknowledge the limitation.

Context:
{context}"""),
    ("human", "{question}"),
])


class RaptorRAGLangChain(BaseRAG):
    """
    RAPTOR using LangChain: recursive cluster-and-summarize builds a summary
    tree; one collapsed vector index serves retrieval across all levels.

    Best for: Large corpora where some questions need fine-grained facts while
    others need section- or corpus-level synthesis — one index serves both.
    """

    TECHNIQUE_NAME = "raptor_rag"
    FRAMEWORK = "langchain"

    def _build_pipeline(self) -> None:
        """Initialize LLM, embeddings, and tree-building settings."""
        cfg = ConfigLoader.get()
        tech_cfg = cfg.get_technique_config("raptor_rag")

        self.llm = get_langchain_llm()
        self.embeddings = get_langchain_embeddings()
        self.text_splitter = get_text_splitter()
        self.top_k = cfg.retrieval.get("top_k", 5)

        self.max_depth = tech_cfg.get("max_depth", 3)
        self.cluster_size_min = tech_cfg.get("cluster_size_min", 2)
        self.max_clusters = tech_cfg.get("cluster_size_max", 10)
        self.summary_char_budget = tech_cfg.get("summary_chunk_size", 1000)
        self.clustering_algorithm = tech_cfg.get("clustering_algorithm", "gmm")

        if tech_cfg.get("dimensionality_reduction", "umap"):
            logger.warning(
                "[RAPTOR/LC] dimensionality_reduction is not implemented — "
                "clustering runs in full embedding space."
            )
        if self.clustering_algorithm != "gmm":
            logger.warning(
                f"[RAPTOR/LC] Unknown clustering '{self.clustering_algorithm}' "
                "— using deterministic 'agglomerative' instead."
            )
            self.clustering_algorithm = "agglomerative"

        self.tree_nodes: dict[str, dict] = {}
        self.vector_store = None

        logger.info(
            f"[RAPTOR/LC] max_depth={self.max_depth}, "
            f"cluster_size_min={self.cluster_size_min}, algo={self.clustering_algorithm}"
        )

    def index(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        """Chunk documents, recursively build the summary tree, index all levels."""
        logger.info(f"[RAPTOR/LC] Indexing {len(documents)} documents...")

        lc_docs = load_texts(documents, metadatas)
        chunks = self.text_splitter.split_documents(lc_docs)

        current_level = [
            {"id": f"L0-{i}", "text": c.page_content, "level": 0, "children": []}
            for i, c in enumerate(chunks)
        ]
        for node in current_level:
            self.tree_nodes[node["id"]] = node

        for level in range(1, self.max_depth + 1):
            parents = self._build_level(current_level, level)
            if _should_stop(level, len(parents), len(current_level), self.max_depth):
                break
            current_level = parents

        self._index_all_nodes()
        self._is_indexed = True
        logger.info(f"[RAPTOR/LC] Tree built: {len(self.tree_nodes)} total nodes ✓")

    def _build_level(self, nodes: list[dict], level: int) -> list[dict]:
        """Cluster one level's texts and summarize each sizable cluster upward."""
        texts = [n["text"] for n in nodes]
        vectors = self.embeddings.embed_documents(texts)
        labels = _cluster_texts(
            vectors, algorithm=self.clustering_algorithm, max_clusters=self.max_clusters
        )

        chain = SUMMARY_PROMPT | self.llm | StrOutputParser()
        parents: list[dict] = []
        for _cluster_idx, indices in enumerate(_group_by_label(labels)):
            if len(indices) < self.cluster_size_min:
                continue
            members = [nodes[i] for i in indices]
            joined = "\n\n".join(m["text"] for m in members)
            summary = _strip_thinking(
                chain.invoke({"text": _truncate(joined, self.summary_char_budget)})
            )
            parent = {
                "id": f"L{level}-{len(parents)}",
                "text": summary,
                "level": level,
                "children": [m["id"] for m in members],
            }
            self.tree_nodes[parent["id"]] = parent
            parents.append(parent)

        logger.info(
            f"[RAPTOR/LC] Level {level}: {len(labels)} items clustered into "
            f"{len(set(labels))} groups → {len(parents)} summary nodes"
        )
        return parents

    def _index_all_nodes(self) -> None:
        """Embed every tree node (all levels) into one collapsed vector store."""
        lc_documents = [
            LCDocument(
                page_content=node["text"],
                metadata={"node_id": node["id"], "level": node["level"]},
            )
            for node in self.tree_nodes.values()
        ]
        self.vector_store = build_langchain_vector_store(
            lc_documents, collection_name="raptor_rag"
        )
        levels: dict[int, int] = {}
        for node in self.tree_nodes.values():
            levels[node["level"]] = levels.get(node["level"], 0) + 1
        logger.info(f"[RAPTOR/LC] Collapsed index: nodes per level={dict(sorted(levels.items()))}")

    def _query(self, question: str) -> RAGResult:
        """Full RAPTOR query: dense retrieval over the collapsed tree, then generate."""
        if not self.vector_store:
            raise RuntimeError("Call index() before querying.")

        retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
        hits = retriever.invoke(question)

        contexts, source_levels = [], []
        for hit in hits:
            contexts.append(hit.page_content)
            source_levels.append(hit.metadata.get("level", 0))

        chain = ANSWER_PROMPT | self.llm | StrOutputParser()
        answer = chain.invoke({"context": "\n\n---\n\n".join(contexts), "question": question})

        return RAGResult(
            query=question,
            answer=answer,
            source_documents=[
                Document(content=hit.page_content, metadata=dict(hit.metadata))
                for hit in hits
            ],
            intermediate_steps=[{
                "step": "collapsed_tree_retrieval",
                "levels_retrieved": sorted(set(source_levels)),
                "num_sources": len(hits),
            }],
            metadata={
                "levels_retrieved": sorted(set(source_levels)),
                "tree_height": max((n["level"] for n in self.tree_nodes.values()), default=0),
                "total_tree_nodes": len(self.tree_nodes),
            },
        )
