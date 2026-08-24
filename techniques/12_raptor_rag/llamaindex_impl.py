"""
RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval — LlamaIndex
==========================================================================================
Same pipeline as the LangChain version: recursive cluster-and-summarize builds
a summary tree (GMM w/ BIC selection or deterministic agglomerative fallback),
and every node from every level lands in ONE vector index so retrieval spans
all abstraction levels at once -- implemented with direct LLM `.complete()`
calls for parity with the LangChain implementation.
"""

import logging
import re

from core.base_rag import BaseRAG, Document, RAGResult
from core.config_loader import ConfigLoader
from core.embeddings import get_llamaindex_embeddings
from core.llm_client import get_llamaindex_llm

logger = logging.getLogger(__name__)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning traces some LMStudio models emit in raw completions."""
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


SUMMARY_PROMPT = """Write a dense, self-contained summary of the following related \
passages. Capture every distinct fact, entity, and claim — the summary will be \
retrieved INSTEAD of these passages for big-picture questions.

Passages:
{text}

Summary:"""

ANSWER_PROMPT = """Answer the question based ONLY on the provided context. \
The context mixes original chunks with multi-level summaries — trust detail over \
summary when they conflict. If the context doesn't contain sufficient information, \
acknowledge the limitation.

Context:
{context}

Question: {question}
Answer:"""


class RaptorRAGLlamaIndex(BaseRAG):
    """
    RAPTOR using LlamaIndex: recursive cluster-and-summarize builds a summary
    tree; one collapsed vector index serves retrieval across all levels.

    Best for: Large corpora where some questions need fine-grained facts while
    others need section- or corpus-level synthesis — one index serves both.
    """

    TECHNIQUE_NAME = "raptor_rag"
    FRAMEWORK = "llamaindex"

    def _build_pipeline(self) -> None:
        """Initialize LLM, embeddings, and tree-building settings."""
        from llama_index.core import Settings
        from llama_index.core.node_parser import SentenceSplitter

        cfg = ConfigLoader.get()
        tech_cfg = cfg.get_technique_config("raptor_rag")
        doc_cfg = cfg.document

        self.llm = get_llamaindex_llm()
        Settings.llm = self.llm
        self.embed_model = get_llamaindex_embeddings()
        Settings.embed_model = self.embed_model
        Settings.node_parser = SentenceSplitter(
            chunk_size=doc_cfg.get("chunk_size", 512),
            chunk_overlap=doc_cfg.get("chunk_overlap", 50),
        )

        self.top_k = cfg.retrieval.get("top_k", 5)
        self.max_depth = tech_cfg.get("max_depth", 3)
        self.cluster_size_min = tech_cfg.get("cluster_size_min", 2)
        self.max_clusters = tech_cfg.get("cluster_size_max", 10)
        self.summary_char_budget = tech_cfg.get("summary_chunk_size", 1000)
        self.clustering_algorithm = tech_cfg.get("clustering_algorithm", "gmm")

        if tech_cfg.get("dimensionality_reduction", "umap"):
            logger.warning(
                "[RAPTOR/LI] dimensionality_reduction is not implemented — "
                "clustering runs in full embedding space."
            )
        if self.clustering_algorithm != "gmm":
            logger.warning(
                f"[RAPTOR/LI] Unknown clustering '{self.clustering_algorithm}' "
                "— using deterministic 'agglomerative' instead."
            )
            self.clustering_algorithm = "agglomerative"

        self.tree_nodes: dict[str, dict] = {}
        self.vector_index = None
        self.retriever = None

        logger.info(
            f"[RAPTOR/LI] max_depth={self.max_depth}, "
            f"cluster_size_min={self.cluster_size_min}, algo={self.clustering_algorithm}"
        )

    def index(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        """Chunk documents, recursively build the summary tree, index all levels."""
        from llama_index.core import Settings
        from llama_index.core.schema import Document as LIDocument

        logger.info(f"[RAPTOR/LI] Indexing {len(documents)} documents...")

        metas = metadatas or [{}] * len(documents)
        li_docs = [LIDocument(text=t, metadata=m) for t, m in zip(documents, metas, strict=True)]
        nodes = Settings.node_parser.get_nodes_from_documents(li_docs)

        current_level = [
            {"id": f"L0-{i}", "text": n.get_content(), "level": 0, "children": []}
            for i, n in enumerate(nodes)
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
        logger.info(f"[RAPTOR/LI] Tree built: {len(self.tree_nodes)} total nodes ✓")

    def _build_level(self, nodes: list[dict], level: int) -> list[dict]:
        """Cluster one level's texts and summarize each sizable cluster upward."""
        texts = [n["text"] for n in nodes]
        vectors = self.embed_model.get_text_embedding_batch(texts)
        labels = _cluster_texts(
            vectors, algorithm=self.clustering_algorithm, max_clusters=self.max_clusters
        )

        parents: list[dict] = []
        for _cluster_idx, indices in enumerate(_group_by_label(labels)):
            if len(indices) < self.cluster_size_min:
                continue
            members = [nodes[i] for i in indices]
            joined = "\n\n".join(m["text"] for m in members)
            raw = self.llm.complete(
                SUMMARY_PROMPT.format(text=_truncate(joined, self.summary_char_budget))
            ).text
            parent = {
                "id": f"L{level}-{len(parents)}",
                "text": _strip_thinking(raw),
                "level": level,
                "children": [m["id"] for m in members],
            }
            self.tree_nodes[parent["id"]] = parent
            parents.append(parent)

        logger.info(
            f"[RAPTOR/LI] Level {level}: {len(labels)} items clustered into "
            f"{len(set(labels))} groups → {len(parents)} summary nodes"
        )
        return parents

    def _index_all_nodes(self) -> None:
        """Embed every tree node (all levels) into one collapsed vector index."""
        from llama_index.core import VectorStoreIndex
        from llama_index.core.schema import Document as LIDocument

        li_documents = [
            LIDocument(
                text=node["text"],
                metadata={"node_id": node["id"], "level": node["level"]},
            )
            for node in self.tree_nodes.values()
        ]
        self.vector_index = VectorStoreIndex(li_documents, show_progress=True)
        self.retriever = self.vector_index.as_retriever(similarity_top_k=self.top_k)

        levels: dict[int, int] = {}
        for node in self.tree_nodes.values():
            levels[node["level"]] = levels.get(node["level"], 0) + 1
        logger.info(f"[RAPTOR/LI] Collapsed index: nodes per level={dict(sorted(levels.items()))}")

    def _query(self, question: str) -> RAGResult:
        """Full RAPTOR query: dense retrieval over the collapsed tree, then generate."""
        if not self.retriever:
            raise RuntimeError("Call index() before querying.")

        scored = self.retriever.retrieve(question)
        contexts, source_levels = [], []
        for hit in scored:
            contexts.append(hit.get_content())
            source_levels.append(hit.metadata.get("level", 0))

        response = self.llm.complete(
            ANSWER_PROMPT.format(context="\n\n---\n\n".join(contexts), question=question)
        )
        answer = _strip_thinking(response.text)

        return RAGResult(
            query=question,
            answer=answer,
            source_documents=[
                Document(content=hit.get_content(), metadata=dict(hit.metadata))
                for hit in scored
            ],
            intermediate_steps=[{
                "step": "collapsed_tree_retrieval",
                "levels_retrieved": sorted(set(source_levels)),
                "num_sources": len(scored),
            }],
            metadata={
                "levels_retrieved": sorted(set(source_levels)),
                "tree_height": max((n["level"] for n in self.tree_nodes.values()), default=0),
                "total_tree_nodes": len(self.tree_nodes),
            },
        )
