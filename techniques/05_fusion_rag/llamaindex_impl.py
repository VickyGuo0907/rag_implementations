"""
Fusion RAG (RAG-Fusion) — LlamaIndex Implementation
======================================================
Uses LlamaIndex's built-in QueryFusionRetriever in RECIPROCAL_RANK mode,
which natively implements the RAG-Fusion pattern: generate query variants,
retrieve with each configured retriever, and fuse all ranked lists via
Reciprocal Rank Fusion (RRF). Hybrid search is achieved by handing it both
a dense (vector) retriever and a sparse (BM25) retriever.

Note: QueryFusionRetriever's RRF implementation hard-codes k=60 (the value
the original RRF paper found near-optimal), which matches this project's
`rrf_k: 60` default in config.yaml.
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

ANSWER_PROMPT = """You are a knowledgeable assistant. Answer the question based ONLY on the \
provided context, which was assembled by fusing results from several query \
reformulations and retrieval methods via Reciprocal Rank Fusion. If the \
context doesn't contain sufficient information, acknowledge the limitation.

Context:
{context}

Question: {question}
Answer:"""


class FusionRAGLlamaIndex(BaseRAG):
    """
    Fusion RAG (RAG-Fusion) using LlamaIndex's QueryFusionRetriever.

    Combines multi-query expansion with hybrid dense+sparse retrieval, fusing
    every resulting ranked list via Reciprocal Rank Fusion (RRF) rather than
    simple union+dedup — documents that rank highly across multiple queries
    and/or retrieval methods surface to the top.

    Best for: Cases where a single retrieval pass misses relevant docs,
    diverse document sets, and production search systems that want robust
    ranking without a separate cross-encoder reranking stage.
    """

    TECHNIQUE_NAME = "fusion_rag"
    FRAMEWORK = "llamaindex"

    def _build_pipeline(self) -> None:
        """Initialize LLM, embeddings, and RAG-Fusion settings."""
        from llama_index.core import Settings
        from llama_index.core.node_parser import SentenceSplitter

        cfg = ConfigLoader.get()
        tech_cfg = cfg.get_technique_config("fusion_rag")
        doc_cfg = cfg.document

        self.llm = get_llamaindex_llm()
        Settings.llm = self.llm
        Settings.embed_model = get_llamaindex_embeddings()
        Settings.node_parser = SentenceSplitter(
            chunk_size=doc_cfg.get("chunk_size", 512),
            chunk_overlap=doc_cfg.get("chunk_overlap", 50),
        )

        self.top_k = cfg.retrieval.get("top_k", 5)
        self.num_queries = tech_cfg.get("num_queries", 4)  # total, including the original
        self.rrf_k = tech_cfg.get("rrf_k", 60)
        self.hybrid_search = tech_cfg.get("hybrid_search", True)

        self.vector_index = None
        self.fusion_retriever = None

        logger.info(
            f"[FusionRAG/LI] num_queries={self.num_queries}, rrf_k={self.rrf_k}, "
            f"hybrid_search={self.hybrid_search}"
        )

    def index(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        """Build a VectorStoreIndex, optionally paired with a BM25 sparse retriever, fused via RRF."""  # noqa: E501
        from llama_index.core import VectorStoreIndex
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
        from llama_index.core.schema import Document as LIDocument

        logger.info(f"[FusionRAG/LI] Indexing {len(documents)} documents...")

        metas = metadatas or [{}] * len(documents)
        li_docs = [LIDocument(text=t, metadata=m) for t, m in zip(documents, metas, strict=True)]

        self.vector_index = VectorStoreIndex.from_documents(li_docs, show_progress=True)
        vector_retriever = self.vector_index.as_retriever(similarity_top_k=self.top_k)

        retrievers = [vector_retriever]
        if self.hybrid_search:
            bm25_retriever = self._build_bm25_retriever()
            if bm25_retriever is not None:
                retrievers.append(bm25_retriever)

        self.fusion_retriever = QueryFusionRetriever(
            retrievers=retrievers,
            llm=self.llm,
            mode=FUSION_MODES.RECIPROCAL_RANK,
            similarity_top_k=self.top_k,
            num_queries=self.num_queries,
        )

        self._is_indexed = True
        logger.info(f"[FusionRAG/LI] Indexed ✓ ({len(retrievers)} retriever(s) fused via RRF)")

    def _build_bm25_retriever(self):
        """Build the sparse (keyword) retriever half of hybrid search."""
        try:
            from llama_index.retrievers.bm25 import BM25Retriever

            docstore = self.vector_index.docstore
            return BM25Retriever.from_defaults(
                docstore=docstore,
                similarity_top_k=self.top_k,
            )
        except Exception as e:
            logger.warning(f"[FusionRAG/LI] BM25 retriever unavailable: {e}. Falling back to dense-only retrieval.")  # noqa: E501
            self.hybrid_search = False
            return None

    def _query(self, question: str) -> RAGResult:
        """Full RAG-Fusion pipeline: QueryFusionRetriever (multi-query + hybrid retrieval + RRF) → answer."""  # noqa: E501
        if not self.fusion_retriever:
            raise RuntimeError("Call index() before querying.")

        retrieved_nodes = self.fusion_retriever.retrieve(question)
        top_nodes = retrieved_nodes[: self.top_k]

        context = "\n\n---\n\n".join(node.get_content() for node in top_nodes)
        response = self.llm.complete(ANSWER_PROMPT.format(context=context, question=question))
        answer = _strip_thinking(response.text)

        return RAGResult(
            query=question,
            answer=answer,
            source_documents=[
                Document(content=node.get_content(), metadata=node.metadata, score=node.score)
                for node in top_nodes
            ],
            metadata={
                "num_queries": self.num_queries,
                "hybrid_search": self.hybrid_search,
                "rrf_k": self.rrf_k,
                "num_docs_retrieved": len(retrieved_nodes),
            },
        )
