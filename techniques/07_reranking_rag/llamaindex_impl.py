"""
Reranking RAG — LlamaIndex Implementation
============================================
Retrieves a wide candidate pool via standard dense vector search, then
reranks with LlamaIndex's SentenceTransformerRerank node postprocessor before
generation. Same isolated single-improvement idea as the LangChain version —
no query rewriting, no multi-query, just a cross-encoder rerank stage.
"""

from typing import Dict, List, Optional
import logging
import re

from core.base_rag import BaseRAG, RAGResult, Document
from core.config_loader import ConfigLoader
from core.llm_client import get_llamaindex_llm
from core.embeddings import get_llamaindex_embeddings

logger = logging.getLogger(__name__)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning traces some LMStudio models emit in raw completions."""
    return _THINK_TAG_RE.sub("", text).strip()


class RerankingRAGLlamaIndex(BaseRAG):
    """
    Reranking RAG using LlamaIndex.

    Pipeline:
        Query → Vector Search (wide candidate pool) →
        SentenceTransformerRerank (cross-encoder) → Top-K → LLM → Answer

    Best for: High-precision requirements where the initial bi-encoder
    similarity search ranks relevant documents outside the top-K.
    """

    TECHNIQUE_NAME = "reranking_rag"
    FRAMEWORK = "llamaindex"

    def _build_pipeline(self) -> None:
        """Initialize LLM, embeddings, and reranking settings."""
        from llama_index.core import Settings
        from llama_index.core.node_parser import SentenceSplitter

        cfg = ConfigLoader.get()
        tech_cfg = cfg.get_technique_config("reranking_rag")
        doc_cfg = cfg.document

        self.llm = get_llamaindex_llm()
        Settings.llm = self.llm
        Settings.embed_model = get_llamaindex_embeddings()
        Settings.node_parser = SentenceSplitter(
            chunk_size=doc_cfg.get("chunk_size", 512),
            chunk_overlap=doc_cfg.get("chunk_overlap", 50),
        )

        self.initial_k = tech_cfg.get("initial_top_k", 20)
        self.rerank_top_k = tech_cfg.get("rerank_top_k", 5)
        self.reranker_model_name = tech_cfg.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.vector_index = None
        self.query_engine = None
        self.reranking_enabled = True

        logger.info(
            f"[Reranking/LI] initial_top_k={self.initial_k}, rerank_top_k={self.rerank_top_k}, "
            f"model={self.reranker_model_name}"
        )

    def index(self, documents: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        """Build a VectorStoreIndex and a query engine with the cross-encoder reranker attached."""
        from llama_index.core import VectorStoreIndex
        from llama_index.core.schema import Document as LIDocument

        logger.info(f"[Reranking/LI] Indexing {len(documents)} documents...")

        metas = metadatas or [{}] * len(documents)
        li_docs = [LIDocument(text=t, metadata=m) for t, m in zip(documents, metas)]

        self.vector_index = VectorStoreIndex.from_documents(li_docs, show_progress=True)

        postprocessors = []
        try:
            from llama_index.postprocessor.sbert_rerank import (
                SentenceTransformerRerank,
            )
            reranker = SentenceTransformerRerank(
                model=self.reranker_model_name,
                top_n=self.rerank_top_k,
            )
            postprocessors.append(reranker)
            logger.info(f"[Reranking/LI] Reranker loaded: {self.reranker_model_name}")
        except Exception as e:
            logger.warning(f"[Reranking/LI] Reranker unavailable: {e}. Falling back to dense-only top-K.")
            self.reranking_enabled = False

        self.query_engine = self.vector_index.as_query_engine(
            similarity_top_k=self.initial_k,
            node_postprocessors=postprocessors if postprocessors else None,
        )

        self._is_indexed = True
        logger.info("[Reranking/LI] Indexing complete ✓")

    def _query(self, question: str) -> RAGResult:
        """Run the reranking query engine."""
        if not self.query_engine:
            raise RuntimeError("Call index() before querying.")

        response = self.query_engine.query(question)
        answer = _strip_thinking(str(response))

        top_nodes = (response.source_nodes or [])[: self.rerank_top_k]

        return RAGResult(
            query=question,
            answer=answer,
            source_documents=[
                Document(content=node.get_content(), metadata=node.metadata, score=node.score)
                for node in top_nodes
            ],
            metadata={
                "initial_top_k": self.initial_k,
                "rerank_top_k": self.rerank_top_k,
                "reranking_enabled": self.reranking_enabled,
                "reranker_model": self.reranker_model_name,
            },
        )
