"""
Adaptive RAG: Dynamic strategy routing by query complexity — LangChain Implementation
==========================================================================================
Classifies query complexity and routes to the appropriate RAG strategy: no-retrieval, naive, or advanced/agentic.

Status: 🔧 Stub — follow the pattern from 01_naive_rag/langchain_impl.py

Reference: https://github.com/NirDiamant/RAG_Techniques
"""

from typing import Dict, List, Optional
import logging
from core.base_rag import BaseRAG, RAGResult, Document
from core.config_loader import ConfigLoader
from core.llm_client import get_langchain_llm
from core.embeddings import get_langchain_embeddings
from core.document_loader import load_texts, get_text_splitter

logger = logging.getLogger(__name__)


class AdaptiveRAGLangChain(BaseRAG):
    """
    Classifies query complexity and routes to the appropriate RAG strategy: no-retrieval, naive, or advanced/agentic.

    Best for: TODO — see README.md for use case guide.
    """

    TECHNIQUE_NAME = "adaptive_rag"
    FRAMEWORK = "langchain"

    def _build_pipeline(self) -> None:
        cfg = ConfigLoader.get()
        self.llm = get_langchain_llm()
        self.embeddings = get_langchain_embeddings()
        self.text_splitter = get_text_splitter()
        self.top_k = cfg.retrieval.get("top_k", 5)
        self.vector_store = None
        # TODO: Initialize technique-specific components
        logger.info(f"[adaptive_rag/LC] Initialized (stub)")

    def index(self, documents: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        """TODO: Implement chunking, embedding, and indexing logic."""
        raise NotImplementedError(
            "Implement index() for AdaptiveRAGLangChain. "
            "See 01_naive_rag/langchain_impl.py for reference pattern."
        )

    def _query(self, question: str) -> RAGResult:
        """TODO: Implement the full adaptive_rag query pipeline."""
        raise NotImplementedError(
            "Implement _query() for AdaptiveRAGLangChain. "
            "See 01_naive_rag/langchain_impl.py for reference pattern."
        )


if __name__ == "__main__":
    # TODO: Add demo code after implementing above methods
    print("AdaptiveRAGLangChain stub — implement _build_pipeline, index, and _query first.")
