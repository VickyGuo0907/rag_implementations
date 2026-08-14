"""
Multi-modal RAG: Retrieval across text, images, and tables — LangChain Implementation
======================================================================================
Processes and retrieves across text, images, and tables by generating
summaries/captions for non-text content.

Status: 🔧 Stub — follow the pattern from 01_naive_rag/langchain_impl.py

Reference: https://github.com/NirDiamant/RAG_Techniques
"""

import logging

from core.base_rag import BaseRAG, RAGResult
from core.config_loader import ConfigLoader
from core.document_loader import get_text_splitter
from core.embeddings import get_langchain_embeddings
from core.llm_client import get_langchain_llm

logger = logging.getLogger(__name__)


class MultimodalRAGLangChain(BaseRAG):
    """
    Processes and retrieves across text, images, and tables by generating
    summaries/captions for non-text content.

    Best for: TODO — see README.md for use case guide.
    """

    TECHNIQUE_NAME = "multimodal_rag"
    FRAMEWORK = "langchain"

    def _build_pipeline(self) -> None:
        cfg = ConfigLoader.get()
        self.llm = get_langchain_llm()
        self.embeddings = get_langchain_embeddings()
        self.text_splitter = get_text_splitter()
        self.top_k = cfg.retrieval.get("top_k", 5)
        self.vector_store = None
        # TODO: Initialize technique-specific components
        logger.info("[multimodal_rag/LC] Initialized (stub)")

    def index(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        """TODO: Implement chunking, embedding, and indexing logic."""
        raise NotImplementedError(
            "Implement index() for MultimodalRAGLangChain. "
            "See 01_naive_rag/langchain_impl.py for reference pattern."
        )

    def _query(self, question: str) -> RAGResult:
        """TODO: Implement the full multimodal_rag query pipeline."""
        raise NotImplementedError(
            "Implement _query() for MultimodalRAGLangChain. "
            "See 01_naive_rag/langchain_impl.py for reference pattern."
        )


if __name__ == "__main__":
    # TODO: Add demo code after implementing above methods
    print("MultimodalRAGLangChain stub — implement _build_pipeline, index, and _query first.")
