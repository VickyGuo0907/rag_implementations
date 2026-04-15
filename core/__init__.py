"""
Core Infrastructure Package
============================
Provides shared utilities for all RAG implementations:
  - ConfigLoader     : Singleton YAML config accessor
  - BaseRAG          : Abstract base class for all RAG techniques
  - LLM Client       : LMStudio LLM factory (LangChain + LlamaIndex)
  - Embeddings       : Embedding model factory
  - Vector Store     : Vector store factory (ChromaDB, FAISS, Qdrant)
  - Document Loader  : Document loading and chunking utilities
"""

from core.config_loader import ConfigLoader
from core.base_rag import BaseRAG, RAGResult, Document

__all__ = ["ConfigLoader", "BaseRAG", "RAGResult", "Document"]
