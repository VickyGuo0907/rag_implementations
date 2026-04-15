"""
{TECHNIQUE_NAME} — {FRAMEWORK} Implementation
{DESCRIPTION}

Status: 🔧 Stub — implement by following the pattern in 01_naive_rag/

Steps to implement:
  1. Inherit from BaseRAG
  2. Set TECHNIQUE_NAME and FRAMEWORK class attributes
  3. Implement _build_pipeline() — initialize LLM, embeddings, any extra components
  4. Implement index() — chunk, embed, store documents
  5. Implement _query() — full RAG pipeline, return RAGResult

Reference: https://github.com/NirDiamant/RAG_Techniques
"""

from typing import Dict, List, Optional
from core.base_rag import BaseRAG, RAGResult, Document
from core.config_loader import ConfigLoader
from core.llm_client import get_langchain_llm       # or get_llamaindex_llm
from core.embeddings import get_langchain_embeddings # or get_llamaindex_embeddings
from core.document_loader import load_texts, get_text_splitter


class {CLASS_NAME}(BaseRAG):
    """
    {TECHNIQUE_DESCRIPTION}

    Best for: TODO
    """

    TECHNIQUE_NAME = "{technique_key}"
    FRAMEWORK = "{framework}"

    def _build_pipeline(self) -> None:
        cfg = ConfigLoader.get()
        self.llm = get_langchain_llm()
        self.embeddings = get_langchain_embeddings()
        self.text_splitter = get_text_splitter()
        self.top_k = cfg.retrieval.get("top_k", 5)
        # TODO: Add technique-specific components here

    def index(self, documents: List[str], metadatas=None) -> None:
        # TODO: Implement indexing logic
        raise NotImplementedError("Implement index() for {CLASS_NAME}")

    def _query(self, question: str) -> RAGResult:
        # TODO: Implement query logic
        raise NotImplementedError("Implement _query() for {CLASS_NAME}")
