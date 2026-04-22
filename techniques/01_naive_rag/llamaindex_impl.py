"""
Naive RAG — LlamaIndex Implementation
=======================================
Same baseline pipeline implemented with LlamaIndex:
  1. Load documents → node parser (chunking)
  2. Embed nodes → VectorStoreIndex
  3. Query engine: embed query → retrieve top-K → LLM generates answer

LlamaIndex provides higher-level abstractions that handle much of the
boilerplate, making the code very concise.
"""

from typing import Dict, List, Optional
import logging

from core.base_rag import BaseRAG, RAGResult, Document
from core.config_loader import ConfigLoader
from core.llm_client import get_llamaindex_llm
from core.embeddings import get_llamaindex_embeddings

logger = logging.getLogger(__name__)


class NaiveRAGLlamaIndex(BaseRAG):
    """
    Naive (Simple) RAG using LlamaIndex.

    LlamaIndex automatically handles:
      - Document chunking via NodeParser
      - Embedding and indexing via VectorStoreIndex
      - Retrieval + generation via QueryEngine

    Best for: Prototyping, baseline testing, simple Q&A over small document sets.
    """

    TECHNIQUE_NAME = "naive_rag"
    FRAMEWORK = "llamaindex"

    def _build_pipeline(self) -> None:
        """Initialize LLM and embeddings; configure LlamaIndex Settings."""
        from llama_index.core import Settings
        from llama_index.core.node_parser import SentenceSplitter

        cfg = ConfigLoader.get()
        doc_cfg = cfg.document

        # Set global LlamaIndex settings (replaces ServiceContext in v0.10+)
        Settings.llm = get_llamaindex_llm()
        Settings.embed_model = get_llamaindex_embeddings()
        Settings.node_parser = SentenceSplitter(
            chunk_size=doc_cfg.get("chunk_size", 512),
            chunk_overlap=doc_cfg.get("chunk_overlap", 50),
        )

        self.top_k = cfg.retrieval.get("top_k", 5)
        self.query_engine = None

        logger.info(f"[NaiveRAG/LI] Initialized | model={cfg.lmstudio.get('model')}")

    def index(self, documents: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        """
        Build a VectorStoreIndex from raw text strings.

        Args:
            documents: Raw text strings to index.
            metadatas: Optional per-document metadata.
        """
        from llama_index.core import VectorStoreIndex
        from llama_index.core.schema import Document as LIDocument

        logger.info(f"[NaiveRAG/LI] Indexing {len(documents)} documents...")

        # 1. Wrap strings in LlamaIndex Document objects
        metas = metadatas or [{}] * len(documents)
        li_docs = [
            LIDocument(text=text, metadata=meta)
            for text, meta in zip(documents, metas)
        ]

        # 2. Build vector index (handles chunking + embedding automatically)
        self.vector_index = VectorStoreIndex.from_documents(
            li_docs,
            show_progress=True,
        )

        # 3. Create query engine
        self.query_engine = self.vector_index.as_query_engine(
            similarity_top_k=self.top_k,
            streaming=False,
        )

        self._is_indexed = True
        logger.info("[NaiveRAG/LI] Indexing complete ✓")

    def _query(self, question: str) -> RAGResult:
        """Run the LlamaIndex naive RAG query engine."""
        if not self.query_engine:
            raise RuntimeError("Call index() before querying.")

        response = self.query_engine.query(question)

        # Extract source nodes for attribution
        source_documents = [
            Document(
                content=node.get_content(),
                metadata=node.metadata,
                score=node.score,
            )
            for node in (response.source_nodes or [])
        ]

        return RAGResult(
            query=question,
            answer=str(response),
            source_documents=source_documents,
            metadata={"response_metadata": response.metadata or {}},
        )
