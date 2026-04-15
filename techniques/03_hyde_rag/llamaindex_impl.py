"""
HyDE RAG — LlamaIndex Implementation
=======================================
Uses LlamaIndex's built-in HyDEQueryTransform for a clean, pipeline-native
implementation of Hypothetical Document Embeddings.
"""

from typing import Dict, List, Optional
import logging

from core.base_rag import BaseRAG, RAGResult, Document
from core.config_loader import ConfigLoader
from core.llm_client import get_llamaindex_llm
from core.embeddings import get_llamaindex_embeddings

logger = logging.getLogger(__name__)


class HyDERAGLlamaIndex(BaseRAG):
    """
    HyDE RAG using LlamaIndex's HyDEQueryTransform.

    LlamaIndex natively supports HyDE via the HyDEQueryTransform,
    which plugs directly into the RetrieverQueryEngine pipeline.

    Best for: Vocabulary mismatch between queries and documents.
    """

    TECHNIQUE_NAME = "hyde_rag"
    FRAMEWORK = "llamaindex"

    def _build_pipeline(self) -> None:
        from llama_index.core import Settings

        cfg = ConfigLoader.get()
        Settings.llm = get_llamaindex_llm()
        Settings.embed_model = get_llamaindex_embeddings()

        self.top_k = cfg.retrieval.get("top_k", 5)
        self.vector_index = None
        self.query_engine = None

    def index(self, documents: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        from llama_index.core import VectorStoreIndex
        from llama_index.core.schema import Document as LIDocument

        metas = metadatas or [{}] * len(documents)
        li_docs = [LIDocument(text=t, metadata=m) for t, m in zip(documents, metas)]
        self.vector_index = VectorStoreIndex.from_documents(li_docs, show_progress=True)
        self._is_indexed = True
        logger.info(f"[HyDE/LI] Indexed {len(documents)} documents ✓")

    def _query(self, question: str) -> RAGResult:
        from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform
        from llama_index.core.query_engine import TransformQueryEngine

        if not self.vector_index:
            raise RuntimeError("Call index() before querying.")

        # Build HyDE-wrapped query engine
        hyde_transform = HyDEQueryTransform(include_original=True)
        base_engine = self.vector_index.as_query_engine(similarity_top_k=self.top_k)
        hyde_engine = TransformQueryEngine(base_engine, hyde_transform)

        response = hyde_engine.query(question)

        return RAGResult(
            query=question,
            answer=str(response),
            source_documents=[
                Document(
                    content=node.get_content(),
                    metadata=node.metadata,
                    score=node.score,
                )
                for node in (response.source_nodes or [])
            ],
            metadata={"transform": "HyDE"},
        )
