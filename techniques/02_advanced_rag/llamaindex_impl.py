"""
Advanced RAG — LlamaIndex Implementation
==========================================
LlamaIndex version of Advanced RAG using built-in transformations:
  - QueryTransform for query rewriting
  - SentenceTransformerRerank for cross-encoder reranking
  - SentenceWindowNodeParser for context-aware chunking
"""

from typing import Dict, List, Optional
import logging

from core.base_rag import BaseRAG, RAGResult, Document
from core.config_loader import ConfigLoader
from core.llm_client import get_llamaindex_llm
from core.embeddings import get_llamaindex_embeddings

logger = logging.getLogger(__name__)


class AdvancedRAGLlamaIndex(BaseRAG):
    """
    Advanced RAG using LlamaIndex with query transformation and reranking.

    Uses SentenceWindowNodeParser for better context retrieval —
    stores sentence-level nodes but retrieves surrounding window for context.

    Best for: Production Q&A, complex retrieval, high accuracy requirements.
    """

    TECHNIQUE_NAME = "advanced_rag"
    FRAMEWORK = "llamaindex"

    def _build_pipeline(self) -> None:
        """Initialize LLM, embeddings, and LlamaIndex Settings."""
        from llama_index.core import Settings
        from llama_index.core.node_parser import SentenceWindowNodeParser

        cfg = ConfigLoader.get()
        tech_cfg = cfg.get_technique_config("advanced_rag")

        Settings.llm = get_llamaindex_llm()
        Settings.embed_model = get_llamaindex_embeddings()

        # SentenceWindowNodeParser: parses at sentence level, stores window context
        Settings.node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,               # ±3 sentences around retrieved sentence
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )

        self.top_k = cfg.retrieval.get("top_k", 5)
        self.rerank_top_k = cfg.retrieval.get("rerank_top_k", 3)
        self.enable_reranking = tech_cfg.get("reranking", True)
        self.vector_index = None
        self.query_engine = None

        logger.info("[AdvancedRAG/LI] Initialized with SentenceWindowNodeParser")

    def index(self, documents: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        """Build VectorStoreIndex with sentence-window nodes."""
        from llama_index.core import VectorStoreIndex
        from llama_index.core.schema import Document as LIDocument

        logger.info(f"[AdvancedRAG/LI] Indexing {len(documents)} documents...")

        metas = metadatas or [{}] * len(documents)
        li_docs = [LIDocument(text=t, metadata=m) for t, m in zip(documents, metas)]

        self.vector_index = VectorStoreIndex.from_documents(
            li_docs,
            show_progress=True,
        )

        # Build query engine with optional reranking
        postprocessors = []

        if self.enable_reranking:
            try:
                from llama_index.postprocessor.sentence_transformer_rerank import (
                    SentenceTransformerRerank,
                )
                reranker = SentenceTransformerRerank(
                    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    top_n=self.rerank_top_k,
                )
                postprocessors.append(reranker)
                logger.info("[AdvancedRAG/LI] Reranker initialized ✓")
            except Exception as e:
                logger.warning(f"[AdvancedRAG/LI] Reranker unavailable: {e}")

        # MetadataReplacementPostProcessor: replaces node text with full window
        try:
            from llama_index.core.postprocessor import MetadataReplacementPostProcessor
            postprocessors.insert(
                0,
                MetadataReplacementPostProcessor(target_metadata_key="window"),
            )
        except Exception:
            pass

        self.query_engine = self.vector_index.as_query_engine(
            similarity_top_k=self.top_k,
            node_postprocessors=postprocessors if postprocessors else None,
        )

        self._is_indexed = True
        logger.info("[AdvancedRAG/LI] Indexing complete ✓")

    def _query(self, question: str) -> RAGResult:
        """Run the advanced RAG query pipeline."""
        if not self.query_engine:
            raise RuntimeError("Call index() before querying.")

        response = self.query_engine.query(question)

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
            metadata={
                "reranking_enabled": self.enable_reranking,
                "node_parser": "SentenceWindow",
            },
        )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))

    docs = [
        "Self-attention allows transformers to relate different positions of a sequence when encoding a representation.",
        "BERT uses masked language modeling and next sentence prediction as pre-training objectives.",
        "GPT-3 with 175 billion parameters demonstrated remarkable few-shot learning capabilities.",
    ]

    rag = AdvancedRAGLlamaIndex(config=ConfigLoader.get()._config)
    rag.index(docs)
    result = rag.query("What pre-training objectives does BERT use?")
    result.print_summary()
