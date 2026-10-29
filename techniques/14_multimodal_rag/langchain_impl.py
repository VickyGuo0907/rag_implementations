"""
Multi-modal RAG: Text + image + table retrieval — LangChain Implementation
==========================================================================
Images and tables are first-class retrievable units. A vision LLM captions
images and summarizes tables at index time; those summaries are embedded next to
the text chunks (so any text embedding model works). At query time the original
tables and the actual images behind the retrieved captions are handed to the
vision LLM together with the text context, so it can read the chart, not just
its caption.

Shared pipeline pieces live in multimodal.py.

Reference: https://github.com/NirDiamant/RAG_Techniques
"""

import logging
from pathlib import Path

from langchain_core.documents import Document as LCDocument

from core.base_rag import BaseRAG, Document, RAGResult
from core.config_loader import ConfigLoader
from core.document_loader import get_text_splitter
from core.embeddings import get_langchain_embeddings
from core.vector_store import build_langchain_vector_store

from .multimodal import Hit, VisionLLM, build_context, build_items

logger = logging.getLogger(__name__)


class MultimodalRAGLangChain(BaseRAG):
    """
    Multi-modal RAG using LangChain: caption images / summarize tables into the
    index, retrieve across modalities, answer with text + original images.

    Best for: PDFs and markdown reports with charts, figures, and tables where
    the answer lives in a visual or tabular element rather than prose.
    """

    TECHNIQUE_NAME = "multimodal_rag"
    FRAMEWORK = "langchain"

    def _build_pipeline(self) -> None:
        """Initialize the vision LLM, embeddings, splitter, and modality settings."""
        cfg = ConfigLoader.get()
        self.tech_cfg = cfg.get_technique_config("multimodal_rag")
        self.vision = VisionLLM(cfg.lmstudio, self.tech_cfg.get("vision_model"))
        self.embeddings = get_langchain_embeddings()
        self.text_splitter = get_text_splitter()
        self.top_k = cfg.retrieval.get("top_k", 5)
        self.max_images = self.tech_cfg.get("max_answer_images", 2)
        self.vector_store = None
        logger.info(f"[MultimodalRAG/LC] vision_model={self.vision.model}, top_k={self.top_k}")

    def index(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        """Extract tables/images, describe them, then embed text + descriptions."""
        logger.info(f"[MultimodalRAG/LC] Indexing {len(documents)} documents...")
        items = build_items(documents, metadatas, self.vision, self.tech_cfg)

        chunks: list[LCDocument] = []
        for item in items:
            doc = LCDocument(page_content=item.text, metadata=item.metadata)
            chunks += self.text_splitter.split_documents([doc]) if item.kind == "text" else [doc]

        self.vector_store = build_langchain_vector_store(chunks, collection_name="multimodal_rag")
        self._is_indexed = True
        counts = {k: sum(i.kind == k for i in items) for k in ("text", "table", "image")}
        logger.info(f"[MultimodalRAG/LC] Indexed {len(chunks)} units ✓ (documents: {counts})")

    def _query(self, question: str) -> RAGResult:
        """Retrieve across modalities, then answer with text, tables, and images."""
        if not self.vector_store:
            raise RuntimeError("Call index() before querying.")

        docs = self.vector_store.as_retriever(search_kwargs={"k": self.top_k}).invoke(question)
        hits = [
            Hit(d.metadata.get("modality", "text"), d.page_content,
                d.metadata.get("raw", d.page_content), d.metadata)
            for d in docs
        ]
        context, images = build_context(hits, self.max_images)
        answer = self.vision.answer(question, context, images)

        return RAGResult(
            query=question,
            answer=answer,
            source_documents=[
                Document(content=h.text, metadata={**h.metadata, "modality": h.kind}) for h in hits
            ],
            intermediate_steps=[{
                "step": "multimodal_retrieval",
                "modalities": [h.kind for h in hits],
                "attached_images": [Path(p).name for p in images],
            }],
            metadata={
                "modalities_retrieved": sorted({h.kind for h in hits}),
                "images_attached": len(images),
            },
        )
