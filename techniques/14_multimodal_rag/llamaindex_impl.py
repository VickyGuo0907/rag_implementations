"""
Multi-modal RAG: Text + image + table retrieval — LlamaIndex Implementation
===========================================================================
Same pipeline as the LangChain version (shared multimodal.py): a vision LLM
captions images and summarizes tables into the text index, and the original
tables and images are handed back to the vision LLM at answer time.
"""

import logging
from pathlib import Path

from core.base_rag import BaseRAG, Document, RAGResult
from core.config_loader import ConfigLoader
from core.embeddings import get_llamaindex_embeddings
from core.llm_client import get_llamaindex_llm

from .multimodal import Hit, VisionLLM, build_context, build_items

logger = logging.getLogger(__name__)

_NON_EMBEDDED_KEYS = ["raw", "source", "modality"]


class MultimodalRAGLlamaIndex(BaseRAG):
    """
    Multi-modal RAG using LlamaIndex: caption images / summarize tables into the
    index, retrieve across modalities, answer with text + original images.

    Best for: PDFs and markdown reports with charts, figures, and tables where
    the answer lives in a visual or tabular element rather than prose.
    """

    TECHNIQUE_NAME = "multimodal_rag"
    FRAMEWORK = "llamaindex"

    def _build_pipeline(self) -> None:
        """Initialize the vision LLM, embeddings, node parser, and modality settings."""
        from llama_index.core import Settings
        from llama_index.core.node_parser import SentenceSplitter

        cfg = ConfigLoader.get()
        doc_cfg = cfg.document
        self.tech_cfg = cfg.get_technique_config("multimodal_rag")
        self.vision = VisionLLM(cfg.lmstudio, self.tech_cfg.get("vision_model"))

        Settings.llm = get_llamaindex_llm()
        Settings.embed_model = get_llamaindex_embeddings()
        self.node_parser = Settings.node_parser = SentenceSplitter(
            chunk_size=doc_cfg.get("chunk_size", 512),
            chunk_overlap=doc_cfg.get("chunk_overlap", 50),
        )
        self.top_k = cfg.retrieval.get("top_k", 5)
        self.max_images = self.tech_cfg.get("max_answer_images", 2)
        self.vector_index = None
        self.retriever = None
        logger.info(f"[MultimodalRAG/LI] vision_model={self.vision.model}, top_k={self.top_k}")

    def index(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        """Extract tables/images, describe them, then embed text + descriptions."""
        from llama_index.core import VectorStoreIndex
        from llama_index.core.schema import Document as LIDocument
        from llama_index.core.schema import TextNode

        logger.info(f"[MultimodalRAG/LI] Indexing {len(documents)} documents...")
        items = build_items(documents, metadatas, self.vision, self.tech_cfg)

        nodes: list = []
        for item in items:
            if item.kind == "text":
                nodes += self.node_parser.get_nodes_from_documents(
                    [LIDocument(text=item.text, metadata=item.metadata,
                                excluded_embed_metadata_keys=_NON_EMBEDDED_KEYS,
                                excluded_llm_metadata_keys=_NON_EMBEDDED_KEYS)]
                )
            else:
                nodes.append(TextNode(text=item.text, metadata=item.metadata,
                                      excluded_embed_metadata_keys=_NON_EMBEDDED_KEYS,
                                      excluded_llm_metadata_keys=_NON_EMBEDDED_KEYS))

        self.vector_index = VectorStoreIndex(nodes, show_progress=True)
        self.retriever = self.vector_index.as_retriever(similarity_top_k=self.top_k)
        self._is_indexed = True
        logger.info(f"[MultimodalRAG/LI] Indexed {len(nodes)} units ✓")

    def _query(self, question: str) -> RAGResult:
        """Retrieve across modalities, then answer with text, tables, and images."""
        if not self.retriever:
            raise RuntimeError("Call index() before querying.")

        nodes = self.retriever.retrieve(question)
        hits = [
            Hit(n.metadata.get("modality", "text"), n.get_content(),
                n.metadata.get("raw", n.get_content()), n.metadata)
            for n in nodes
        ]
        context, images = build_context(hits, self.max_images)
        answer = self.vision.answer(question, context, images)

        return RAGResult(
            query=question,
            answer=answer,
            source_documents=[
                Document(content=h.text, metadata={**h.metadata, "modality": h.kind}, score=n.score)
                for h, n in zip(hits, nodes, strict=True)
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
