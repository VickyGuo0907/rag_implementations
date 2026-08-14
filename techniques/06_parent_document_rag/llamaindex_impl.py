"""
Parent Document RAG — LlamaIndex Implementation
==================================================
Same small-to-big retrieval idea as the LangChain version, implemented
directly with LlamaIndex's node/splitter primitives rather than its
auto-merging hierarchy machinery (which supports N-level merge thresholds --
more machinery than this technique needs). Each document is split into large
parent chunks; each parent is further split into small child chunks that get
embedded and searched. Retrieval hits map back to their parent via metadata,
and the (deduplicated) parent text is what actually grounds the answer.
"""

import logging
import re

from core.base_rag import BaseRAG, Document, RAGResult
from core.config_loader import ConfigLoader
from core.embeddings import get_llamaindex_embeddings
from core.llm_client import get_llamaindex_llm

logger = logging.getLogger(__name__)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning traces some LMStudio models emit in raw completions."""
    return _THINK_TAG_RE.sub("", text).strip()


ANSWER_PROMPT = """You are a knowledgeable assistant. Answer the question based ONLY on the \
provided context, which consists of full parent documents surrounding the \
specific passages that matched your query. If the context doesn't contain \
sufficient information, acknowledge the limitation.

Context:
{context}

Question: {question}
Answer:"""


class ParentDocumentRAGLlamaIndex(BaseRAG):
    """
    Parent Document RAG using LlamaIndex.

    Pipeline:
        Documents → Parent Splitter (large chunks) → Child Splitter (small chunks) →
        Embed children only → Query → Retrieve by child similarity → Return parents → LLM → Answer

    Best for: Long documents needing precise retrieval, limited context windows,
    technical manuals/books/reports where a small chunk alone lacks context.
    """

    TECHNIQUE_NAME = "parent_document_rag"
    FRAMEWORK = "llamaindex"

    def _build_pipeline(self) -> None:
        """Initialize LLM, embeddings, and parent/child chunk sizes."""
        from llama_index.core import Settings

        cfg = ConfigLoader.get()
        tech_cfg = cfg.get_technique_config("parent_document_rag")
        doc_cfg = cfg.document

        self.llm = get_llamaindex_llm()
        Settings.llm = self.llm
        Settings.embed_model = get_llamaindex_embeddings()

        self.parent_chunk_size = tech_cfg.get("parent_chunk_size", 2000)
        self.child_chunk_size = tech_cfg.get("child_chunk_size", 200)
        self.child_chunk_overlap = doc_cfg.get("chunk_overlap", 50)
        self.top_k = cfg.retrieval.get("top_k", 5)

        self.parent_lookup: dict[str, str] = {}
        self.vector_index = None
        self.retriever = None

        logger.info(
            f"[ParentDoc/LI] parent_chunk_size={self.parent_chunk_size}, "
            f"child_chunk_size={self.child_chunk_size}"
        )

    def index(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        """Split into parent chunks, then child chunks; embed only the children."""
        from llama_index.core import VectorStoreIndex
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.schema import Document as LIDocument

        logger.info(f"[ParentDoc/LI] Indexing {len(documents)} documents...")

        metas = metadatas or [{}] * len(documents)
        li_docs = [LIDocument(text=t, metadata=m) for t, m in zip(documents, metas, strict=True)]

        parent_splitter = SentenceSplitter(chunk_size=self.parent_chunk_size, chunk_overlap=0)
        child_splitter = SentenceSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_chunk_overlap,
        )

        parent_nodes = parent_splitter.get_nodes_from_documents(li_docs)

        self.parent_lookup = {}
        child_nodes = []
        for parent in parent_nodes:
            self.parent_lookup[parent.node_id] = parent.get_content()
            children = child_splitter.get_nodes_from_documents(
                [LIDocument(text=parent.get_content(), metadata=parent.metadata)]
            )
            for child in children:
                child.metadata["parent_id"] = parent.node_id
                child_nodes.append(child)

        self.vector_index = VectorStoreIndex(nodes=child_nodes)
        self.retriever = self.vector_index.as_retriever(similarity_top_k=self.top_k)

        self._is_indexed = True
        logger.info(f"[ParentDoc/LI] Indexed {len(parent_nodes)} parents → {len(child_nodes)} child chunks ✓")  # noqa: E501

    def _query(self, question: str) -> RAGResult:
        """Retrieve by child-chunk similarity, but return the (deduplicated) parent chunks for context."""  # noqa: E501
        if not self.retriever:
            raise RuntimeError("Call index() before querying.")

        child_hits = self.retriever.retrieve(question)

        seen = set()
        parent_texts = []
        for node in child_hits:
            parent_id = node.metadata.get("parent_id")
            if parent_id and parent_id not in seen:
                seen.add(parent_id)
                parent_texts.append(self.parent_lookup.get(parent_id, node.get_content()))

        context = "\n\n---\n\n".join(parent_texts)
        response = self.llm.complete(ANSWER_PROMPT.format(context=context, question=question))
        answer = _strip_thinking(response.text)

        return RAGResult(
            query=question,
            answer=answer,
            source_documents=[Document(content=text, metadata={}) for text in parent_texts],
            metadata={
                "parent_chunk_size": self.parent_chunk_size,
                "child_chunk_size": self.child_chunk_size,
                "num_child_hits": len(child_hits),
                "num_parents_returned": len(parent_texts),
            },
        )
