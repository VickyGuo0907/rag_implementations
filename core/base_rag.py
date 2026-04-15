"""
Base RAG Abstract Class
=======================
All RAG technique implementations inherit from BaseRAG.
Provides a consistent interface across LangChain and LlamaIndex implementations.
Includes built-in timing, logging, and result formatting.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared Data Models
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """Represents a retrieved document chunk."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None
    doc_id: Optional[str] = None

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return f"Document(score={self.score:.3f}, preview='{preview}...')"


@dataclass
class RAGResult:
    """Standardized result object returned by all RAG implementations."""
    query: str
    answer: str
    source_documents: List[Document] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics (auto-populated by BaseRAG)
    latency_ms: float = 0.0
    technique: str = ""
    framework: str = ""            # "langchain" or "llamaindex"

    # Intermediate steps (for inspection / debugging)
    intermediate_steps: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "technique": self.technique,
            "framework": self.framework,
            "latency_ms": self.latency_ms,
            "num_sources": len(self.source_documents),
            "sources": [
                {"content": d.content[:200], "score": d.score, "metadata": d.metadata}
                for d in self.source_documents
            ],
            "metadata": self.metadata,
        }

    def print_summary(self) -> None:
        """Print a formatted result summary."""
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"  Technique : {self.technique} ({self.framework})")
        print(f"  Query     : {self.query}")
        print(f"  Latency   : {self.latency_ms:.1f} ms")
        print(f"  Sources   : {len(self.source_documents)}")
        print(sep)
        print(f"\n{self.answer}\n")
        if self.source_documents:
            print("── Retrieved Sources ──")
            for i, doc in enumerate(self.source_documents, 1):
                score_str = f"[score={doc.score:.3f}] " if doc.score else ""
                print(f"  [{i}] {score_str}{doc.content[:120]}...")
        print()


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------

class BaseRAG(ABC):
    """
    Abstract base class for all RAG technique implementations.

    Every RAG implementation (LangChain or LlamaIndex) must:
      1. Inherit from BaseRAG
      2. Implement `_build_pipeline()` to initialize the chain/engine
      3. Implement `index(documents)` to ingest documents
      4. Implement `_query(question)` to run the RAG pipeline

    The public `query()` method wraps `_query()` with timing, logging,
    and result normalization automatically.
    """

    TECHNIQUE_NAME: str = "base"      # Override in each subclass
    FRAMEWORK: str = "base"           # "langchain" or "llamaindex"

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Full config dict (from ConfigLoader) or technique-specific subset.
        """
        self.config = config
        self.pipeline = None
        self._is_indexed = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self._build_pipeline()

    @abstractmethod
    def _build_pipeline(self) -> None:
        """
        Initialize the LLM, embeddings, vector store, and retrieval chain.
        Called once during __init__. Store the built pipeline in self.pipeline.
        """
        ...

    @abstractmethod
    def index(self, documents: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        """
        Chunk, embed, and store documents in the vector store.

        Args:
            documents: List of raw text strings to index.
            metadatas: Optional metadata dicts aligned with documents.
        """
        ...

    @abstractmethod
    def _query(self, question: str) -> RAGResult:
        """
        Core RAG query logic. Must return a RAGResult.
        Implement the full retrieve → augment → generate pipeline here.
        """
        ...

    def query(self, question: str) -> RAGResult:
        """
        Public query interface. Wraps _query() with timing and logging.
        Do NOT override this in subclasses — override _query() instead.
        """
        if not self._is_indexed:
            self.logger.warning(
                "No documents indexed yet. Call index() before querying."
            )

        self.logger.info(f"[{self.TECHNIQUE_NAME}] Query: {question[:80]}")
        start = time.perf_counter()

        result = self._query(question)

        result.latency_ms = (time.perf_counter() - start) * 1000
        result.technique = self.TECHNIQUE_NAME
        result.framework = self.FRAMEWORK

        self.logger.info(
            f"[{self.TECHNIQUE_NAME}] Answered in {result.latency_ms:.1f}ms "
            f"| Sources: {len(result.source_documents)}"
        )
        return result

    def get_info(self) -> Dict[str, str]:
        """Return metadata about this RAG implementation."""
        return {
            "technique": self.TECHNIQUE_NAME,
            "framework": self.FRAMEWORK,
            "description": self.__doc__ or "",
            "indexed": str(self._is_indexed),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(technique={self.TECHNIQUE_NAME}, framework={self.FRAMEWORK})"
