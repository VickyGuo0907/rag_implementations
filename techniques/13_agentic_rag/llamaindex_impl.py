"""
Agentic RAG: Autonomous tool-using agent for retrieval — LlamaIndex Implementation
==================================================================================
Same ReAct agent as the LangChain version (shared loop in react_loop.py), with
LlamaIndex supplying the LLM and the vector retriever. The agent decides when
and what to retrieve, can iterate with refined queries, and can use a calculator.
"""

import logging

from llama_index.core.llms import ChatMessage

from core.base_rag import BaseRAG, Document, RAGResult
from core.config_loader import ConfigLoader
from core.embeddings import get_llamaindex_embeddings
from core.llm_client import get_llamaindex_llm

from .react_loop import Tool, run_react, safe_calculate

logger = logging.getLogger(__name__)


class AgenticRAGLlamaIndex(BaseRAG):
    """
    Agentic RAG using LlamaIndex: a ReAct agent that chooses when and what to
    retrieve, can iterate with refined queries, and can compute with a calculator.

    Best for: Multi-step questions where one retrieval pass is not enough —
    comparisons, lookups followed by arithmetic, or queries that need
    follow-up searches based on what earlier searches returned.
    """

    TECHNIQUE_NAME = "agentic_rag"
    FRAMEWORK = "llamaindex"

    def _build_pipeline(self) -> None:
        """Initialize LLM, embeddings, and agent settings from config."""
        from llama_index.core import Settings
        from llama_index.core.node_parser import SentenceSplitter

        cfg = ConfigLoader.get()
        tech_cfg = cfg.get_technique_config("agentic_rag")
        doc_cfg = cfg.document

        self.llm = get_llamaindex_llm()
        Settings.llm = self.llm
        Settings.embed_model = get_llamaindex_embeddings()
        Settings.node_parser = SentenceSplitter(
            chunk_size=doc_cfg.get("chunk_size", 512),
            chunk_overlap=doc_cfg.get("chunk_overlap", 50),
        )

        self.top_k = cfg.retrieval.get("top_k", 5)
        self.max_iterations = tech_cfg.get("max_iterations", 5)
        self.enabled_tools = set(tech_cfg.get("tools", [])) | {"vector_retrieval"}
        self.vector_index = None
        self.retriever = None

        logger.info(
            f"[AgenticRAG/LI] max_iterations={self.max_iterations}, "
            f"tools={sorted(self.enabled_tools)}"
        )

    def index(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        """Build a VectorStoreIndex from raw text strings."""
        from llama_index.core import VectorStoreIndex
        from llama_index.core.schema import Document as LIDocument

        logger.info(f"[AgenticRAG/LI] Indexing {len(documents)} documents...")
        metas = metadatas or [{}] * len(documents)
        li_docs = [LIDocument(text=t, metadata=m) for t, m in zip(documents, metas, strict=True)]
        self.vector_index = VectorStoreIndex.from_documents(li_docs, show_progress=True)
        self.retriever = self.vector_index.as_retriever(similarity_top_k=self.top_k)
        self._is_indexed = True
        logger.info("[AgenticRAG/LI] Indexing complete ✓")

    def _complete(self, prompt: str) -> str:
        """One ReAct LLM turn via the chat API, so the model's chat template applies."""
        return self.llm.chat([ChatMessage(role="user", content=prompt)]).message.content or ""

    def _build_tools(self, retrieved: list) -> dict[str, Tool]:
        """Build the enabled tools; retrieval appends every hit to `retrieved`."""

        def vector_retrieval(query: str) -> str:
            nodes = self.retriever.retrieve(query)
            retrieved.extend(nodes)
            return "\n\n".join(f"[{i}] {n.get_content()}" for i, n in enumerate(nodes, 1))

        tools = {"vector_retrieval": Tool(
            "vector_retrieval",
            "Search the document knowledge base. Input: a search query.",
            vector_retrieval,
        )}
        if "calculator" in self.enabled_tools:
            tools["calculator"] = Tool(
                "calculator",
                "Evaluate an arithmetic expression such as (3 + 4) * 2. Input: the expression.",
                safe_calculate,
            )
        return tools

    def _query(self, question: str) -> RAGResult:
        """Run the ReAct agent and package its answer, sources, and trace."""
        if not self.retriever:
            raise RuntimeError("Call index() before querying.")

        retrieved: list = []
        answer, steps = run_react(
            self._complete, self._build_tools(retrieved), question, self.max_iterations
        )

        unique = {n.get_content(): n for n in retrieved}.values()
        return RAGResult(
            query=question,
            answer=answer,
            source_documents=[
                Document(content=n.get_content(), metadata=n.metadata, score=n.score)
                for n in unique
            ],
            intermediate_steps=[{"step": "agent_action", **s} for s in steps],
            metadata={
                "num_agent_steps": len(steps),
                "tools_used": sorted({s["action"] for s in steps}),
                "retrieval_calls": sum(s["action"] == "vector_retrieval" for s in steps),
            },
        )
