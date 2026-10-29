"""
Agentic RAG: Autonomous tool-using agent for retrieval — LangChain Implementation
=================================================================================
Instead of a fixed retrieve → generate pipeline, an LLM agent runs a ReAct loop
(see react_loop.py): it decides whether to retrieve, what to search for, can
retrieve repeatedly with refined queries, and can use a calculator, before
committing to a final answer.

Tools: vector_retrieval (always on) and calculator, per config `agentic_rag.tools`.

Reference: https://github.com/NirDiamant/RAG_Techniques
Paper: "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022)
"""

import logging

from langchain_core.output_parsers import StrOutputParser

from core.base_rag import BaseRAG, Document, RAGResult
from core.config_loader import ConfigLoader
from core.document_loader import get_text_splitter, load_texts
from core.embeddings import get_langchain_embeddings
from core.llm_client import get_langchain_llm
from core.vector_store import build_langchain_vector_store

from .react_loop import Tool, run_react, safe_calculate

logger = logging.getLogger(__name__)


class AgenticRAGLangChain(BaseRAG):
    """
    Agentic RAG using LangChain: a ReAct agent that chooses when and what to
    retrieve, can iterate with refined queries, and can compute with a calculator.

    Best for: Multi-step questions where one retrieval pass is not enough —
    comparisons, lookups followed by arithmetic, or queries that need
    follow-up searches based on what earlier searches returned.
    """

    TECHNIQUE_NAME = "agentic_rag"
    FRAMEWORK = "langchain"

    def _build_pipeline(self) -> None:
        """Initialize LLM, embeddings, and agent settings from config."""
        cfg = ConfigLoader.get()
        tech_cfg = cfg.get_technique_config("agentic_rag")

        self.llm = get_langchain_llm()
        self.embeddings = get_langchain_embeddings()
        self.text_splitter = get_text_splitter()
        self.top_k = cfg.retrieval.get("top_k", 5)
        self.max_iterations = tech_cfg.get("max_iterations", 5)
        self.enabled_tools = set(tech_cfg.get("tools", [])) | {"vector_retrieval"}
        self.vector_store = None

        logger.info(
            f"[AgenticRAG/LC] max_iterations={self.max_iterations}, "
            f"tools={sorted(self.enabled_tools)}"
        )

    def index(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        """Chunk, embed, and store documents in the configured vector store."""
        logger.info(f"[AgenticRAG/LC] Indexing {len(documents)} documents...")
        chunks = self.text_splitter.split_documents(load_texts(documents, metadatas))
        self.vector_store = build_langchain_vector_store(chunks, collection_name="agentic_rag")
        self._is_indexed = True
        logger.info(f"[AgenticRAG/LC] Indexed {len(chunks)} chunks ✓")

    def _complete(self, prompt: str) -> str:
        """Single LLM completion used by the ReAct loop."""
        return (self.llm | StrOutputParser()).invoke(prompt)

    def _build_tools(self, retrieved: list) -> dict[str, Tool]:
        """Build the enabled tools; retrieval appends every hit to `retrieved`."""

        def vector_retrieval(query: str) -> str:
            docs = self.vector_store.as_retriever(search_kwargs={"k": self.top_k}).invoke(query)
            retrieved.extend(docs)
            return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))

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
        if not self.vector_store:
            raise RuntimeError("Call index() before querying.")

        retrieved: list = []
        answer, steps = run_react(
            self._complete, self._build_tools(retrieved), question, self.max_iterations
        )

        unique = {d.page_content: d for d in retrieved}.values()
        return RAGResult(
            query=question,
            answer=answer,
            source_documents=[
                Document(content=d.page_content, metadata=d.metadata) for d in unique
            ],
            intermediate_steps=[{"step": "agent_action", **s} for s in steps],
            metadata={
                "num_agent_steps": len(steps),
                "tools_used": sorted({s["action"] for s in steps}),
                "retrieval_calls": sum(s["action"] == "vector_retrieval" for s in steps),
            },
        )
