"""
Corrective RAG (CRAG) — LlamaIndex Implementation
====================================================
Same single-pass quality gate as the LangChain version: grade the initial
retrieval once, classify it as correct / ambiguous / incorrect, then refine
the relevant documents and/or fall back to web search accordingly --
implemented with direct LLM `.complete()` calls for parity with the
LangChain implementation.
"""

import logging
import os
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


def _parse_score(text: str, default: float = 0.5) -> float:
    match = re.search(r"(\d*\.?\d+)", _strip_thinking(text))
    if not match:
        return default
    try:
        return max(0.0, min(1.0, float(match.group(1))))
    except ValueError:
        return default


def _parse_scores(text: str, count: int, default: float = 0.5) -> list[float]:
    lines = [line for line in _strip_thinking(text).split("\n") if line.strip()]
    scores = [_parse_score(line, default) for line in lines]
    while len(scores) < count:
        scores.append(default)
    return scores[:count]


RELEVANCE_GRADE_PROMPT = """For each numbered document below, judge how relevant it is \
to answering the question. Respond with ONLY the scores, one per line, in the same \
order, each between 0.0 (not relevant) and 1.0 (highly relevant). No other text.

Question: {question}

Documents:
{documents}

Relevance scores (one per line, in order):"""

REFINEMENT_PROMPT = """Extract ONLY the sentences from the following documents that are \
directly relevant to answering the question. Discard irrelevant sentences and filler. \
Preserve the original wording of the sentences you keep. Return the relevant sentences \
as a bulleted list, nothing else.

Question: {question}

Documents:
{documents}

Relevant knowledge strips:"""

ANSWER_PROMPT = """Answer the question based ONLY on the provided context. \
If the context doesn't contain sufficient information, acknowledge the limitation.

Context:
{context}

Question: {question}
Answer:"""

NO_CONTEXT_PROMPT = """No reliable retrieved or external context was available for this question. \
Answer directly from your own knowledge, being concise and noting any uncertainty.

Question: {question}
Answer:"""


class CorrectiveRAGLlamaIndex(BaseRAG):
    """
    Corrective RAG (CRAG) using LlamaIndex.

    Pipeline:
        Query → Retrieve → Grade Relevance (batched) → Classify (correct/
        ambiguous/incorrect) → Refine relevant docs and/or web search →
        Generate → Answer

    Best for: Knowledge bases that may be incomplete, high-stakes factual
    retrieval, and time-sensitive information where web search fallback
    matters.
    """

    TECHNIQUE_NAME = "corrective_rag"
    FRAMEWORK = "llamaindex"

    def _build_pipeline(self) -> None:
        """Initialize LLM, embeddings, CRAG thresholds, and the optional web search tool."""
        from llama_index.core import Settings

        cfg = ConfigLoader.get()
        tech_cfg = cfg.get_technique_config("corrective_rag")

        self.llm = get_llamaindex_llm()
        Settings.llm = self.llm
        Settings.embed_model = get_llamaindex_embeddings()

        self.top_k = cfg.retrieval.get("top_k", 5)
        self.relevance_threshold = tech_cfg.get("relevance_threshold", 0.7)
        self.web_search_fallback = tech_cfg.get("web_search_fallback", False)
        self.tavily_api_key = tech_cfg.get("tavily_api_key")

        self.vector_index = None
        self.retriever = None
        self.web_search_tool = self._build_web_search_tool()

        logger.info(
            f"[CorrectiveRAG/LI] relevance_threshold={self.relevance_threshold}, "
            f"web_search_fallback={self.web_search_fallback and bool(self.web_search_tool)}"
        )

    def _build_web_search_tool(self):
        """Build the optional web search fallback tool. Returns None if disabled or unconfigured."""
        if not self.web_search_fallback:
            return None
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults

            api_key = self.tavily_api_key or os.environ.get("TAVILY_API_KEY")
            if not api_key:
                logger.warning(
                    "[CorrectiveRAG/LI] web_search_fallback is enabled but no Tavily API key is "
                    "configured. Falling back to parametric knowledge instead of web search."
                )
                return None
            os.environ.setdefault("TAVILY_API_KEY", api_key)
            return TavilySearchResults(max_results=3)
        except Exception as e:
            logger.warning(f"[CorrectiveRAG/LI] Web search unavailable: {e}. Falling back to parametric knowledge.")  # noqa: E501
            return None

    def index(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        """Build a VectorStoreIndex from raw text strings."""
        from llama_index.core import VectorStoreIndex
        from llama_index.core.schema import Document as LIDocument

        logger.info(f"[CorrectiveRAG/LI] Indexing {len(documents)} documents...")

        metas = metadatas or [{}] * len(documents)
        li_docs = [LIDocument(text=t, metadata=m) for t, m in zip(documents, metas, strict=True)]

        self.vector_index = VectorStoreIndex.from_documents(li_docs, show_progress=True)
        self.retriever = self.vector_index.as_retriever(similarity_top_k=self.top_k)

        self._is_indexed = True
        logger.info("[CorrectiveRAG/LI] Indexing complete ✓")

    # ------------------------------------------------------------------
    # Corrective steps
    # ------------------------------------------------------------------

    def _grade_relevance(self, question: str, candidates: list) -> list[float]:
        if not candidates:
            return []
        docs_text = "\n\n".join(f"[{i + 1}] {node.get_content()}" for i, node in enumerate(candidates))  # noqa: E501
        response = self.llm.complete(RELEVANCE_GRADE_PROMPT.format(question=question, documents=docs_text))  # noqa: E501
        return _parse_scores(response.text, count=len(candidates))

    @staticmethod
    def _classify_action(scores: list[float], threshold: float) -> str:
        if not scores:
            return "incorrect"
        passing = sum(1 for s in scores if s >= threshold)
        if passing == len(scores):
            return "correct"
        if passing == 0:
            return "incorrect"
        return "ambiguous"

    def _refine_knowledge(self, question: str, nodes: list) -> list[str]:
        if not nodes:
            return []
        docs_text = "\n\n".join(f"[{i + 1}] {node.get_content()}" for i, node in enumerate(nodes))
        response = self.llm.complete(REFINEMENT_PROMPT.format(question=question, documents=docs_text))  # noqa: E501
        text = _strip_thinking(response.text)
        return [text] if text else []

    def _web_search(self, question: str) -> list[str]:
        if not self.web_search_tool:
            return []
        try:
            results = self.web_search_tool.invoke({"query": question})
            texts = []
            for r in results:
                if isinstance(r, dict):
                    texts.append(r.get("content", str(r)))
                else:
                    texts.append(str(r))
            return texts
        except Exception as e:
            logger.warning(f"[CorrectiveRAG/LI] Web search call failed: {e}")
            return []

    def _generate_with_context(self, question: str, context_blocks: list[str]) -> str:
        context = "\n\n---\n\n".join(context_blocks)
        response = self.llm.complete(ANSWER_PROMPT.format(context=context, question=question))
        return _strip_thinking(response.text)

    def _generate_without_context(self, question: str) -> str:
        response = self.llm.complete(NO_CONTEXT_PROMPT.format(question=question))
        return _strip_thinking(response.text)

    def _query(self, question: str) -> RAGResult:
        """Full CRAG pipeline: retrieve → grade → classify → refine/websearch → generate."""
        if not self.retriever:
            raise RuntimeError("Call index() before querying.")

        candidates = self.retriever.retrieve(question)
        scores = self._grade_relevance(question, candidates)
        action = self._classify_action(scores, self.relevance_threshold)
        correct_nodes = [node for node, score in zip(candidates, scores, strict=True) if score >= self.relevance_threshold]  # noqa: E501

        intermediate_steps = [{
            "step": "relevance_grading",
            "num_candidates": len(candidates),
            "scores": scores,
            "action": action,
        }]
        logger.debug(f"[CorrectiveRAG/LI] action={action}, {len(correct_nodes)}/{len(candidates)} nodes relevant")  # noqa: E501

        web_results: list[str] = []
        if action in ("incorrect", "ambiguous"):
            web_results = self._web_search(question)
            intermediate_steps.append({
                "step": "web_search",
                "attempted": self.web_search_tool is not None,
                "num_results": len(web_results),
            })

        context_blocks: list[str] = []
        if action in ("correct", "ambiguous") and correct_nodes:
            refined = self._refine_knowledge(question, correct_nodes)
            context_blocks.extend(refined)
            intermediate_steps.append({"step": "knowledge_refinement", "num_source_nodes": len(correct_nodes)})  # noqa: E501
        context_blocks.extend(web_results)

        if context_blocks:
            answer = self._generate_with_context(question, context_blocks)
        else:
            answer = self._generate_without_context(question)
            intermediate_steps.append({"step": "fallback_generation", "reason": "no usable internal or external context"})  # noqa: E501

        source_documents = [
            Document(content=node.get_content(), metadata=node.metadata, score=node.score)
            for node in correct_nodes
        ]
        source_documents.extend(
            Document(content=text, metadata={"source": "web_search"}) for text in web_results
        )

        return RAGResult(
            query=question,
            answer=answer,
            source_documents=source_documents,
            intermediate_steps=intermediate_steps,
            metadata={
                "action": action,
                "web_search_used": bool(web_results),
                "num_correct_docs": len(correct_nodes),
            },
        )
