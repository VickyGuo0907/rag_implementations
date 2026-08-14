"""
Corrective RAG (CRAG) — LangChain Implementation
===================================================
Unlike Self-RAG's iterative reflect-and-retry loop, CRAG is a single-pass
quality gate: grade the initial retrieval once, then branch into one of
three corrective actions based on how much of it actually cleared the bar:

  - CORRECT   (all candidates relevant)   — refine and use the internal docs
  - AMBIGUOUS (some candidates relevant)  — refine the relevant internal docs
                                             AND supplement with web search
  - INCORRECT (no candidates relevant)    — discard internal docs entirely,
                                             fall back to web search

"Refine" here means the paper's decompose-then-recompose idea, simplified to
a single LLM call: extract only the sentences actually relevant to the
question from the surviving documents, discarding filler.

Web search is optional and disabled by default (config: web_search_fallback).
Without it, INCORRECT/AMBIGUOUS branches fall back to the model's own
parametric knowledge rather than failing outright.

Reference: https://github.com/NirDiamant/RAG_Techniques
Paper: "Corrective Retrieval Augmented Generation" (Yan et al., 2024)
       https://arxiv.org/abs/2401.15884
"""

import logging
import os
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.base_rag import BaseRAG, Document, RAGResult
from core.config_loader import ConfigLoader
from core.document_loader import get_text_splitter, load_texts
from core.embeddings import get_langchain_embeddings
from core.llm_client import get_langchain_llm
from core.vector_store import build_langchain_vector_store

logger = logging.getLogger(__name__)


def _parse_score(text: str, default: float = 0.5) -> float:
    """Extract a single 0.0-1.0 score from an LLM response, clamped and defaulted on parse failure."""  # noqa: E501
    match = re.search(r"(\d*\.?\d+)", text)
    if not match:
        return default
    try:
        return max(0.0, min(1.0, float(match.group(1))))
    except ValueError:
        return default


def _parse_scores(text: str, count: int, default: float = 0.5) -> list[float]:
    """Extract one 0.0-1.0 score per line, padding/truncating to the expected count."""
    lines = [line for line in text.strip().split("\n") if line.strip()]
    scores = [_parse_score(line, default) for line in lines]
    while len(scores) < count:
        scores.append(default)
    return scores[:count]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RELEVANCE_GRADE_PROMPT = ChatPromptTemplate.from_template(
"""For each numbered document below, judge how relevant it is to answering the question.
Respond with ONLY the scores, one per line, in the same order, each between 0.0
(not relevant) and 1.0 (highly relevant). No other text.

Question: {question}

Documents:
{documents}

Relevance scores (one per line, in order):"""
)

REFINEMENT_PROMPT = ChatPromptTemplate.from_template(
"""Extract ONLY the sentences from the following documents that are directly relevant
to answering the question. Discard irrelevant sentences and filler. Preserve the
original wording of the sentences you keep. Return the relevant sentences as a
bulleted list, nothing else.

Question: {question}

Documents:
{documents}

Relevant knowledge strips:"""
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Answer the question based ONLY on the provided context.
If the context doesn't contain sufficient information, acknowledge the limitation.

Context:
{context}"""),
    ("human", "{question}"),
])

NO_CONTEXT_PROMPT = ChatPromptTemplate.from_template(
"""No reliable retrieved or external context was available for this question.
Answer directly from your own knowledge, being concise and noting any uncertainty.

Question: {question}
Answer:"""
)


class CorrectiveRAGLangChain(BaseRAG):
    """
    Corrective RAG (CRAG) using LangChain.

    Pipeline:
        Query → Retrieve → Grade Relevance (batched) → Classify (correct/
        ambiguous/incorrect) → Refine relevant docs and/or web search →
        Generate → Answer

    Best for: Knowledge bases that may be incomplete, high-stakes factual
    retrieval, and time-sensitive information where web search fallback
    matters.
    """

    TECHNIQUE_NAME = "corrective_rag"
    FRAMEWORK = "langchain"

    def _build_pipeline(self) -> None:
        """Initialize LLM, embeddings, CRAG thresholds, and the optional web search tool."""
        cfg = ConfigLoader.get()
        tech_cfg = cfg.get_technique_config("corrective_rag")

        self.llm = get_langchain_llm()
        self.embeddings = get_langchain_embeddings()
        self.text_splitter = get_text_splitter()
        self.top_k = cfg.retrieval.get("top_k", 5)

        self.relevance_threshold = tech_cfg.get("relevance_threshold", 0.7)
        self.web_search_fallback = tech_cfg.get("web_search_fallback", False)
        self.tavily_api_key = tech_cfg.get("tavily_api_key")

        self.vector_store = None
        self.web_search_tool = self._build_web_search_tool()

        logger.info(
            f"[CorrectiveRAG/LC] relevance_threshold={self.relevance_threshold}, "
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
                    "[CorrectiveRAG/LC] web_search_fallback is enabled but no Tavily API key is "
                    "configured (rag_techniques.corrective_rag.tavily_api_key or TAVILY_API_KEY env "  # noqa: E501
                    "var). Falling back to parametric knowledge instead of web search."
                )
                return None
            os.environ.setdefault("TAVILY_API_KEY", api_key)
            return TavilySearchResults(max_results=3)
        except Exception as e:
            logger.warning(f"[CorrectiveRAG/LC] Web search unavailable: {e}. Falling back to parametric knowledge.")  # noqa: E501
            return None

    def index(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        """Chunk, embed, and store documents in the configured vector store."""
        logger.info(f"[CorrectiveRAG/LC] Indexing {len(documents)} documents...")

        lc_docs = load_texts(documents, metadatas)
        chunks = self.text_splitter.split_documents(lc_docs)

        self.vector_store = build_langchain_vector_store(
            chunks,
            collection_name="corrective_rag",
        )

        self._is_indexed = True
        logger.info(f"[CorrectiveRAG/LC] Indexed {len(chunks)} chunks ✓")

    # ------------------------------------------------------------------
    # Corrective steps
    # ------------------------------------------------------------------

    def _retrieve(self, query: str) -> list:
        retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
        return retriever.invoke(query)

    def _grade_relevance(self, question: str, candidates: list) -> list[float]:
        """Batch-score every candidate's relevance to the question in one LLM call."""
        if not candidates:
            return []
        docs_text = "\n\n".join(f"[{i + 1}] {doc.page_content}" for i, doc in enumerate(candidates))
        chain = RELEVANCE_GRADE_PROMPT | self.llm | StrOutputParser()
        result = chain.invoke({"question": question, "documents": docs_text})
        return _parse_scores(result, count=len(candidates))

    @staticmethod
    def _classify_action(scores: list[float], threshold: float) -> str:
        """Classify overall retrieval quality as correct / ambiguous / incorrect."""
        if not scores:
            return "incorrect"
        passing = sum(1 for s in scores if s >= threshold)
        if passing == len(scores):
            return "correct"
        if passing == 0:
            return "incorrect"
        return "ambiguous"

    def _refine_knowledge(self, question: str, docs: list) -> list[str]:
        """Decompose-then-recompose: extract only the sentences relevant to the question."""
        if not docs:
            return []
        docs_text = "\n\n".join(f"[{i + 1}] {doc.page_content}" for i, doc in enumerate(docs))
        chain = REFINEMENT_PROMPT | self.llm | StrOutputParser()
        result = chain.invoke({"question": question, "documents": docs_text})
        return [result.strip()] if result.strip() else []

    def _web_search(self, question: str) -> list[str]:
        """Run the optional web search fallback. Returns [] if unavailable or it fails."""
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
            logger.warning(f"[CorrectiveRAG/LC] Web search call failed: {e}")
            return []

    def _generate_with_context(self, question: str, context_blocks: list[str]) -> str:
        context = "\n\n---\n\n".join(context_blocks)
        chain = ANSWER_PROMPT | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})

    def _generate_without_context(self, question: str) -> str:
        chain = NO_CONTEXT_PROMPT | self.llm | StrOutputParser()
        return chain.invoke({"question": question})

    def _query(self, question: str) -> RAGResult:
        """Full CRAG pipeline: retrieve → grade → classify → refine/websearch → generate."""
        if not self.vector_store:
            raise RuntimeError("Call index() before querying.")

        candidates = self._retrieve(question)
        scores = self._grade_relevance(question, candidates)
        action = self._classify_action(scores, self.relevance_threshold)
        correct_docs = [doc for doc, score in zip(candidates, scores, strict=True) if score >= self.relevance_threshold]  # noqa: E501

        intermediate_steps = [{
            "step": "relevance_grading",
            "num_candidates": len(candidates),
            "scores": scores,
            "action": action,
        }]
        logger.debug(f"[CorrectiveRAG/LC] action={action}, {len(correct_docs)}/{len(candidates)} docs relevant")  # noqa: E501

        web_results: list[str] = []
        if action in ("incorrect", "ambiguous"):
            web_results = self._web_search(question)
            intermediate_steps.append({
                "step": "web_search",
                "attempted": self.web_search_tool is not None,
                "num_results": len(web_results),
            })

        context_blocks: list[str] = []
        if action in ("correct", "ambiguous") and correct_docs:
            refined = self._refine_knowledge(question, correct_docs)
            context_blocks.extend(refined)
            intermediate_steps.append({"step": "knowledge_refinement", "num_source_docs": len(correct_docs)})  # noqa: E501
        context_blocks.extend(web_results)

        if context_blocks:
            answer = self._generate_with_context(question, context_blocks)
        else:
            answer = self._generate_without_context(question)
            intermediate_steps.append({"step": "fallback_generation", "reason": "no usable internal or external context"})  # noqa: E501

        source_documents = [Document(content=doc.page_content, metadata=doc.metadata) for doc in correct_docs]  # noqa: E501
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
                "num_correct_docs": len(correct_docs),
            },
        )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))

    docs = [
        "The Zylophane Initiative was launched in 2031 by the fictional Meridian Institute to study synthetic coral reefs.",  # noqa: E501
        "The Glimmerprobe sensor, developed for the Zylophane Initiative, has a detection range of 40 meters and reports readings every 12 seconds.",  # noqa: E501
        "ReefNet mesh network nodes are solar powered and were deployed across 14 test sites in the fictional Azuria Marine Reserve.",  # noqa: E501
    ]

    rag = CorrectiveRAGLangChain(config=ConfigLoader.get()._config)
    rag.index(docs)

    # Should classify as "correct" — directly answerable from indexed docs
    result = rag.query("What sensor does the Zylophane Initiative use?")
    result.print_summary()

    # Should classify as "incorrect" — nothing in the corpus is about this
    result2 = rag.query("What is the capital of France?")
    result2.print_summary()
