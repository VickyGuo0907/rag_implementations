"""
Self-RAG (Self-Reflective RAG) — LlamaIndex Implementation
=============================================================
Same structured-prompting simulation of Self-RAG's reflection tokens as the
LangChain version: decide whether retrieval is needed, grade candidate
relevance, generate, critique groundedness, and iterate with a refined query
if the answer isn't well-supported -- implemented with direct LLM
`.complete()` calls for parity with the LangChain implementation.
"""

from typing import Dict, List, Optional, Tuple
import logging
import re

from core.base_rag import BaseRAG, RAGResult, Document
from core.config_loader import ConfigLoader
from core.llm_client import get_llamaindex_llm
from core.embeddings import get_llamaindex_embeddings

logger = logging.getLogger(__name__)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning traces some LMStudio models emit in raw completions."""
    return _THINK_TAG_RE.sub("", text).strip()


def _parse_score(text: str, default: float = 0.5) -> float:
    """Extract a single 0.0-1.0 score from an LLM response, clamped and defaulted on parse failure."""
    match = re.search(r"(\d*\.?\d+)", _strip_thinking(text))
    if not match:
        return default
    try:
        return max(0.0, min(1.0, float(match.group(1))))
    except ValueError:
        return default


def _parse_scores(text: str, count: int, default: float = 0.5) -> List[float]:
    """Extract one 0.0-1.0 score per line, padding/truncating to the expected count."""
    lines = [line for line in _strip_thinking(text).split("\n") if line.strip()]
    scores = [_parse_score(line, default) for line in lines]
    while len(scores) < count:
        scores.append(default)
    return scores[:count]


RETRIEVAL_DECISION_PROMPT = """Decide whether answering the following question requires retrieving external \
documents, or whether it can be answered accurately from general knowledge alone. \
Respond with ONLY a single number between 0.0 and 1.0 -- 0.0 means retrieval is \
definitely not needed, 1.0 means retrieval is essential.

Question: {question}
Retrieval necessity score:"""

RELEVANCE_GRADE_PROMPT = """For each numbered passage below, judge how relevant it is to answering the question. \
Respond with ONLY the scores, one per line, in the same order, each between 0.0 \
(not relevant) and 1.0 (highly relevant). No other text.

Question: {question}

Passages:
{passages}

Relevance scores (one per line, in order):"""

SUPPORT_GRADE_PROMPT = """Judge how well the following answer is supported by the provided context. An \
answer is well-supported if every claim in it can be verified against the context. \
Respond with ONLY a single number between 0.0 (not supported / hallucinated) and \
1.0 (fully supported).

Context:
{context}

Question: {question}
Answer: {answer}

Support score:"""

REFINE_QUERY_PROMPT = """The following answer to a question was not well-supported by the retrieved \
context, suggesting the search didn't find the right documents. Generate a \
better, more specific search query to find documents that would support \
answering this question. Return ONLY the new query.

Original question: {question}
Weakly-supported answer: {answer}
Better search query:"""

ANSWER_PROMPT = """Answer the question based ONLY on the provided context. \
If the context doesn't contain sufficient information, acknowledge the limitation.

Context:
{context}

Question: {question}
Answer:"""

NO_RETRIEVAL_PROMPT = """Answer the following question directly using your own knowledge. Be concise and accurate.

Question: {question}
Answer:"""


class SelfRAGLlamaIndex(BaseRAG):
    """
    Self-RAG using LlamaIndex, with reflection simulated via structured prompting.

    Best for: Minimizing retrieval overhead on questions that don't need it,
    mixed fact-based/reasoning question sets, and reducing hallucinations by
    explicitly grading relevance and groundedness rather than trusting
    retrieval blindly.
    """

    TECHNIQUE_NAME = "self_rag"
    FRAMEWORK = "llamaindex"

    def _build_pipeline(self) -> None:
        """Initialize LLM, embeddings, and Self-RAG's reflection thresholds."""
        from llama_index.core import Settings
        from llama_index.core.node_parser import SentenceSplitter

        cfg = ConfigLoader.get()
        tech_cfg = cfg.get_technique_config("self_rag")
        doc_cfg = cfg.document

        self.llm = get_llamaindex_llm()
        Settings.llm = self.llm
        Settings.embed_model = get_llamaindex_embeddings()
        Settings.node_parser = SentenceSplitter(
            chunk_size=doc_cfg.get("chunk_size", 512),
            chunk_overlap=doc_cfg.get("chunk_overlap", 50),
        )

        self.top_k = cfg.retrieval.get("top_k", 5)
        self.max_iterations = tech_cfg.get("max_iterations", 3)
        self.retrieve_token_threshold = tech_cfg.get("retrieve_token_threshold", 0.5)
        self.relevance_threshold = tech_cfg.get("relevance_threshold", 0.6)
        self.support_threshold = tech_cfg.get("support_threshold", 0.7)

        self.vector_index = None
        self.retriever = None

        logger.info(
            f"[SelfRAG/LI] max_iterations={self.max_iterations}, "
            f"retrieve_threshold={self.retrieve_token_threshold}, "
            f"relevance_threshold={self.relevance_threshold}, "
            f"support_threshold={self.support_threshold}"
        )

    def index(self, documents: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        """Build a VectorStoreIndex from raw text strings."""
        from llama_index.core import VectorStoreIndex
        from llama_index.core.schema import Document as LIDocument

        logger.info(f"[SelfRAG/LI] Indexing {len(documents)} documents...")

        metas = metadatas or [{}] * len(documents)
        li_docs = [LIDocument(text=t, metadata=m) for t, m in zip(documents, metas)]

        self.vector_index = VectorStoreIndex.from_documents(li_docs, show_progress=True)
        self.retriever = self.vector_index.as_retriever(similarity_top_k=self.top_k)

        self._is_indexed = True
        logger.info("[SelfRAG/LI] Indexing complete ✓")

    # ------------------------------------------------------------------
    # Reflection steps
    # ------------------------------------------------------------------

    def _decide_retrieval(self, question: str) -> float:
        response = self.llm.complete(RETRIEVAL_DECISION_PROMPT.format(question=question))
        return _parse_score(response.text, default=1.0)

    def _grade_relevance(self, question: str, candidates: List) -> Tuple[List, List[float]]:
        if not candidates:
            return [], []
        passages_text = "\n\n".join(f"[{i + 1}] {node.get_content()}" for i, node in enumerate(candidates))
        response = self.llm.complete(RELEVANCE_GRADE_PROMPT.format(question=question, passages=passages_text))
        scores = _parse_scores(response.text, count=len(candidates))
        relevant = [node for node, score in zip(candidates, scores) if score >= self.relevance_threshold]
        return relevant, scores

    def _generate_with_context(self, question: str, nodes: List) -> str:
        context = "\n\n---\n\n".join(node.get_content() for node in nodes)
        response = self.llm.complete(ANSWER_PROMPT.format(context=context, question=question))
        return _strip_thinking(response.text)

    def _generate_without_retrieval(self, question: str) -> str:
        response = self.llm.complete(NO_RETRIEVAL_PROMPT.format(question=question))
        return _strip_thinking(response.text)

    def _grade_support(self, question: str, nodes: List, answer: str) -> float:
        context = "\n\n---\n\n".join(node.get_content() for node in nodes)
        response = self.llm.complete(SUPPORT_GRADE_PROMPT.format(context=context, question=question, answer=answer))
        return _parse_score(response.text, default=0.5)

    def _refine_query(self, question: str, weak_answer: str) -> str:
        response = self.llm.complete(REFINE_QUERY_PROMPT.format(question=question, answer=weak_answer))
        return _strip_thinking(response.text)

    def _query(self, question: str) -> RAGResult:
        """Full Self-RAG pipeline: decide → (retrieve → grade → generate → critique)* → answer."""
        if not self.retriever:
            raise RuntimeError("Call index() before querying.")

        intermediate_steps = []

        retrieval_score = self._decide_retrieval(question)
        needs_retrieval = retrieval_score >= self.retrieve_token_threshold
        intermediate_steps.append({
            "step": "retrieval_decision",
            "retrieval_necessity_score": retrieval_score,
            "needs_retrieval": needs_retrieval,
        })
        logger.debug(f"[SelfRAG/LI] retrieval_necessity={retrieval_score:.2f} → needs_retrieval={needs_retrieval}")

        if not needs_retrieval:
            answer = self._generate_without_retrieval(question)
            return RAGResult(
                query=question,
                answer=answer,
                source_documents=[],
                intermediate_steps=intermediate_steps,
                metadata={"retrieval_used": False, "iterations": 0},
            )

        best_answer, best_nodes, best_support = None, [], -1.0
        retrieval_query = question
        iterations_run = 0

        for i in range(self.max_iterations):
            iterations_run += 1
            candidates = self.retriever.retrieve(retrieval_query)
            relevant_nodes, relevance_scores = self._grade_relevance(question, candidates)
            used_nodes = relevant_nodes if relevant_nodes else candidates

            answer = self._generate_with_context(question, used_nodes)
            support_score = self._grade_support(question, used_nodes, answer)

            intermediate_steps.append({
                "step": f"iteration_{i + 1}",
                "retrieval_query": retrieval_query,
                "num_candidates": len(candidates),
                "num_relevant": len(relevant_nodes),
                "support_score": support_score,
            })
            logger.debug(
                f"[SelfRAG/LI] iteration {i + 1}: {len(relevant_nodes)}/{len(candidates)} relevant, "
                f"support={support_score:.2f}"
            )

            if support_score > best_support:
                best_answer, best_nodes, best_support = answer, used_nodes, support_score

            if support_score >= self.support_threshold:
                break

            if i < self.max_iterations - 1:
                retrieval_query = self._refine_query(question, answer)

        return RAGResult(
            query=question,
            answer=best_answer,
            source_documents=[
                Document(content=node.get_content(), metadata=node.metadata, score=node.score)
                for node in best_nodes
            ],
            intermediate_steps=intermediate_steps,
            metadata={
                "retrieval_used": True,
                "iterations": iterations_run,
                "final_support_score": best_support,
            },
        )
