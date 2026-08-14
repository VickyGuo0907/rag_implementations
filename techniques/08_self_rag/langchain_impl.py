"""
Self-RAG (Self-Reflective RAG) — LangChain Implementation
============================================================
The original Self-RAG paper (Asai et al., 2023) fine-tunes a model to emit
special "reflection tokens" (Retrieve / ISREL / ISSUP / ISUSE) that control
retrieval and critique generation inline. Without a model fine-tuned for
those tokens, this implementation reproduces the same control flow via
structured prompting against a general-purpose chat LLM:

  1. Retrieval decision  — does this question even need external documents?
  2. Relevance grading   — score each retrieved candidate, keep only the relevant ones
  3. Generation          — answer from the relevant context (or from parametric
                            knowledge alone, if retrieval wasn't needed)
  4. Support critique    — score how well the answer is grounded in the context
  5. Iterate             — if support is weak, reformulate the retrieval query
                            and try again, up to max_iterations times

Reference: https://github.com/NirDiamant/RAG_Techniques
Paper: "Self-RAG: Learning to Retrieve, Generate, and Critique through
Self-Reflection" (Asai et al., 2023) — https://arxiv.org/abs/2310.11511
"""

from typing import Dict, List, Optional, Tuple
import logging
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.base_rag import BaseRAG, RAGResult, Document
from core.config_loader import ConfigLoader
from core.llm_client import get_langchain_llm
from core.embeddings import get_langchain_embeddings
from core.document_loader import load_texts, get_text_splitter
from core.vector_store import build_langchain_vector_store

logger = logging.getLogger(__name__)


def _parse_score(text: str, default: float = 0.5) -> float:
    """Extract a single 0.0-1.0 score from an LLM response, clamped and defaulted on parse failure."""
    match = re.search(r"(\d*\.?\d+)", text)
    if not match:
        return default
    try:
        return max(0.0, min(1.0, float(match.group(1))))
    except ValueError:
        return default


def _parse_scores(text: str, count: int, default: float = 0.5) -> List[float]:
    """Extract one 0.0-1.0 score per line, padding/truncating to the expected count."""
    lines = [line for line in text.strip().split("\n") if line.strip()]
    scores = [_parse_score(line, default) for line in lines]
    while len(scores) < count:
        scores.append(default)
    return scores[:count]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RETRIEVAL_DECISION_PROMPT = ChatPromptTemplate.from_template(
    """Decide whether answering the following question requires retrieving external
documents, or whether it can be answered accurately from general knowledge alone.
Respond with ONLY a single number between 0.0 and 1.0 -- 0.0 means retrieval is
definitely not needed, 1.0 means retrieval is essential.

Question: {question}
Retrieval necessity score:"""
)

RELEVANCE_GRADE_PROMPT = ChatPromptTemplate.from_template(
"""For each numbered passage below, judge how relevant it is to answering the question.
Respond with ONLY the scores, one per line, in the same order, each between 0.0
(not relevant) and 1.0 (highly relevant). No other text.

Question: {question}

Passages:
{passages}

Relevance scores (one per line, in order):"""
)

SUPPORT_GRADE_PROMPT = ChatPromptTemplate.from_template(
"""Judge how well the following answer is supported by the provided context. An
answer is well-supported if every claim in it can be verified against the context.
Respond with ONLY a single number between 0.0 (not supported / hallucinated) and
1.0 (fully supported).

Context:
{context}

Question: {question}
Answer: {answer}

Support score:"""
)

REFINE_QUERY_PROMPT = ChatPromptTemplate.from_template(
"""The following answer to a question was not well-supported by the retrieved
context, suggesting the search didn't find the right documents. Generate a
better, more specific search query to find documents that would support
answering this question. Return ONLY the new query.

Original question: {question}
Weakly-supported answer: {answer}
Better search query:"""
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Answer the question based ONLY on the provided context.
If the context doesn't contain sufficient information, acknowledge the limitation.

Context:
{context}"""),
    ("human", "{question}"),
])

NO_RETRIEVAL_PROMPT = ChatPromptTemplate.from_template(
    """Answer the following question directly using your own knowledge. Be concise and accurate.

Question: {question}
Answer:"""
)


class SelfRAGLangChain(BaseRAG):
    """
    Self-RAG using LangChain, with reflection simulated via structured prompting.

    Best for: Minimizing retrieval overhead on questions that don't need it,
    mixed fact-based/reasoning question sets, and reducing hallucinations by
    explicitly grading relevance and groundedness rather than trusting
    retrieval blindly.
    """

    TECHNIQUE_NAME = "self_rag"
    FRAMEWORK = "langchain"

    def _build_pipeline(self) -> None:
        """Initialize LLM, embeddings, and Self-RAG's reflection thresholds."""
        cfg = ConfigLoader.get()
        tech_cfg = cfg.get_technique_config("self_rag")

        self.llm = get_langchain_llm()
        self.embeddings = get_langchain_embeddings()
        self.text_splitter = get_text_splitter()
        self.top_k = cfg.retrieval.get("top_k", 5)

        self.max_iterations = tech_cfg.get("max_iterations", 3)
        self.retrieve_token_threshold = tech_cfg.get("retrieve_token_threshold", 0.5)
        self.relevance_threshold = tech_cfg.get("relevance_threshold", 0.6)
        self.support_threshold = tech_cfg.get("support_threshold", 0.7)

        self.vector_store = None

        logger.info(
            f"[SelfRAG/LC] max_iterations={self.max_iterations}, "
            f"retrieve_threshold={self.retrieve_token_threshold}, "
            f"relevance_threshold={self.relevance_threshold}, "
            f"support_threshold={self.support_threshold}"
        )

    def index(self, documents: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        """Chunk, embed, and store documents in the configured vector store."""
        logger.info(f"[SelfRAG/LC] Indexing {len(documents)} documents...")

        lc_docs = load_texts(documents, metadatas)
        chunks = self.text_splitter.split_documents(lc_docs)

        self.vector_store = build_langchain_vector_store(
            chunks,
            collection_name="self_rag",
        )

        self._is_indexed = True
        logger.info(f"[SelfRAG/LC] Indexed {len(chunks)} chunks ✓")

    # ------------------------------------------------------------------
    # Reflection steps
    # ------------------------------------------------------------------

    def _decide_retrieval(self, question: str) -> float:
        """Score (0-1) how necessary retrieval is for this question."""
        chain = RETRIEVAL_DECISION_PROMPT | self.llm | StrOutputParser()
        result = chain.invoke({"question": question})
        return _parse_score(result, default=1.0)  # default to retrieving if the model is unclear

    def _retrieve(self, query: str) -> List:
        retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
        return retriever.invoke(query)

    def _grade_relevance(self, question: str, candidates: List) -> Tuple[List, List[float]]:
        """Batch-score every candidate's relevance in one LLM call; keep only those above threshold."""
        if not candidates:
            return [], []
        passages_text = "\n\n".join(f"[{i + 1}] {doc.page_content}" for i, doc in enumerate(candidates))
        chain = RELEVANCE_GRADE_PROMPT | self.llm | StrOutputParser()
        result = chain.invoke({"question": question, "passages": passages_text})
        scores = _parse_scores(result, count=len(candidates))
        relevant = [doc for doc, score in zip(candidates, scores) if score >= self.relevance_threshold]
        return relevant, scores

    def _generate_with_context(self, question: str, docs: List) -> str:
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        chain = ANSWER_PROMPT | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})

    def _generate_without_retrieval(self, question: str) -> str:
        chain = NO_RETRIEVAL_PROMPT | self.llm | StrOutputParser()
        return chain.invoke({"question": question})

    def _grade_support(self, question: str, docs: List, answer: str) -> float:
        """Score (0-1) how well the answer is grounded in the given context."""
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        chain = SUPPORT_GRADE_PROMPT | self.llm | StrOutputParser()
        result = chain.invoke({"context": context, "question": question, "answer": answer})
        return _parse_score(result, default=0.5)

    def _refine_query(self, question: str, weak_answer: str) -> str:
        """Reformulate the retrieval query after a weakly-supported attempt."""
        chain = REFINE_QUERY_PROMPT | self.llm | StrOutputParser()
        return chain.invoke({"question": question, "answer": weak_answer}).strip()

    def _query(self, question: str) -> RAGResult:
        """Full Self-RAG pipeline: decide → (retrieve → grade → generate → critique)* → answer."""
        if not self.vector_store:
            raise RuntimeError("Call index() before querying.")

        intermediate_steps = []

        retrieval_score = self._decide_retrieval(question)
        needs_retrieval = retrieval_score >= self.retrieve_token_threshold
        intermediate_steps.append({
            "step": "retrieval_decision",
            "retrieval_necessity_score": retrieval_score,
            "needs_retrieval": needs_retrieval,
        })
        logger.debug(f"[SelfRAG/LC] retrieval_necessity={retrieval_score:.2f} → needs_retrieval={needs_retrieval}")

        if not needs_retrieval:
            answer = self._generate_without_retrieval(question)
            return RAGResult(
                query=question,
                answer=answer,
                source_documents=[],
                intermediate_steps=intermediate_steps,
                metadata={"retrieval_used": False, "iterations": 0},
            )

        best_answer, best_docs, best_support = None, [], -1.0
        retrieval_query = question
        iterations_run = 0

        for i in range(self.max_iterations):
            iterations_run += 1
            candidates = self._retrieve(retrieval_query)
            relevant_docs, relevance_scores = self._grade_relevance(question, candidates)
            used_docs = relevant_docs if relevant_docs else candidates

            answer = self._generate_with_context(question, used_docs)
            support_score = self._grade_support(question, used_docs, answer)

            intermediate_steps.append({
                "step": f"iteration_{i + 1}",
                "retrieval_query": retrieval_query,
                "num_candidates": len(candidates),
                "num_relevant": len(relevant_docs),
                "support_score": support_score,
            })
            logger.debug(
                f"[SelfRAG/LC] iteration {i + 1}: {len(relevant_docs)}/{len(candidates)} relevant, "
                f"support={support_score:.2f}"
            )

            if support_score > best_support:
                best_answer, best_docs, best_support = answer, used_docs, support_score

            if support_score >= self.support_threshold:
                break

            if i < self.max_iterations - 1:
                retrieval_query = self._refine_query(question, answer)

        return RAGResult(
            query=question,
            answer=best_answer,
            source_documents=[
                Document(content=doc.page_content, metadata=doc.metadata) for doc in best_docs
            ],
            intermediate_steps=intermediate_steps,
            metadata={
                "retrieval_used": True,
                "iterations": iterations_run,
                "final_support_score": best_support,
            },
        )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))

    docs = [
        "Quantum entanglement is a phenomenon where two or more particles become correlated such that the quantum state of each particle cannot be described independently of the others, even when separated by large distances.",
        "Bell's theorem proves that quantum mechanics predicts correlations between measurements that cannot be explained by local hidden variable theories.",
        "Quantum computing leverages superposition and entanglement to perform computations that would be intractable for classical computers.",
    ]

    rag = SelfRAGLangChain(config=ConfigLoader.get()._config)
    rag.index(docs)

    result = rag.query("How does entanglement help quantum computers and what does Bell's theorem have to do with it?")
    result.print_summary()

    # A question that shouldn't need retrieval at all
    result2 = rag.query("What is 2 + 2?")
    result2.print_summary()
