"""
Adaptive RAG: Dynamic strategy routing by query complexity — LangChain Implementation
==========================================================================================
Classifies each incoming query into a complexity level and routes it to the
cheapest pipeline that can answer it well:

  1. Classify   — one LLM call sorts the query into simple / moderate / complex
  2. Route      — per config, simple → answer from parametric knowledge alone,
                  moderate → single-pass dense retrieval + generation,
                  complex → multi-query expansion + dense retrieval + RRF fusion
  3. Generate   — the routed strategy produces context (or not) and an answer

The payoff is cost/latency proportionality: trivial questions never pay for
retrieval, hard questions get the heavy machinery.

Reference: https://github.com/NirDiamant/RAG_Techniques
Paper: "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language
Models through Question Complexity" (Jeong et al., 2024) — https://arxiv.org/abs/2403.14403
"""

import logging
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

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_VALID_COMPLEXITIES = ("simple", "moderate", "complex")


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning traces some LMStudio models emit."""
    return _THINK_TAG_RE.sub("", text).strip()


def _parse_complexity(text: str, default: str = "moderate") -> str:
    """
    Extract a complexity label (simple/moderate/complex) from an LLM response.

    Falls back to `default` when no known label appears in the response.
    """
    cleaned = _strip_thinking(text).lower()
    for label in _VALID_COMPLEXITIES:
        if label in cleaned:
            return label
    return default


def _parse_query_variants(text: str) -> list[str]:
    """Extract one query variant per line, stripping bullets and blank lines."""
    lines = _strip_thinking(text).split("\n")
    return [line.strip("- ").strip() for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CLASSIFY_PROMPT = ChatPromptTemplate.from_template(
    """Classify the following question into exactly one complexity level:

- "simple": a factual or common-knowledge question that can be answered
  correctly without any documents (e.g., definitions, arithmetic,
  well-known facts).
- "moderate": a question about specific subject matter that standard
  retrieval over documents can answer directly.
- "complex": a multi-part, comparative, or multi-hop question that needs
  evidence gathered from several angles before it can be answered.

Respond with ONLY one word: simple, moderate, or complex.

Question: {question}
Complexity:"""
)

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(
    """You are an AI assistant that generates multiple search query variations.
Generate {num_variants} different rephrasings of the following question to
improve document retrieval. Each version should use different wording or
emphasize a different aspect of the question.
Return ONLY the queries, one per line, no numbering.

Original question: {question}
Query variations:"""
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


class AdaptiveRAGLangChain(BaseRAG):
    """
    Adaptive RAG using LangChain: classify query complexity, route to the
    configured strategy (no-retrieval / naive / fusion), then execute it.

    Best for: Production systems facing mixed workloads — frequent easy
    questions that shouldn't pay retrieval costs, plus occasional hard ones
    that deserve multi-query evidence gathering — all through one endpoint.
    """

    TECHNIQUE_NAME = "adaptive_rag"
    FRAMEWORK = "langchain"

    def _build_pipeline(self) -> None:
        """Initialize LLM, embeddings, and the complexity→strategy routing map."""
        cfg = ConfigLoader.get()
        tech_cfg = cfg.get_technique_config("adaptive_rag")

        self.llm = get_langchain_llm()
        self.embeddings = get_langchain_embeddings()
        self.text_splitter = get_text_splitter()
        self.top_k = cfg.retrieval.get("top_k", 5)

        strategies_cfg = tech_cfg.get("strategies", {})
        self.strategies = {
            "simple": strategies_cfg.get("simple", "no_retrieval"),
            "moderate": strategies_cfg.get("moderate", "naive_rag"),
            "complex": strategies_cfg.get("complex", "fusion_rag"),
        }

        # Settings for the built-in complex strategy (RAG-Fusion style)
        self.num_queries = tech_cfg.get("num_queries", 3)
        self.rrf_k = tech_cfg.get("rrf_k", 60)

        self.vector_store = None

        logger.info(
            f"[AdaptiveRAG/LC] strategies={self.strategies}, "
            f"num_queries={self.num_queries}, rrf_k={self.rrf_k}"
        )

    def index(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        """Chunk, embed, and store documents in the configured vector store."""
        logger.info(f"[AdaptiveRAG/LC] Indexing {len(documents)} documents...")

        lc_docs = load_texts(documents, metadatas)
        chunks = self.text_splitter.split_documents(lc_docs)

        self.vector_store = build_langchain_vector_store(
            chunks,
            collection_name="adaptive_rag",
        )

        self._is_indexed = True
        logger.info(f"[AdaptiveRAG/LC] Indexed {len(chunks)} chunks ✓")

    # ------------------------------------------------------------------
    # Classification + shared building blocks
    # ------------------------------------------------------------------

    def _classify(self, question: str) -> str:
        """Sort the question into simple / moderate / complex via one LLM call."""
        chain = CLASSIFY_PROMPT | self.llm | StrOutputParser()
        result = chain.invoke({"question": question})
        return _parse_complexity(result, default="moderate")

    def _retrieve(self, query: str) -> list:
        retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
        return retriever.invoke(query)

    def _generate_with_context(self, question: str, docs: list) -> str:
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        chain = ANSWER_PROMPT | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})

    def _generate_without_retrieval(self, question: str) -> str:
        chain = NO_RETRIEVAL_PROMPT | self.llm | StrOutputParser()
        return chain.invoke({"question": question})

    # ------------------------------------------------------------------
    # Complex-strategy helpers (RAG-Fusion style, kept self-contained)
    # ------------------------------------------------------------------

    def _generate_query_variants(self, question: str) -> list[str]:
        """Generate (num_queries - 1) rephrasings; the original is added separately."""
        num_variants = max(self.num_queries - 1, 0)
        if num_variants == 0:
            return []
        chain = MULTI_QUERY_PROMPT | self.llm | StrOutputParser()
        result = chain.invoke({"question": question, "num_variants": num_variants})
        return _parse_query_variants(result)[:num_variants]

    def _reciprocal_rank_fusion(self, ranked_lists: list[list]) -> list:
        """Fuse multiple ranked lists into one, sorted by RRF score descending."""
        scores: dict[int, float] = {}
        doc_lookup: dict[int, object] = {}

        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list):
                key = hash(doc.page_content[:100])
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                doc_lookup.setdefault(key, doc)

        fused_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        return [doc_lookup[k] for k in fused_keys]

    # ------------------------------------------------------------------
    # Strategies — each returns (answer, source_docs)
    # ------------------------------------------------------------------

    def _strategy_no_retrieval(self, question: str) -> tuple[str, list]:
        """Simple route: answer straight from parametric knowledge."""
        return self._generate_without_retrieval(question), []

    def _strategy_naive(self, question: str) -> tuple[str, list]:
        """Moderate route: one dense retrieval pass, generate from results."""
        docs = self._retrieve(question)
        return self._generate_with_context(question, docs), docs

    def _strategy_fusion(self, question: str) -> tuple[str, list]:
        """Complex route: multi-query expansion → dense retrieval per variant → RRF."""
        queries = [question] + self._generate_query_variants(question)
        ranked_lists = [self._retrieve(q) for q in queries]
        fused_docs = self._reciprocal_rank_fusion(ranked_lists)
        top_docs = fused_docs[: self.top_k]
        return self._generate_with_context(question, top_docs), top_docs

    def _resolve_strategy(self, strategy_name: str):
        """Map a configured strategy name to its method; fall back to naive on unknowns."""
        dispatch = {
            "no_retrieval": self._strategy_no_retrieval,
            "naive_rag": self._strategy_naive,
            "fusion_rag": self._strategy_fusion,
        }
        strategy_fn = dispatch.get(strategy_name)
        if strategy_fn is None:
            logger.warning(
                "[AdaptiveRAG/LC] Unknown strategy '%s' — falling back to 'naive_rag'.",
                strategy_name,
            )
            return self._strategy_naive
        return strategy_fn

    def _query(self, question: str) -> RAGResult:
        """Full Adaptive RAG pipeline: classify → route → execute the chosen strategy."""
        if not self.vector_store:
            raise RuntimeError("Call index() before querying.")

        complexity = self._classify(question)
        strategy_name = self.strategies.get(complexity, "naive_rag")
        strategy_fn = self._resolve_strategy(strategy_name)

        logger.debug(f"[AdaptiveRAG/LC] complexity={complexity} → strategy={strategy_name}")
        answer, source_docs = strategy_fn(question)

        return RAGResult(
            query=question,
            answer=answer,
            source_documents=[
                Document(content=doc.page_content, metadata=doc.metadata)
                for doc in source_docs
            ],
            intermediate_steps=[{
                "step": "routing",
                "complexity": complexity,
                "routed_strategy": strategy_name,
            }],
            metadata={
                "complexity": complexity,
                "routed_strategy": strategy_name,
                "retrieval_used": strategy_name != "no_retrieval",
            },
        )
