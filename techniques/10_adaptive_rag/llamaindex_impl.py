"""
Adaptive RAG: Dynamic strategy routing by query complexity — LlamaIndex Implementation
==========================================================================================
Same classify-then-route pipeline as the LangChain version: one LLM call sorts
the question into simple / moderate / complex, then the configured strategy
runs — parametric-knowledge answering for simple queries, single-pass dense
retrieval for moderate ones, and multi-query expansion with Reciprocal Rank
Fusion for complex ones -- implemented with direct LLM `.complete()` calls
for parity with the LangChain implementation.
"""

import logging
import re

from core.base_rag import BaseRAG, Document, RAGResult
from core.config_loader import ConfigLoader
from core.embeddings import get_llamaindex_embeddings
from core.llm_client import get_llamaindex_llm

logger = logging.getLogger(__name__)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_VALID_COMPLEXITIES = ("simple", "moderate", "complex")


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning traces some LMStudio models emit in raw completions."""
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


CLASSIFY_PROMPT = """Classify the following question into exactly one complexity level:

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

MULTI_QUERY_PROMPT = """You are an AI assistant that generates multiple search \
query variations. Generate {num_variants} different rephrasings of the following \
question to improve document retrieval. Each version should use different wording \
or emphasize a different aspect of the question. Return ONLY the queries, one per \
line, no numbering.

Original question: {question}
Query variations:"""

ANSWER_PROMPT = """Answer the question based ONLY on the provided context. \
If the context doesn't contain sufficient information, acknowledge the limitation.

Context:
{context}

Question: {question}
Answer:"""

NO_RETRIEVAL_PROMPT = """Answer the following question directly using your own knowledge. \
Be concise and accurate.

Question: {question}
Answer:"""


class AdaptiveRAGLlamaIndex(BaseRAG):
    """
    Adaptive RAG using LlamaIndex: classify query complexity, route to the
    configured strategy (no-retrieval / naive / fusion), then execute it.

    Best for: Production systems facing mixed workloads — frequent easy
    questions that shouldn't pay retrieval costs, plus occasional hard ones
    that deserve multi-query evidence gathering — all through one endpoint.
    """

    TECHNIQUE_NAME = "adaptive_rag"
    FRAMEWORK = "llamaindex"

    def _build_pipeline(self) -> None:
        """Initialize LLM, embeddings, and the complexity→strategy routing map."""
        from llama_index.core import Settings
        from llama_index.core.node_parser import SentenceSplitter

        cfg = ConfigLoader.get()
        tech_cfg = cfg.get_technique_config("adaptive_rag")
        doc_cfg = cfg.document

        self.llm = get_llamaindex_llm()
        Settings.llm = self.llm
        Settings.embed_model = get_llamaindex_embeddings()
        Settings.node_parser = SentenceSplitter(
            chunk_size=doc_cfg.get("chunk_size", 512),
            chunk_overlap=doc_cfg.get("chunk_overlap", 50),
        )

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

        self.vector_index = None
        self.retriever = None

        logger.info(
            f"[AdaptiveRAG/LI] strategies={self.strategies}, "
            f"num_queries={self.num_queries}, rrf_k={self.rrf_k}"
        )

    def index(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        """Build a VectorStoreIndex from raw text strings."""
        from llama_index.core import VectorStoreIndex
        from llama_index.core.schema import Document as LIDocument

        logger.info(f"[AdaptiveRAG/LI] Indexing {len(documents)} documents...")

        metas = metadatas or [{}] * len(documents)
        li_docs = [LIDocument(text=t, metadata=m) for t, m in zip(documents, metas, strict=True)]

        self.vector_index = VectorStoreIndex.from_documents(li_docs, show_progress=True)
        self.retriever = self.vector_index.as_retriever(similarity_top_k=self.top_k)

        self._is_indexed = True
        logger.info("[AdaptiveRAG/LI] Indexing complete ✓")

    # ------------------------------------------------------------------
    # Classification + shared building blocks
    # ------------------------------------------------------------------

    def _classify(self, question: str) -> str:
        """Sort the question into simple / moderate / complex via one LLM call."""
        response = self.llm.complete(CLASSIFY_PROMPT.format(question=question))
        return _parse_complexity(response.text, default="moderate")

    def _retrieve(self, query: str) -> list:
        return self.retriever.retrieve(query)

    def _generate_with_context(self, question: str, nodes: list) -> str:
        context = "\n\n---\n\n".join(node.get_content() for node in nodes)
        response = self.llm.complete(ANSWER_PROMPT.format(context=context, question=question))
        return _strip_thinking(response.text)

    def _generate_without_retrieval(self, question: str) -> str:
        response = self.llm.complete(NO_RETRIEVAL_PROMPT.format(question=question))
        return _strip_thinking(response.text)

    # ------------------------------------------------------------------
    # Complex-strategy helpers (RAG-Fusion style, kept self-contained)
    # ------------------------------------------------------------------

    def _generate_query_variants(self, question: str) -> list[str]:
        """Generate (num_queries - 1) rephrasings; the original is added separately."""
        num_variants = max(self.num_queries - 1, 0)
        if num_variants == 0:
            return []
        response = self.llm.complete(
            MULTI_QUERY_PROMPT.format(question=question, num_variants=num_variants)
        )
        return _parse_query_variants(response.text)[:num_variants]

    def _reciprocal_rank_fusion(self, ranked_lists: list[list]) -> list:
        """Fuse multiple ranked lists into one, sorted by RRF score descending."""
        scores: dict[int, float] = {}
        node_lookup: dict[int, object] = {}

        for ranked_list in ranked_lists:
            for rank, node in enumerate(ranked_list):
                key = hash(node.get_content()[:100])
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                node_lookup.setdefault(key, node)

        fused_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        return [node_lookup[k] for k in fused_keys]

    # ------------------------------------------------------------------
    # Strategies — each returns (answer, source_nodes)
    # ------------------------------------------------------------------

    def _strategy_no_retrieval(self, question: str) -> tuple[str, list]:
        """Simple route: answer straight from parametric knowledge."""
        return self._generate_without_retrieval(question), []

    def _strategy_naive(self, question: str) -> tuple[str, list]:
        """Moderate route: one dense retrieval pass, generate from results."""
        nodes = self._retrieve(question)
        return self._generate_with_context(question, nodes), nodes

    def _strategy_fusion(self, question: str) -> tuple[str, list]:
        """Complex route: multi-query expansion → dense retrieval per variant → RRF."""
        queries = [question] + self._generate_query_variants(question)
        ranked_lists = [self._retrieve(q) for q in queries]
        fused_nodes = self._reciprocal_rank_fusion(ranked_lists)
        top_nodes = fused_nodes[: self.top_k]
        return self._generate_with_context(question, top_nodes), top_nodes

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
                "[AdaptiveRAG/LI] Unknown strategy '%s' — falling back to 'naive_rag'.",
                strategy_name,
            )
            return self._strategy_naive
        return strategy_fn

    def _query(self, question: str) -> RAGResult:
        """Full Adaptive RAG pipeline: classify → route → execute the chosen strategy."""
        if not self.retriever:
            raise RuntimeError("Call index() before querying.")

        complexity = self._classify(question)
        strategy_name = self.strategies.get(complexity, "naive_rag")
        strategy_fn = self._resolve_strategy(strategy_name)

        logger.debug(f"[AdaptiveRAG/LI] complexity={complexity} → strategy={strategy_name}")
        answer, source_nodes = strategy_fn(question)

        return RAGResult(
            query=question,
            answer=answer,
            source_documents=[
                Document(content=node.get_content(), metadata=node.metadata, score=node.score)
                for node in source_nodes
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
