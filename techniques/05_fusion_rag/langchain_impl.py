"""
Fusion RAG (RAG-Fusion) — LangChain Implementation
=====================================================
Generates multiple query variants, retrieves for each via both dense (vector)
and sparse (BM25) search, then fuses every ranked list with Reciprocal Rank
Fusion (RRF) — a principled alternative to Advanced/Query Transform RAG's
simple union+dedup, since RRF rewards documents that rank highly across
several retrieval passes instead of treating every hit as equally relevant.

RRF score for a document d across ranked lists L:
    score(d) = sum over lists l in L of  1 / (rrf_k + rank_of(d, l))

Reference: https://github.com/NirDiamant/RAG_Techniques
Paper: "Reciprocal Rank Fusion outperforms Condorcet and individual Rank
Learning Methods" (Cormack et al., 2009) — https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
"""

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.base_rag import BaseRAG, Document, RAGResult
from core.config_loader import ConfigLoader
from core.document_loader import get_text_splitter, load_texts
from core.embeddings import get_langchain_embeddings
from core.llm_client import get_langchain_llm
from core.vector_store import build_langchain_vector_store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

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
    ("system", """You are a knowledgeable assistant. Answer the question based
ONLY on the provided context, which was assembled by fusing results from
several query reformulations and retrieval methods via Reciprocal Rank Fusion.
If the context doesn't contain sufficient information, acknowledge the limitation.

Context:
{context}"""),
    ("human", "{question}"),
])


class FusionRAGLangChain(BaseRAG):
    """
    Fusion RAG (RAG-Fusion) using LangChain.

    Combines multi-query expansion with hybrid dense+sparse retrieval, fusing
    every resulting ranked list via Reciprocal Rank Fusion (RRF) rather than
    simple union+dedup — documents that rank highly across multiple queries
    and/or retrieval methods surface to the top, even if no single pass
    ranked them first.

    Best for: Cases where a single retrieval pass misses relevant docs,
    diverse document sets, and production search systems that want robust
    ranking without a separate cross-encoder reranking stage.
    """

    TECHNIQUE_NAME = "fusion_rag"
    FRAMEWORK = "langchain"

    def _build_pipeline(self) -> None:
        """Initialize LLM, embeddings, and RAG-Fusion settings."""
        cfg = ConfigLoader.get()
        tech_cfg = cfg.get_technique_config("fusion_rag")

        self.llm = get_langchain_llm()
        self.embeddings = get_langchain_embeddings()
        self.text_splitter = get_text_splitter()

        self.top_k = cfg.retrieval.get("top_k", 5)
        self.num_queries = tech_cfg.get("num_queries", 4)  # total, including the original
        self.rrf_k = tech_cfg.get("rrf_k", 60)
        self.hybrid_search = tech_cfg.get("hybrid_search", True)

        self.vector_store = None
        self.bm25_retriever = None

        logger.info(
            f"[FusionRAG/LC] num_queries={self.num_queries}, rrf_k={self.rrf_k}, "
            f"hybrid_search={self.hybrid_search}"
        )

    def index(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        """Chunk, embed, and store documents. Also builds a BM25 sparse index if hybrid search is enabled."""  # noqa: E501
        logger.info(f"[FusionRAG/LC] Indexing {len(documents)} documents...")

        lc_docs = load_texts(documents, metadatas)
        chunks = self.text_splitter.split_documents(lc_docs)

        self.vector_store = build_langchain_vector_store(
            chunks,
            collection_name="fusion_rag",
        )

        if self.hybrid_search:
            self._build_bm25_retriever(chunks)

        self._is_indexed = True
        logger.info(f"[FusionRAG/LC] Indexed {len(chunks)} chunks ✓")

    def _build_bm25_retriever(self, chunks: list) -> None:
        """Build the sparse (keyword) retriever half of hybrid search."""
        try:
            from langchain_community.retrievers import BM25Retriever

            self.bm25_retriever = BM25Retriever.from_documents(chunks)
            self.bm25_retriever.k = self.top_k
            logger.info("[FusionRAG/LC] BM25 sparse retriever built ✓")
        except Exception as e:
            logger.warning(f"[FusionRAG/LC] BM25 retriever unavailable: {e}. Falling back to dense-only retrieval.")  # noqa: E501
            self.bm25_retriever = None
            self.hybrid_search = False

    # ------------------------------------------------------------------
    # Multi-query generation
    # ------------------------------------------------------------------

    def _generate_query_variants(self, question: str) -> list[str]:
        """Generate (num_queries - 1) rephrasings; the original question is added separately."""
        num_variants = max(self.num_queries - 1, 0)
        if num_variants == 0:
            return []
        chain = MULTI_QUERY_PROMPT | self.llm | StrOutputParser()
        result = chain.invoke({"question": question, "num_variants": num_variants})
        variants = [line.strip("- ").strip() for line in result.strip().split("\n") if line.strip()]
        logger.debug(f"[FusionRAG/LC] Query variants: {variants}")
        return variants[:num_variants]

    # ------------------------------------------------------------------
    # Hybrid retrieval + Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _dense_retrieve(self, query: str) -> list:
        retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
        return retriever.invoke(query)

    def _sparse_retrieve(self, query: str) -> list:
        return self.bm25_retriever.invoke(query)

    def _reciprocal_rank_fusion(self, ranked_lists: list[list]) -> list:
        """Fuse multiple ranked lists into one, scoring by RRF and returning docs sorted by fused score."""  # noqa: E501
        scores: dict[int, float] = {}
        doc_lookup: dict[int, object] = {}

        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list):
                key = hash(doc.page_content[:100])
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                doc_lookup.setdefault(key, doc)

        fused_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        return [doc_lookup[k] for k in fused_keys]

    def _query(self, question: str) -> RAGResult:
        """Full RAG-Fusion pipeline: multi-query → hybrid retrieval per variant → RRF → answer."""
        if not self.vector_store:
            raise RuntimeError("Call index() before querying.")

        queries = [question] + self._generate_query_variants(question)

        ranked_lists = []
        for q in queries:
            ranked_lists.append(self._dense_retrieve(q))
            if self.hybrid_search and self.bm25_retriever:
                ranked_lists.append(self._sparse_retrieve(q))

        fused_docs = self._reciprocal_rank_fusion(ranked_lists)
        top_docs = fused_docs[: self.top_k]

        intermediate_steps = [
            {"step": "multi_query", "queries": queries},
            {
                "step": "rrf_fusion",
                "num_ranked_lists": len(ranked_lists),
                "num_unique_docs": len(fused_docs),
                "rrf_k": self.rrf_k,
            },
        ]
        logger.debug(
            f"[FusionRAG/LC] {len(queries)} queries × "
            f"{'hybrid' if self.hybrid_search else 'dense-only'} → "
            f"{len(ranked_lists)} ranked lists → {len(fused_docs)} fused docs"
        )

        context = "\n\n---\n\n".join(doc.page_content for doc in top_docs)
        answer_chain = ANSWER_PROMPT | self.llm | StrOutputParser()
        answer = answer_chain.invoke({"context": context, "question": question})

        return RAGResult(
            query=question,
            answer=answer,
            source_documents=[
                Document(content=doc.page_content, metadata=doc.metadata)
                for doc in top_docs
            ],
            intermediate_steps=intermediate_steps,
            metadata={
                "num_query_variants": len(queries),
                "hybrid_search": self.hybrid_search,
                "rrf_k": self.rrf_k,
                "num_docs_retrieved": len(fused_docs),
            },
        )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))

    docs = [
        "Quantum entanglement is a phenomenon where two or more particles become correlated such that the quantum state of each particle cannot be described independently of the others, even when separated by large distances.",  # noqa: E501
        "Bell's theorem proves that quantum mechanics predicts correlations between measurements that cannot be explained by local hidden variable theories.",  # noqa: E501
        "Quantum computing leverages superposition and entanglement to perform computations that would be intractable for classical computers.",  # noqa: E501
    ]

    rag = FusionRAGLangChain(config=ConfigLoader.get()._config)
    rag.index(docs)
    result = rag.query("How does entanglement help quantum computers and what does Bell's theorem have to do with it?")  # noqa: E501
    result.print_summary()
