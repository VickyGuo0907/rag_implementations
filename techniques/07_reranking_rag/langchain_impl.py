"""
Reranking RAG — LangChain Implementation
===========================================
Retrieves a wide candidate pool via standard dense vector search, then
reranks every (query, candidate) pair with a cross-encoder before generation.

Unlike Advanced RAG (which bundles reranking with query rewriting and
multi-query retrieval) or Fusion RAG (which fuses multiple ranked lists via
RRF), Reranking RAG isolates the single highest-ROI improvement over Naive
RAG: a cross-encoder jointly encodes the query and each candidate document,
producing far more accurate relevance scores than the bi-encoder similarity
used for the initial retrieval pass.

Reference: https://github.com/NirDiamant/RAG_Techniques
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


ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a knowledgeable assistant. Answer the question based
ONLY on the provided context, which has been reranked by relevance to the
question. If the context doesn't contain sufficient information, acknowledge
the limitation.

Context:
{context}"""),
    ("human", "{question}"),
])


class RerankingRAGLangChain(BaseRAG):
    """
    Reranking RAG using LangChain.

    Pipeline:
        Query → Vector Search (wide candidate pool) →
        Cross-Encoder Rerank (query, doc) pairs → Top-K → Prompt → LLM → Answer

    Best for: High-precision requirements where the initial bi-encoder
    similarity search ranks relevant documents outside the top-K -- a
    cross-encoder re-scores every candidate jointly with the query instead
    of comparing pre-computed embeddings independently.
    """

    TECHNIQUE_NAME = "reranking_rag"
    FRAMEWORK = "langchain"

    def _build_pipeline(self) -> None:
        """Initialize LLM, embeddings, text splitter, and the cross-encoder reranker."""
        cfg = ConfigLoader.get()
        tech_cfg = cfg.get_technique_config("reranking_rag")

        self.llm = get_langchain_llm()
        self.embeddings = get_langchain_embeddings()
        self.text_splitter = get_text_splitter()

        self.initial_k = tech_cfg.get("initial_top_k", 20)
        self.rerank_top_k = tech_cfg.get("rerank_top_k", 5)
        self.reranker_model_name = tech_cfg.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")  # noqa: E501

        self.vector_store = None
        self.reranker = None
        self._build_reranker()

        logger.info(
            f"[Reranking/LC] initial_top_k={self.initial_k}, rerank_top_k={self.rerank_top_k}, "
            f"model={self.reranker_model_name}"
        )

    def _build_reranker(self) -> None:
        """Load the cross-encoder reranker (local, no API needed)."""
        try:
            try:
                from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import (
                    CrossEncoderReranker,
                )
            except (ImportError, ModuleNotFoundError):
                from langchain_community.document_compressors import CrossEncoderReranker
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder

            model = HuggingFaceCrossEncoder(model_name=self.reranker_model_name)
            self.reranker = CrossEncoderReranker(model=model, top_n=self.rerank_top_k)
            logger.info(f"[Reranking/LC] Reranker loaded: {self.reranker_model_name}")
        except Exception as e:
            logger.warning(f"[Reranking/LC] Reranker unavailable: {e}. Falling back to dense-only top-K.")  # noqa: E501
            self.reranker = None

    def index(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        """Chunk, embed, and store documents in the configured vector store."""
        logger.info(f"[Reranking/LC] Indexing {len(documents)} documents...")

        lc_docs = load_texts(documents, metadatas)
        chunks = self.text_splitter.split_documents(lc_docs)

        self.vector_store = build_langchain_vector_store(
            chunks,
            collection_name="reranking_rag",
        )

        self._is_indexed = True
        logger.info(f"[Reranking/LC] Indexed {len(chunks)} chunks ✓")

    def _query(self, question: str) -> RAGResult:
        """Retrieve a wide candidate pool, rerank with the cross-encoder, then answer."""
        if not self.vector_store:
            raise RuntimeError("Call index() before querying.")

        retriever = self.vector_store.as_retriever(search_kwargs={"k": self.initial_k})
        candidates = retriever.invoke(question)

        if self.reranker:
            top_docs = self.reranker.compress_documents(candidates, question)
        else:
            top_docs = candidates[: self.rerank_top_k]

        intermediate_steps = [
            {"step": "retrieval", "num_candidates": len(candidates)},
            {
                "step": "rerank",
                "reranking_enabled": bool(self.reranker),
                "reranker_model": self.reranker_model_name,
                "num_after_rerank": len(top_docs),
            },
        ]
        logger.debug(f"[Reranking/LC] {len(candidates)} candidates → {len(top_docs)} after rerank")

        context = "\n\n---\n\n".join(doc.page_content for doc in top_docs)
        answer_chain = ANSWER_PROMPT | self.llm | StrOutputParser()
        answer = answer_chain.invoke({"context": context, "question": question})

        return RAGResult(
            query=question,
            answer=answer,
            source_documents=[
                Document(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=doc.metadata.get("relevance_score"),
                )
                for doc in top_docs
            ],
            intermediate_steps=intermediate_steps,
            metadata={
                "initial_top_k": self.initial_k,
                "rerank_top_k": self.rerank_top_k,
                "reranking_enabled": bool(self.reranker),
                "reranker_model": self.reranker_model_name,
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

    rag = RerankingRAGLangChain(config=ConfigLoader.get()._config)
    rag.index(docs)
    result = rag.query("How does entanglement help quantum computers and what does Bell's theorem have to do with it?")  # noqa: E501
    result.print_summary()
