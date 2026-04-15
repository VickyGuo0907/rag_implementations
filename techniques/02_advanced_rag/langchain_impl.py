"""
Advanced RAG — LangChain Implementation
=========================================
Enhances naive RAG with three stages of optimization:

PRE-RETRIEVAL:
  - Query rewriting: Rephrase the query for better retrieval
  - Step-back prompting: Generate a more abstract version of the query

RETRIEVAL:
  - Multi-query retrieval: Generate N query variants, retrieve for each, deduplicate

POST-RETRIEVAL:
  - Reranking: Cross-encoder reranks retrieved chunks by true relevance
  - Contextual compression: LLM filters out irrelevant parts of retrieved chunks

Reference: https://github.com/NirDiamant/RAG_Techniques
"""

from typing import Dict, List, Optional
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.base_rag import BaseRAG, RAGResult, Document
from core.config_loader import ConfigLoader
from core.llm_client import get_langchain_llm
from core.embeddings import get_langchain_embeddings
from core.document_loader import load_texts, get_text_splitter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert at reformulating search queries to improve document retrieval.

Given the original query, rewrite it to be more specific and likely to find relevant documents.
Provide ONLY the rewritten query, no explanations.

Original query: {query}
Rewritten query:"""
)

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(
    """You are an AI assistant that generates multiple search query variations.
Generate {num_queries} different versions of the following query to improve document retrieval.
Each version should approach the question from a different angle.
Return ONLY the queries, one per line, no numbering.

Original query: {query}
Query variations:"""
)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a knowledgeable assistant. Answer based ONLY on the provided context.
Be precise and cite which parts of the context support your answer.
If the answer isn't in the context, say "The provided context doesn't contain this information."

Context:
{context}"""),
    ("human", "{question}"),
])


# ---------------------------------------------------------------------------
# Advanced RAG Implementation
# ---------------------------------------------------------------------------

class AdvancedRAGLangChain(BaseRAG):
    """
    Advanced RAG with pre-retrieval query enhancement and post-retrieval reranking.

    Pipeline:
        Query → Rewrite → Multi-query expand → Retrieve (all variants) →
        Deduplicate → Rerank → Compress → LLM → Answer

    Best for: Production Q&A systems, complex multi-document retrieval,
    high accuracy requirements.
    """

    TECHNIQUE_NAME = "advanced_rag"
    FRAMEWORK = "langchain"

    def _build_pipeline(self) -> None:
        """Initialize all components."""
        cfg = ConfigLoader.get()
        tech_cfg = cfg.get_technique_config("advanced_rag")

        self.llm = get_langchain_llm()
        self.embeddings = get_langchain_embeddings()
        self.text_splitter = get_text_splitter()

        self.top_k = cfg.retrieval.get("top_k", 5)
        self.initial_k = tech_cfg.get("initial_top_k", 20)
        self.enable_reranking = tech_cfg.get("reranking", True)
        self.enable_compression = tech_cfg.get("compression", True)
        self.enable_query_rewriting = tech_cfg.get("query_rewriting", True)
        self.num_queries = cfg.get_value("rag_techniques", "query_transform_rag", "num_queries", default=3)

        self.reranker = None
        self.retriever = None
        self.vector_store = None

        logger.info(f"[AdvancedRAG/LC] reranking={self.enable_reranking}, "
                    f"compression={self.enable_compression}, "
                    f"query_rewriting={self.enable_query_rewriting}")

    def index(self, documents: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        """Chunk, embed, and store documents. Build reranker and retriever."""
        from langchain_chroma import Chroma
        from langchain.retrievers.multi_query import MultiQueryRetriever

        logger.info(f"[AdvancedRAG/LC] Indexing {len(documents)} documents...")

        lc_docs = load_texts(documents, metadatas)
        chunks = self.text_splitter.split_documents(lc_docs)

        cfg = ConfigLoader.get()
        vs_cfg = cfg.vector_store

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=f"advanced_rag_{vs_cfg.get('collection_name', 'docs')}",
            persist_directory=vs_cfg.get("persist_directory", "./data/vector_store"),
        )

        # Base retriever (retrieves more candidates for reranking)
        base_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.initial_k}
        )

        # Multi-query retriever: generates N query variants automatically
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.llm,
        )

        # Build cross-encoder reranker (local, no API needed)
        if self.enable_reranking:
            self._build_reranker()

        self._is_indexed = True
        logger.info(f"[AdvancedRAG/LC] Indexed {len(chunks)} chunks ✓")

    def _build_reranker(self) -> None:
        """Initialize cross-encoder reranker for post-retrieval ranking."""
        try:
            from langchain.retrievers.document_compressors import CrossEncoderReranker
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder

            cfg = ConfigLoader.get()
            reranker_model = cfg.get_value(
                "rag_techniques", "reranking_rag", "reranker_model",
                default="cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            model = HuggingFaceCrossEncoder(model_name=reranker_model)
            self.reranker = CrossEncoderReranker(model=model, top_n=self.top_k)
            logger.info(f"[AdvancedRAG/LC] Reranker loaded: {reranker_model}")
        except Exception as e:
            logger.warning(f"[AdvancedRAG/LC] Reranker unavailable: {e}. Skipping.")
            self.enable_reranking = False

    def _rewrite_query(self, query: str) -> str:
        """Use LLM to rewrite query for better retrieval."""
        chain = QUERY_REWRITE_PROMPT | self.llm | StrOutputParser()
        rewritten = chain.invoke({"query": query})
        logger.debug(f"[AdvancedRAG/LC] Rewritten query: {rewritten}")
        return rewritten.strip()

    def _retrieve_and_rerank(self, query: str) -> List:
        """Full retrieval pipeline: multi-query → retrieve → rerank → compress."""
        # Step 1: Multi-query retrieval (auto-generates query variants)
        docs = self.retriever.invoke(query)

        # Deduplicate by content
        seen = set()
        unique_docs = []
        for doc in docs:
            h = hash(doc.page_content[:100])
            if h not in seen:
                seen.add(h)
                unique_docs.append(doc)

        logger.debug(f"[AdvancedRAG/LC] Retrieved {len(docs)} → deduplicated to {len(unique_docs)}")

        # Step 2: Rerank with cross-encoder
        if self.enable_reranking and self.reranker:
            try:
                reranked = self.reranker.compress_documents(unique_docs, query)
                return reranked[:self.top_k]
            except Exception as e:
                logger.warning(f"[AdvancedRAG/LC] Reranking failed: {e}")

        return unique_docs[:self.top_k]

    def _query(self, question: str) -> RAGResult:
        """Full advanced RAG pipeline."""
        if not self.retriever:
            raise RuntimeError("Call index() before querying.")

        intermediate_steps = []

        # Step 1: Query rewriting (pre-retrieval)
        effective_query = question
        if self.enable_query_rewriting:
            effective_query = self._rewrite_query(question)
            intermediate_steps.append({
                "step": "query_rewriting",
                "original": question,
                "rewritten": effective_query,
            })

        # Step 2: Multi-query retrieval + reranking (post-retrieval)
        retrieved_docs = self._retrieve_and_rerank(effective_query)
        intermediate_steps.append({
            "step": "retrieval",
            "num_docs": len(retrieved_docs),
        })

        # Step 3: Build context and generate answer
        context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)

        chain = RAG_PROMPT | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        return RAGResult(
            query=question,
            answer=answer,
            source_documents=[
                Document(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=getattr(doc, "metadata", {}).get("relevance_score"),
                )
                for doc in retrieved_docs
            ],
            intermediate_steps=intermediate_steps,
            metadata={
                "rewritten_query": effective_query,
                "reranking_enabled": self.enable_reranking,
                "num_retrieved": len(retrieved_docs),
            },
        )


# ---------------------------------------------------------------------------
# Quick Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))

    docs = [
        "The transformer architecture introduced in 'Attention is All You Need' (2017) revolutionized NLP by replacing recurrent networks with self-attention mechanisms.",
        "BERT (Bidirectional Encoder Representations from Transformers) pre-trains deep bidirectional representations from unlabeled text by conditioning on both left and right context.",
        "GPT models use a decoder-only transformer architecture and are trained using causal language modeling, predicting the next token given all previous tokens.",
        "LLaMA (Large Language Meta AI) models are open-source language models trained on publicly available data, demonstrating that smaller models trained on more data can outperform larger ones.",
        "RAG enhances LLMs by retrieving relevant documents at inference time, reducing hallucinations and enabling knowledge updates without retraining.",
    ]

    rag = AdvancedRAGLangChain(config=ConfigLoader.get()._config)
    rag.index(docs)
    result = rag.query("How do transformer models differ from recurrent networks?")
    result.print_summary()
