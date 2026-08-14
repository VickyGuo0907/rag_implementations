"""
HyDE RAG (Hypothetical Document Embeddings) — LangChain Implementation
========================================================================
Instead of embedding the raw query for retrieval, HyDE:
  1. Asks the LLM to generate a hypothetical document that WOULD answer the query
  2. Embeds that hypothetical document
  3. Uses those embeddings to find real documents that are similar to the ideal answer

This bridges the vocabulary gap between short queries and long documents.
Paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al. 2022)
Reference: https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/HyDe_RAG.ipynb
"""

from typing import Dict, List, Optional
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.base_rag import BaseRAG, RAGResult, Document
from core.config_loader import ConfigLoader
from core.llm_client import get_langchain_llm
from core.embeddings import get_langchain_embeddings
from core.document_loader import load_texts, get_text_splitter
from core.vector_store import build_langchain_vector_store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

HYDE_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert document writer. Write a detailed, factual passage that
directly answers the following question. Write it as if it were an excerpt from
an authoritative reference document. Be specific and informative.

Question: {question}

Hypothetical document passage (2-4 paragraphs):"""
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Answer based ONLY on the provided context.
If the context doesn't contain sufficient information, acknowledge the limitation.

Context:
{context}"""),
    ("human", "{question}"),
])


class HyDERAGLangChain(BaseRAG):
    """
    HyDE RAG using LangChain.

    Key insight: The embedding of an ideal answer is closer in vector space
    to relevant documents than the embedding of a short user query.

    Best for:
      - When query phrasing differs from document style (e.g., questions vs. statements)
      - Technical documentation retrieval
      - Academic paper search
      - When Naive RAG has poor recall due to vocabulary mismatch
    """

    TECHNIQUE_NAME = "hyde_rag"
    FRAMEWORK = "langchain"

    def _build_pipeline(self) -> None:
        cfg = ConfigLoader.get()
        tech_cfg = cfg.get_technique_config("hyde_rag")

        self.llm = get_langchain_llm()
        self.embeddings = get_langchain_embeddings()
        self.text_splitter = get_text_splitter()
        self.top_k = cfg.retrieval.get("top_k", 5)
        self.num_hypothetical = tech_cfg.get("num_hypothetical_docs", 3)
        self.vector_store = None

        logger.info(f"[HyDE/LC] num_hypothetical={self.num_hypothetical}")

    def index(self, documents: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        """Standard indexing — real documents are stored as-is."""
        lc_docs = load_texts(documents, metadatas)
        chunks = self.text_splitter.split_documents(lc_docs)

        self.vector_store = build_langchain_vector_store(
            chunks,
            collection_name="hyde_rag",
        )
        self._is_indexed = True
        logger.info(f"[HyDE/LC] Indexed {len(chunks)} chunks ✓")

    def _generate_hypothetical_document(self, question: str) -> str:
        """Use LLM to generate a hypothetical answer document."""
        chain = HYDE_PROMPT | self.llm | StrOutputParser()
        return chain.invoke({"question": question})

    def _query(self, question: str) -> RAGResult:
        """HyDE retrieval: generate hypothetical doc → embed → retrieve real docs."""
        if not self.vector_store:
            raise RuntimeError("Call index() before querying.")

        # Step 1: Generate hypothetical document
        hypothetical_doc = self._generate_hypothetical_document(question)
        logger.debug(f"[HyDE/LC] Hypothetical doc: {hypothetical_doc[:100]}...")

        # Step 2: Embed the hypothetical document (not the original query)
        hyp_embedding = self.embeddings.embed_query(hypothetical_doc)

        # Step 3: Retrieve real documents using hypothetical embedding
        retrieved_docs = self.vector_store.similarity_search_by_vector(
            embedding=hyp_embedding,
            k=self.top_k,
        )

        # Step 4: Generate answer from real retrieved context
        context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)
        answer_chain = ANSWER_PROMPT | self.llm | StrOutputParser()
        answer = answer_chain.invoke({"context": context, "question": question})

        return RAGResult(
            query=question,
            answer=answer,
            source_documents=[
                Document(content=doc.page_content, metadata=doc.metadata)
                for doc in retrieved_docs
            ],
            intermediate_steps=[
                {"step": "hypothetical_doc", "content": hypothetical_doc}
            ],
            metadata={"hypothetical_document_preview": hypothetical_doc[:200]},
        )

