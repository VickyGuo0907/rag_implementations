"""
Naive RAG — LangChain Implementation
======================================
The baseline RAG pipeline:
  1. Load & chunk documents
  2. Embed chunks → vector store
  3. Embed query → retrieve top-K similar chunks
  4. Stuff chunks into prompt → LLM generates answer

This is the starting point for all RAG systems.
Reference: https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/simple_rag.ipynb
"""

from typing import Dict, List, Optional
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from core.base_rag import BaseRAG, RAGResult, Document
from core.config_loader import ConfigLoader
from core.llm_client import get_langchain_llm
from core.embeddings import get_langchain_embeddings
from core.document_loader import load_texts, get_text_splitter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant. Answer the user's question based ONLY on the provided context.
    If the answer is not in the context, say "I don't have enough information to answer this question."
    Context:
    {context}

    Question:
    {question}
    """
)


def format_docs(docs) -> str:
    """Format retrieved LangChain documents into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------------------------
# Naive RAG Implementation
# ---------------------------------------------------------------------------

class NaiveRAGLangChain(BaseRAG):
    """
    Naive (Simple) RAG using LangChain.

    Pipeline:
        Documents → TextSplitter → Embeddings → ChromaDB
        Query → Embed → Retrieve top-K → Prompt → LLM → Answer

    Best for: Prototyping, baseline testing, simple Q&A over small document sets.
    """

    TECHNIQUE_NAME = "naive_rag"
    FRAMEWORK = "langchain"

    def _build_pipeline(self) -> None:
        """Initialize LLM, embeddings, and text splitter. Vector store built at index time."""
        self.llm = get_langchain_llm()
        self.embeddings = get_langchain_embeddings()
        self.text_splitter = get_text_splitter()
        self.retriever = None
        self.chain = None

        cfg = ConfigLoader.get()
        self.top_k = cfg.retrieval.get("top_k", 5)

        logger.info(f"[NaiveRAG/LC] Initialized | model={cfg.lmstudio.get('model')}")

    def index(self, documents: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        """
        Chunk, embed, and store documents in ChromaDB.

        Args:
            documents: Raw text strings to index.
            metadatas: Optional per-document metadata.
        """
        from langchain_chroma import Chroma

        logger.info(f"[NaiveRAG/LC] Indexing {len(documents)} documents...")

        # 1. Wrap raw strings in LangChain Document objects
        lc_docs = load_texts(documents, metadatas)

        # 2. Chunk documents
        chunks = self.text_splitter.split_documents(lc_docs)
        logger.info(f"[NaiveRAG/LC] Split into {len(chunks)} chunks")

        # 3. Embed and store in ChromaDB
        cfg = ConfigLoader.get()
        vs_cfg = cfg.vector_store
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=f"naive_rag_{vs_cfg.get('collection_name', 'docs')}",
            persist_directory=vs_cfg.get("persist_directory", "./data/vector_store"),
        )

        # 4. Build retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k},
        )

        # 5. Build LCEL chain: retrieve → format → prompt → LLM → parse
        self.chain = (
                {
                    "context": self.retriever | format_docs,
                    "question": RunnablePassthrough(),
                }
                | RAG_PROMPT
                | self.llm
                | StrOutputParser()
        )

        self._is_indexed = True
        logger.info("[NaiveRAG/LC] Indexing complete ✓")

    def _query(self, question: str) -> RAGResult:
        """Run the full naive RAG pipeline."""
        if not self.chain:
            raise RuntimeError("Call index() before querying.")

        # Generate answer via LCEL chain
        answer = self.chain.invoke(question)

        # Retrieve source documents for result attribution
        source_docs = self.retriever.invoke(question)

        return RAGResult(
            query=question,
            answer=answer,
            source_documents=[
                Document(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=None,  # Basic similarity search doesn't return scores
                )
                for doc in source_docs
            ],
            metadata={"num_chunks_indexed": len(self.vector_store.get()["ids"])},
        )
