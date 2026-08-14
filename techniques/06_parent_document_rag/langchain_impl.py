"""
Parent Document RAG — LangChain Implementation
=================================================
Splits documents into large "parent" chunks, then further splits each parent
into small "child" chunks. Only the child chunks are embedded and searched —
small, focused text embeds and matches queries far better than a large,
diffuse chunk. But when a child chunk is retrieved, its *parent* is returned
to the LLM instead of the tiny child snippet, giving the model the full
surrounding context the child alone would lack.

Uses LangChain's built-in ParentDocumentRetriever, which handles the
child-embed / parent-return bookkeeping internally via a vectorstore (for
child embeddings) paired with a docstore (holding parent documents).

Reference: https://github.com/NirDiamant/RAG_Techniques
"""

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.base_rag import BaseRAG, Document, RAGResult
from core.config_loader import ConfigLoader
from core.document_loader import load_texts
from core.embeddings import get_langchain_embeddings
from core.llm_client import get_langchain_llm
from core.vector_store import get_langchain_vector_store

logger = logging.getLogger(__name__)


ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a knowledgeable assistant. Answer the question based
ONLY on the provided context, which consists of full parent documents
surrounding the specific passages that matched your query.
If the context doesn't contain sufficient information, acknowledge the limitation.

Context:
{context}"""),
    ("human", "{question}"),
])


class ParentDocumentRAGLangChain(BaseRAG):
    """
    Parent Document RAG using LangChain's ParentDocumentRetriever.

    Pipeline:
        Documents → Parent Splitter (large chunks) → Child Splitter (small chunks) →
        Embed children only → Query → Retrieve by child similarity → Return parents → LLM → Answer

    Best for: Long documents needing precise retrieval, limited context windows,
    technical manuals/books/reports where a small chunk alone lacks context.
    """

    TECHNIQUE_NAME = "parent_document_rag"
    FRAMEWORK = "langchain"

    def _build_pipeline(self) -> None:
        """Initialize LLM, embeddings, and parent/child chunk sizes."""
        cfg = ConfigLoader.get()
        tech_cfg = cfg.get_technique_config("parent_document_rag")
        doc_cfg = cfg.document

        self.llm = get_langchain_llm()
        self.embeddings = get_langchain_embeddings()

        self.parent_chunk_size = tech_cfg.get("parent_chunk_size", 2000)
        self.child_chunk_size = tech_cfg.get("child_chunk_size", 200)
        self.child_chunk_overlap = doc_cfg.get("chunk_overlap", 50)
        self.top_k = cfg.retrieval.get("top_k", 5)

        self.vector_store = None
        self.docstore = None
        self.retriever = None

        logger.info(
            f"[ParentDoc/LC] parent_chunk_size={self.parent_chunk_size}, "
            f"child_chunk_size={self.child_chunk_size}"
        )

    def index(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        """Build a ParentDocumentRetriever: embed child chunks, store parent chunks for return."""
        from langchain_classic.retrievers import ParentDocumentRetriever
        from langchain_core.stores import InMemoryStore
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        logger.info(f"[ParentDoc/LC] Indexing {len(documents)} documents...")

        lc_docs = load_texts(documents, metadatas)

        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=self.parent_chunk_size, chunk_overlap=0)  # noqa: E501
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_chunk_overlap,
        )

        self.vector_store = get_langchain_vector_store(
            collection_name="parent_document_rag",
        )
        self.docstore = InMemoryStore()

        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vector_store,
            docstore=self.docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={"k": self.top_k},
        )
        self.retriever.add_documents(lc_docs)

        self._is_indexed = True
        num_parents = len(list(self.docstore.yield_keys()))
        logger.info(f"[ParentDoc/LC] Indexed {num_parents} parent chunks ✓")

    def _query(self, question: str) -> RAGResult:
        """Retrieve by child-chunk similarity, but return the parent chunks for context."""
        if not self.retriever:
            raise RuntimeError("Call index() before querying.")

        parent_docs = self.retriever.invoke(question)

        context = "\n\n---\n\n".join(doc.page_content for doc in parent_docs)
        answer_chain = ANSWER_PROMPT | self.llm | StrOutputParser()
        answer = answer_chain.invoke({"context": context, "question": question})

        return RAGResult(
            query=question,
            answer=answer,
            source_documents=[
                Document(content=doc.page_content, metadata=doc.metadata)
                for doc in parent_docs
            ],
            metadata={
                "parent_chunk_size": self.parent_chunk_size,
                "child_chunk_size": self.child_chunk_size,
                "num_parents_returned": len(parent_docs),
            },
        )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))

    docs = [
        "Quantum entanglement is a phenomenon where two or more particles become correlated such that the quantum state of each particle cannot be described independently of the others, even when separated by large distances. "  # noqa: E501
        "This correlation persists regardless of the distance separating the particles, a property Einstein famously called 'spooky action at a distance.' "  # noqa: E501
        "Bell's theorem proves that quantum mechanics predicts correlations between measurements that cannot be explained by local hidden variable theories, providing a rigorous mathematical foundation for the non-classical nature of entanglement. "  # noqa: E501
        "Quantum computing leverages superposition and entanglement to perform computations that would be intractable for classical computers, using entangled qubits to represent and manipulate exponentially large state spaces.",  # noqa: E501
    ]

    rag = ParentDocumentRAGLangChain(config=ConfigLoader.get()._config)
    rag.index(docs)
    result = rag.query("How does entanglement help quantum computers and what does Bell's theorem have to do with it?")  # noqa: E501
    result.print_summary()
