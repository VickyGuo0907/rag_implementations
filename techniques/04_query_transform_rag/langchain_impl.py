"""
Query Transform RAG — LangChain Implementation
=================================================
Reformulates the user's query through three complementary strategies before
retrieval, then merges all results into one deduplicated context:

  - Step-back: Abstract the query to a higher-level, more general question
    (bridges overly-specific queries to broader document context).
  - Decompose: Break a complex question into simpler sub-questions and
    retrieve for each (helps multi-hop / compound questions).
  - Multi-query: Generate N differently-worded variants of the query and
    retrieve for each (classic recall-boosting trick, same idea as
    Advanced RAG's multi-query retriever).

Each strategy is independently toggleable via config.yaml
(rag_techniques.query_transform_rag.strategies).

Reference: https://github.com/NirDiamant/RAG_Techniques
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

STEP_BACK_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert at world knowledge. Your task is to step back and
paraphrase a question into a more generic, higher-level "step-back" question
that is easier to look up in a document. Return ONLY the step-back question.

Original question: {question}
Step-back question:"""
)

DECOMPOSE_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert at decomposing complex questions. Break the following
question into {num_subquestions} simpler sub-questions that, if each were
answered, would together answer the original question.
Return ONLY the sub-questions, one per line, no numbering.

Original question: {question}
Sub-questions:"""
)

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(
    """You are an AI assistant that generates multiple search query variations.
Generate {num_queries} different rephrasings of the following question to
improve document retrieval. Each version should use different wording or
emphasize a different aspect of the question.
Return ONLY the queries, one per line, no numbering.

Original question: {question}
Query variations:"""
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a knowledgeable assistant. Answer the question based
ONLY on the provided context, which was gathered from several reformulations
of the original question (step-back, decomposition, and/or multi-query).
Synthesize a single, coherent answer.
If the context doesn't contain sufficient information, acknowledge the limitation.

Context:
{context}"""),
    ("human", "{question}"),
])


class QueryTransformRAGLangChain(BaseRAG):
    """
    Query Transform RAG using LangChain.

    Combines step-back abstraction, query decomposition, and multi-query
    expansion to retrieve a broader, more relevant candidate set before
    answering — each strategy contributes additional retrieval queries,
    and results are deduplicated before being handed to the LLM.

    Best for: Complex multi-hop questions, ambiguous/vague queries, and
    generally improving recall over Naive RAG's single-query retrieval.
    """

    TECHNIQUE_NAME = "query_transform_rag"
    FRAMEWORK = "langchain"

    def _build_pipeline(self) -> None:
        """Initialize LLM, embeddings, and enabled query-transform strategies."""
        cfg = ConfigLoader.get()
        tech_cfg = cfg.get_technique_config("query_transform_rag")

        self.llm = get_langchain_llm()
        self.embeddings = get_langchain_embeddings()
        self.text_splitter = get_text_splitter()

        self.top_k = cfg.retrieval.get("top_k", 5)
        self.strategies = tech_cfg.get("strategies", ["step_back", "decompose", "multi_query"])
        self.num_queries = tech_cfg.get("num_queries", 3)
        self.vector_store = None

        logger.info(f"[QueryTransform/LC] strategies={self.strategies}, num_queries={self.num_queries}")

    def index(self, documents: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        """Chunk, embed, and store documents (standard indexing, unchanged by query transforms)."""
        logger.info(f"[QueryTransform/LC] Indexing {len(documents)} documents...")

        lc_docs = load_texts(documents, metadatas)
        chunks = self.text_splitter.split_documents(lc_docs)

        self.vector_store = build_langchain_vector_store(
            chunks,
            collection_name="query_transform_rag",
        )

        self._is_indexed = True
        logger.info(f"[QueryTransform/LC] Indexed {len(chunks)} chunks ✓")

    # ------------------------------------------------------------------
    # Query transformation strategies
    # ------------------------------------------------------------------

    def _generate_step_back(self, question: str) -> str:
        """Generate a more abstract/general version of the query."""
        chain = STEP_BACK_PROMPT | self.llm | StrOutputParser()
        step_back_query = chain.invoke({"question": question}).strip()
        logger.debug(f"[QueryTransform/LC] Step-back query: {step_back_query}")
        return step_back_query

    def _decompose_query(self, question: str) -> List[str]:
        """Break the query into simpler sub-questions."""
        chain = DECOMPOSE_PROMPT | self.llm | StrOutputParser()
        result = chain.invoke({"question": question, "num_subquestions": self.num_queries})
        sub_questions = [line.strip("- ").strip() for line in result.strip().split("\n") if line.strip()]
        logger.debug(f"[QueryTransform/LC] Sub-questions: {sub_questions}")
        return sub_questions

    def _generate_multi_query(self, question: str) -> List[str]:
        """Generate N differently-worded variants of the query."""
        chain = MULTI_QUERY_PROMPT | self.llm | StrOutputParser()
        result = chain.invoke({"question": question, "num_queries": self.num_queries})
        variants = [line.strip("- ").strip() for line in result.strip().split("\n") if line.strip()]
        logger.debug(f"[QueryTransform/LC] Multi-query variants: {variants}")
        return variants

    def _retrieve_deduplicated(self, queries: List[str]) -> List:
        """Retrieve top-K docs for each query, merging and deduplicating by content."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})

        seen = set()
        unique_docs = []
        for q in queries:
            for doc in retriever.invoke(q):
                h = hash(doc.page_content[:100])
                if h not in seen:
                    seen.add(h)
                    unique_docs.append(doc)
        return unique_docs

    def _query(self, question: str) -> RAGResult:
        """Full query-transform pipeline: transform → retrieve (all queries) → dedupe → answer."""
        if not self.vector_store:
            raise RuntimeError("Call index() before querying.")

        all_queries = [question]
        intermediate_steps = []

        if "step_back" in self.strategies:
            step_back_query = self._generate_step_back(question)
            all_queries.append(step_back_query)
            intermediate_steps.append({"step": "step_back", "query": step_back_query})

        if "decompose" in self.strategies:
            sub_questions = self._decompose_query(question)
            all_queries.extend(sub_questions)
            intermediate_steps.append({"step": "decompose", "sub_questions": sub_questions})

        if "multi_query" in self.strategies:
            variants = self._generate_multi_query(question)
            all_queries.extend(variants)
            intermediate_steps.append({"step": "multi_query", "variants": variants})

        retrieved_docs = self._retrieve_deduplicated(all_queries)
        intermediate_steps.append({
            "step": "retrieval",
            "num_queries": len(all_queries),
            "num_docs": len(retrieved_docs),
        })
        logger.debug(f"[QueryTransform/LC] {len(all_queries)} queries → {len(retrieved_docs)} unique docs")

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
            intermediate_steps=intermediate_steps,
            metadata={
                "strategies_used": self.strategies,
                "num_queries_generated": len(all_queries),
                "num_docs_retrieved": len(retrieved_docs),
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

    rag = QueryTransformRAGLangChain(config=ConfigLoader.get()._config)
    rag.index(docs)
    result = rag.query("How does entanglement help quantum computers and what does Bell's theorem have to do with it?")
    result.print_summary()
