"""
Query Transform RAG — LlamaIndex Implementation
==================================================
Same three-strategy query reformulation as the LangChain version — step-back
abstraction, decomposition, and multi-query expansion — implemented with
LlamaIndex's VectorStoreIndex and a direct LLM completion call per strategy
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


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning traces some LMStudio models emit in raw completions."""
    return _THINK_TAG_RE.sub("", text).strip()


STEP_BACK_PROMPT = """You are an expert at world knowledge. Your task is to step back and \
paraphrase a question into a more generic, higher-level "step-back" question \
that is easier to look up in a document. Return ONLY the step-back question.

Original question: {question}
Step-back question:"""

DECOMPOSE_PROMPT = """You are an expert at decomposing complex questions. Break the following \
question into {num_subquestions} simpler sub-questions that, if each were \
answered, would together answer the original question. \
Return ONLY the sub-questions, one per line, no numbering.

Original question: {question}
Sub-questions:"""

MULTI_QUERY_PROMPT = """You are an AI assistant that generates multiple search query variations. \
Generate {num_queries} different rephrasings of the following question to \
improve document retrieval. Each version should use different wording or \
emphasize a different aspect of the question. \
Return ONLY the queries, one per line, no numbering.

Original question: {question}
Query variations:"""

ANSWER_PROMPT = """You are a knowledgeable assistant. Answer the question based ONLY on the \
provided context, which was gathered from several reformulations of the \
original question (step-back, decomposition, and/or multi-query). Synthesize \
a single, coherent answer. If the context doesn't contain sufficient \
information, acknowledge the limitation.

Context:
{context}

Question: {question}
Answer:"""


class QueryTransformRAGLlamaIndex(BaseRAG):
    """
    Query Transform RAG using LlamaIndex.

    Combines step-back abstraction, query decomposition, and multi-query
    expansion to retrieve a broader, more relevant candidate set before
    answering — each strategy contributes additional retrieval queries,
    and results are deduplicated before being handed to the LLM.

    Best for: Complex multi-hop questions, ambiguous/vague queries, and
    generally improving recall over Naive RAG's single-query retrieval.
    """

    TECHNIQUE_NAME = "query_transform_rag"
    FRAMEWORK = "llamaindex"

    def _build_pipeline(self) -> None:
        """Initialize LLM, embeddings, and enabled query-transform strategies."""
        from llama_index.core import Settings
        from llama_index.core.node_parser import SentenceSplitter

        cfg = ConfigLoader.get()
        tech_cfg = cfg.get_technique_config("query_transform_rag")
        doc_cfg = cfg.document

        self.llm = get_llamaindex_llm()
        Settings.llm = self.llm
        Settings.embed_model = get_llamaindex_embeddings()
        Settings.node_parser = SentenceSplitter(
            chunk_size=doc_cfg.get("chunk_size", 512),
            chunk_overlap=doc_cfg.get("chunk_overlap", 50),
        )

        self.top_k = cfg.retrieval.get("top_k", 5)
        self.strategies = tech_cfg.get("strategies", ["step_back", "decompose", "multi_query"])
        self.num_queries = tech_cfg.get("num_queries", 3)
        self.vector_index = None
        self.retriever = None

        logger.info(f"[QueryTransform/LI] strategies={self.strategies}, num_queries={self.num_queries}")  # noqa: E501

    def index(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        """Build a VectorStoreIndex from raw text strings (standard indexing, unchanged by query transforms)."""  # noqa: E501
        from llama_index.core import VectorStoreIndex
        from llama_index.core.schema import Document as LIDocument

        logger.info(f"[QueryTransform/LI] Indexing {len(documents)} documents...")

        metas = metadatas or [{}] * len(documents)
        li_docs = [LIDocument(text=t, metadata=m) for t, m in zip(documents, metas, strict=True)]

        self.vector_index = VectorStoreIndex.from_documents(li_docs, show_progress=True)
        self.retriever = self.vector_index.as_retriever(similarity_top_k=self.top_k)

        self._is_indexed = True
        logger.info("[QueryTransform/LI] Indexing complete ✓")

    # ------------------------------------------------------------------
    # Query transformation strategies
    # ------------------------------------------------------------------

    def _generate_step_back(self, question: str) -> str:
        """Generate a more abstract/general version of the query."""
        response = self.llm.complete(STEP_BACK_PROMPT.format(question=question))
        step_back_query = _strip_thinking(response.text)
        logger.debug(f"[QueryTransform/LI] Step-back query: {step_back_query}")
        return step_back_query

    def _decompose_query(self, question: str) -> list[str]:
        """Break the query into simpler sub-questions."""
        response = self.llm.complete(
            DECOMPOSE_PROMPT.format(question=question, num_subquestions=self.num_queries)
        )
        text = _strip_thinking(response.text)
        sub_questions = [line.strip("- ").strip() for line in text.split("\n") if line.strip()]
        logger.debug(f"[QueryTransform/LI] Sub-questions: {sub_questions}")
        return sub_questions

    def _generate_multi_query(self, question: str) -> list[str]:
        """Generate N differently-worded variants of the query."""
        response = self.llm.complete(
            MULTI_QUERY_PROMPT.format(question=question, num_queries=self.num_queries)
        )
        text = _strip_thinking(response.text)
        variants = [line.strip("- ").strip() for line in text.split("\n") if line.strip()]
        logger.debug(f"[QueryTransform/LI] Multi-query variants: {variants}")
        return variants

    def _retrieve_deduplicated(self, queries: list[str]) -> list:
        """Retrieve top-K nodes for each query, merging and deduplicating by content."""
        seen = set()
        unique_nodes = []
        for q in queries:
            for node in self.retriever.retrieve(q):
                h = hash(node.get_content()[:100])
                if h not in seen:
                    seen.add(h)
                    unique_nodes.append(node)
        return unique_nodes

    def _query(self, question: str) -> RAGResult:
        """Full query-transform pipeline: transform → retrieve (all queries) → dedupe → answer."""
        if not self.retriever:
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

        retrieved_nodes = self._retrieve_deduplicated(all_queries)
        intermediate_steps.append({
            "step": "retrieval",
            "num_queries": len(all_queries),
            "num_docs": len(retrieved_nodes),
        })
        logger.debug(f"[QueryTransform/LI] {len(all_queries)} queries → {len(retrieved_nodes)} unique nodes")  # noqa: E501

        context = "\n\n---\n\n".join(node.get_content() for node in retrieved_nodes)
        response = self.llm.complete(ANSWER_PROMPT.format(context=context, question=question))
        answer = _strip_thinking(response.text)

        return RAGResult(
            query=question,
            answer=answer,
            source_documents=[
                Document(content=node.get_content(), metadata=node.metadata, score=node.score)
                for node in retrieved_nodes
            ],
            intermediate_steps=intermediate_steps,
            metadata={
                "strategies_used": self.strategies,
                "num_queries_generated": len(all_queries),
                "num_docs_retrieved": len(retrieved_nodes),
            },
        )
