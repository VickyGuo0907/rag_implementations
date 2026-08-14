"""
Technique Registry — Single Source of Truth
============================================
Centralizes every RAG technique's metadata and its (framework → class)
mappings so the CLI (main.py), the web UI (app.py), and the auto-discovery
package (techniques/__init__.py) all read from one place instead of three
drifting copies.

Framework values: "langchain" | "llamaindex"
Status values:    "implemented" | "stub"

Usage:
    from techniques.registry import get_techniques, load_class, is_implemented

    for key, meta in get_techniques().items():
        print(key, meta["status"])

    rag_cls = load_class("naive_rag", "langchain")
"""


FRAMEWORKS = ("langchain", "llamaindex")

# ---------------------------------------------------------------------------
# Technique metadata (all 14 techniques, including stubs)
# ---------------------------------------------------------------------------

TECHNIQUES: dict[str, dict] = {
    # ---- Tier 1 — Foundational ----
    "naive_rag": {
        "name": "Naive RAG",
        "tier": 1,
        "complexity": 1,
        "latency": "Low",
        "accuracy": "Moderate",
        "status": "implemented",
        "description": "Basic retrieval-augmented generation with vector similarity search",
    },
    "advanced_rag": {
        "name": "Advanced RAG",
        "tier": 1,
        "complexity": 3,
        "latency": "Medium",
        "accuracy": "High",
        "status": "implemented",
        "description": "Enhanced RAG with query rewriting, reranking, and contextual compression",
    },
    # ---- Tier 2 — Query & Retrieval Enhancement ----
    "hyde_rag": {
        "name": "HyDE RAG",
        "tier": 2,
        "complexity": 2,
        "latency": "Medium",
        "accuracy": "High",
        "status": "implemented",
        "description": "Hypothetical document embeddings for vocabulary mismatch resolution",
    },
    "query_transform_rag": {
        "name": "Query Transform RAG",
        "tier": 2,
        "complexity": 2,
        "latency": "Medium",
        "accuracy": "High",
        "status": "implemented",
        "description": "Step-back, decomposition, and multi-query transformation for improved recall",  # noqa: E501
    },
    "fusion_rag": {
        "name": "Fusion RAG",
        "tier": 2,
        "complexity": 3,
        "latency": "Medium",
        "accuracy": "High",
        "status": "implemented",
        "description": "Multi-query generation + hybrid dense/sparse search + Reciprocal Rank Fusion",  # noqa: E501
    },
    "parent_document_rag": {
        "name": "Parent Document RAG",
        "tier": 2,
        "complexity": 2,
        "latency": "Low",
        "accuracy": "High",
        "status": "implemented",
        "description": "Small child chunks for precise retrieval, large parent chunks for generation context",  # noqa: E501
    },
    "reranking_rag": {
        "name": "Reranking RAG",
        "tier": 2,
        "complexity": 2,
        "latency": "Medium",
        "accuracy": "High",
        "status": "implemented",
        "description": "Wide candidate retrieval + cross-encoder reranking for high-precision results",  # noqa: E501
    },
    # ---- Tier 3 — Adaptive & Self-Reflective ----
    "self_rag": {
        "name": "Self-RAG",
        "tier": 3,
        "complexity": 4,
        "latency": "High",
        "accuracy": "High",
        "status": "implemented",
        "description": "Prompted reflection: decide if retrieval is needed, grade relevance, critique groundedness",  # noqa: E501
    },
    "corrective_rag": {
        "name": "Corrective RAG",
        "tier": 3,
        "complexity": 4,
        "latency": "High",
        "accuracy": "Very High",
        "status": "implemented",
        "description": "Grade retrieval quality once, then correct via knowledge refinement and/or web search fallback",  # noqa: E501
    },
    "adaptive_rag": {
        "name": "Adaptive RAG",
        "tier": 3,
        "complexity": 3,
        "latency": "Varies",
        "accuracy": "High",
        "status": "stub",
        "description": "Classify query complexity, route to the optimal RAG strategy",
    },
    # ---- Tier 4 — Structural & Specialized ----
    "graph_rag": {
        "name": "GraphRAG",
        "tier": 4,
        "complexity": 5,
        "latency": "High",
        "accuracy": "Very High",
        "status": "stub",
        "description": "Extract entities/relations, build a knowledge graph, query with graph traversal",  # noqa: E501
    },
    "raptor_rag": {
        "name": "RAPTOR",
        "tier": 4,
        "complexity": 5,
        "latency": "High",
        "accuracy": "Very High",
        "status": "stub",
        "description": "Cluster docs → summarize clusters → build a recursive tree → query at depth",  # noqa: E501
    },
    "agentic_rag": {
        "name": "Agentic RAG",
        "tier": 4,
        "complexity": 5,
        "latency": "High",
        "accuracy": "Very High",
        "status": "stub",
        "description": "LLM agent decides when/what to retrieve and uses multiple tools",
    },
    "multimodal_rag": {
        "name": "Multi-modal RAG",
        "tier": 4,
        "complexity": 4,
        "latency": "High",
        "accuracy": "High",
        "status": "stub",
        "description": "Process images/tables alongside text and retrieve cross-modally",
    },
}

# ---------------------------------------------------------------------------
# (technique, framework) → dotted class path
# ---------------------------------------------------------------------------

IMPLEMENTATIONS: dict[tuple[str, str], str] = {
    ("naive_rag", "langchain"): "techniques.01_naive_rag.langchain_impl.NaiveRAGLangChain",
    ("naive_rag", "llamaindex"): "techniques.01_naive_rag.llamaindex_impl.NaiveRAGLlamaIndex",
    ("advanced_rag", "langchain"): "techniques.02_advanced_rag.langchain_impl.AdvancedRAGLangChain",
    ("advanced_rag", "llamaindex"): "techniques.02_advanced_rag.llamaindex_impl.AdvancedRAGLlamaIndex",  # noqa: E501
    ("hyde_rag", "langchain"): "techniques.03_hyde_rag.langchain_impl.HyDERAGLangChain",
    ("hyde_rag", "llamaindex"): "techniques.03_hyde_rag.llamaindex_impl.HyDERAGLlamaIndex",
    ("query_transform_rag", "langchain"): "techniques.04_query_transform_rag.langchain_impl.QueryTransformRAGLangChain",  # noqa: E501
    ("query_transform_rag", "llamaindex"): "techniques.04_query_transform_rag.llamaindex_impl.QueryTransformRAGLlamaIndex",  # noqa: E501
    ("fusion_rag", "langchain"): "techniques.05_fusion_rag.langchain_impl.FusionRAGLangChain",
    ("fusion_rag", "llamaindex"): "techniques.05_fusion_rag.llamaindex_impl.FusionRAGLlamaIndex",
    ("parent_document_rag", "langchain"): "techniques.06_parent_document_rag.langchain_impl.ParentDocumentRAGLangChain",  # noqa: E501
    ("parent_document_rag", "llamaindex"): "techniques.06_parent_document_rag.llamaindex_impl.ParentDocumentRAGLlamaIndex",  # noqa: E501
    ("reranking_rag", "langchain"): "techniques.07_reranking_rag.langchain_impl.RerankingRAGLangChain",  # noqa: E501
    ("reranking_rag", "llamaindex"): "techniques.07_reranking_rag.llamaindex_impl.RerankingRAGLlamaIndex",  # noqa: E501
    ("self_rag", "langchain"): "techniques.08_self_rag.langchain_impl.SelfRAGLangChain",
    ("self_rag", "llamaindex"): "techniques.08_self_rag.llamaindex_impl.SelfRAGLlamaIndex",
    ("corrective_rag", "langchain"): "techniques.09_corrective_rag.langchain_impl.CorrectiveRAGLangChain",  # noqa: E501
    ("corrective_rag", "llamaindex"): "techniques.09_corrective_rag.llamaindex_impl.CorrectiveRAGLlamaIndex",  # noqa: E501
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_techniques(status: str | None = None) -> dict[str, dict]:
    """
    Return all technique metadata, optionally filtered by status.

    Args:
        status: "implemented", "stub", or None for all techniques.

    Returns:
        Dict of technique_key → metadata dict.
    """
    if status is None:
        return dict(TECHNIQUES)
    return {k: v for k, v in TECHNIQUES.items() if v["status"] == status}


def get_implementations(framework: str | None = None) -> list[tuple[str, str]]:
    """
    Return registered (technique, framework) key pairs.

    Args:
        framework: Restrict to one framework, or None for all.

    Returns:
        List of (technique, framework) tuples.
    """
    return [
        (tech, fw) for (tech, fw) in IMPLEMENTATIONS
        if framework is None or fw == framework
    ]


def load_class(technique: str, framework: str):
    """
    Dynamically import and return the implementation class for a technique+framework.

    Args:
        technique: Technique key (e.g. "naive_rag").
        framework: Framework key ("langchain" or "llamaindex").

    Returns:
        The BaseRAG subclass for that implementation.

    Raises:
        KeyError: If the combination is not implemented.
    """
    dotted_path = IMPLEMENTATIONS[(technique, framework)]
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def is_implemented(technique: str) -> bool:
    """Return True if the technique has at least one implemented framework."""
    return TECHNIQUES.get(technique, {}).get("status") == "implemented"


def is_registered(technique: str, framework: str) -> bool:
    """Return True if the (technique, framework) combo has a registered class."""
    return (technique, framework) in IMPLEMENTATIONS
