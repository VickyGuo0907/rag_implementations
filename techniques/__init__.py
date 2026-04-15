"""
RAG Techniques Registry
========================
Auto-discovers all implemented RAG techniques.
Import individual techniques or use the registry for bulk operations.

Usage:
    from techniques import get_all_techniques, get_technique
    from techniques.naive_rag.langchain_impl import NaiveRAGLangChain
"""

from pathlib import Path
import importlib

TECHNIQUE_DIRS = {
    "naive_rag":            "techniques.01_naive_rag",
    "advanced_rag":         "techniques.02_advanced_rag",
    "hyde_rag":             "techniques.03_hyde_rag",
    "query_transform_rag":  "techniques.04_query_transform_rag",
    "fusion_rag":           "techniques.05_fusion_rag",
    "parent_document_rag":  "techniques.06_parent_document_rag",
    "reranking_rag":        "techniques.07_reranking_rag",
    "self_rag":             "techniques.08_self_rag",
    "corrective_rag":       "techniques.09_corrective_rag",
    "adaptive_rag":         "techniques.10_adaptive_rag",
    "graph_rag":            "techniques.11_graph_rag",
    "raptor_rag":           "techniques.12_raptor_rag",
    "agentic_rag":          "techniques.13_agentic_rag",
    "multimodal_rag":       "techniques.14_multimodal_rag",
}

TECHNIQUE_METADATA = {
    "naive_rag":            {"tier": 1, "complexity": 1, "status": "implemented"},
    "advanced_rag":         {"tier": 1, "complexity": 3, "status": "implemented"},
    "hyde_rag":             {"tier": 2, "complexity": 2, "status": "implemented"},
    "query_transform_rag":  {"tier": 2, "complexity": 2, "status": "stub"},
    "fusion_rag":           {"tier": 2, "complexity": 3, "status": "stub"},
    "parent_document_rag":  {"tier": 2, "complexity": 2, "status": "stub"},
    "reranking_rag":        {"tier": 2, "complexity": 2, "status": "stub"},
    "self_rag":             {"tier": 3, "complexity": 4, "status": "stub"},
    "corrective_rag":       {"tier": 3, "complexity": 4, "status": "stub"},
    "adaptive_rag":         {"tier": 3, "complexity": 3, "status": "stub"},
    "graph_rag":            {"tier": 4, "complexity": 5, "status": "stub"},
    "raptor_rag":           {"tier": 4, "complexity": 5, "status": "stub"},
    "agentic_rag":          {"tier": 4, "complexity": 5, "status": "stub"},
    "multimodal_rag":       {"tier": 4, "complexity": 4, "status": "stub"},
}


def list_techniques():
    """Return list of all available technique names."""
    return list(TECHNIQUE_DIRS.keys())


def get_technique_status():
    """Return implementation status for all techniques."""
    return {k: v["status"] for k, v in TECHNIQUE_METADATA.items()}
