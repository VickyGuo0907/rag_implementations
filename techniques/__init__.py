"""
RAG Techniques Package
======================
The canonical technique metadata and framework→class mappings live in
`techniques.registry` — the single source of truth used by main.py and
app.py. This module re-exports the registry helpers for convenience.

Usage:
    from techniques import get_techniques, load_class
    from techniques.naive_rag.langchain_impl import NaiveRAGLangChain
"""

from techniques.registry import (
    FRAMEWORKS,
    IMPLEMENTATIONS,
    TECHNIQUES,
    get_implementations,
    get_techniques,
    is_implemented,
    is_registered,
    load_class,
)

__all__ = [
    "TECHNIQUES",
    "IMPLEMENTATIONS",
    "FRAMEWORKS",
    "get_techniques",
    "get_implementations",
    "load_class",
    "is_implemented",
    "is_registered",
]
