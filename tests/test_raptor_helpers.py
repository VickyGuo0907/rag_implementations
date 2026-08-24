"""
Unit tests for RAPTOR pure helpers: GMM/agglomerative clustering, label
grouping, recursion stopping rules, and truncation. Clustering tests use
small synthetic vectors — no LLM, embeddings, or vector store involved.
"""

import importlib

import pytest

raptor_lc = importlib.import_module("techniques.12_raptor_rag.langchain_impl")

_select_best_gmm = raptor_lc._select_best_gmm
_cluster_texts = raptor_lc._cluster_texts
_group_by_label = raptor_lc._group_by_label
_should_stop = raptor_lc._should_stop
_truncate = raptor_lc._truncate
_strip_thinking = raptor_lc._strip_thinking


def _separable_vectors():
    """
    Two well-separated gaussian clusters (30 points each).

    Sized so BIC's preference for the true k=2 dominates its tendency to
    over-split tiny samples — verified empirically.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    a = rng.normal(0.0, 0.05, (30, 2))
    b = rng.normal(0.0, 0.05, (30, 2)) + [14.0, 14.0]
    return np.vstack([a, b]).tolist()


def _perpendicular_vectors():
    """Two tight clusters at ~90° in cosine terms — separable by cosine distance."""
    return [
        [1.0, 0.0], [1.0, 0.05], [0.95, 0.0], [1.02, 0.02],
        [0.98, 0.04], [1.01, 0.01],
        [0.0, 1.0], [0.05, 1.0], [0.0, 0.95], [0.02, 1.02],
        [0.04, 0.98], [0.01, 1.01],
    ]


# ---------------------------------------------------------------------------
# GMM + clustering dispatch
# ---------------------------------------------------------------------------

def test_gmm_separates_two_tight_clusters():
    labels = _select_best_gmm(_separable_vectors(), max_clusters=4)
    assert len(labels) == 60
    assert len(set(labels)) == 2
    assert len(set(labels[:30])) == 1
    assert len(set(labels[30:])) == 1
    assert labels[0] != labels[-1]


def test_gmm_single_sample_returns_one_cluster():
    assert _select_best_gmm([[1.0, 2.0]], max_clusters=3) == [0]


def test_cluster_texts_agglomerative_fallback_separates_clusters():
    labels = _cluster_texts(_perpendicular_vectors(), algorithm="agglomerative")
    assert len(set(labels)) == 2
    assert labels[0] == labels[1]
    assert labels[6] != labels[0]


def test_cluster_texts_unknown_algorithm_falls_back():
    labels = _cluster_texts(_perpendicular_vectors(), algorithm="mystery")
    assert len(labels) == 12


# ---------------------------------------------------------------------------
# Label grouping
# ---------------------------------------------------------------------------

def test_group_by_label_preserves_index_order():
    groups = _group_by_label([1, 0, 1, 0])
    assert groups == [[1, 3], [0, 2]]


def test_group_by_label_empty_input():
    assert _group_by_label([]) == []


# ---------------------------------------------------------------------------
# Recursion stopping rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level,parent,prev,max_depth,expected", [
    (1, 3, 9, 3, False),   # healthy recursion continues
    (4, 3, 9, 3, True),    # depth exceeded
    (2, 0, 5, 3, True),    # nothing to build on
    (2, 5, 5, 3, True),    # summaries stopped shrinking the frontier
    (2, 4, 5, 3, False),   # still shrinking
])
def test_should_stop_rules(level, parent, prev, max_depth, expected):
    assert _should_stop(level, parent, prev, max_depth) is expected


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def test_truncate_short_text_untouched():
    assert _truncate("hello", 100) == "hello"


def test_truncate_long_text_capped_with_marker():
    out = _truncate("x" * 50, 10)
    assert out == "xxxxxxxxxx..."


def test_strip_thinking_removes_reasoning_trace():
    assert _strip_thinking("<think>hmm</think>Summary.") == "Summary."
