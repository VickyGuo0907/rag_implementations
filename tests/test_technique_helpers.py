"""
Unit tests for pure helper logic shared across technique implementations:
score parsers, action classifiers, Reciprocal Rank Fusion, and content
helpers. These run without any LLM or vector store.
"""

import importlib

import pytest

self_rag_lc = importlib.import_module("techniques.08_self_rag.langchain_impl")
corrective_lc = importlib.import_module("techniques.09_corrective_rag.langchain_impl")
fusion_lc = importlib.import_module("techniques.05_fusion_rag.langchain_impl")
naive_lc = importlib.import_module("techniques.01_naive_rag.langchain_impl")

_parse_score = self_rag_lc._parse_score
_parse_scores = self_rag_lc._parse_scores
CorrectiveRAGLangChain = corrective_lc.CorrectiveRAGLangChain
FusionRAGLangChain = fusion_lc.FusionRAGLangChain
format_docs = naive_lc.format_docs
_classify_action = CorrectiveRAGLangChain._classify_action


# ---------------------------------------------------------------------------
# Score parsers (Self-RAG / CRAG share the same helpers)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("0.85", 0.85),
    ("score: 0.3", 0.3),
    ("1.0", 1.0),
    ("0", 0.0),
])
def test_parse_score_parses_numbers(text, expected):
    assert _parse_score(text) == pytest.approx(expected)


def test_parse_score_clamps_to_unit_interval():
    assert _parse_score("2.5") == 1.0
    assert _parse_score("-0.4") == 0.4  # regex grabs magnitude, ignores sign


def test_parse_score_defaults_on_garbage():
    assert _parse_score("no numbers here", default=0.7) == 0.7
    assert _parse_score("", default=0.3) == 0.3


def test_parse_scores_pads_to_count():
    scores = _parse_scores("0.9\n0.5", count=4, default=0.5)
    assert len(scores) == 4
    assert scores[0] == 0.9
    assert scores[2] == 0.5


def test_parse_scores_truncates_to_count():
    scores = _parse_scores("0.9\n0.5\n0.1\n0.2", count=2)
    assert len(scores) == 2


def test_parse_scores_empty_input():
    assert _parse_scores("", count=3, default=0.4) == [0.4, 0.4, 0.4]


# ---------------------------------------------------------------------------
# CRAG action classifier
# ---------------------------------------------------------------------------

def test_classify_correct_all_pass():
    assert _classify_action([0.9, 0.8, 0.95], threshold=0.7) == "correct"


def test_classify_ambiguous_some_pass():
    assert _classify_action([0.9, 0.3, 0.8], threshold=0.7) == "ambiguous"


def test_classify_incorrect_none_pass():
    assert _classify_action([0.1, 0.2, 0.3], threshold=0.7) == "incorrect"


def test_classify_incorrect_empty():
    assert _classify_action([], threshold=0.7) == "incorrect"


# ---------------------------------------------------------------------------
# Naive RAG format_docs
# ---------------------------------------------------------------------------

def test_format_docs_joins_with_separators():
    docs = type("D", (), {"page_content": "one"}), type("D", (), {"page_content": "two"})
    out = format_docs(docs)
    assert "one" in out and "two" in out
    assert out.count("---") == 1


# ---------------------------------------------------------------------------
# Fusion RAG — Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _rrf_instance():
    inst = FusionRAGLangChain.__new__(FusionRAGLangChain)
    inst.rrf_k = 60
    return inst


class _Doc:
    def __init__(self, content):
        self.page_content = content


def test_rrf_ranks_docs_present_in_more_lists_higher():
    inst = _rrf_instance()
    doc_a = _Doc("alpha content")
    doc_b = _Doc("beta content")
    doc_c = _Doc("gamma content")

    list1 = [doc_a, doc_b]   # a ranked #1
    list2 = [doc_a, doc_c]   # a ranked #1 again
    list3 = [doc_b, doc_c]   # b ranked #1 here

    fused = inst._reciprocal_rank_fusion([list1, list2, list3])
    contents = [d.page_content for d in fused]

    # a appears at rank 1 in two lists → highest RRF score
    assert contents[0] == "alpha content"


def test_rrf_deduplicates_across_lists():
    inst = _rrf_instance()
    same = _Doc("same content")
    other = _Doc("other content")
    fused = inst._reciprocal_rank_fusion([[same, other], [same]])
    assert len(fused) == 2


def test_rrf_k_parameter_respected():
    """A larger rrf_k flattens score differences but keeps relative order."""
    inst_small = _rrf_instance()
    inst_small.rrf_k = 1
    inst_big = _rrf_instance()
    inst_big.rrf_k = 1000

    doc_a = _Doc("alpha")
    doc_b = _Doc("beta")
    lists = [[doc_a, doc_b], [doc_a]]

    order_small = [d.page_content for d in inst_small._reciprocal_rank_fusion(lists)]
    order_big = [d.page_content for d in inst_big._reciprocal_rank_fusion(lists)]
    assert order_small[0] == "alpha"
    assert order_big[0] == "alpha"
