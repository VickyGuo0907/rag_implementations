"""
Unit tests for Adaptive RAG pure helpers: complexity-label parsing,
query-variant parsing, and the routing dispatch fallback. These run
without any LLM or vector store.
"""

import importlib

import pytest

adaptive_lc = importlib.import_module("techniques.10_adaptive_rag.langchain_impl")

_parse_complexity = adaptive_lc._parse_complexity
_parse_query_variants = adaptive_lc._parse_query_variants
_strip_thinking = adaptive_lc._strip_thinking


# ---------------------------------------------------------------------------
# _parse_complexity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("simple", "simple"),
    ("moderate", "moderate"),
    ("complex", "complex"),
    ("SIMPLE", "simple"),
    ("The complexity is Complex.", "complex"),
    ("<think>hmm let me decide</think>\nmoderate", "moderate"),
])
def test_parse_complexity_extracts_label(text, expected):
    assert _parse_complexity(text) == expected


def test_parse_complexity_uses_fixed_priority_order():
    # Labels are checked in a fixed order (simple → moderate → complex),
    # not by their position in the response.
    assert _parse_complexity("not simple or complex at all") == "simple"


def test_parse_complexity_defaults_on_unknown():
    assert _parse_complexity("I cannot classify this", default="moderate") == "moderate"


def test_parse_complexity_empty_input_uses_default():
    assert _parse_complexity("", default="complex") == "complex"


# ---------------------------------------------------------------------------
# _strip_thinking
# ---------------------------------------------------------------------------

def test_strip_thinking_removes_reasoning_trace():
    text = "<think>reasoning here</think>The answer is 4."
    assert _strip_thinking(text) == "The answer is 4."


def test_strip_thinking_leaves_plain_text_untouched():
    assert _strip_thinking("plain response") == "plain response"


# ---------------------------------------------------------------------------
# _parse_query_variants
# ---------------------------------------------------------------------------

def test_parse_query_variants_one_per_line():
    text = "What is RAG?\nHow does retrieval augmented generation work?\nDefine RAG."
    variants = _parse_query_variants(text)
    assert len(variants) == 3
    assert variants[0] == "What is RAG?"


def test_parse_query_variants_strips_bullets_and_blank_lines():
    text = "- variant one\n\n- variant two\n"
    variants = _parse_query_variants(text)
    assert variants == ["variant one", "variant two"]


def test_parse_query_variants_empty_response():
    assert _parse_query_variants("") == []
