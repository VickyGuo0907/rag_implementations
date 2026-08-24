"""
Unit tests for GraphRAG pure helpers: extraction-output parsing, section
extraction, and entity normalization. These run without any LLM, graph
backend, or vector store.
"""

import importlib

import pytest

graph_lc = importlib.import_module("techniques.11_graph_rag.langchain_impl")

_parse_entities = graph_lc._parse_entities
_parse_relationships = graph_lc._parse_relationships
_parse_question_entities = graph_lc._parse_question_entities
_extract_section = graph_lc._extract_section
_normalize_entity = graph_lc._normalize_entity
_split_pipe = graph_lc._split_pipe


# ---------------------------------------------------------------------------
# Extraction parsing
# ---------------------------------------------------------------------------

EXTRACTION_OK = """ENTITIES:
- Alice Chen | PERSON
- Acme Corp | ORGANIZATION
RELATIONSHIPS:
- Alice Chen | works for | Acme Corp"""


def test_parse_entities_names_and_types():
    assert _parse_entities(EXTRACTION_OK) == [
        ("Alice Chen", "PERSON"),
        ("Acme Corp", "ORGANIZATION"),
    ]


def test_parse_relationships_triples():
    assert _parse_relationships(EXTRACTION_OK) == [
        ("Alice Chen", "works for", "Acme Corp"),
    ]


def test_parse_handles_thinking_tags_and_prose():
    noisy = "<think>scan text</think>Here you go:\nENTITIES:\n- X | CONCEPT\n" \
        "RELATIONSHIPS:\n- X | relates to | Y"
    assert _parse_entities(noisy) == [("X", "CONCEPT")]
    assert _parse_relationships(noisy) == [("X", "relates to", "Y")]


def test_parse_skips_malformed_lines():
    messy = """ENTITIES:
- Missing Type
-
- Valid | EVENT
RELATIONSHIPS:
- OnlyTwo | parts
- A | rel | B"""
    assert _parse_entities(messy) == [("Valid", "EVENT")]
    assert _parse_relationships(messy) == [("A", "rel", "B")]


def test_parse_empty_response_returns_empty_lists():
    assert _parse_entities("") == []
    assert _parse_relationships("no markers at all") == []


def test_parse_question_entities_plain_names():
    text = "ENTITIES:\n- RAG\n- vector databases\n- 2024"
    assert _parse_question_entities(text) == ["RAG", "vector databases", "2024"]


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

def test_extract_section_stops_at_stop_marker():
    body = _extract_section("ENTITIES:\n- a\nRELATIONSHIPS:\n- b", "ENTITIES", "RELATIONSHIPS")
    assert "- a" in body
    assert "- b" not in body


def test_extract_section_runs_to_end_without_stop():
    body = _extract_section("ENTITIES:\n- a\n- b", "ENTITIES", None)
    assert "- a" in body and "- b" in body


def test_extract_section_missing_marker_returns_empty():
    assert _extract_section("nothing here", "ENTITIES", None) == ""


# ---------------------------------------------------------------------------
# Normalization + pipe splitting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Alice Chen", "alice chen"),
    ("  RAG   systems  ", "rag systems"),
    ("ACME", "acme"),
])
def test_normalize_entity(raw, expected):
    assert _normalize_entity(raw) == expected


def test_split_pipe_pads_to_three():
    assert _split_pipe("a | b", 2) == ["a", "b", ""]


def test_split_pipe_rejects_too_few_fields():
    assert _split_pipe("only-one", 2) == []
    assert _split_pipe("| missing name", 1) == []
