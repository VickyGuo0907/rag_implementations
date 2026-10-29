"""
Unit tests for Multi-modal RAG's framework-agnostic helpers: table and image
extraction, item building with a fake describer, and answer-context assembly.
No LLM, vector store, or network required.
"""

import importlib
from pathlib import Path

import pytest

mm = importlib.import_module("techniques.14_multimodal_rag.multimodal")

TABLE = "| Region | Units |\n|--------|-------|\n| EU | 10 |\n| US | 20 |"
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32


class FakeVision:
    """Stands in for VisionLLM: deterministic descriptions, records calls."""

    def __init__(self, fail=False):
        self.fail = fail
        self.captioned: list[Path] = []

    def caption_image(self, path):
        if self.fail:
            raise ConnectionError("server down")
        self.captioned.append(path)
        return "a bar chart of revenue"

    def summarize_table(self, table):
        if self.fail:
            raise ConnectionError("server down")
        return "units by region"


# ---------------------------------------------------------------------------
# extract_tables
# ---------------------------------------------------------------------------

def test_extract_tables_finds_pipe_table():
    text = f"Intro\n\n{TABLE}\n\nOutro"
    assert mm.extract_tables(text) == [TABLE]


def test_extract_tables_table_at_end_of_text():
    assert mm.extract_tables(f"x\n{TABLE}") == [TABLE]


def test_extract_tables_multiple():
    assert len(mm.extract_tables(f"{TABLE}\n\ntext\n\n{TABLE}")) == 2


@pytest.mark.parametrize("text", [
    "no table here",
    "| just | a header |",
    "| a | b |\n| c | d |",            # pipes but no separator row
    "a | b | c",                        # inline pipes
])
def test_extract_tables_ignores_non_tables(text):
    assert mm.extract_tables(text) == []


# ---------------------------------------------------------------------------
# extract_markdown_images
# ---------------------------------------------------------------------------

def test_extract_markdown_images_resolves_relative_path(tmp_path):
    (tmp_path / "chart.png").write_bytes(PNG)
    found = mm.extract_markdown_images("![A chart](chart.png)", tmp_path)
    assert found == [("A chart", (tmp_path / "chart.png").resolve())]


def test_extract_markdown_images_skips_remote_missing_and_non_images(tmp_path):
    (tmp_path / "notes.txt").write_text("x")
    text = (
        "![r](https://x.io/a.png) ![m](missing.png) ![t](notes.txt) "
        "![d](data:image/png;base64,AA)"
    )
    assert mm.extract_markdown_images(text, tmp_path) == []


# ---------------------------------------------------------------------------
# build_items
# ---------------------------------------------------------------------------

def _doc(tmp_path, body):
    md = tmp_path / "report.md"
    md.write_text(body)
    return body, {"source": str(md), "page": None}


def test_build_items_emits_text_table_and_image(tmp_path):
    (tmp_path / "c.png").write_bytes(PNG)
    body, meta = _doc(tmp_path, f"Intro\n![Revenue](c.png)\n\n{TABLE}\n")
    items = mm.build_items([body], [meta], FakeVision(), {})
    assert [i.kind for i in items] == ["text", "table", "image"]
    table, image = items[1], items[2]
    assert table.text == "units by region" and table.metadata["raw"] == TABLE
    assert "Revenue" in image.text and "a bar chart of revenue" in image.text
    assert image.metadata["raw"].endswith("c.png")


def test_build_items_respects_config_switches(tmp_path):
    (tmp_path / "c.png").write_bytes(PNG)
    body, meta = _doc(tmp_path, f"![x](c.png)\n{TABLE}")
    cfg = {"table_extraction": False, "image_captioning": False}
    assert [i.kind for i in mm.build_items([body], [meta], FakeVision(), cfg)] == ["text"]


def test_build_items_without_metadata_is_text_only():
    items = mm.build_items(["plain"], None, FakeVision(), {})
    assert [i.kind for i in items] == ["text"]


def test_build_items_degrades_when_vision_fails(tmp_path):
    (tmp_path / "c.png").write_bytes(PNG)
    body, meta = _doc(tmp_path, f"![Revenue](c.png)\n\n{TABLE}")
    items = mm.build_items([body], [meta], FakeVision(fail=True), {})
    assert [i.kind for i in items] == ["text", "table", "image"]
    assert items[1].text.startswith("| Region")      # falls back to raw table prefix
    assert items[2].text.strip().startswith("Revenue")  # falls back to alt text


def test_build_items_drops_non_scalar_metadata():
    items = mm.build_items(["x"], [{"source": "s", "tags": ["a"], "n": 1}], FakeVision(), {})
    assert items[0].metadata == {"modality": "text", "source": "s", "n": 1}


# ---------------------------------------------------------------------------
# build_context
# ---------------------------------------------------------------------------

def test_build_context_mixes_modalities_and_caps_images():
    hits = [
        mm.Hit("text", "chunk", "chunk text"),
        mm.Hit("table", "summary", TABLE),
        mm.Hit("image", "cap a", "/tmp/a.png"),
        mm.Hit("image", "cap a again", "/tmp/a.png"),      # duplicate file
        mm.Hit("image", "cap b", "/tmp/b.png"),
        mm.Hit("image", "cap c", "/tmp/c.png"),
    ]
    context, images = mm.build_context(hits, max_images=2)
    assert "chunk text" in context and f"[Table]\n{TABLE}" in context
    assert "[Image: cap a]" in context and "[Image: cap c]" in context
    assert images == [Path("/tmp/a.png"), Path("/tmp/b.png")]


def test_build_context_text_only_attaches_nothing():
    _, images = mm.build_context([mm.Hit("text", "t", "t")], max_images=2)
    assert images == []


# ---------------------------------------------------------------------------
# image_data_uri / strip_thinking
# ---------------------------------------------------------------------------

def test_image_data_uri_encodes_mime_and_payload(tmp_path):
    p = tmp_path / "a.png"
    p.write_bytes(PNG)
    uri = mm.image_data_uri(p)
    assert uri.startswith("data:image/png;base64,") and len(uri) > 30


def test_strip_thinking_removes_trace():
    assert mm.strip_thinking("<think>hmm</think>\nanswer") == "answer"
