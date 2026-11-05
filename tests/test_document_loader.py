"""
Unit tests for the shared document loading and text-splitting helpers.
"""

import pytest

from core.document_loader import load_documents, load_texts


def test_load_texts_wraps_strings():
    """Raw strings become LangChain Documents with metadata."""
    docs = load_texts(["a", "b"], [{"source": "x"}, {"source": "y"}])
    assert len(docs) == 2
    assert docs[0].page_content == "a"
    assert docs[0].metadata == {"source": "x"}


def test_load_texts_no_shared_metadata():
    """
    Without metadata, each document gets its own dict — mutating one
    must not leak into the others (guards the [{}]*len() aliasing bug).
    """
    docs = load_texts(["a", "b", "c"])
    docs[0].metadata["note"] = "only here"
    assert "note" not in docs[1].metadata
    assert "note" not in docs[2].metadata


def test_load_texts_no_shared_metadata_with_override():
    """The same isolation holds when metadata is explicitly provided."""
    docs = load_texts(["a", "b"], [{"k": 1}, {"k": 2}])
    docs[0].metadata["extra"] = "value"
    assert "extra" not in docs[1].metadata


def test_load_documents_missing_path_raises():
    """A non-existent path raises a clear error naming it, not a generic loader error."""
    with pytest.raises(FileNotFoundError, match="Document source not found"):
        load_documents("/nonexistent/path/xyz")


def test_load_documents_sample_docs():
    """The bundled sample documents load without error."""
    docs = load_documents("data/sample_docs")
    assert len(docs) > 0
    assert all(getattr(d, "page_content", None) for d in docs)


def test_load_documents_directory_skips_image_assets(tmp_path):
    """Images referenced by documents are assets, not documents: they must not hit TextLoader."""
    (tmp_path / "report.md").write_text("# Report\n![c](chart.png)")
    (tmp_path / "chart.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe\x00")
    docs = load_documents(str(tmp_path))
    assert len(docs) == 1
    assert docs[0].metadata["source"].endswith("report.md")

