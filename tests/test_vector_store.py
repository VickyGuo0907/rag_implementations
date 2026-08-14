"""
Unit tests for the vector store factory.

Uses a fake embeddings object so tests run without LMStudio or a
downloaded embedding model. Chroma (the default provider) is exercised
against a temp directory.
"""

import numpy as np
import pytest

from core.document_loader import load_texts
from core.vector_store import _resolve, build_langchain_vector_store


class FakeEmbeddings:
    """Deterministic fake embeddings that never touch the network."""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        rng = np.random.RandomState(hash(text) % (2**32))
        vec = rng.rand(self.dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-9)


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()


@pytest.fixture
def sample_chunks():
    return load_texts(
        ["RAG combines retrieval with generation.",
         "The vector store holds embedded document chunks.",
         "Query transforms improve retrieval recall."],
        [{"source": "a"}, {"source": "b"}, {"source": "c"}],
    )


def test_resolve_defaults_from_config():
    """_resolve falls back to config defaults when no overrides given."""
    provider, coll = _resolve(None, None)
    assert provider in ("chroma", "faiss", "qdrant")
    assert coll == "rag_documents"


def test_resolve_with_overrides():
    """_resolve honors explicit overrides."""
    provider, coll = _resolve("my_coll", "faiss")
    assert provider == "faiss"
    assert coll == "my_coll"


def test_build_langchain_vector_store_chroma(sample_chunks, fake_embeddings, tmp_path, monkeypatch):
    """build_langchain_vector_store populates a Chroma store with the chunks."""
    monkeypatch.setenv("RAG_CONFIG_PATH", "config/config.yaml")
    store = build_langchain_vector_store(
        sample_chunks,
        collection_name="test_coll",
        provider="chroma",
        embeddings=fake_embeddings,
    )
    assert store is not None
    hits = store.similarity_search("vector store", k=2)
    assert len(hits) == 2
    assert hits[0].page_content


def test_build_langchain_vector_store_unknown_provider(sample_chunks, fake_embeddings):
    """An unknown provider raises ValueError."""
    with pytest.raises(ValueError, match="Unknown vector store provider"):
        build_langchain_vector_store(
            sample_chunks,
            collection_name="c",
            provider="mystery",
            embeddings=fake_embeddings,
        )
