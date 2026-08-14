"""
Vector Store Factory
====================
Creates and populates vector stores for both LangChain and LlamaIndex,
dispatching on the configured provider (chroma, faiss, qdrant).

LangChain techniques call `build_langchain_vector_store()` to embed and store
document chunks in one step. Techniques that need an *empty* store to manage
documents themselves (e.g. Parent Document RAG) use `get_langchain_vector_store()`.

LlamaIndex techniques use its idiomatic in-memory `VectorStoreIndex` directly;
the factory previously exposed a dead `get_llamaindex_vector_store()` and it
was removed — see each LlamaIndex technique for the canonical pattern.

Usage:
    from core.vector_store import build_langchain_vector_store
    store = build_langchain_vector_store(chunks, collection_name="naive_rag")
"""

import logging
import os
from pathlib import Path

from core.config_loader import ConfigLoader
from core.embeddings import get_langchain_embeddings

logger = logging.getLogger(__name__)


def _resolve(collection_name: str | None, provider: str | None):
    """Resolve collection name and provider from config with overrides."""
    cfg = ConfigLoader.get()
    vs_cfg = cfg.vector_store
    prov = provider or vs_cfg.get("provider", "chroma")
    coll = collection_name or vs_cfg.get("collection_name", "rag_documents")
    return prov, coll


def _infer_embedding_dimension(embeddings) -> int:
    """
    Derive the embedding dimension from the configured model.

    Falls back to the configured ``embeddings.dimension`` if probing fails.
    Hardcoding the dimension (e.g. 768) silently breaks FAISS when the
    configured model actually emits a different size (e.g. 384 for
    all-MiniLM-L6-v2).
    """
    try:
        probe = embeddings.embed_query("dimension probe")
        return len(probe)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"Could not infer embedding dimension from model ({e}); "
            f"falling back to config embeddings.dimension."
        )
        return int(ConfigLoader.get().embeddings.get("dimension", 768))


def get_langchain_vector_store(
    collection_name: str | None = None,
    provider: str | None = None,
    embeddings=None,
):
    """
    Returns an empty LangChain VectorStore instance (no documents yet).

    Callers are responsible for adding documents via ``add_documents()`` /
    ``add_texts()`` (e.g. ParentDocumentRetriever does this internally).

    Providers: chroma, faiss, qdrant

    Args:
        collection_name: Override collection name from config.
        provider: Override provider from config.
        embeddings: Override the embeddings instance (defaults to config).

    Returns:
        LangChain VectorStore instance
    """
    cfg = ConfigLoader.get()
    vs_cfg = cfg.vector_store
    prov, coll = _resolve(collection_name, provider)
    embeddings = embeddings or get_langchain_embeddings()

    if prov == "chroma":
        from langchain_chroma import Chroma
        persist_dir = vs_cfg.get("persist_directory", "./data/vector_store")
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=coll,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )

    elif prov == "faiss":
        import faiss
        from langchain_community.docstore.in_memory import InMemoryDocstore
        from langchain_community.vectorstores import FAISS

        dim = _infer_embedding_dimension(embeddings)
        index = faiss.IndexFlatL2(dim)
        return FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    elif prov == "qdrant":
        from langchain_qdrant import Qdrant
        from qdrant_client import QdrantClient
        q_cfg = vs_cfg.get("qdrant", {})
        client = QdrantClient(url=q_cfg.get("url", "http://localhost:6333"))
        return Qdrant(
            client=client,
            collection_name=coll,
            embeddings=embeddings,
        )

    else:
        raise ValueError(f"Unknown vector store provider: {prov}")


def build_langchain_vector_store(
    documents: list,
    collection_name: str | None = None,
    provider: str | None = None,
    embeddings=None,
):
    """
    Embed and store document chunks, returning a populated LangChain VectorStore.

    This is the canonical indexing helper for LangChain techniques — it replaces
    the inline ``Chroma.from_documents(...)`` boilerplate duplicated across every
    implementation and respects the configured provider (chroma / faiss / qdrant).

    Args:
        documents: LangChain Document chunks to embed and store.
        collection_name: Override collection name from config. A per-technique
            prefix (e.g. "naive_rag") is recommended to avoid cross-technique
            collisions in the same persistent store.
        provider: Override provider from config.
        embeddings: Override the embeddings instance (defaults to config).

    Returns:
        LangChain VectorStore instance containing the embedded documents.
    """
    cfg = ConfigLoader.get()
    vs_cfg = cfg.vector_store
    prov, coll = _resolve(collection_name, provider)
    embeddings = embeddings or get_langchain_embeddings()

    if prov == "chroma":
        from langchain_chroma import Chroma
        persist_dir = vs_cfg.get("persist_directory", "./data/vector_store")
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=coll,
            persist_directory=persist_dir,
        )

    elif prov == "faiss":
        from langchain_community.vectorstores import FAISS
        faiss_path = vs_cfg.get("faiss", {}).get("save_path", "./data/faiss_index")
        if os.path.exists(f"{faiss_path}.faiss"):
            logger.info(f"[VectorStore] Loading FAISS index from {faiss_path}")
            return FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        return FAISS.from_documents(documents, embeddings)

    elif prov == "qdrant":
        from langchain_qdrant import Qdrant
        from qdrant_client import QdrantClient
        q_cfg = vs_cfg.get("qdrant", {})
        client = QdrantClient(url=q_cfg.get("url", "http://localhost:6333"))
        return Qdrant.from_documents(
            documents=documents,
            embedding=embeddings,
            client=client,
            collection_name=coll,
        )

    else:
        raise ValueError(f"Unknown vector store provider: {prov}")
