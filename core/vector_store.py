"""
Vector Store Factory
====================
Creates and manages vector store instances (ChromaDB, FAISS, Qdrant).
Provides a unified interface for both LangChain and LlamaIndex.

Usage:
    from core.vector_store import get_langchain_vector_store, get_llamaindex_vector_store
"""

import os
from typing import Optional, List
from pathlib import Path

from core.config_loader import ConfigLoader
from core.embeddings import get_langchain_embeddings, get_llamaindex_embeddings


def get_langchain_vector_store(
    collection_name: Optional[str] = None,
    provider: Optional[str] = None,
):
    """
    Returns a LangChain VectorStore instance.

    Providers: chroma, faiss, qdrant

    Args:
        collection_name: Override collection name from config.
        provider: Override provider from config.

    Returns:
        LangChain VectorStore instance
    """
    cfg = ConfigLoader.get()
    vs_cfg = cfg.vector_store
    prov = provider or vs_cfg.get("provider", "chroma")
    coll = collection_name or vs_cfg.get("collection_name", "rag_documents")
    embeddings = get_langchain_embeddings()

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
        from langchain_community.vectorstores import FAISS
        faiss_path = vs_cfg.get("faiss", {}).get("save_path", "./data/faiss_index")
        if os.path.exists(f"{faiss_path}.faiss"):
            return FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        # Return empty FAISS — will be populated during indexing
        return None  # Caller handles creation with from_documents()

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


def get_llamaindex_vector_store(
    collection_name: Optional[str] = None,
    provider: Optional[str] = None,
):
    """
    Returns a LlamaIndex VectorStore instance and StorageContext.

    Returns:
        Tuple of (VectorStore, StorageContext)
    """
    from llama_index.core import StorageContext

    cfg = ConfigLoader.get()
    vs_cfg = cfg.vector_store
    prov = provider or vs_cfg.get("provider", "chroma")
    coll = collection_name or vs_cfg.get("collection_name", "rag_documents")

    if prov == "chroma":
        import chromadb
        from llama_index.vector_stores.chroma import ChromaVectorStore

        persist_dir = vs_cfg.get("persist_directory", "./data/vector_store")
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = chroma_client.get_or_create_collection(coll)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return vector_store, storage_context

    elif prov == "faiss":
        import faiss
        from llama_index.vector_stores.faiss import FaissVectorStore

        emb_cfg = ConfigLoader.get().embeddings
        dim = emb_cfg.get("dimension", 768)
        faiss_index = faiss.IndexFlatL2(dim)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return vector_store, storage_context

    else:
        raise ValueError(f"Unknown vector store provider: {prov}")
