"""
Embedding Model Factory
========================
Creates LangChain and LlamaIndex embedding instances from config.
Supports LMStudio (OpenAI-compatible), HuggingFace, and sentence-transformers.

Usage:
    from core.embeddings import get_langchain_embeddings, get_llamaindex_embeddings
    embeddings = get_langchain_embeddings()
"""

from typing import Optional
from core.config_loader import ConfigLoader


def get_langchain_embeddings(provider: Optional[str] = None):
    """
    Returns a LangChain embeddings instance based on config.

    Providers:
        - "lmstudio"          → OpenAI-compatible API (LMStudio)
        - "huggingface"       → Local HuggingFace model
        - "sentence_transformers" → sentence-transformers library

    Returns:
        LangChain Embeddings instance
    """
    cfg = ConfigLoader.get()
    emb_cfg = cfg.embeddings
    prov = provider or emb_cfg.get("provider", "lmstudio")

    if prov == "lmstudio":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            base_url=emb_cfg.get("base_url", "http://localhost:1234/v1"),
            api_key=emb_cfg.get("api_key", "lm-studio"),
            model=emb_cfg.get("model", "nomic-embed-text-v1.5"),
            timeout=120,
            max_retries=2,
        )

    elif prov == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        hf_cfg = emb_cfg.get("huggingface_fallback", {})
        return HuggingFaceEmbeddings(
            model_name=hf_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
            model_kwargs={"device": hf_cfg.get("device", "cpu")},
            encode_kwargs={"normalize_embeddings": True},
        )

    elif prov == "sentence_transformers":
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        hf_cfg = emb_cfg.get("huggingface_fallback", {})
        return SentenceTransformerEmbeddings(
            model_name=hf_cfg.get("model", "all-MiniLM-L6-v2")
        )

    else:
        raise ValueError(f"Unknown embedding provider: {prov}. "
                         "Choose from: lmstudio, huggingface, sentence_transformers")


def get_llamaindex_embeddings(provider: Optional[str] = None):
    """
    Returns a LlamaIndex embedding instance based on config.

    Returns:
        LlamaIndex BaseEmbedding instance
    """
    cfg = ConfigLoader.get()
    emb_cfg = cfg.embeddings
    prov = provider or emb_cfg.get("provider", "lmstudio")

    if prov == "lmstudio":
        from llama_index.embeddings.openai import OpenAIEmbedding
        return OpenAIEmbedding(
            api_base=emb_cfg.get("base_url", "http://localhost:1234/v1"),
            api_key=emb_cfg.get("api_key", "lm-studio"),
            model=emb_cfg.get("model", "nomic-embed-text-v1.5"),
        )

    elif prov in ("huggingface", "sentence_transformers"):
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        hf_cfg = emb_cfg.get("huggingface_fallback", {})
        return HuggingFaceEmbedding(
            model_name=hf_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        )

    else:
        raise ValueError(f"Unknown embedding provider: {prov}")
