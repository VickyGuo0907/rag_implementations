"""
LMStudio LLM Client Factory
============================
Creates LangChain and LlamaIndex LLM instances pointing to a local LMStudio
server. LMStudio exposes an OpenAI-compatible REST API at localhost:1234.

Usage:
    from core.llm_client import get_langchain_llm, get_llamaindex_llm
    llm = get_langchain_llm()
"""

from typing import Optional
from core.config_loader import ConfigLoader


def get_langchain_llm(config_key: str = "lmstudio"):
    """
    Returns a LangChain ChatOpenAI instance pointed at LMStudio.

    Args:
        config_key: "lmstudio" for main model, "lmstudio_small" for fast model.

    Returns:
        langchain_openai.ChatOpenAI instance
    """
    from langchain_openai import ChatOpenAI

    cfg = ConfigLoader.get()
    lm_cfg = cfg[config_key]

    return ChatOpenAI(
        base_url=lm_cfg["base_url"],
        api_key=lm_cfg["api_key"],
        model=lm_cfg["model"],
        temperature=lm_cfg.get("temperature", 0.1),
        max_tokens=lm_cfg.get("max_tokens", 2048),
        timeout=lm_cfg.get("timeout", 120),
        streaming=lm_cfg.get("streaming", False),
    )


def get_llamaindex_llm(config_key: str = "lmstudio"):
    """
    Returns a LlamaIndex OpenAI LLM instance pointed at LMStudio.

    Args:
        config_key: "lmstudio" for main model, "lmstudio_small" for fast model.

    Returns:
        llama_index.llms.openai.OpenAI instance
    """
    from llama_index.llms.openai import OpenAI

    cfg = ConfigLoader.get()
    lm_cfg = cfg[config_key]

    return OpenAI(
        api_base=lm_cfg["base_url"],
        api_key=lm_cfg["api_key"],
        model=lm_cfg["model"],
        temperature=lm_cfg.get("temperature", 0.1),
        max_tokens=lm_cfg.get("max_tokens", 2048),
        timeout=lm_cfg.get("timeout", 120),
    )


def get_langchain_llm_with_fallback(config_key: str = "lmstudio"):
    """
    Attempts LMStudio connection; prints helpful error if server is not running.
    """
    try:
        llm = get_langchain_llm(config_key)
        # Quick connectivity test
        llm.invoke("ping")
        return llm
    except Exception as e:
        print(
            f"\n⚠️  Could not connect to LMStudio at "
            f"{ConfigLoader.get()[config_key]['base_url']}\n"
            f"   Make sure LMStudio is running and a model is loaded.\n"
            f"   Error: {e}\n"
        )
        raise
