"""
LMStudio LLM Client Factory
============================
Creates LangChain and LlamaIndex LLM instances pointing to a local LMStudio
server. LMStudio exposes an OpenAI-compatible REST API at localhost:1234.

Usage:
    from core.llm_client import get_langchain_llm, get_llamaindex_llm
    llm = get_langchain_llm()
"""

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

    Note: Patches LlamaIndex's model validation to allow custom model names
    from LMStudio (which aren't in OpenAI's official list).
    """
    from llama_index.llms.openai import OpenAI
    from llama_index.llms.openai import utils, base

    # Get the original function from utils module
    original_func = utils.openai_modelname_to_contextsize

    def patched_validation(model_name: str):
        """Allow custom model names, default to gpt-3.5-turbo context if unknown."""
        try:
            return original_func(model_name)
        except ValueError:
            # For unknown models (like LMStudio custom names), use gpt-3.5-turbo size
            return original_func("gpt-3.5-turbo")

    # Patch in all places the function is imported
    utils.openai_modelname_to_contextsize = patched_validation
    base.openai_modelname_to_contextsize = patched_validation

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
