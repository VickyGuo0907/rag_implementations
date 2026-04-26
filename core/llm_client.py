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


class LMStudioOpenAI:
    """Wrapper around LlamaIndex OpenAI client for LMStudio compatibility.

    Solves the problem that LlamaIndex validates model names against OpenAI's list,
    but LMStudio needs the actual model name in API calls.
    """

    def __init__(self, base_url: str, api_key: str, actual_model: str, **kwargs):
        """Initialize wrapper with actual model for API calls."""
        from llama_index.llms.openai import OpenAI

        self.actual_model = actual_model
        # Create client with placeholder for LlamaIndex validation
        self._llm = OpenAI(
            api_base=base_url,
            api_key=api_key,
            model="gpt-3.5-turbo",
            **kwargs
        )

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped LLM."""
        return getattr(self._llm, name)

    def complete(self, prompt: str, **kwargs):
        """Override complete to use actual model name."""
        # Temporarily change model for the API call
        original_model = self._llm.model
        self._llm.model = self.actual_model
        try:
            return self._llm.complete(prompt, **kwargs)
        finally:
            self._llm.model = original_model

    def chat(self, messages, **kwargs):
        """Override chat to use actual model name."""
        original_model = self._llm.model
        self._llm.model = self.actual_model
        try:
            return self._llm.chat(messages, **kwargs)
        finally:
            self._llm.model = original_model

    def stream_complete(self, prompt: str, **kwargs):
        """Override stream_complete to use actual model name."""
        original_model = self._llm.model
        self._llm.model = self.actual_model
        try:
            return self._llm.stream_complete(prompt, **kwargs)
        finally:
            self._llm.model = original_model

    def stream_chat(self, messages, **kwargs):
        """Override stream_chat to use actual model name."""
        original_model = self._llm.model
        self._llm.model = self.actual_model
        try:
            return self._llm.stream_chat(messages, **kwargs)
        finally:
            self._llm.model = original_model


def get_llamaindex_llm(config_key: str = "lmstudio"):
    """
    Returns a LlamaIndex OpenAI LLM instance pointed at LMStudio.

    Args:
        config_key: "lmstudio" for main model, "lmstudio_small" for fast model.

    Returns:
        LMStudioOpenAI wrapper instance

    Note: Uses gpt-3.5-turbo for LlamaIndex's model validation while sending
    the actual configured model name to LMStudio in API calls.
    """
    cfg = ConfigLoader.get()
    lm_cfg = cfg[config_key]

    return LMStudioOpenAI(
        base_url=lm_cfg["base_url"],
        api_key=lm_cfg["api_key"],
        actual_model=lm_cfg["model"],
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
