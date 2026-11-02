"""Tests for the LlamaIndex LLM factory (no LLM server required)."""

from core.llm_client import get_llamaindex_llm


def test_llamaindex_llm_is_chat_model_for_lmstudio_names():
    """LMStudio model names must use the chat API, not raw /v1/completions."""
    assert get_llamaindex_llm().metadata.is_chat_model is True
