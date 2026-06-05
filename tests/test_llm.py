"""Tests for chat LLM factory configuration."""

from __future__ import annotations

from config import get_settings


def test_create_chat_model_uses_ollama_effective_settings(monkeypatch):
    from agent import llm as llm_module

    captured: dict[str, object] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")

    model = llm_module.create_chat_model(get_settings())

    assert isinstance(model, FakeChatOpenAI)
    assert captured == {
        "model": "qwen2.5:7b",
        "api_key": "ollama",
        "base_url": "http://localhost:11434/v1",
        "temperature": 0,
    }


def test_create_chat_model_uses_openai_compatible_effective_settings(monkeypatch):
    from agent import llm as llm_module

    captured: dict[str, object] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-chat")

    llm_module.create_chat_model(get_settings())

    assert captured == {
        "model": "deepseek-chat",
        "api_key": "test-key",
        "base_url": "https://api.deepseek.com/v1",
        "temperature": 0,
    }
