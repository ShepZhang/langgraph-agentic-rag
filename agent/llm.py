"""Chat LLM factory for OpenAI-compatible and local Ollama providers."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from config import Settings


def create_chat_model(settings: Settings) -> ChatOpenAI:
    """Create the chat model used by Agentic and naive RAG flows."""

    settings.require_llm_config()
    return ChatOpenAI(
        model=settings.effective_llm_model,
        api_key=settings.effective_llm_api_key,
        base_url=settings.effective_llm_base_url,
        temperature=settings.temperature,
    )
