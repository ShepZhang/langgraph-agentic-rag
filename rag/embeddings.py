"""Embedding model initialization."""

from __future__ import annotations

from typing import Any

from config import Settings, get_settings


class UnsupportedEmbeddingProviderError(ValueError):
    """Raised when the configured embedding provider is unsupported."""


def get_embedding_model(settings: Settings | None = None) -> Any:
    """Create the configured embedding model."""

    resolved_settings = settings or get_settings()
    provider = resolved_settings.embedding_provider.lower()

    if provider in {"sentence_transformers", "huggingface", "local"}:
        embeddings_cls = _load_huggingface_embeddings_class()
        return embeddings_cls(model_name=resolved_settings.embedding_model)

    raise UnsupportedEmbeddingProviderError(
        f"Unsupported embedding provider {resolved_settings.embedding_provider!r}. "
        "Supported provider: sentence_transformers"
    )


def _load_huggingface_embeddings_class() -> type[Any]:
    """Load the HuggingFace embeddings class from available LangChain packages."""

    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings
