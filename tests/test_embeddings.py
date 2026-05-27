"""Tests for embedding model initialization."""

from __future__ import annotations

from dataclasses import replace

import pytest

from config import get_settings
from rag import embeddings
from rag.embeddings import UnsupportedEmbeddingProviderError, get_embedding_model


def test_get_embedding_model_uses_sentence_transformers_provider(monkeypatch):
    settings = replace(
        get_settings(),
        embedding_provider="sentence_transformers",
        embedding_model="sentence-transformers/test-model",
    )
    calls = {}

    class FakeEmbeddings:
        def __init__(self, model_name: str):
            calls["model_name"] = model_name

    monkeypatch.setattr(
        embeddings,
        "_load_huggingface_embeddings_class",
        lambda: FakeEmbeddings,
    )

    model = get_embedding_model(settings)

    assert isinstance(model, FakeEmbeddings)
    assert calls["model_name"] == "sentence-transformers/test-model"


def test_get_embedding_model_rejects_unsupported_provider():
    settings = replace(get_settings(), embedding_provider="unsupported")

    with pytest.raises(UnsupportedEmbeddingProviderError, match="Unsupported embedding provider"):
        get_embedding_model(settings)
