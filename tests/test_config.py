"""Tests for runtime configuration validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import get_settings


def test_get_settings_accepts_max_retry_count(monkeypatch):
    monkeypatch.setenv("MAX_RETRY_COUNT", "4")

    settings = get_settings()

    assert settings.max_retry_count == 4
    assert settings.max_rewrite_attempts == 4


def test_get_settings_keeps_legacy_max_rewrite_attempts_env(monkeypatch):
    monkeypatch.delenv("MAX_RETRY_COUNT", raising=False)
    monkeypatch.setenv("MAX_REWRITE_ATTEMPTS", "3")

    settings = get_settings()

    assert settings.max_retry_count == 3


def test_get_settings_rejects_invalid_chunk_overlap(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")

    with pytest.raises(ValueError, match="CHUNK_OVERLAP must be smaller"):
        get_settings()


def test_get_settings_rejects_invalid_top_k(monkeypatch):
    monkeypatch.setenv("TOP_K", "0")

    with pytest.raises(ValueError, match="TOP_K must be greater than 0"):
        get_settings()


def test_get_settings_rejects_invalid_temperature(monkeypatch):
    monkeypatch.setenv("OPENAI_TEMPERATURE", "2.5")

    with pytest.raises(ValueError, match="OPENAI_TEMPERATURE"):
        get_settings()


def test_get_settings_defaults_reranker_off(monkeypatch):
    monkeypatch.delenv("RERANKER_ENABLED", raising=False)

    settings = get_settings()

    assert settings.reranker_enabled is False
    assert settings.reranker_candidate_top_k >= settings.top_k


def test_get_settings_defaults_hybrid_retrieval_off(monkeypatch):
    monkeypatch.delenv("HYBRID_RETRIEVAL_ENABLED", raising=False)
    monkeypatch.delenv("DENSE_TOP_K", raising=False)
    monkeypatch.delenv("BM25_TOP_K", raising=False)
    monkeypatch.delenv("FUSION_TOP_K", raising=False)

    settings = get_settings()

    assert settings.hybrid_retrieval_enabled is False
    assert settings.dense_top_k == 20
    assert settings.bm25_top_k == 20
    assert settings.fusion_top_k == 20


def test_get_settings_accepts_hybrid_retrieval_config(monkeypatch):
    monkeypatch.setenv("HYBRID_RETRIEVAL_ENABLED", "true")
    monkeypatch.setenv("DENSE_TOP_K", "12")
    monkeypatch.setenv("BM25_TOP_K", "14")
    monkeypatch.setenv("FUSION_TOP_K", "7")

    settings = get_settings()

    assert settings.hybrid_retrieval_enabled is True
    assert settings.dense_top_k == 12
    assert settings.bm25_top_k == 14
    assert settings.fusion_top_k == 7


@pytest.mark.parametrize(
    ("env_name", "message"),
    [
        ("DENSE_TOP_K", "DENSE_TOP_K"),
        ("BM25_TOP_K", "BM25_TOP_K"),
        ("FUSION_TOP_K", "FUSION_TOP_K"),
    ],
)
def test_get_settings_rejects_invalid_hybrid_top_k(
    monkeypatch,
    env_name,
    message,
):
    monkeypatch.setenv(env_name, "0")

    with pytest.raises(ValueError, match=message):
        get_settings()


def test_get_settings_accepts_reranker_config(monkeypatch):
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    monkeypatch.setenv("RERANKER_TOP_N", "3")
    monkeypatch.setenv("RERANKER_CANDIDATE_TOP_K", "8")

    settings = get_settings()

    assert settings.reranker_enabled is True
    assert settings.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert settings.reranker_top_n == 3
    assert settings.reranker_candidate_top_k == 8


def test_get_settings_defaults_reranker_top_n(monkeypatch):
    monkeypatch.delenv("RERANKER_TOP_N", raising=False)

    settings = get_settings()

    assert settings.reranker_top_n == 5


def test_get_settings_rejects_invalid_reranker_top_n(monkeypatch):
    monkeypatch.setenv("RERANKER_TOP_N", "0")

    with pytest.raises(ValueError, match="RERANKER_TOP_N"):
        get_settings()


def test_get_settings_rejects_enabled_reranker_with_too_small_candidate_top_k(
    monkeypatch,
):
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_TOP_N", "5")
    monkeypatch.setenv("RERANKER_CANDIDATE_TOP_K", "3")

    with pytest.raises(ValueError, match="RERANKER_CANDIDATE_TOP_K"):
        get_settings()


def test_get_settings_rejects_invalid_reranker_enabled_value(monkeypatch):
    monkeypatch.setenv("RERANKER_ENABLED", "sometimes")

    with pytest.raises(ValueError, match="RERANKER_ENABLED"):
        get_settings()


def test_get_settings_supports_ollama_without_openai_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")

    settings = get_settings()

    assert settings.has_llm_config is True
    assert settings.llm_provider == "ollama"
    assert settings.effective_llm_api_key == "ollama"
    assert settings.effective_llm_base_url == "http://localhost:11434/v1"
    assert settings.effective_llm_model == "qwen2.5:7b"


def test_get_settings_keeps_openai_compatible_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-chat")

    settings = get_settings()

    assert settings.has_llm_config is True
    assert settings.llm_provider == "openai_compatible"
    assert settings.effective_llm_api_key == "test-key"
    assert settings.effective_llm_base_url == "https://api.deepseek.com/v1"
    assert settings.effective_llm_model == "deepseek-chat"


def test_get_settings_rejects_unknown_llm_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local_magic")

    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        get_settings()


def test_get_settings_requires_ollama_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "")

    settings = get_settings()

    assert settings.has_llm_config is False
    with pytest.raises(RuntimeError, match="OLLAMA_MODEL"):
        settings.require_llm_config()


def test_get_settings_accepts_llm_temperature_alias(monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "0.3")
    monkeypatch.setenv("OPENAI_TEMPERATURE", "1.7")

    settings = get_settings()

    assert settings.temperature == 0.3


def test_settings_loads_evaluation_history_defaults(monkeypatch):
    for name in (
        "EVALUATION_HISTORY_ENABLED",
        "EVALUATION_HISTORY_DB",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = get_settings()

    assert settings.evaluation_history_enabled is True
    assert settings.evaluation_history_db == Path("./data/evaluation_history.sqlite3")


def test_settings_loads_evaluation_history_overrides(monkeypatch, tmp_path):
    db_path = tmp_path / "history.sqlite3"
    monkeypatch.setenv("EVALUATION_HISTORY_ENABLED", "false")
    monkeypatch.setenv("EVALUATION_HISTORY_DB", str(db_path))

    settings = get_settings()

    assert settings.evaluation_history_enabled is False
    assert settings.evaluation_history_db == db_path


def test_settings_rejects_empty_evaluation_history_db(monkeypatch):
    monkeypatch.setenv("EVALUATION_HISTORY_DB", "   ")

    with pytest.raises(ValueError, match="EVALUATION_HISTORY_DB"):
        get_settings()
