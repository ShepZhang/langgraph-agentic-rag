"""Tests for runtime configuration validation."""

from __future__ import annotations

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
