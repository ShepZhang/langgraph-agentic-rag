"""Tests for independent evaluation Judge configuration."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from evaluation.judge_config import (
    EvaluationJudgeSettings,
    build_judge_runtime_metadata,
    create_evaluation_judge_model,
    load_evaluation_judge_settings,
)


def _enabled_environment() -> dict[str, str]:
    return {
        "EVALUATION_JUDGE_ENABLED": "true",
        "EVALUATION_JUDGE_API_KEY": "judge-secret",
        "EVALUATION_JUDGE_BASE_URL": "https://judge.example/v1",
        "EVALUATION_JUDGE_MODEL": "deepseek-chat",
    }


def test_judge_settings_default_disabled_without_reusing_system_llm_environment():
    settings = load_evaluation_judge_settings(
        {
            "OPENAI_API_KEY": "system-secret",
            "OPENAI_BASE_URL": "https://system.example/v1",
            "OPENAI_MODEL": "system-model",
            "OPENAI_TEMPERATURE": "1.5",
            "OLLAMA_BASE_URL": "http://localhost:11434",
            "OLLAMA_MODEL": "qwen2.5:7b",
            "LLM_PROVIDER": "ollama",
            "LLM_TEMPERATURE": "0.7",
        }
    )

    assert settings == EvaluationJudgeSettings()
    assert settings.enabled is False
    assert settings.api_key == ""
    assert settings.base_url == ""
    assert settings.model == ""
    assert settings.temperature == 0.0


def test_judge_settings_are_frozen():
    settings = EvaluationJudgeSettings()

    with pytest.raises(FrozenInstanceError):
        settings.enabled = True  # type: ignore[misc]


def test_disabled_judge_ignores_invalid_unused_temperature():
    settings = load_evaluation_judge_settings(
        {
            "EVALUATION_JUDGE_ENABLED": "false",
            "EVALUATION_JUDGE_TEMPERATURE": "not-a-number",
        }
    )

    assert settings.temperature == 0.0


@pytest.mark.parametrize(
    "missing_name",
    [
        "EVALUATION_JUDGE_API_KEY",
        "EVALUATION_JUDGE_BASE_URL",
        "EVALUATION_JUDGE_MODEL",
    ],
)
def test_enabled_judge_requires_each_independent_field(missing_name):
    environ = _enabled_environment()
    environ[missing_name] = "   "

    with pytest.raises(ValueError, match=missing_name):
        load_evaluation_judge_settings(environ)


@pytest.mark.parametrize(
    "raw_temperature",
    ["-0.1", "2.1", "not-a-number"],
)
def test_enabled_judge_rejects_invalid_temperatures(raw_temperature):
    environ = _enabled_environment()
    environ["EVALUATION_JUDGE_TEMPERATURE"] = raw_temperature

    with pytest.raises(ValueError, match="EVALUATION_JUDGE_TEMPERATURE"):
        load_evaluation_judge_settings(environ)


def test_enabled_judge_defaults_temperature_to_zero():
    settings = load_evaluation_judge_settings(_enabled_environment())

    assert settings.temperature == 0.0


@pytest.mark.parametrize(
    ("raw_enabled", "expected"),
    [
        ("1", True),
        ("TRUE", True),
        ("Yes", True),
        ("y", True),
        ("ON", True),
        ("0", False),
        ("FALSE", False),
        ("No", False),
        ("n", False),
        ("OFF", False),
    ],
)
def test_judge_enabled_accepts_project_boolean_forms(raw_enabled, expected):
    environ = _enabled_environment() if expected else {}
    environ["EVALUATION_JUDGE_ENABLED"] = raw_enabled

    settings = load_evaluation_judge_settings(environ)

    assert settings.enabled is expected


def test_judge_enabled_rejects_invalid_boolean():
    with pytest.raises(ValueError, match="EVALUATION_JUDGE_ENABLED"):
        load_evaluation_judge_settings({"EVALUATION_JUDGE_ENABLED": "sometimes"})


@pytest.mark.parametrize("raw_enabled", ["", "   "])
def test_judge_enabled_rejects_explicit_blank_boolean(raw_enabled):
    with pytest.raises(ValueError, match="EVALUATION_JUDGE_ENABLED"):
        load_evaluation_judge_settings({"EVALUATION_JUDGE_ENABLED": raw_enabled})


def test_disabled_judge_does_not_construct_client():
    calls = 0

    def client_factory(**_kwargs):
        nonlocal calls
        calls += 1

    with pytest.raises(RuntimeError, match="disabled"):
        create_evaluation_judge_model(
            EvaluationJudgeSettings(),
            client_factory=client_factory,
        )

    assert calls == 0


def test_enabled_judge_constructs_injected_client_with_exact_kwargs():
    captured: dict[str, object] = {}
    expected_client = object()

    def client_factory(**kwargs):
        captured.update(kwargs)
        return expected_client

    settings = EvaluationJudgeSettings(
        enabled=True,
        api_key="judge-secret",
        base_url="https://judge.example/v1",
        model="deepseek-chat",
        temperature=0.0,
    )

    client = create_evaluation_judge_model(
        settings,
        client_factory=client_factory,
    )

    assert client is expected_client
    assert captured == {
        "model": "deepseek-chat",
        "api_key": "judge-secret",
        "base_url": "https://judge.example/v1",
        "temperature": 0.0,
    }


def test_judge_runtime_metadata_excludes_secret_and_base_url():
    settings = EvaluationJudgeSettings(
        enabled=True,
        api_key="judge-secret",
        base_url="https://judge.example/v1",
        model="deepseek-chat",
        temperature=0.0,
    )

    metadata = build_judge_runtime_metadata(settings)

    assert metadata == {
        "enabled": True,
        "provider": "openai_compatible",
        "model": "deepseek-chat",
        "temperature": 0.0,
    }
    assert "api_key" not in metadata
    assert "base_url" not in metadata


def test_disabled_judge_runtime_metadata_uses_none_for_empty_model():
    metadata = build_judge_runtime_metadata(EvaluationJudgeSettings())

    assert metadata == {
        "enabled": False,
        "provider": "openai_compatible",
        "model": None,
        "temperature": 0.0,
    }
