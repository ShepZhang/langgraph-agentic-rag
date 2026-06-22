"""Independent configuration and model construction for the evaluation Judge."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationJudgeSettings:
    """Settings for an optional OpenAI-compatible evaluation Judge."""

    enabled: bool = False
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    temperature: float = 0.0


def _parse_enabled(raw_value: str | None) -> bool:
    if raw_value is None:
        return False

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"EVALUATION_JUDGE_ENABLED must be a boolean, got {raw_value!r}")


def load_evaluation_judge_settings(
    environ: Mapping[str, str] | None = None,
) -> EvaluationJudgeSettings:
    """Load evaluation Judge settings from their dedicated environment variables."""

    source = os.environ if environ is None else environ
    enabled = _parse_enabled(source.get("EVALUATION_JUDGE_ENABLED"))
    api_key = source.get("EVALUATION_JUDGE_API_KEY", "").strip()
    base_url = source.get("EVALUATION_JUDGE_BASE_URL", "").strip()
    model = source.get("EVALUATION_JUDGE_MODEL", "").strip()

    if not enabled:
        return EvaluationJudgeSettings(
            enabled=False,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0.0,
        )

    required_values = {
        "EVALUATION_JUDGE_API_KEY": api_key,
        "EVALUATION_JUDGE_BASE_URL": base_url,
        "EVALUATION_JUDGE_MODEL": model,
    }
    for name, value in required_values.items():
        if not value:
            raise ValueError(
                f"{name} must not be empty when EVALUATION_JUDGE_ENABLED is true"
            )

    raw_temperature = source.get("EVALUATION_JUDGE_TEMPERATURE", "0.0").strip()
    try:
        temperature = float(raw_temperature)
    except ValueError as exc:
        raise ValueError(
            f"EVALUATION_JUDGE_TEMPERATURE must be a number, got {raw_temperature!r}"
        ) from exc
    if not 0 <= temperature <= 2:
        raise ValueError("EVALUATION_JUDGE_TEMPERATURE must be between 0 and 2")

    return EvaluationJudgeSettings(
        enabled=True,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
    )


def create_evaluation_judge_model(
    settings: EvaluationJudgeSettings,
    *,
    client_factory: Callable[..., Any] | None = None,
) -> Any:
    """Construct the independently configured evaluation Judge model."""

    if not settings.enabled:
        raise RuntimeError(
            "Evaluation Judge is disabled; enable EVALUATION_JUDGE_ENABLED "
            "before constructing its model"
        )

    if client_factory is None:
        from langchain_openai import ChatOpenAI

        client_factory = ChatOpenAI

    return client_factory(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=settings.temperature,
    )


def build_judge_runtime_metadata(
    settings: EvaluationJudgeSettings,
) -> dict[str, bool | str | float | None]:
    """Return sanitized Judge configuration metadata."""

    return {
        "enabled": settings.enabled,
        "provider": "openai_compatible",
        "model": settings.model or None,
        "temperature": settings.temperature,
    }
