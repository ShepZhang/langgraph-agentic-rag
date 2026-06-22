"""Sanitized runtime configuration snapshots for evaluation artifacts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from agent.features import AgentFeatureFlags
from config import Settings, get_settings
from evaluation.judge_config import (
    EvaluationJudgeSettings,
    build_judge_runtime_metadata,
    load_evaluation_judge_settings,
)
from evaluation.schemas import RuntimeMetadata
from prompting import get_active_prompt_manifest


EVALUATION_SCHEMA_VERSION = 3
EVALUATOR_VERSION = "p5a"


def build_runtime_config_snapshot(
    settings: Settings | None = None,
    features: AgentFeatureFlags | None = None,
    judge_settings: EvaluationJudgeSettings | None = None,
    judge_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return reproducibility metadata without secrets or local paths."""

    return build_runtime_metadata(
        settings=settings,
        features=features,
        judge_settings=judge_settings,
        judge_metadata=judge_metadata,
    ).to_dict()


def build_runtime_metadata(
    settings: Settings | None = None,
    features: AgentFeatureFlags | None = None,
    judge_settings: EvaluationJudgeSettings | None = None,
    judge_metadata: Mapping[str, Any] | None = None,
) -> RuntimeMetadata:
    """Return versioned reproducibility metadata without secrets or local paths."""

    resolved = settings or get_settings()
    resolved_features = features or AgentFeatureFlags()
    if judge_metadata is None:
        resolved_judge = (
            judge_settings
            if judge_settings is not None
            else load_evaluation_judge_settings()
        )
        resolved_judge_metadata = build_judge_runtime_metadata(resolved_judge)
    else:
        resolved_judge_metadata = _sanitize_judge_runtime_metadata(judge_metadata)
    return RuntimeMetadata(
        schema_version=EVALUATION_SCHEMA_VERSION,
        evaluator_version=EVALUATOR_VERSION,
        config={
            "agent_features": resolved_features.to_dict(),
            "judge": resolved_judge_metadata,
            "llm": {
                "provider": resolved.llm_provider,
                "model": resolved.effective_llm_model,
                "temperature": resolved.temperature,
            },
            "prompts": get_active_prompt_manifest(),
            "retriever": {
                "top_k": resolved.top_k,
                "hybrid_retrieval_enabled": resolved.hybrid_retrieval_enabled,
                "dense_top_k": resolved.dense_top_k,
                "bm25_top_k": resolved.bm25_top_k,
                "fusion_top_k": resolved.fusion_top_k,
            },
            "reranker": {
                "enabled": resolved.reranker_enabled,
                "model": resolved.reranker_model,
                "top_n": resolved.reranker_top_n,
                "candidate_top_k": resolved.reranker_candidate_top_k,
            },
            "vectorstore": {
                "collection_name": resolved.chroma_collection_name,
            },
        },
    )


def _sanitize_judge_runtime_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, bool | str | float | None]:
    enabled = metadata.get("enabled") is True
    provider_value = metadata.get("provider")
    provider = (
        provider_value.strip()
        if isinstance(provider_value, str) and provider_value.strip()
        else "injected"
    )
    model_value = metadata.get("model")
    model = (
        model_value.strip()
        if isinstance(model_value, str) and model_value.strip()
        else None
    )
    temperature_value = metadata.get("temperature")
    temperature: float | None = None
    if isinstance(temperature_value, int | float) and not isinstance(
        temperature_value, bool
    ):
        try:
            candidate_temperature = float(temperature_value)
        except OverflowError:
            candidate_temperature = None
        if (
            candidate_temperature is not None
            and math.isfinite(candidate_temperature)
        ):
            temperature = candidate_temperature
    return {
        "enabled": enabled,
        "provider": provider,
        "model": model if enabled else None,
        "temperature": temperature if enabled else 0.0,
    }
