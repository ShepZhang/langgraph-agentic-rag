"""Sanitized runtime configuration snapshots for evaluation artifacts."""

from __future__ import annotations

from typing import Any

from agent.features import AgentFeatureFlags
from config import Settings, get_settings
from evaluation.schemas import RuntimeMetadata
from prompting import get_active_prompt_manifest


EVALUATION_SCHEMA_VERSION = 2
EVALUATOR_VERSION = "p4d"


def build_runtime_config_snapshot(
    settings: Settings | None = None,
    features: AgentFeatureFlags | None = None,
) -> dict[str, Any]:
    """Return reproducibility metadata without secrets or local paths."""

    return build_runtime_metadata(settings=settings, features=features).to_dict()


def build_runtime_metadata(
    settings: Settings | None = None,
    features: AgentFeatureFlags | None = None,
) -> RuntimeMetadata:
    """Return versioned reproducibility metadata without secrets or local paths."""

    resolved = settings or get_settings()
    resolved_features = features or AgentFeatureFlags()
    return RuntimeMetadata(
        schema_version=EVALUATION_SCHEMA_VERSION,
        evaluator_version=EVALUATOR_VERSION,
        config={
            "agent_features": resolved_features.to_dict(),
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
