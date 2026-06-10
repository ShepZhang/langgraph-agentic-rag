"""Sanitized runtime configuration snapshots for evaluation artifacts."""

from __future__ import annotations

from typing import Any

from config import Settings, get_settings


def build_runtime_config_snapshot(
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Return reproducibility metadata without secrets or local paths."""

    resolved = settings or get_settings()
    return {
        "llm": {
            "provider": resolved.llm_provider,
            "model": resolved.effective_llm_model,
            "temperature": resolved.temperature,
        },
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
    }
