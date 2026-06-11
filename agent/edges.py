"""Conditional routing for the Agentic RAG graph."""

from __future__ import annotations

import logging
from typing import Literal

from agent.features import AgentFeatureFlags
from config import Settings, get_settings
from agent.state import AgentState

AgentRoute = Literal["generate_answer", "rewrite_query", "fallback"]
AnswerRoute = Literal["extract_claims", "finalize_answer", "fallback"]
ExtractionRoute = Literal["verify_citations", "finalize_answer", "fallback"]
VerificationRoute = Literal["finalize_answer", "revise_answer", "fallback"]
RevisionRoute = Literal["extract_claims", "finalize_answer", "fallback"]
logger = logging.getLogger(__name__)


def route_after_grading(
    state: AgentState,
    settings: Settings | None = None,
    features: AgentFeatureFlags | None = None,
) -> AgentRoute:
    """Route after retrieval grading."""

    if state.get("relevant_documents"):
        logger.info("Route decision: generate_answer")
        return "generate_answer"

    resolved_features = features or AgentFeatureFlags()
    if not resolved_features.conditional_retry_enabled:
        logger.info("Route decision: fallback conditional retry disabled")
        return "fallback"

    resolved_settings = settings or get_settings()
    max_retry_count = state.get("max_retry_count", resolved_settings.max_retry_count)

    if state.get("retry_count", 0) < max_retry_count:
        logger.info(
            "Route decision: rewrite_query retry_count=%s max_retry_count=%s",
            state.get("retry_count", 0),
            max_retry_count,
        )
        return "rewrite_query"
    logger.info(
        "Route decision: fallback retry_count=%s max_retry_count=%s",
        state.get("retry_count", 0),
        max_retry_count,
    )
    return "fallback"


def route_after_answer_generation(state: AgentState) -> AnswerRoute:
    """Route after draft answer generation."""

    route = state.get("route")
    if route in {"extract_claims", "finalize_answer", "fallback"}:
        logger.info("Route decision after answer generation: %s", route)
        return route  # type: ignore[return-value]
    logger.info(
        "Route decision after answer generation: fallback invalid_route=%s",
        route,
    )
    return "fallback"


def route_after_claim_extraction(state: AgentState) -> ExtractionRoute:
    """Route after claim extraction."""

    route = state.get("route")
    if route in {"verify_citations", "finalize_answer", "fallback"}:
        logger.info("Route decision after claim extraction: %s", route)
        return route  # type: ignore[return-value]
    logger.info(
        "Route decision after claim extraction: fallback invalid_route=%s",
        route,
    )
    return "fallback"


def route_after_citation_verification(state: AgentState) -> VerificationRoute:
    """Route after claim-level citation verification."""

    route = state.get("route")
    if route in {"finalize_answer", "revise_answer", "fallback"}:
        logger.info("Route decision after citation verification: %s", route)
        return route  # type: ignore[return-value]
    logger.info(
        "Route decision after citation verification: fallback invalid_route=%s",
        route,
    )
    return "fallback"


def route_after_answer_revision(state: AgentState) -> RevisionRoute:
    """Route after answer revision."""

    route = state.get("route")
    if route in {"extract_claims", "finalize_answer", "fallback"}:
        logger.info("Route decision after answer revision: %s", route)
        return route  # type: ignore[return-value]
    logger.info(
        "Route decision after answer revision: fallback invalid_route=%s",
        route,
    )
    return "fallback"
