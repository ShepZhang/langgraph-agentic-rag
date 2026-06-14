"""Conditional routing for the Agentic RAG graph."""

from __future__ import annotations

import logging
from typing import Literal

from agent.state import AgentState
from config import Settings, get_settings

AgentRoute = Literal["generate_answer", "rewrite_query", "fallback"]
logger = logging.getLogger(__name__)


def route_after_grading(
    state: AgentState,
    settings: Settings | None = None,
) -> AgentRoute:
    """Route after retrieval grading."""

    if state.get("relevant_documents"):
        logger.info("Route decision: generate_answer")
        return "generate_answer"

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
