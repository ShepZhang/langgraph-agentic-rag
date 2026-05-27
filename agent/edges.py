"""Conditional routing for the Agentic RAG graph."""

from __future__ import annotations

from typing import Literal

from config import Settings, get_settings
from agent.state import AgentState

AgentRoute = Literal["generate_answer", "rewrite_query", "fallback"]


def route_after_grading(
    state: AgentState,
    settings: Settings | None = None,
) -> AgentRoute:
    """Route after retrieval grading."""

    if state.get("is_relevant") is True:
        return "generate_answer"

    resolved_settings = settings or get_settings()

    if state.get("rewrite_count", 0) < resolved_settings.max_rewrite_attempts:
        return "rewrite_query"
    return "fallback"
