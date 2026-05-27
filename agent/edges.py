"""Conditional routing for the Agentic RAG graph."""

from __future__ import annotations

from config import Settings, get_settings
from agent.state import AgentState


def route_after_grading(
    state: AgentState,
    settings: Settings | None = None,
) -> str:
    """Route after retrieval grading."""

    resolved_settings = settings or get_settings()

    if state.get("is_relevant"):
        return "generate_answer"
    if state.get("rewrite_count", 0) < resolved_settings.max_rewrite_attempts:
        return "rewrite_query"
    return "fallback"
