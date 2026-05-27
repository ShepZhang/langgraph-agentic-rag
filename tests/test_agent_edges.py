"""Tests for Agent graph routing."""

from __future__ import annotations

from dataclasses import replace

from config import get_settings
from agent.edges import route_after_grading
from agent.state import create_initial_state


def test_route_after_grading_generates_when_relevant():
    state = create_initial_state("question")
    state["is_relevant"] = True

    route = route_after_grading(state)

    assert route == "generate_answer"


def test_route_after_grading_rewrites_when_under_attempt_limit():
    settings = replace(get_settings(), max_rewrite_attempts=2)
    state = create_initial_state("question")
    state["is_relevant"] = False
    state["rewrite_count"] = 1

    route = route_after_grading(state, settings=settings)

    assert route == "rewrite_query"


def test_route_after_grading_falls_back_at_attempt_limit():
    settings = replace(get_settings(), max_rewrite_attempts=2)
    state = create_initial_state("question")
    state["is_relevant"] = False
    state["rewrite_count"] = 2

    route = route_after_grading(state, settings=settings)

    assert route == "fallback"
