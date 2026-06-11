"""Tests for Agent graph routing."""

from __future__ import annotations

from dataclasses import replace

from config import get_settings
from agent.edges import route_after_grading
from agent.features import AgentFeatureFlags
from agent.state import create_initial_state


def test_agent_feature_flags_default_to_complete_workflow():
    flags = AgentFeatureFlags()

    assert flags.query_transformation_enabled is True
    assert flags.retrieval_grading_enabled is True
    assert flags.conditional_retry_enabled is True
    assert flags.citation_verification_enabled is True
    assert flags.to_dict() == {
        "query_transformation_enabled": True,
        "retrieval_grading_enabled": True,
        "conditional_retry_enabled": True,
        "citation_verification_enabled": True,
    }


def test_route_after_grading_generates_when_relevant_documents_exist_without_settings_lookup(
    monkeypatch,
):
    def fail_get_settings():
        raise AssertionError("get_settings should not be called for relevant state")

    monkeypatch.setattr("agent.edges.get_settings", fail_get_settings)
    state = create_initial_state("question")
    state["relevant_documents"] = [{"content": "answer context"}]

    route = route_after_grading(state)

    assert route == "generate_answer"


def test_route_after_grading_generates_when_relevant_documents_exist():
    state = create_initial_state("question")
    state["relevant_documents"] = [{"content": "answer context"}]

    route = route_after_grading(state)

    assert route == "generate_answer"


def test_route_after_grading_rewrites_when_under_attempt_limit():
    settings = replace(get_settings(), max_retry_count=2)
    state = create_initial_state("question")
    state["relevant_documents"] = []
    state["retry_count"] = 1

    route = route_after_grading(state, settings=settings)

    assert route == "rewrite_query"


def test_route_after_grading_falls_back_at_attempt_limit():
    settings = replace(get_settings(), max_retry_count=2)
    state = create_initial_state("question")
    state["relevant_documents"] = []
    state["retry_count"] = 2

    route = route_after_grading(state, settings=settings)

    assert route == "fallback"


def test_route_after_grading_falls_back_when_retry_is_disabled():
    state = create_initial_state("question")
    state["relevant_documents"] = []
    state["retry_count"] = 0

    route = route_after_grading(
        state,
        features=AgentFeatureFlags(conditional_retry_enabled=False),
    )

    assert route == "fallback"


def test_route_after_grading_uses_state_max_retry_count_when_present():
    state = create_initial_state("question", max_retry_count=1)
    state["relevant_documents"] = []
    state["retry_count"] = 1

    route = route_after_grading(state)

    assert route == "fallback"
