"""Tests for structured query transformation parsing."""

from __future__ import annotations

from agent.query_transform import (
    build_query_transform_prompt,
    fallback_query_transform,
    parse_query_transform_response,
)


def test_parse_query_transform_response_reads_structured_json():
    raw = (
        '{"strategy": "multi_query", '
        '"rewritten_query": "What advantages does Agentic RAG have?", '
        '"expanded_queries": ["Agentic RAG benefits", "Agentic RAG vs naive RAG"], '
        '"sub_questions": ["ignored for multi query"], '
        '"reason": "The question is ambiguous and benefits from expansion."}'
    )

    result = parse_query_transform_response(
        raw,
        original_question="What are its advantages?",
    )

    assert result == {
        "strategy": "multi_query",
        "rewritten_query": "What advantages does Agentic RAG have?",
        "expanded_queries": [
            "Agentic RAG benefits",
            "Agentic RAG vs naive RAG",
        ],
        "sub_questions": [],
        "reason": "The question is ambiguous and benefits from expansion.",
    }


def test_parse_query_transform_response_reads_fenced_json():
    raw = (
        "```json\n"
        '{"strategy": "decomposition", '
        '"rewritten_query": "Compare naive RAG and Agentic RAG reliability.", '
        '"expanded_queries": ["ignored for decomposition"], '
        '"sub_questions": ["What is naive RAG?", "What reliability controls does Agentic RAG add?"], '
        '"reason": "The question asks for a comparison."}'
        "\n```"
    )

    result = parse_query_transform_response(
        raw,
        original_question="Compare them.",
    )

    assert result["strategy"] == "decomposition"
    assert result["rewritten_query"] == "Compare naive RAG and Agentic RAG reliability."
    assert result["expanded_queries"] == []
    assert result["sub_questions"] == [
        "What is naive RAG?",
        "What reliability controls does Agentic RAG add?",
    ]


def test_parse_query_transform_response_falls_back_for_plain_text():
    result = parse_query_transform_response(
        "What is Agentic RAG?",
        original_question="What is it?",
    )

    assert result["strategy"] == "rewrite"
    assert result["rewritten_query"] == "What is Agentic RAG?"
    assert result["expanded_queries"] == []
    assert result["sub_questions"] == []
    assert "plain text" in result["reason"].lower()


def test_parse_query_transform_response_falls_back_for_invalid_strategy():
    raw = (
        '{"strategy": "planning", '
        '"rewritten_query": "What is Agentic RAG?", '
        '"expanded_queries": ["x"], '
        '"sub_questions": ["y"], '
        '"reason": "invalid"}'
    )

    result = parse_query_transform_response(raw, original_question="What is it?")

    assert result["strategy"] == "rewrite"
    assert result["rewritten_query"] == "What is Agentic RAG?"
    assert result["expanded_queries"] == []
    assert result["sub_questions"] == []
    assert "invalid strategy" in result["reason"].lower()


def test_parse_query_transform_response_uses_original_question_for_blank_output():
    result = parse_query_transform_response("   ", original_question="What is RAG?")

    assert result == fallback_query_transform(
        "What is RAG?",
        reason="Blank query transform response; using original question.",
    )


def test_build_query_transform_prompt_contains_router_contract():
    prompt = build_query_transform_prompt(
        question="How does it compare?",
        chat_history=[{"role": "user", "content": "Discuss Agentic RAG."}],
    )

    assert "rewrite" in prompt
    assert "multi_query" in prompt
    assert "decomposition" in prompt
    assert "Return JSON only" in prompt
    assert "Discuss Agentic RAG." in prompt
    assert "How does it compare?" in prompt
