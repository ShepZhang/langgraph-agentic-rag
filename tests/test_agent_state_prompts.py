"""Tests for Agent state and prompt formatting."""

from __future__ import annotations

from agent.prompts import (
    ANSWER_GENERATION_PROMPT,
    QUERY_REWRITE_PROMPT,
    RETRIEVAL_GRADING_PROMPT,
    format_chat_history,
    format_documents,
)
from agent.state import create_initial_state


def test_create_initial_state_sets_defaults():
    state = create_initial_state("What is RAG?")

    assert state["question"] == "What is RAG?"
    assert state["rewritten_question"] == ""
    assert state["chat_history"] == []
    assert state["documents"] == []
    assert state["answer"] == ""
    assert state["citations"] == []
    assert state["rewrite_count"] == 0
    assert state["is_relevant"] is False
    assert state["route"] == ""


def test_create_initial_state_preserves_chat_history():
    history = [{"role": "user", "content": "Tell me about LangGraph"}]

    state = create_initial_state("How does it help?", chat_history=history)

    assert state["chat_history"] == history


def test_format_chat_history_handles_empty_and_nonempty_history():
    assert format_chat_history([]) == "No prior chat history."

    formatted = format_chat_history(
        [
            {"role": "user", "content": "What is RAG?"},
            {"role": "assistant", "content": "Retrieval augmented generation."},
        ]
    )

    assert "user: What is RAG?" in formatted
    assert "assistant: Retrieval augmented generation." in formatted


def test_format_documents_includes_metadata_and_content():
    docs = [
        {
            "content": "Chunk text",
            "source": "paper.pdf",
            "page": 2,
            "chunk_id": "paper.pdf:p2:c1",
            "score": 0.87,
        }
    ]

    formatted = format_documents(docs)

    assert "[1]" in formatted
    assert "source=paper.pdf" in formatted
    assert "page=2" in formatted
    assert "chunk_id=paper.pdf:p2:c1" in formatted
    assert "score=0.87" in formatted
    assert "Chunk text" in formatted


def test_prompts_contain_required_guardrails():
    assert "standalone" in QUERY_REWRITE_PROMPT.lower()
    assert "retrieved chunks" in ANSWER_GENERATION_PROMPT.lower()
    assert "json" in RETRIEVAL_GRADING_PROMPT.lower()
    assert "keyword" in RETRIEVAL_GRADING_PROMPT.lower()
