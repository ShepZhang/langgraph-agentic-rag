"""Tests for evaluation dataset loading and normalization."""

from __future__ import annotations

import json
from typing import Any

import pytest

from evaluation.dataset import load_questions, normalize_question, normalize_questions
from evaluation.schemas import EvaluationQuestion


def test_load_questions_returns_typed_questions_with_legacy_source_and_default_mode(
    tmp_path,
):
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question": "What is Agentic RAG?",
                    "expected_source": "legacy.md",
                    "expected_keywords": ["agentic"],
                    "chat_history": [
                        {
                            "role": "user",
                            "content": "Discuss retrieval.",
                            "turn_id": "t1",
                        }
                    ],
                    "difficulty": "easy",
                }
            ]
        ),
        encoding="utf-8",
    )

    questions = load_questions(path)

    assert len(questions) == 1
    question = questions[0]
    assert isinstance(question, EvaluationQuestion)
    assert question.id == "q001"
    assert question.expected_sources == ["legacy.md"]
    assert question.source_match_mode == "any"
    assert question.answerable is True
    assert question.expected_behavior == "answer_with_citation"
    assert question.chat_history == [
        {"role": "user", "content": "Discuss retrieval.", "turn_id": "t1"}
    ]
    assert question.extra_fields["expected_source"] == "legacy.md"
    assert question.extra_fields["difficulty"] == "easy"
    assert question.to_compat_dict()["expected_source"] == "legacy.md"
    assert question.to_compat_dict()["should_answer"] is True


def test_normalize_question_preserves_all_source_match_mode():
    question = normalize_question(
        {
            "question": "How do two files relate?",
            "expected_sources": ["first.md", "second.md"],
            "source_match_mode": "all",
        },
        0,
    )

    assert question.source_match_mode == "all"
    assert question.to_compat_dict()["source_match_mode"] == "all"


def test_normalize_question_rejects_invalid_source_match_mode():
    with pytest.raises(ValueError, match="source_match_mode"):
        normalize_question(
            {
                "question": "Bad mode?",
                "expected_sources": ["notes.md"],
                "source_match_mode": "either",
            },
            0,
        )


def test_normalize_question_rejects_unhashable_source_match_mode():
    with pytest.raises(ValueError, match="source_match_mode"):
        normalize_question(
            {
                "question": "Bad mode type?",
                "expected_sources": ["notes.md"],
                "source_match_mode": [],
            },
            0,
        )


def test_normalize_question_rejects_all_mode_with_fewer_than_two_sources():
    with pytest.raises(ValueError, match="source_match_mode"):
        normalize_question(
            {
                "question": "Bad all mode?",
                "expected_sources": ["notes.md"],
                "source_match_mode": "all",
            },
            0,
        )


def test_normalize_question_rejects_conflicting_answerable_and_should_answer():
    with pytest.raises(ValueError, match="answerable"):
        normalize_question(
            {
                "question": "Conflict?",
                "answerable": True,
                "should_answer": False,
            },
            0,
        )


@pytest.mark.parametrize(
    "chat_history",
    [
        "not a list",
        ["user said this"],
        [{"role": "user"}],
        [{"role": "user", "content": 123}],
    ],
)
def test_normalize_question_rejects_bad_chat_history(chat_history: Any):
    with pytest.raises(ValueError, match="chat_history"):
        normalize_question(
            {
                "question": "Bad history?",
                "chat_history": chat_history,
            },
            0,
        )


def test_normalize_questions_rejects_non_list_root():
    with pytest.raises(ValueError, match="evaluation questions must be a list"):
        normalize_questions({"question": "Not a list"})
