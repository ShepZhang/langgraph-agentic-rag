"""Evaluation dataset loading and normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

from agent.state import ChatMessage
from evaluation.schemas import EvaluationQuestion


DEFAULT_EVAL_PATH = Path(__file__).with_name("eval_questions.json")
SourceMatchMode = Literal["any", "all"]


def load_questions(path: str | Path = DEFAULT_EVAL_PATH) -> list[EvaluationQuestion]:
    """Load evaluation questions from JSON and return typed records."""

    with Path(path).open(encoding="utf-8") as question_file:
        records = json.load(question_file)

    return normalize_questions(records)


def normalize_questions(records: Any) -> list[EvaluationQuestion]:
    """Normalize raw evaluation question records into typed records."""

    if not isinstance(records, list):
        raise ValueError("evaluation questions must be a list")

    return [
        normalize_question(record, index)
        for index, record in enumerate(records)
    ]


def normalize_question(record: dict[str, Any], index: int) -> EvaluationQuestion:
    """Normalize one raw evaluation question into a typed record."""

    if not isinstance(record, dict):
        raise ValueError(f"evaluation question at index {index} must be an object")

    question = record.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"evaluation question at index {index} requires question")

    requires_rewrite = record.get("requires_rewrite", False)
    if not isinstance(requires_rewrite, bool):
        raise ValueError("requires_rewrite must be a boolean")

    expected_sources = _normalize_expected_sources(record)
    source_match_mode = record.get("source_match_mode", "any")
    if source_match_mode not in {"any", "all"}:
        raise ValueError("source_match_mode must be 'any' or 'all'")
    if source_match_mode == "all" and len(expected_sources) < 2:
        raise ValueError(
            "source_match_mode 'all' requires at least two expected_sources"
        )

    answerable = _get_answerable(record)
    expected_behavior = _normalize_expected_behavior(record, answerable)

    return EvaluationQuestion(
        id=str(record.get("id") or f"q{index + 1:03d}"),
        question=question,
        question_type=str(record.get("question_type") or "unspecified").strip(),
        gold_answer=str(record.get("gold_answer") or "").strip(),
        expected_sources=expected_sources,
        expected_keywords=_normalize_string_list(
            record.get("expected_keywords"),
            field_name="expected_keywords",
        ),
        source_match_mode=cast(SourceMatchMode, source_match_mode),
        answerable=answerable,
        expected_behavior=expected_behavior,
        chat_history=_normalize_chat_history(record.get("chat_history", [])),
        requires_rewrite=requires_rewrite,
        extra_fields=dict(record),
    )


def _normalize_expected_sources(record: dict[str, Any]) -> list[str]:
    if "expected_sources" in record:
        return _normalize_string_list(
            record.get("expected_sources"),
            field_name="expected_sources",
        )
    return _normalize_string_list(
        record.get("expected_source"),
        field_name="expected_source",
    )


def _normalize_expected_behavior(
    record: dict[str, Any],
    answerable: bool,
) -> str:
    expected_behavior = record.get("expected_behavior")
    if expected_behavior is None:
        return "answer_with_citation" if answerable else "fallback"
    if not isinstance(expected_behavior, str):
        raise ValueError(
            "expected_behavior must be one of answer_with_citation or fallback"
        )

    expected_behavior = expected_behavior.strip()
    if expected_behavior not in {"answer_with_citation", "fallback"}:
        raise ValueError(
            "expected_behavior must be one of answer_with_citation or fallback"
        )
    if answerable and expected_behavior == "fallback":
        raise ValueError("expected_behavior must match answerable")
    if not answerable and expected_behavior == "answer_with_citation":
        raise ValueError("expected_behavior must match answerable")
    return expected_behavior


def _get_answerable(item: dict[str, Any]) -> bool:
    has_answerable = "answerable" in item
    has_should_answer = "should_answer" in item

    if has_answerable:
        answerable = item.get("answerable")
        if not isinstance(answerable, bool):
            raise ValueError("answerable must be a boolean")
        if has_should_answer:
            should_answer = item.get("should_answer")
            if not isinstance(should_answer, bool):
                raise ValueError("should_answer must be a boolean")
            if answerable != should_answer:
                raise ValueError("answerable and should_answer must match")
        return answerable

    if has_should_answer:
        should_answer = item.get("should_answer")
        if not isinstance(should_answer, bool):
            raise ValueError("should_answer must be a boolean")
        return should_answer

    return True


def _normalize_chat_history(value: Any) -> list[ChatMessage]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("chat_history must be a list")

    normalized_history: list[ChatMessage] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(
                "chat_history must contain dict entries with role and content"
            )
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not role.strip():
            raise ValueError("chat_history entries require string role and content")
        if not isinstance(content, str):
            raise ValueError("chat_history entries require string role and content")
        normalized_history.append(cast(ChatMessage, dict(item)))

    return normalized_history


def _normalize_string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return value
        raise ValueError(f"{field_name} must contain only strings")
    raise ValueError(f"{field_name} must be a string or list of strings")
