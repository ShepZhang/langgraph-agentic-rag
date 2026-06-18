"""Structured query transformation utilities for retrieval preparation."""

from __future__ import annotations

import json
import re
from typing import Any, Literal, TypedDict

from agent.prompts import format_chat_history
from agent.state import ChatMessage
from prompting import render_prompt


QueryTransformStrategy = Literal["rewrite", "multi_query", "decomposition"]
VALID_QUERY_TRANSFORM_STRATEGIES = {"rewrite", "multi_query", "decomposition"}


class QueryTransformResult(TypedDict):
    """Structured output from the query transformation step."""

    strategy: QueryTransformStrategy
    rewritten_query: str
    expanded_queries: list[str]
    sub_questions: list[str]
    reason: str


def build_query_transform_prompt(
    question: str,
    chat_history: list[ChatMessage],
) -> str:
    """Build the initial query transformation prompt."""

    return render_prompt(
        "agent.query_transform",
        chat_history=format_chat_history(chat_history),
        question=question,
    )


def fallback_query_transform(
    question: str,
    reason: str = "Fallback direct rewrite.",
) -> QueryTransformResult:
    """Build a safe direct rewrite transform."""

    return {
        "strategy": "rewrite",
        "rewritten_query": question.strip() or question,
        "expanded_queries": [],
        "sub_questions": [],
        "reason": reason,
    }


def parse_query_transform_response(
    raw_text: str,
    original_question: str,
) -> QueryTransformResult:
    """Parse structured query transformation output with safe fallback."""

    text = raw_text.strip()
    if not text:
        return fallback_query_transform(
            original_question,
            reason="Blank query transform response; using original question.",
        )

    json_text = _extract_json_object(text)
    if json_text is None:
        return fallback_query_transform(
            text,
            reason="Plain text query transform response treated as direct rewrite.",
        )

    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError:
        return fallback_query_transform(
            original_question,
            reason="Invalid JSON query transform response; using original question.",
        )

    if not isinstance(payload, dict):
        return fallback_query_transform(
            original_question,
            reason="Query transform response was not an object; using original question.",
        )

    strategy = str(payload.get("strategy", "rewrite")).strip()
    rewritten_query = _clean_string(payload.get("rewritten_query")) or original_question
    if strategy not in VALID_QUERY_TRANSFORM_STRATEGIES:
        return fallback_query_transform(
            rewritten_query,
            reason=f"Invalid strategy {strategy!r}; using direct rewrite.",
        )

    expanded_queries = _clean_string_list(payload.get("expanded_queries"))
    sub_questions = _clean_string_list(payload.get("sub_questions"))
    reason = _clean_string(payload.get("reason")) or "Structured query transform parsed."

    if strategy == "rewrite":
        expanded_queries = []
        sub_questions = []
    elif strategy == "multi_query":
        sub_questions = []
    elif strategy == "decomposition":
        expanded_queries = []

    return {
        "strategy": strategy,  # type: ignore[typeddict-item]
        "rewritten_query": rewritten_query,
        "expanded_queries": expanded_queries,
        "sub_questions": sub_questions,
        "reason": reason,
    }


def _extract_json_object(text: str) -> str | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start : end + 1]


def _clean_string(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _clean_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = _clean_string(item)
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned
