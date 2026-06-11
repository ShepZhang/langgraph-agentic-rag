"""Structured retrieval grading parser and normalizer."""

from __future__ import annotations

import json
from typing import Any, Literal, TypedDict


RelevanceLabel = Literal["relevant", "partially_relevant", "irrelevant"]
VALID_RELEVANCE_LABELS = {"relevant", "partially_relevant", "irrelevant"}
DEFAULT_GRADING_REASON = "No grading reason provided."


class DocumentGrade(TypedDict):
    """Chunk-level retrieval grading result."""

    document_index: int
    relevance: RelevanceLabel
    confidence: float
    reason: str


class GradingResult(TypedDict):
    """Normalized retrieval grading output."""

    grades: list[DocumentGrade]
    relevant_indices: list[int]
    partially_relevant_indices: list[int]
    reason: str


def parse_retrieval_grading_response(
    raw_result: str,
    document_count: int,
) -> GradingResult:
    """Parse structured or legacy retrieval grading JSON."""

    parsed = _extract_first_json_object(raw_result)
    if parsed is None:
        return _empty_result("Could not parse retrieval grading JSON.")

    if isinstance(parsed.get("grades"), list):
        return _parse_structured_grades(parsed, document_count=document_count)

    return _parse_legacy_grading(parsed, document_count=document_count)


def build_legacy_document_grades(
    relevant_indices: list[int],
    document_count: int,
    reason: str,
) -> list[DocumentGrade]:
    """Build structured grades from legacy relevant index output."""

    grades: list[DocumentGrade] = []
    for index in relevant_indices:
        if index < 1 or index > document_count:
            continue
        grades.append(
            {
                "document_index": index,
                "relevance": "relevant",
                "confidence": 1.0,
                "reason": reason or DEFAULT_GRADING_REASON,
            }
        )
    return grades


def _parse_structured_grades(
    parsed: dict[str, Any],
    document_count: int,
) -> GradingResult:
    grades: list[DocumentGrade] = []
    for raw_grade in parsed.get("grades", []):
        if not isinstance(raw_grade, dict):
            continue
        document_index = _coerce_document_index(raw_grade.get("document_index"))
        if document_index is None or document_index < 1 or document_index > document_count:
            continue
        relevance = _normalize_relevance(raw_grade.get("relevance"))
        grades.append(
            {
                "document_index": document_index,
                "relevance": relevance,
                "confidence": _clamp_confidence(raw_grade.get("confidence")),
                "reason": _normalize_reason(raw_grade.get("reason")),
            }
        )

    relevant_indices = [
        grade["document_index"]
        for grade in grades
        if grade["relevance"] == "relevant"
    ]
    partially_relevant_indices = [
        grade["document_index"]
        for grade in grades
        if grade["relevance"] == "partially_relevant"
    ]
    return {
        "grades": grades,
        "relevant_indices": _dedupe_indices(relevant_indices),
        "partially_relevant_indices": _dedupe_indices(partially_relevant_indices),
        "reason": _normalize_reason(parsed.get("reason")),
    }


def _parse_legacy_grading(
    parsed: dict[str, Any],
    document_count: int,
) -> GradingResult:
    raw_indices = parsed.get("relevant_indices", [])
    if not isinstance(raw_indices, list):
        raw_indices = []
    relevant_indices: list[int] = []
    for raw_index in raw_indices:
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            continue
        if raw_index < 1 or raw_index > document_count:
            continue
        relevant_indices.append(raw_index)

    if parsed.get("relevant") is not True:
        relevant_indices = []
    relevant_indices = _dedupe_indices(relevant_indices)
    reason = _normalize_reason(parsed.get("reason"))
    return {
        "grades": build_legacy_document_grades(
            relevant_indices,
            document_count=document_count,
            reason=reason,
        ),
        "relevant_indices": relevant_indices,
        "partially_relevant_indices": [],
        "reason": reason,
    }


def _empty_result(reason: str) -> GradingResult:
    return {
        "grades": [],
        "relevant_indices": [],
        "partially_relevant_indices": [],
        "reason": reason,
    }


def _extract_first_json_object(raw_result: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(raw_result):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw_result[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _coerce_document_index(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _normalize_relevance(value: Any) -> RelevanceLabel:
    if not isinstance(value, str):
        return "irrelevant"
    normalized = value.strip().lower()
    if normalized in VALID_RELEVANCE_LABELS:
        return normalized  # type: ignore[return-value]
    return "irrelevant"


def _clamp_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _normalize_reason(value: Any) -> str:
    if not isinstance(value, str):
        return DEFAULT_GRADING_REASON
    normalized = value.strip()
    return normalized or DEFAULT_GRADING_REASON


def _dedupe_indices(indices: list[int]) -> list[int]:
    deduped: list[int] = []
    for index in indices:
        if index not in deduped:
            deduped.append(index)
    return deduped
