"""Strict parsing for semantic judge responses."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass


_TOP_LEVEL_KEYS = {"semantic_correctness", "groundedness"}
_SEMANTIC_KEYS = {"score", "reason"}
_GROUNDED_KEYS = {"applicable", "score", "reason"}


@dataclass(frozen=True)
class ParsedJudgeResult:
    raw_scores: dict[str, int | None]
    scores: dict[str, float | None]
    reasons: dict[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_scores", deepcopy(self.raw_scores))
        object.__setattr__(self, "scores", deepcopy(self.scores))
        object.__setattr__(self, "reasons", deepcopy(self.reasons))


def parse_semantic_judge_response(
    raw_text: str,
    *,
    fallback_triggered: bool,
) -> ParsedJudgeResult:
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError(
            "raw_text must be a non-blank string; blank values are not allowed"
        )

    try:
        parsed = json.loads(raw_text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("raw_text must be strict JSON") from exc

    if not isinstance(parsed, dict):
        raise ValueError("root must be a JSON object")

    _validate_top_level_keys(parsed)

    semantic = _require_dict(parsed["semantic_correctness"], "semantic_correctness")
    grounded = _require_dict(parsed["groundedness"], "groundedness")

    semantic_score = _parse_int_score(
        semantic.get("score"),
        field="semantic_correctness.score",
    )
    semantic_reason = _parse_reason(
        semantic.get("reason"),
        field="semantic_correctness.reason",
    )
    _require_exact_keys(semantic, _SEMANTIC_KEYS, "semantic_correctness")

    applicable = _parse_applicable(grounded.get("applicable"))
    grounded_score_raw = grounded.get("score")
    grounded_reason = _parse_reason(
        grounded.get("reason"),
        field="groundedness.reason",
    )
    _require_exact_keys(grounded, _GROUNDED_KEYS, "groundedness")

    expected_applicable = not fallback_triggered
    if applicable != expected_applicable:
        raise ValueError("groundedness.applicable does not match fallback_triggered")

    if applicable:
        grounded_score = _parse_int_score(
            grounded_score_raw,
            field="groundedness.score",
        )
    else:
        if grounded_score_raw is not None:
            raise ValueError("groundedness.score must be None when groundedness.applicable is false")
        grounded_score = None

    raw_scores = {
        "semantic_correctness": semantic_score,
        "groundedness": grounded_score,
    }
    scores = {
        "semantic_correctness": semantic_score / 4,
        "groundedness": None if grounded_score is None else grounded_score / 4,
    }
    reasons = {
        "semantic_correctness": semantic_reason,
        "groundedness": grounded_reason,
    }
    return ParsedJudgeResult(raw_scores=raw_scores, scores=scores, reasons=reasons)


def _validate_top_level_keys(payload: dict[str, object]) -> None:
    keys = set(payload)
    if keys != _TOP_LEVEL_KEYS:
        raise ValueError("top-level keys must be exactly semantic_correctness and groundedness")


def _require_dict(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _require_exact_keys(payload: dict[str, object], expected: set[str], field: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{field} must contain exactly {', '.join(sorted(expected))}")


def _parse_int_score(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0 or value > 4:
        raise ValueError(f"{field} must be an int from 0 to 4")
    return value


def _parse_applicable(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError("groundedness.applicable must be a bool")
    return value


def _parse_reason(value: object, *, field: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{field} must be a non-empty string")
    reason = value.strip()
    if not reason:
        raise ValueError(f"{field} must be a non-empty string")
    return reason
