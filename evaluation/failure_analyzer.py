"""Deterministic failed-case attribution for evaluation results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any, Literal


FailureType = Literal[
    "no_failure",
    "tool_failure",
    "fallback_failure",
    "query_rewrite_failure",
    "retrieval_failure",
    "reranking_failure",
    "citation_failure",
    "generation_failure",
]

SUCCESS_REASON = "The case satisfied correctness, fallback, and evidence checks."
SUCCESS_SUGGESTION = "No action required."


def analyze_failure(question: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, str]:
    """Attribute one evaluation result to the earliest deterministic failure stage."""

    question_id = _question_id(question, result)
    expected_sources = _expected_sources(question)

    if _is_successful_case(question, result):
        return _analysis(
            question_id,
            "no_failure",
            SUCCESS_REASON,
            SUCCESS_SUGGESTION,
        )

    error = _clean_text(result.get("error"))
    if error:
        return _analysis(
            question_id,
            "tool_failure",
            f"Tool or runtime error occurred: {error}",
            "Inspect the execution trace, tool inputs, and exception logs.",
        )

    answerable = _answerable(question)
    fallback_triggered = bool(result.get("fallback_triggered"))
    answer_returned = bool(result.get("answer_returned"))
    if answerable and fallback_triggered:
        return _analysis(
            question_id,
            "fallback_failure",
            "Answerable question triggered fallback instead of returning an answer.",
            "Inspect fallback gating, retrieval confidence thresholds, and evidence checks.",
        )
    if not answerable and answer_returned and not fallback_triggered:
        return _analysis(
            question_id,
            "fallback_failure",
            "Unanswerable question returned an answer instead of falling back.",
            "Tighten answerability detection and require fallback when evidence is insufficient.",
        )

    retrieved_has_expected = _has_source(
        expected_sources,
        result.get("retrieved_documents", []),
    )
    evidence_has_expected = _has_source(
        expected_sources,
        _evidence_records(result),
    )
    source_hit = _source_hit(result, expected_sources)
    retry_count = _safe_int(result.get("retry_count", result.get("rewrite_count", 0)))

    if (
        _requires_query_rewrite(question)
        and not source_hit
        and retry_count == 0
        and not bool(result.get("rewrite_triggered"))
    ):
        return _analysis(
            question_id,
            "query_rewrite_failure",
            "Question required rewrite but no retry retrieved the expected source.",
            "Generate a standalone rewritten query before retrieval and retry on missing source hits.",
        )

    if expected_sources and not retrieved_has_expected and not source_hit:
        return _analysis(
            question_id,
            "retrieval_failure",
            "Expected source was not retrieved in candidate documents.",
            "Try query expansion, hybrid retrieval, or increasing retrieval top_k.",
        )

    if (
        expected_sources
        and retrieved_has_expected
        and not evidence_has_expected
        and not bool(result.get("context_relevant"))
        and not bool(result.get("citation_hit"))
    ):
        return _analysis(
            question_id,
            "reranking_failure",
            "Expected source was retrieved but not selected as relevant evidence or cited.",
            "Tune reranking, relevance grading, or evidence selection thresholds.",
        )

    unsupported_claim_count = _safe_int(result.get("unsupported_claim_count"))
    if unsupported_claim_count > 0:
        return _analysis(
            question_id,
            "citation_failure",
            f"Answer included {unsupported_claim_count} unsupported claim(s).",
            "Strengthen citation verification and require every claim to be supported by evidence.",
        )

    if not bool(result.get("correct")) and _has_any_evidence_hit(result, expected_sources):
        return _analysis(
            question_id,
            "generation_failure",
            "Evidence was available, but the generated answer failed correctness checks.",
            "Revise answer synthesis prompts or post-generation validation against the gold answer.",
        )

    if not bool(result.get("fallback_correct", True)):
        return _analysis(
            question_id,
            "fallback_failure",
            "Fallback behavior did not match the question answerability expectation.",
            "Review answerability labels, fallback policy, and fallback trigger conditions.",
        )

    return _analysis(
        question_id,
        "generation_failure",
        "Case failed correctness checks without an earlier deterministic failure signal.",
        "Inspect the generated answer, expected answer, and evaluation labels.",
    )


def summarize_failure_types(results: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Count failure types already attached to evaluation result records."""

    counts: Counter[str] = Counter()
    for result in results:
        analysis = result.get("failure_analysis")
        if not isinstance(analysis, Mapping):
            counts["tool_failure"] += 1
            continue

        failure_type = _clean_text(analysis.get("failure_type"))
        counts[failure_type or "tool_failure"] += 1

    return dict(counts)


def _analysis(
    question_id: str,
    failure_type: FailureType,
    reason: str,
    suggestion: str,
) -> dict[str, str]:
    return {
        "question_id": question_id,
        "failure_type": failure_type,
        "reason": reason,
        "suggestion": suggestion,
    }


def _is_successful_case(
    question: Mapping[str, Any],
    result: Mapping[str, Any],
) -> bool:
    if _clean_text(result.get("error")):
        return False
    if _safe_int(result.get("unsupported_claim_count")) > 0:
        return False
    if not bool(result.get("correct")):
        return False
    if not bool(result.get("fallback_correct")):
        return False

    if not _answerable(question):
        return bool(result.get("fallback_triggered")) or not bool(result.get("answer_returned"))

    expected_sources = _expected_sources(question)
    if not expected_sources:
        return True

    return _source_hit(result, expected_sources) and (
        bool(result.get("context_relevant"))
        or bool(result.get("citation_hit"))
        or _has_source(expected_sources, _evidence_records(result))
    )


def _answerable(question: Mapping[str, Any]) -> bool:
    answerable = question.get("answerable")
    if isinstance(answerable, bool):
        return answerable

    should_answer = question.get("should_answer")
    if isinstance(should_answer, bool):
        return should_answer

    expected_behavior = _clean_text(question.get("expected_behavior")).lower()
    if expected_behavior == "fallback":
        return False
    return True


def _requires_query_rewrite(question: Mapping[str, Any]) -> bool:
    if question.get("requires_rewrite") is True:
        return True
    question_type = _clean_text(question.get("question_type")).lower()
    return question_type in {"follow_up", "ambiguous"}


def _has_any_evidence_hit(
    result: Mapping[str, Any],
    expected_sources: list[str],
) -> bool:
    return (
        bool(result.get("source_hit"))
        or bool(result.get("context_relevant"))
        or bool(result.get("citation_hit"))
        or _has_source(expected_sources, result.get("retrieved_documents", []))
        or _has_source(expected_sources, _evidence_records(result))
    )


def _source_hit(result: Mapping[str, Any], expected_sources: list[str]) -> bool:
    return bool(result.get("source_hit")) or _has_source(
        expected_sources,
        _candidate_and_evidence_records(result),
    )


def _expected_sources(question: Mapping[str, Any]) -> list[str]:
    if "expected_sources" in question:
        return _string_list(question.get("expected_sources"))
    return _string_list(question.get("expected_source"))


def _candidate_and_evidence_records(result: Mapping[str, Any]) -> list[Any]:
    records: list[Any] = []
    records.extend(_as_records(result.get("retrieved_documents", [])))
    records.extend(_evidence_records(result))
    return records


def _evidence_records(result: Mapping[str, Any]) -> list[Any]:
    records: list[Any] = []
    for field_name in (
        "relevant_documents",
        "citations",
        "evidence",
        "evidence_documents",
        "source_documents",
        "context",
        "contexts",
        "context_documents",
    ):
        records.extend(_as_records(result.get(field_name, [])))
    return records


def _as_records(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return []


def _has_source(expected_sources: Any, records: Any) -> bool:
    expected = _string_list(expected_sources)
    if not expected:
        return False

    return any(
        _source_matches(expected_source, actual_source)
        for expected_source in expected
        for actual_source in _iter_sources(records)
    )


def _iter_sources(records: Any) -> Iterable[str]:
    if isinstance(records, str):
        cleaned = _clean_text(records)
        if cleaned:
            yield cleaned
        return

    if isinstance(records, Mapping):
        yield from _sources_from_document(records)
        return

    if not isinstance(records, Iterable):
        return

    for record in records:
        if isinstance(record, str):
            cleaned = _clean_text(record)
            if cleaned:
                yield cleaned
        elif isinstance(record, Mapping):
            yield from _sources_from_document(record)


def _sources_from_document(document: Mapping[str, Any]) -> Iterable[str]:
    for field_name in ("source", "document_id"):
        for value in _string_list(document.get(field_name)):
            yield value

    metadata = document.get("metadata")
    if isinstance(metadata, Mapping):
        for field_name in ("source", "file_path"):
            for value in _string_list(metadata.get(field_name)):
                yield value


def _source_matches(expected: Any, actual: Any) -> bool:
    expected_normalized = _normalize_source(expected)
    actual_normalized = _normalize_source(actual)
    if not expected_normalized or not actual_normalized:
        return False

    expected_basename = expected_normalized.rsplit("/", maxsplit=1)[-1]
    actual_basename = actual_normalized.rsplit("/", maxsplit=1)[-1]
    return (
        expected_normalized == actual_normalized
        or expected_basename == actual_basename
        or expected_normalized in actual_normalized
        or actual_normalized in expected_normalized
    )


def _normalize_source(value: Any) -> str:
    return _clean_text(value).replace("\\", "/").lower()


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = _clean_text(value)
        return [cleaned] if cleaned else []
    if isinstance(value, Iterable) and not isinstance(value, Mapping):
        values = [_clean_text(item) for item in value]
        return [item for item in values if item]
    return []


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _safe_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _question_id(question: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    return (
        _clean_text(result.get("question_id"))
        or _clean_text(question.get("id"))
        or _clean_text(question.get("question_id"))
        or "unknown"
    )


__all__ = [
    "FailureType",
    "analyze_failure",
    "summarize_failure_types",
]
