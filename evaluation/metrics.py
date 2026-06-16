"""Deterministic per-question evaluation scoring."""

from __future__ import annotations

import math
import re
from typing import Any

from evaluation.failure_analyzer import analyze_failure
from evaluation.schemas import EvaluationQuestion, EvaluationResult


def score_system_output(
    question: EvaluationQuestion,
    system_result: dict[str, Any],
) -> EvaluationResult:
    """Score one system output against one normalized evaluation question."""

    if not isinstance(system_result, dict):
        raise ValueError("agent result must be a dictionary")

    answer = system_result.get("answer", "")
    if not isinstance(answer, str):
        raise ValueError("agent answer must be a string")

    citations = _safe_list(system_result.get("citations", []), "citations")
    claims = _safe_list(system_result.get("claims", []), "claims")
    claim_verification_results = _safe_list(
        system_result.get("claim_verification_results", []),
        "claim_verification_results",
    )
    retrieved_documents = _safe_list(
        system_result.get("retrieved_documents", []),
        "retrieved_documents",
    )
    relevant_documents = _safe_list(
        system_result.get("relevant_documents", []),
        "relevant_documents",
    )
    retry_count = _safe_int(
        system_result.get("retry_count", system_result.get("rewrite_count", 0)),
        "retry_count",
    )
    fallback_reason = system_result.get("fallback_reason", "")
    if fallback_reason is None:
        fallback_reason = ""
    if not isinstance(fallback_reason, str):
        raise ValueError("fallback_reason must be a string")

    fallback_triggered = bool(fallback_reason.strip()) or _is_fallback_answer(answer)
    should_answer = question.answerable
    answer_returned = bool(answer.strip()) and not fallback_triggered
    citation_verification_applicable = bool(
        system_result.get("citation_verification_enabled", True)
    )
    legacy_claim_labels = (
        "citation_verification_enabled" not in system_result
        and not claim_verification_results
    )
    verification_records = (
        claims if legacy_claim_labels else claim_verification_results
    )
    unsupported_claim_count = (
        _count_unsupported_claims(verification_records)
        if citation_verification_applicable
        else None
    )
    supported_claim_count = (
        _count_supported_claims(verification_records)
        if citation_verification_applicable
        else None
    )
    total_claim_count = (
        len(verification_records) if citation_verification_applicable else None
    )
    token_usage = system_result.get("token_usage")
    estimated_cost = _safe_cost(system_result.get("estimated_cost"))

    return EvaluationResult(
        question_id=question.id,
        question_type=question.question_type,
        question=question.question,
        chat_history_supplied=bool(question.chat_history),
        chat_history_used=bool(system_result.get("chat_history_used", False)),
        answer_returned=answer_returned,
        fallback_triggered=fallback_triggered,
        fallback_correct=fallback_triggered is (not should_answer),
        correct=_is_correct_answer(
            answer=answer,
            answer_returned=answer_returned,
            expected_keywords=question.expected_keywords,
            gold_answer=question.gold_answer,
            should_answer=should_answer,
        ),
        context_relevant=_has_expected_source(
            question.expected_sources,
            relevant_documents,
            retrieved_documents,
            question.source_match_mode,
        ),
        citation_hit=_has_expected_source(
            question.expected_sources,
            citations,
            [],
            question.source_match_mode,
        ),
        citation_returned=bool(citations),
        is_verified=bool(system_result.get("is_verified", False)),
        citation_verification_applicable=citation_verification_applicable,
        claim_count=len(claims),
        unsupported_claim_count=unsupported_claim_count,
        supported_claim_count=supported_claim_count,
        total_claim_count=total_claim_count,
        source_hit=_has_expected_source(
            question.expected_sources,
            citations,
            [],
            question.source_match_mode,
        )
        or _has_expected_source(
            question.expected_sources,
            retrieved_documents,
            [],
            question.source_match_mode,
        ),
        keyword_hit=(
            answer_returned and _has_expected_keywords(answer, question.expected_keywords)
        ),
        citation_verification_passed=bool(
            system_result.get(
                "citation_verification_passed",
                system_result.get("is_verified", False),
            )
        ),
        rewrite_triggered=retry_count > 0,
        retry_count=retry_count,
        retrieved_doc_count=len(retrieved_documents),
        relevant_doc_count=len(relevant_documents),
        token_usage=token_usage,
        estimated_cost=estimated_cost,
        answer=answer,
        citations=citations,
        claims=claims,
        claim_verification_results=claim_verification_results,
        retrieved_documents=retrieved_documents,
        relevant_documents=relevant_documents,
    )


def build_error_result(question: EvaluationQuestion) -> EvaluationResult:
    """Build the deterministic result shell used when system execution fails."""

    result = EvaluationResult.empty(
        question_id=question.id,
        question_type=question.question_type,
        question=question.question,
    )
    result.chat_history_supplied = bool(question.chat_history)
    return result


def attach_failure_analysis(
    question: EvaluationQuestion,
    result: EvaluationResult,
) -> EvaluationResult:
    """Attach deterministic failure analysis to a scored result."""

    result.failure_analysis = analyze_failure(
        question.to_compat_dict(),
        result.to_dict(),
    )
    return result


def _safe_list(value: Any, field_name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return value


def _safe_int(value: Any, field_name: str) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _has_expected_source(
    expected_sources: Any,
    citations: list[Any],
    retrieved_documents: list[Any],
    source_match_mode: str = "any",
) -> bool:
    if not expected_sources:
        return False

    evidence = citations if citations else retrieved_documents
    observed_sources = {
        document["source"]
        for document in evidence
        if isinstance(document, dict)
        and isinstance(document.get("source"), str)
    }
    expected_source_set = set(expected_sources)
    if source_match_mode == "all":
        return expected_source_set.issubset(observed_sources)
    return bool(expected_source_set.intersection(observed_sources))


def _has_expected_keywords(answer: Any, expected_keywords: list[Any]) -> bool:
    if not expected_keywords or not isinstance(answer, str):
        return False

    lower_answer = answer.lower()
    return all(str(keyword).lower() in lower_answer for keyword in expected_keywords)


def _is_correct_answer(
    answer: str,
    answer_returned: bool,
    expected_keywords: list[Any],
    gold_answer: str,
    should_answer: bool,
) -> bool:
    if not should_answer:
        return not answer_returned
    if not answer_returned:
        return False
    if expected_keywords:
        return _has_expected_keywords(answer, expected_keywords)
    if gold_answer.strip():
        return _has_gold_answer_overlap(answer, gold_answer)
    return True


def _has_gold_answer_overlap(answer: str, gold_answer: str) -> bool:
    answer_terms = _content_terms(answer)
    gold_terms = _content_terms(gold_answer)
    if not gold_terms:
        return False
    overlap = answer_terms.intersection(gold_terms)
    return len(overlap) / len(gold_terms) >= 0.5


def _content_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9_]+", text.lower())
        if len(token) > 2
    }


def _count_supported_claims(claims: list[Any]) -> int:
    return sum(
        1
        for claim in claims
        if isinstance(claim, dict)
        and (
            claim.get("supported") is True
            or claim.get("verification_label") == "supported"
        )
    )


def _count_unsupported_claims(claims: list[Any]) -> int:
    return sum(
        1
        for claim in claims
        if isinstance(claim, dict)
        and (
            claim.get("supported") is False
            or claim.get("verification_label") == "unsupported"
        )
    )


def _safe_cost(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    cost = float(value)
    if not math.isfinite(cost):
        return None
    return cost


def _is_fallback_answer(answer: str) -> bool:
    lower_answer = answer.lower()
    fallback_markers = [
        "cannot answer from the current documents",
        "cannot answer based on the current documents",
        "provided documents do not contain enough information",
        "documents do not contain enough information",
        "do not contain enough information",
        "don't have enough evidence from the current documents",
        "do not have enough evidence from the current documents",
        "i cannot answer",
        "无法可靠回答",
        "无法根据当前文档回答",
        "当前文档无法回答",
    ]
    return any(marker in lower_answer for marker in fallback_markers)


__all__ = [
    "attach_failure_analysis",
    "build_error_result",
    "score_system_output",
]
