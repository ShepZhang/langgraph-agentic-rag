"""Deterministic per-question evaluation scoring."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from evaluation.failure_analyzer import analyze_failure, summarize_failure_types
from evaluation.schemas import EvaluationQuestion, EvaluationResult, EvaluationSummary


@dataclass(frozen=True)
class SummaryMetric:
    name: str
    compute: Callable[
        [list[EvaluationResult], list[EvaluationQuestion]],
        int | float | None,
    ]


DEFAULT_SUMMARY_METRICS = (
    SummaryMetric(
        "correctness_score",
        lambda results, questions: _rate(
            sum(1 for result in results if result.correct),
            len(results),
        ),
    ),
    SummaryMetric(
        "context_relevance_score",
        lambda results, questions: _rate(
            sum(1 for result in results if result.context_relevant),
            sum(1 for question in questions if question.expected_sources),
        ),
    ),
    SummaryMetric(
        "citation_hit_rate",
        lambda results, questions: _rate(
            sum(1 for result in results if result.citation_hit),
            sum(1 for question in questions if question.expected_sources),
        ),
    ),
    SummaryMetric(
        "fallback_accuracy",
        lambda results, questions: _rate(
            sum(1 for result in results if result.fallback_correct),
            len(results),
        ),
    ),
)


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


def summarize_results(
    results: list[EvaluationResult],
    questions: list[EvaluationQuestion],
) -> EvaluationSummary:
    """Aggregate typed per-question results into the canonical summary."""

    total_questions = len(results)
    if total_questions == 0:
        return EvaluationSummary.empty()

    answer_count = sum(1 for result in results if result.answer_returned)
    fallback_count = sum(1 for result in results if result.fallback_triggered)
    citation_count = sum(1 for result in results if result.citation_returned)
    verified_count = sum(1 for result in results if result.is_verified)
    source_hit_count = sum(1 for result in results if result.source_hit)
    keyword_hit_count = sum(1 for result in results if result.keyword_hit)
    fallback_correct_count = sum(1 for result in results if result.fallback_correct)
    applicable_verification_results = [
        result for result in results if result.citation_verification_applicable
    ]
    if applicable_verification_results:
        unsupported_claim_count: int | None = sum(
            int(result.unsupported_claim_count or 0)
            for result in applicable_verification_results
        )
        supported_claim_count = sum(
            int(result.supported_claim_count or 0)
            for result in applicable_verification_results
        )
        total_claim_count = sum(
            int(result.total_claim_count or 0)
            for result in applicable_verification_results
        )
        verification_pass_count = sum(
            1
            for result in applicable_verification_results
            if result.citation_verification_passed
        )
        supported_claim_ratio: float | None = _rate(
            supported_claim_count,
            total_claim_count,
        )
        citation_verification_pass_rate: float | None = _rate(
            verification_pass_count,
            len(applicable_verification_results),
        )
    else:
        unsupported_claim_count = None
        supported_claim_ratio = None
        citation_verification_pass_rate = None
    source_expected_count = sum(1 for question in questions if question.expected_sources)
    keyword_expected_count = sum(
        1 for question in questions if question.expected_keywords
    )
    rewrite_triggered_count = sum(1 for result in results if result.rewrite_triggered)
    error_count = sum(1 for result in results if result.error)
    retrieved_doc_count = sum(result.retrieved_doc_count for result in results)
    relevant_doc_count = sum(result.relevant_doc_count for result in results)
    token_values = [
        _extract_total_tokens(result.token_usage) or 0
        for result in results
    ]
    estimated_cost = sum(
        _safe_cost(result.estimated_cost) or 0.0
        for result in results
    )
    registered = {
        metric.name: metric.compute(results, questions)
        for metric in DEFAULT_SUMMARY_METRICS
    }

    # --- Semantic judge aggregation ---
    judge_completed = [
        r for r in results
        if r.judge.status == "completed"
    ]
    judge_failed = [
        r for r in results
        if r.judge.status == "failed"
    ]
    judge_attempted = len(judge_completed) + len(judge_failed)

    if judge_attempted > 0:
        judge_completion_rate: float | None = round(
            len(judge_completed) / judge_attempted, 4
        )
        semantic_scores = _judge_scores(judge_completed, "semantic_correctness")
        grounded_scores = _judge_scores(judge_completed, "groundedness")
        grounded_applicable = len(grounded_scores)

        average_semantic: float | None = (
            round(sum(semantic_scores) / len(semantic_scores), 4)
            if semantic_scores
            else None
        )
        average_grounded: float | None = (
            round(sum(grounded_scores) / len(grounded_scores), 4)
            if grounded_scores
            else None
        )
    else:
        judge_completion_rate = None
        average_semantic = None
        average_grounded = None
        grounded_applicable = 0

    return EvaluationSummary(
        total_questions=total_questions,
        answer_rate=_rate(answer_count, total_questions),
        fallback_rate=_rate(fallback_count, total_questions),
        citation_rate=_rate(citation_count, total_questions),
        verification_rate=_rate(verified_count, total_questions),
        average_claim_count=_average(result.claim_count for result in results),
        correctness_score=registered["correctness_score"] or 0,
        context_relevance_score=registered["context_relevance_score"] or 0,
        citation_hit_rate=registered["citation_hit_rate"] or 0,
        fallback_accuracy=registered["fallback_accuracy"] or 0,
        unsupported_claim_count=unsupported_claim_count,
        supported_claim_ratio=supported_claim_ratio,
        citation_verification_pass_rate=citation_verification_pass_rate,
        average_token_usage=_average(token_values),
        estimated_cost=round(estimated_cost, 6),
        source_hit_rate=_rate(source_hit_count, source_expected_count),
        keyword_hit_rate=_rate(keyword_hit_count, keyword_expected_count),
        fallback_correctness_rate=_rate(fallback_correct_count, total_questions),
        average_retry_count=_average(result.retry_count for result in results),
        average_retrieved_docs=_average(
            result.retrieved_doc_count for result in results
        ),
        average_relevant_docs=_average(
            result.relevant_doc_count for result in results
        ),
        relevant_filtering_rate=_rate(
            retrieved_doc_count - relevant_doc_count,
            retrieved_doc_count,
        ),
        average_latency=_average(result.latency for result in results),
        rewrite_triggered_count=rewrite_triggered_count,
        error_count=error_count,
        judge_completed_count=len(judge_completed),
        judge_failed_count=len(judge_failed),
        judge_completion_rate=judge_completion_rate,
        average_semantic_correctness=average_semantic,
        average_groundedness=average_grounded,
        groundedness_applicable_count=grounded_applicable,
        failure_type_counts=summarize_failure_types(
            [result.to_dict() for result in results]
        ),
    )


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


def _safe_float(value: Any) -> float | None:
    """Return value as a finite float, or None if bool/string/NaN/Inf/None."""
    if isinstance(value, bool):
        return None
    if not isinstance(value, int | float):
        return None
    try:
        float_value = float(value)
    except OverflowError:
        return None
    if not math.isfinite(float_value):
        return None
    return float_value


def _safe_cost(value: Any) -> float | None:
    """Return estimated cost as finite float or None. Semantic alias for _safe_float."""
    return _safe_float(value)


def _extract_total_tokens(token_usage: Any) -> int | None:
    if not isinstance(token_usage, dict):
        return None
    value = token_usage.get("total_tokens")
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        return int(value)
    return None


def _judge_scores(
    results: list[EvaluationResult], dimension: str
) -> list[float]:
    """Return valid finite scores for a judge dimension from completed results."""
    scores: list[float] = []
    for r in results:
        score = _safe_float(r.judge.scores.get(dimension))
        if score is not None and 0.0 <= score <= 1.0:
            scores.append(score)
    return scores


def _rate(count: int, denominator: int) -> float:
    if denominator == 0:
        return 0
    return round(count / denominator, 4)


def _average(values: Any) -> float:
    values_list = list(values)
    if not values_list:
        return 0
    return round(sum(values_list) / len(values_list), 4)


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
    "DEFAULT_SUMMARY_METRICS",
    "SummaryMetric",
    "attach_failure_analysis",
    "build_error_result",
    "score_system_output",
    "summarize_results",
]
