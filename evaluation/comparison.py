"""Typed evaluation orchestration for single-system and comparison runs."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from evaluation.metrics import summarize_results
from evaluation.runners import EvaluationRunner, evaluate_question
from evaluation.schemas import (
    ComparisonEvaluationSummary,
    EvaluationQuestion,
    EvaluationReport,
    EvaluationSummary,
    PairedEvaluationResult,
)


def evaluate_single_system(
    questions: list[EvaluationQuestion],
    runner: EvaluationRunner,
    timer: Callable[[], float] = time.perf_counter,
) -> EvaluationReport:
    """Evaluate one system over typed questions."""

    results = [
        evaluate_question(
            question,
            runner,
            timer,
        )
        for question in questions
    ]
    summary = summarize_results(results, questions)
    return EvaluationReport(summary=summary, results=results)


def evaluate_comparison(
    questions: list[EvaluationQuestion],
    agentic_runner: EvaluationRunner,
    naive_runner: EvaluationRunner,
    timer: Callable[[], float] = time.perf_counter,
) -> EvaluationReport:
    """Evaluate naive and agentic systems over the same typed questions."""

    paired_results: list[PairedEvaluationResult] = []
    naive_results = []
    agentic_results = []

    for question in questions:
        naive_result = evaluate_question(question, naive_runner, timer)
        agentic_result = evaluate_question(question, agentic_runner, timer)
        naive_results.append(naive_result)
        agentic_results.append(agentic_result)
        paired_results.append(
            PairedEvaluationResult(
                question=question.question,
                requires_rewrite=question.requires_rewrite,
                naive=naive_result,
                agentic=agentic_result,
            )
        )

    naive_summary = summarize_results(naive_results, questions)
    agentic_summary = summarize_results(agentic_results, questions)
    summary = ComparisonEvaluationSummary(
        total_questions=len(questions),
        naive=naive_summary,
        agentic=agentic_summary,
        comparison=build_comparison_summary(
            naive=naive_summary,
            agentic=agentic_summary,
        ),
    )
    return EvaluationReport(summary=summary, results=paired_results)


def build_comparison_summary(
    naive: EvaluationSummary,
    agentic: EvaluationSummary,
) -> dict[str, Any]:
    """Flatten core comparison metrics for easy JSON/report consumption."""

    return {
        "naive_source_hit_rate": naive.source_hit_rate,
        "agentic_source_hit_rate": agentic.source_hit_rate,
        "naive_keyword_hit_rate": naive.keyword_hit_rate,
        "agentic_keyword_hit_rate": agentic.keyword_hit_rate,
        "naive_citation_rate": naive.citation_rate,
        "agentic_citation_rate": agentic.citation_rate,
        "naive_verification_rate": naive.verification_rate,
        "agentic_verification_rate": agentic.verification_rate,
        "naive_fallback_correctness_rate": naive.fallback_correctness_rate,
        "agentic_fallback_correctness_rate": agentic.fallback_correctness_rate,
        "naive_average_latency": naive.average_latency,
        "agentic_average_latency": agentic.average_latency,
    }
