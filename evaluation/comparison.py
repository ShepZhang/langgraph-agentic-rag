"""Typed evaluation orchestration for single-system and comparison runs."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from evaluation.judges import DisabledJudge, Judge, invoke_judge
from evaluation.metrics import attach_failure_analysis, summarize_results
from evaluation.runners import EvaluationRunner, evaluate_question
from evaluation.schemas import (
    ComparisonEvaluationSummary,
    EvaluationQuestion,
    EvaluationReport,
    EvaluationSummary,
    EvaluationResult,
    JudgeResult,
    PairedEvaluationResult,
)

SYSTEM_RESULT_UNAVAILABLE_ERROR = (
    "SystemResultUnavailable: system execution failed; Judge was not invoked"
)


def evaluate_single_system(
    questions: list[EvaluationQuestion],
    runner: EvaluationRunner,
    timer: Callable[[], float] = time.perf_counter,
    judge: Judge | None = None,
) -> EvaluationReport:
    """Evaluate one system over typed questions."""

    resolved_judge = judge if judge is not None else DisabledJudge()
    results = [
        _finalize_result(
            question,
            evaluate_question(
                question,
                runner,
                timer,
            ),
            resolved_judge,
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
    judge: Judge | None = None,
) -> EvaluationReport:
    """Evaluate naive and agentic systems over the same typed questions."""

    resolved_judge = judge if judge is not None else DisabledJudge()
    paired_results: list[PairedEvaluationResult] = []
    naive_results = []
    agentic_results = []

    for question in questions:
        naive_result = _finalize_result(
            question,
            evaluate_question(question, naive_runner, timer),
            resolved_judge,
        )
        agentic_result = _finalize_result(
            question,
            evaluate_question(question, agentic_runner, timer),
            resolved_judge,
        )
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


def _finalize_result(
    question: EvaluationQuestion,
    result: EvaluationResult,
    judge: Judge,
) -> EvaluationResult:
    if result.error:
        result.judge = JudgeResult.failed(SYSTEM_RESULT_UNAVAILABLE_ERROR)
        return attach_failure_analysis(question, result)

    result.judge = invoke_judge(judge, question, result)
    return attach_failure_analysis(question, result)


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
        "naive_average_semantic_correctness": naive.average_semantic_correctness,
        "agentic_average_semantic_correctness": agentic.average_semantic_correctness,
        "naive_average_groundedness": naive.average_groundedness,
        "agentic_average_groundedness": agentic.average_groundedness,
        "naive_judge_completion_rate": naive.judge_completion_rate,
        "agentic_judge_completion_rate": agentic.judge_completion_rate,
    }
