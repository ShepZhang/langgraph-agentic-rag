"""Lightweight evaluation runner for Agentic RAG."""

from __future__ import annotations

import argparse
import inspect
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent.graph import run_agent
from agent.state import ChatMessage
from evaluation.baselines import run_naive_rag
from evaluation.dataset import (
    DEFAULT_EVAL_PATH,
    load_questions,
    normalize_question,
    normalize_questions,
)
from evaluation.failure_analyzer import analyze_failure
from evaluation.metrics import (
    build_error_result,
    score_system_output,
    summarize_results as summarize_metric_results,
)
from evaluation.runtime_config import build_runtime_config_snapshot
from evaluation.schemas import EvaluationResult


EvaluationRunner = Callable[[str, list[ChatMessage]], dict[str, Any]]


def load_eval_questions(path: str | Path = DEFAULT_EVAL_PATH) -> list[dict[str, Any]]:
    """Load and validate evaluation questions."""

    return [question.to_compat_dict() for question in load_questions(path)]


def evaluate_questions(
    questions: list[dict[str, Any]],
    run_agent_fn: Callable[[str], dict[str, Any]] = run_agent,
    run_naive_fn: Callable[[str], dict[str, Any]] | None = None,
    timer: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Evaluate questions and return per-question results plus summary metrics."""

    typed_questions = normalize_questions(questions)
    normalized_questions = [question.to_compat_dict() for question in typed_questions]

    if run_naive_fn is not None:
        return _evaluate_comparison(
            questions=normalized_questions,
            run_agent_fn=run_agent_fn,
            run_naive_fn=run_naive_fn,
            timer=timer,
        )

    results: list[dict[str, Any]] = []
    for item in normalized_questions:
        results.append(_evaluate_single_system(item, run_agent_fn, timer))

    return {"summary": _summarize(results, normalized_questions), "results": results}


def evaluate_single_system(
    item: dict[str, Any],
    runner: Callable[[str], dict[str, Any]],
    timer: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Evaluate one raw question record with one system."""

    normalized_item = normalize_question(item, 0).to_compat_dict()
    return _evaluate_single_system(normalized_item, runner, timer)


def summarize_results(
    results: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize results after normalizing raw question records."""

    typed_questions = normalize_questions(questions)
    normalized_questions = [question.to_compat_dict() for question in typed_questions]
    return _summarize(results, normalized_questions)


def _normalize_eval_question(record: dict[str, Any], index: int) -> dict[str, Any]:
    """Compatibility facade for legacy evaluation callers."""

    return normalize_question(record, index).to_compat_dict()


def format_report(report: dict[str, Any]) -> str:
    """Format an evaluation report for terminal output."""

    summary = report.get("summary", {})
    if summary.get("mode") == "comparison":
        return _format_comparison_report(report)

    lines = ["Evaluation Report", "", "Summary"]
    for key, value in summary.items():
        lines.append(f"{key}: {value}")

    lines.extend(["", "Questions"])
    for index, result in enumerate(report.get("results", []), start=1):
        lines.append(
            (
                f"{index}. {result.get('question', '')} | "
                f"answer={_format_bool(result.get('answer_returned'))} | "
                f"fallback={_format_bool(result.get('fallback_triggered'))} | "
                f"citation={_format_bool(result.get('citation_returned'))} | "
                f"source_hit={_format_bool(result.get('source_hit'))} | "
                f"keyword_hit={_format_bool(result.get('keyword_hit'))} | "
                f"rewrite={_format_bool(result.get('rewrite_triggered'))} | "
                f"retry_count={result.get('retry_count', 0)} | "
                f"retrieved={result.get('retrieved_doc_count', 0)} | "
                f"relevant={result.get('relevant_doc_count', 0)} | "
                f"latency={float(result.get('latency', 0)):.4f}s | "
                f"error={result.get('error') or ''}"
            )
        )

    return "\n".join(lines)


def main(
    argv: list[str] | None = None,
    run_agent_fn: Callable[[str], dict[str, Any]] = run_agent,
    run_naive_fn: Callable[[str], dict[str, Any]] | None = run_naive_rag,
) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Run Agentic RAG evaluations.")
    parser.add_argument(
        "--questions",
        default=DEFAULT_EVAL_PATH,
        type=Path,
        help="Path to evaluation questions JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Directory for JSON evaluation artifacts.",
    )
    args = parser.parse_args(argv)

    try:
        questions = load_eval_questions(args.questions)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(f"Unable to load evaluation questions: {exc}")

    report = evaluate_questions(
        questions,
        run_agent_fn=run_agent_fn,
        run_naive_fn=run_naive_fn,
    )
    if args.output_dir is not None:
        write_evaluation_artifacts(report, args.output_dir)
    print(format_report(report))
    return 0


def write_evaluation_artifacts(report: dict[str, Any], output_dir: str | Path) -> None:
    """Write structured JSON artifacts for downstream evaluation analysis."""

    artifact_dir = Path(output_dir)
    summary = report.get("summary", {})
    runtime_config = build_runtime_config_snapshot()
    if summary.get("mode") == "comparison":
        paired_results = report.get("results", [])
        _write_json(
            artifact_dir / "baseline_result.json",
            {
                "system": "naive_rag",
                "runtime_config": runtime_config,
                "summary": summary.get("naive", {}),
                "results": [paired.get("naive", {}) for paired in paired_results],
            },
        )
        _write_json(
            artifact_dir / "agentic_result.json",
            {
                "system": "agentic_rag",
                "runtime_config": runtime_config,
                "summary": summary.get("agentic", {}),
                "results": [paired.get("agentic", {}) for paired in paired_results],
            },
        )
        _write_json(
            artifact_dir / "comparison_result.json",
            {"runtime_config": runtime_config, **report},
        )
        return

    _write_json(
        artifact_dir / "agentic_result.json",
        {
            "system": "agentic_rag",
            "runtime_config": runtime_config,
            "summary": report.get("summary", {}),
            "results": report.get("results", []),
        },
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _evaluate_comparison(
    questions: list[dict[str, Any]],
    run_agent_fn: Callable[[str], dict[str, Any]],
    run_naive_fn: Callable[[str], dict[str, Any]],
    timer: Callable[[], float],
) -> dict[str, Any]:
    """Evaluate naive and agentic RAG on the same questions."""

    paired_results: list[dict[str, Any]] = []
    naive_results: list[dict[str, Any]] = []
    agentic_results: list[dict[str, Any]] = []

    for item in questions:
        naive_result = _evaluate_single_system(item, run_naive_fn, timer)
        agentic_result = _evaluate_single_system(item, run_agent_fn, timer)
        naive_results.append(naive_result)
        agentic_results.append(agentic_result)
        paired_results.append(
            {
                "question": item["question"],
                "requires_rewrite": item.get("requires_rewrite", False),
                "naive": naive_result,
                "agentic": agentic_result,
            }
        )

    naive_summary = _summarize(naive_results, questions)
    agentic_summary = _summarize(agentic_results, questions)
    return {
        "summary": {
            "mode": "comparison",
            "total_questions": len(questions),
            "naive": naive_summary,
            "agentic": agentic_summary,
            "comparison": _build_comparison_summary(
                naive_summary=naive_summary,
                agentic_summary=agentic_summary,
            ),
        },
        "results": paired_results,
    }


def _evaluate_single_system(
    item: dict[str, Any],
    runner: Callable[..., dict[str, Any]],
    timer: Callable[[], float],
) -> dict[str, Any]:
    """Evaluate one system for one question and record errors as data."""

    question = item["question"]
    started_at = timer()
    try:
        system_result = _invoke_evaluation_runner(
            runner,
            question,
            item.get("chat_history", []),
        )
        result = _build_success_result(item, system_result)
        error = None
    except Exception as exc:  # noqa: BLE001 - evaluation records system failures.
        result = _build_error_result(item)
        error = _format_error(exc)
    latency = timer() - started_at

    result["latency"] = latency
    result["error"] = error
    result["failure_analysis"] = analyze_failure(item, result)
    return result


def _invoke_evaluation_runner(
    runner: Callable[..., dict[str, Any]],
    question: str,
    chat_history: list[ChatMessage],
) -> dict[str, Any]:
    """Invoke the history-aware runner while preserving injected legacy runners."""

    try:
        parameters = inspect.signature(runner).parameters.values()
    except (TypeError, ValueError):
        return runner(question, chat_history)

    positional_parameters = [
        parameter
        for parameter in parameters
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    ]
    accepts_varargs = any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in parameters
    )
    if accepts_varargs or len(positional_parameters) >= 2:
        return runner(question, chat_history)
    return runner(question)


def _build_success_result(
    item: dict[str, Any],
    agent_result: dict[str, Any],
) -> dict[str, Any]:
    question = normalize_question(item, 0)
    return score_system_output(question, agent_result).to_dict()


def _build_error_result(item: dict[str, Any]) -> dict[str, Any]:
    question = normalize_question(item, 0)
    return build_error_result(question).to_dict()


def _summarize(
    results: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    typed_questions = normalize_questions(questions)
    typed_results = [EvaluationResult(**result) for result in results]
    return summarize_metric_results(typed_results, typed_questions).to_dict()


def _build_comparison_summary(
    naive_summary: dict[str, Any],
    agentic_summary: dict[str, Any],
) -> dict[str, Any]:
    """Flatten core comparison metrics for easy JSON/report consumption."""

    return {
        "naive_source_hit_rate": naive_summary.get("source_hit_rate", "N/A"),
        "agentic_source_hit_rate": agentic_summary.get("source_hit_rate", "N/A"),
        "naive_keyword_hit_rate": naive_summary.get("keyword_hit_rate", "N/A"),
        "agentic_keyword_hit_rate": agentic_summary.get("keyword_hit_rate", "N/A"),
        "naive_citation_rate": naive_summary.get("citation_rate", "N/A"),
        "agentic_citation_rate": agentic_summary.get("citation_rate", "N/A"),
        "naive_verification_rate": naive_summary.get("verification_rate", "N/A"),
        "agentic_verification_rate": agentic_summary.get("verification_rate", "N/A"),
        "naive_fallback_correctness_rate": naive_summary.get(
            "fallback_correctness_rate",
            "N/A",
        ),
        "agentic_fallback_correctness_rate": agentic_summary.get(
            "fallback_correctness_rate",
            "N/A",
        ),
        "naive_average_latency": naive_summary.get("average_latency", "N/A"),
        "agentic_average_latency": agentic_summary.get("average_latency", "N/A"),
    }


def _format_comparison_report(report: dict[str, Any]) -> str:
    """Format a naive-vs-agentic report as readable markdown."""

    summary = report.get("summary", {})
    naive = summary.get("naive", {})
    agentic = summary.get("agentic", {})

    lines = [
        "Evaluation Report",
        "",
        "Comparison Summary",
        "",
        "| Metric | Naive RAG | Agentic RAG |",
        "|---|---:|---:|",
        (
            f"| Source Hit Rate | {naive.get('source_hit_rate', 'N/A')} | "
            f"{agentic.get('source_hit_rate', 'N/A')} |"
        ),
        (
            f"| Keyword Hit Rate | {naive.get('keyword_hit_rate', 'N/A')} | "
            f"{agentic.get('keyword_hit_rate', 'N/A')} |"
        ),
        (
            f"| Citation Rate | {naive.get('citation_rate', 'N/A')} | "
            f"{agentic.get('citation_rate', 'N/A')} |"
        ),
        (
            f"| Claim Verification Rate | {naive.get('verification_rate', 'N/A')} | "
            f"{agentic.get('verification_rate', 'N/A')} |"
        ),
        (
            f"| Fallback Correctness | "
            f"{naive.get('fallback_correctness_rate', 'N/A')} | "
            f"{agentic.get('fallback_correctness_rate', 'N/A')} |"
        ),
        (
            f"| Avg Latency | {naive.get('average_latency', 'N/A')} | "
            f"{agentic.get('average_latency', 'N/A')} |"
        ),
        "",
        "Agentic-specific Metrics",
        f"average_retry_count: {agentic.get('average_retry_count', 'N/A')}",
        f"rewrite_triggered_count: {agentic.get('rewrite_triggered_count', 'N/A')}",
        f"average_retrieved_docs: {agentic.get('average_retrieved_docs', 'N/A')}",
        f"average_relevant_docs: {agentic.get('average_relevant_docs', 'N/A')}",
        f"relevant_filtering_rate: {agentic.get('relevant_filtering_rate', 'N/A')}",
        f"verification_rate: {agentic.get('verification_rate', 'N/A')}",
        f"average_claim_count: {agentic.get('average_claim_count', 'N/A')}",
        "",
        "Questions",
    ]

    for index, result in enumerate(report.get("results", []), start=1):
        naive_result = result.get("naive", {})
        agentic_result = result.get("agentic", {})
        lines.append(
            (
                f"{index}. {result.get('question', '')} | "
                f"naive_answer={_format_bool(naive_result.get('answer_returned'))} | "
                f"agentic_answer={_format_bool(agentic_result.get('answer_returned'))} | "
                f"naive_source_hit={_format_bool(naive_result.get('source_hit'))} | "
                f"agentic_source_hit={_format_bool(agentic_result.get('source_hit'))} | "
                f"retry_count={agentic_result.get('retry_count', 0)} | "
                f"retrieved={agentic_result.get('retrieved_doc_count', 0)} | "
                f"relevant={agentic_result.get('relevant_doc_count', 0)} | "
                f"error={naive_result.get('error') or agentic_result.get('error') or ''}"
            )
        )

    return "\n".join(lines)


def _format_error(exc: Exception) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def _format_bool(value: Any) -> str:
    return "true" if value else "false"


if __name__ == "__main__":
    raise SystemExit(main())
