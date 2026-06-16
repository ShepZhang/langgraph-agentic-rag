"""Lightweight evaluation runner for Agentic RAG."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent.graph import run_agent
from agent.state import ChatMessage
from evaluation.baselines import run_naive_rag
from evaluation.comparison import (
    evaluate_comparison as evaluate_typed_comparison,
    evaluate_single_system as evaluate_typed_single_system,
)
from evaluation.dataset import (
    DEFAULT_EVAL_PATH,
    load_questions,
    normalize_question,
    normalize_questions,
)
from evaluation.metrics import summarize_results as summarize_metric_results
from evaluation.reporting import format_evaluation_report
from evaluation.runners import CallableRunnerAdapter
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
    agentic_runner = CallableRunnerAdapter(run_agent_fn)

    if run_naive_fn is not None:
        return evaluate_typed_comparison(
            questions=typed_questions,
            agentic_runner=agentic_runner,
            naive_runner=CallableRunnerAdapter(run_naive_fn),
            timer=timer,
        ).to_dict()

    return evaluate_typed_single_system(
        typed_questions,
        agentic_runner,
        timer,
    ).to_dict()


def evaluate_single_system(
    item: dict[str, Any],
    runner: Callable[[str], dict[str, Any]],
    timer: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Evaluate one raw question record with one system."""

    report = evaluate_typed_single_system(
        [normalize_question(item, 0)],
        CallableRunnerAdapter(runner),
        timer,
    )
    return report.results[0].to_dict()


def summarize_results(
    results: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize results after normalizing raw question records."""

    typed_questions = normalize_questions(questions)
    typed_results = [EvaluationResult(**result) for result in results]
    return summarize_metric_results(typed_results, typed_questions).to_dict()


def _normalize_eval_question(record: dict[str, Any], index: int) -> dict[str, Any]:
    """Compatibility facade for legacy evaluation callers."""

    return normalize_question(record, index).to_compat_dict()


def format_report(report: dict[str, Any]) -> str:
    """Format an evaluation report for terminal output."""

    return format_evaluation_report(report)


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


if __name__ == "__main__":
    raise SystemExit(main())
