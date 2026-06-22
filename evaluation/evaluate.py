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
from evaluation.judges import Judge, build_configured_judge, describe_judge_runtime
from evaluation.metrics import summarize_results as summarize_metric_results
from evaluation.reporting import format_evaluation_report
from evaluation.runners import CallableRunnerAdapter
from evaluation.runtime_config import build_runtime_config_snapshot
from evaluation.schemas import EvaluationResult
from evaluation.storage import write_compatibility_artifacts


EvaluationRunner = Callable[[str, list[ChatMessage]], dict[str, Any]]


def load_eval_questions(path: str | Path = DEFAULT_EVAL_PATH) -> list[dict[str, Any]]:
    """Load and validate evaluation questions."""

    return [question.to_compat_dict() for question in load_questions(path)]


def evaluate_questions(
    questions: list[dict[str, Any]],
    run_agent_fn: Callable[[str], dict[str, Any]] = run_agent,
    run_naive_fn: Callable[[str], dict[str, Any]] | None = None,
    timer: Callable[[], float] = time.perf_counter,
    judge: Judge | None = None,
) -> dict[str, Any]:
    """Evaluate questions and return per-question results plus summary metrics."""

    typed_questions = normalize_questions(questions)
    agentic_runner = CallableRunnerAdapter(run_agent_fn)
    resolved_judge = judge if judge is not None else build_configured_judge()

    if run_naive_fn is not None:
        typed_report = evaluate_typed_comparison(
            questions=typed_questions,
            agentic_runner=agentic_runner,
            naive_runner=CallableRunnerAdapter(run_naive_fn),
            timer=timer,
            judge=resolved_judge,
        )
    else:
        typed_report = evaluate_typed_single_system(
            typed_questions,
            agentic_runner,
            timer,
            judge=resolved_judge,
        )

    report = typed_report.to_dict()
    runtime_config = build_runtime_config_snapshot(
        judge_metadata=describe_judge_runtime(
            resolved_judge,
            result_model=_find_judge_model(report),
        )
    )
    report["runtime_config"] = runtime_config
    return report


def evaluate_single_system(
    item: dict[str, Any],
    runner: Callable[[str], dict[str, Any]],
    timer: Callable[[], float] = time.perf_counter,
    judge: Judge | None = None,
) -> dict[str, Any]:
    """Evaluate one raw question record with one system."""

    resolved_judge = judge if judge is not None else build_configured_judge()
    report = evaluate_typed_single_system(
        [normalize_question(item, 0)],
        CallableRunnerAdapter(runner),
        timer,
        judge=resolved_judge,
    )
    return report.results[0].to_dict()


def summarize_results(
    results: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize results after normalizing raw question records."""

    typed_questions = normalize_questions(questions)
    typed_results = [EvaluationResult.from_compat_dict(result) for result in results]
    return summarize_metric_results(typed_results, typed_questions).to_dict()


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

    runtime_config = report.get("runtime_config")
    if not isinstance(runtime_config, dict):
        runtime_config = build_runtime_config_snapshot()
    write_compatibility_artifacts(
        report,
        output_dir,
        runtime_config=runtime_config,
    )


def _find_judge_model(value: Any) -> str | None:
    if isinstance(value, dict):
        judge = value.get("judge")
        if isinstance(judge, dict):
            model = judge.get("model")
            if isinstance(model, str) and model.strip():
                return model.strip()
        for nested in value.values():
            model = _find_judge_model(nested)
            if model is not None:
                return model
    elif isinstance(value, list):
        for nested in value:
            model = _find_judge_model(nested)
            if model is not None:
                return model
    return None


if __name__ == "__main__":
    raise SystemExit(main())
