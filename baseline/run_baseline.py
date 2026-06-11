"""CLI for running the naive RAG baseline evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from agent.state import ChatMessage


EvaluationRunner = Callable[[str, list[ChatMessage]], dict[str, Any]]


def main(
    argv: list[str] | None = None,
    run_naive_fn: EvaluationRunner | None = None,
) -> int:
    """Run baseline evaluation and write a JSON artifact."""

    parser = argparse.ArgumentParser(description="Run the naive RAG baseline.")
    parser.add_argument(
        "--questions",
        default=Path("evaluation/eval_questions.json"),
        type=Path,
        help="Path to evaluation questions JSON.",
    )
    parser.add_argument(
        "--output",
        default=Path("experiments/results/baseline_result.json"),
        type=Path,
        help="Path to write baseline result JSON.",
    )
    args = parser.parse_args(argv)

    if run_naive_fn is None:
        run_naive_fn = _load_naive_rag_runner()

    evaluate_questions, load_eval_questions = _load_evaluation_tools()

    questions = load_eval_questions(args.questions)
    report = evaluate_questions(questions, run_naive_fn)
    payload = {
        "system": "naive_rag",
        "summary": report["summary"],
        "results": report["results"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote baseline results to {args.output}")
    return 0


def _load_naive_rag_runner() -> EvaluationRunner:
    from baseline.naive_rag import run_naive_rag

    return run_naive_rag


def _load_evaluation_tools() -> tuple[
    Callable[[list[dict[str, Any]], EvaluationRunner], dict[str, Any]],
    Callable[[str | Path], list[dict[str, Any]]],
]:
    from evaluation.evaluate import evaluate_questions, load_eval_questions

    return evaluate_questions, load_eval_questions


if __name__ == "__main__":
    raise SystemExit(main())
