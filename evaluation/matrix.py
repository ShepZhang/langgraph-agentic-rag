"""Three-variant evaluation matrix for benchmark reporting."""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from agent.graph import run_agent
from config import Settings, get_settings
from evaluation.baselines import run_naive_rag
from evaluation.evaluate import (
    DEFAULT_EVAL_PATH,
    evaluate_single_system,
    load_eval_questions,
    summarize_results,
)
from rag.embeddings import get_embedding_model
from rag.retriever import Retriever
from rag.vectorstore import VectorStoreManager

Runner = Callable[[str], dict[str, Any]]


class BenchmarkConfigurationError(RuntimeError):
    """Raised when a benchmark variant cannot use its configured components."""


VARIANT_LABELS = {
    "naive": "Naive RAG",
    "agentic": "Agentic RAG",
    "agentic_reranker": "Agentic + Reranker",
}
VARIANT_ORDER = tuple(VARIANT_LABELS)

MATRIX_METRICS = (
    ("Retrieval Source Hit Rate", "source_hit_rate"),
    ("Keyword Hit Rate", "keyword_hit_rate"),
    ("Citation Rate", "citation_rate"),
    ("Claim Verification Rate", "verification_rate"),
    ("Fallback Correctness", "fallback_correctness_rate"),
    ("Average Retry Count", "average_retry_count"),
    ("Average Retrieved Docs", "average_retrieved_docs"),
    ("Average Relevant Docs", "average_relevant_docs"),
    ("Average Latency", "average_latency"),
    ("Error Count", "error_count"),
)


def evaluate_matrix(
    questions: list[dict[str, Any]],
    runners: dict[str, Runner],
    timer: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Evaluate every question against the required benchmark variants."""

    expected_variants = set(VARIANT_ORDER)
    received_variants = set(runners)
    if received_variants != expected_variants:
        missing = sorted(expected_variants - received_variants)
        extra = sorted(received_variants - expected_variants)
        raise ValueError(
            "runners must contain exactly naive, agentic, and agentic_reranker; "
            f"missing={missing}, extra={extra}"
        )

    variant_results: dict[str, list[dict[str, Any]]] = {
        name: [] for name in VARIANT_ORDER
    }
    matrix_results = []

    for item in questions:
        systems = {}
        for name in VARIANT_ORDER:
            runner = runners[name]
            result = evaluate_single_system(item, runner, timer)
            variant_results[name].append(result)
            systems[name] = result
        matrix_results.append(
            {
                "question": item["question"],
                "requires_rewrite": item.get("requires_rewrite", False),
                "systems": systems,
            }
        )

    return {
        "summary": {
            "mode": "matrix",
            "total_questions": len(questions),
            "variants": {
                name: summarize_results(variant_results[name], questions)
                for name in VARIANT_ORDER
            },
        },
        "results": matrix_results,
    }


def format_matrix_report(report: dict[str, Any]) -> str:
    """Format matrix summary metrics as a Markdown table."""

    variants = report.get("summary", {}).get("variants", {})
    lines = [
        "| Metric | Naive RAG | Agentic RAG | Agentic + Reranker |",
        "|---|---:|---:|---:|",
    ]
    for label, metric_name in MATRIX_METRICS:
        values = [
            str(variants.get(name, {}).get(metric_name, "N/A"))
            for name in VARIANT_ORDER
        ]
        lines.append(f"| {label} | {' | '.join(values)} |")
    return "\n".join(lines)


def build_benchmark_runners(
    settings: Settings | None = None,
) -> dict[str, Runner]:
    """Build isolated runners with reranking disabled or enabled as required."""

    base = settings or get_settings()
    base.require_llm_config()

    without_reranker = replace(base, reranker_enabled=False)
    with_reranker = replace(base, reranker_enabled=True)
    embedding_model = get_embedding_model(without_reranker)
    naive_manager = VectorStoreManager(
        settings=without_reranker,
        embedding_model=embedding_model,
    )
    agentic_manager = VectorStoreManager(
        settings=without_reranker,
        embedding_model=embedding_model,
    )
    reranked_manager = VectorStoreManager(
        settings=with_reranker,
        embedding_model=embedding_model,
    )
    naive_retriever = Retriever(
        vectorstore_manager=naive_manager,
        settings=without_reranker,
    ).retrieve
    agentic_retriever = Retriever(
        vectorstore_manager=agentic_manager,
        settings=without_reranker,
    ).retrieve
    try:
        reranked_retriever = Retriever(
            vectorstore_manager=reranked_manager,
            settings=with_reranker,
        ).retrieve
    except Exception as exc:  # noqa: BLE001 - hide backend details at this boundary.
        raise BenchmarkConfigurationError(
            f"Unable to initialize reranker model {with_reranker.reranker_model!r}."
        ) from exc

    def run_naive(question: str) -> dict[str, Any]:
        return run_naive_rag(
            question,
            retriever_fn=naive_retriever,
            settings=without_reranker,
        )

    def run_agentic(question: str) -> dict[str, Any]:
        return run_agent(
            question,
            retriever_fn=agentic_retriever,
            settings=without_reranker,
        )

    def run_agentic_reranker(question: str) -> dict[str, Any]:
        return run_agent(
            question,
            retriever_fn=reranked_retriever,
            settings=with_reranker,
        )

    return {
        "naive": run_naive,
        "agentic": run_agentic,
        "agentic_reranker": run_agentic_reranker,
    }


def main(
    argv: list[str] | None = None,
    runner_builder: Callable[[], dict[str, Runner]] | None = None,
) -> int:
    """CLI entrypoint for the three-variant evaluation matrix."""

    parser = argparse.ArgumentParser(
        description="Compare naive, agentic, and reranked Agentic RAG."
    )
    parser.add_argument(
        "--questions",
        default=DEFAULT_EVAL_PATH,
        type=Path,
        help="Path to evaluation questions JSON.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for the full JSON report.",
    )
    args = parser.parse_args(argv)

    try:
        questions = load_eval_questions(args.questions)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        parser.error(f"Unable to load evaluation questions: {exc}")

    try:
        runners = (runner_builder or build_benchmark_runners)()
    except BenchmarkConfigurationError as exc:
        parser.error(str(exc))
    except Exception:  # noqa: BLE001 - CLI must not expose construction details.
        parser.error(
            "Unable to build benchmark runners. "
            "Check LLM and reranker configuration."
        )

    report = evaluate_matrix(questions, runners)

    if args.json_output is not None:
        try:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(
                    _build_persistable_report(report),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as exc:
            parser.error(f"Unable to write JSON output: {exc}")

    print(format_matrix_report(report))
    return 0


def _build_persistable_report(report: dict[str, Any]) -> dict[str, Any]:
    persisted_report = copy.deepcopy(report)
    for result in persisted_report.get("results", []):
        for system_result in result.get("systems", {}).values():
            error = system_result.get("error")
            if error:
                system_result["error"] = str(error).split(":", maxsplit=1)[0]
    return persisted_report


if __name__ == "__main__":
    raise SystemExit(main())
