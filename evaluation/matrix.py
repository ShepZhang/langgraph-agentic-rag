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
from agent.state import ChatMessage
from config import Settings, get_settings
from evaluation.baselines import run_naive_rag
from evaluation.dataset import normalize_questions
from evaluation.evaluate import (
    DEFAULT_EVAL_PATH,
    evaluate_single_system,
    load_eval_questions,
    summarize_results,
)
from evaluation.judges import Judge, build_configured_judge
from rag.embeddings import get_embedding_model
from rag.retriever import Retriever
from rag.vectorstore import VectorStoreManager

Runner = Callable[..., dict[str, Any]]


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
    ("Semantic Correctness", "average_semantic_correctness"),
    ("Groundedness", "average_groundedness"),
    ("Judge Completion Rate", "judge_completion_rate"),
    ("Average Latency", "average_latency"),
    ("Error Count", "error_count"),
)


def evaluate_matrix(
    questions: list[dict[str, Any]],
    runners: dict[str, Runner],
    timer: Callable[[], float] = time.perf_counter,
    judge: Judge | None = None,
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
    resolved_judge = judge if judge is not None else build_configured_judge()
    normalized_questions = [
        question.to_compat_dict()
        for question in normalize_questions(questions)
    ]

    for item in normalized_questions:
        systems = {}
        for name in VARIANT_ORDER:
            runner = runners[name]
            result = evaluate_single_system(
                item,
                runner,
                timer,
                judge=resolved_judge,
            )
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
            "total_questions": len(normalized_questions),
            "variants": {
                name: summarize_results(variant_results[name], normalized_questions)
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
            str(variant_value)
            if (variant_value := variants.get(name, {}).get(metric_name)) is not None
            else "N/A"
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

    def run_naive(
        question: str,
        chat_history: list[ChatMessage] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "retriever_fn": naive_retriever,
            "settings": without_reranker,
        }
        if chat_history is not None:
            kwargs["chat_history"] = chat_history
        return run_naive_rag(question, **kwargs)

    def run_agentic(
        question: str,
        chat_history: list[ChatMessage] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "retriever_fn": agentic_retriever,
            "settings": without_reranker,
        }
        if chat_history is not None:
            kwargs["chat_history"] = chat_history
        return run_agent(question, **kwargs)

    def run_agentic_reranker(
        question: str,
        chat_history: list[ChatMessage] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "retriever_fn": reranked_retriever,
            "settings": with_reranker,
        }
        if chat_history is not None:
            kwargs["chat_history"] = chat_history
        return run_agent(question, **kwargs)

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
                raw_error = str(error)
                safe_error = raw_error.split(":", maxsplit=1)[0]
                sensitive_detail = raw_error.removeprefix(safe_error).lstrip(": ")
                _redact_error_details(
                    system_result,
                    raw_error=raw_error,
                    sensitive_detail=sensitive_detail,
                    safe_error=safe_error,
                )
                system_result["error"] = safe_error
    return persisted_report


def _redact_error_details(
    value: Any,
    *,
    raw_error: str,
    sensitive_detail: str,
    safe_error: str,
) -> None:
    """Remove runtime error details from nested persisted report fields."""

    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str):
                redacted = item.replace(raw_error, safe_error)
                if sensitive_detail:
                    redacted = redacted.replace(sensitive_detail, safe_error)
                value[key] = redacted
            else:
                _redact_error_details(
                    item,
                    raw_error=raw_error,
                    sensitive_detail=sensitive_detail,
                    safe_error=safe_error,
                )
    elif isinstance(value, list):
        for item in value:
            _redact_error_details(
                item,
                raw_error=raw_error,
                sensitive_detail=sensitive_detail,
                safe_error=safe_error,
            )


if __name__ == "__main__":
    raise SystemExit(main())
