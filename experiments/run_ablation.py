"""Run reproducible cumulative P0b ablation experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from agent.state import ChatMessage
from config import Settings, get_settings
from evaluation.evaluate import evaluate_questions, load_eval_questions
from evaluation.runtime_config import build_runtime_config_snapshot
from experiments.variants import (
    AblationVariant,
    create_variant_runner,
    load_ablation_variants,
)


DEFAULT_CONFIG_DIR = Path(__file__).with_name("configs")
EvaluationRunner = Callable[[str, list[ChatMessage]], dict[str, Any]]
VariantRunnerFactory = Callable[[AblationVariant, Settings], EvaluationRunner]
FAILURE_COUNT_COLUMNS = [
    "no_failure",
    "retrieval_failure",
    "reranking_failure",
    "query_rewrite_failure",
    "generation_failure",
    "citation_failure",
    "fallback_failure",
    "tool_failure",
]


def main(
    argv: list[str] | None = None,
    variant_runner_factory: VariantRunnerFactory | None = None,
) -> int:
    """Run configured ablation variants and write recoverable artifacts."""

    parser = argparse.ArgumentParser(description="Run P0b ablation evaluations.")
    parser.add_argument(
        "--questions",
        required=True,
        type=Path,
        help="Path to evaluation questions JSON.",
    )
    parser.add_argument(
        "--config-dir",
        default=DEFAULT_CONFIG_DIR,
        type=Path,
        help="Directory containing cumulative ablation configs.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for ablation artifacts.",
    )
    parser.add_argument(
        "--report",
        default=None,
        type=Path,
        help="Markdown report path. Defaults to output-dir/ablation_report.md.",
    )
    parser.add_argument(
        "--question-ids",
        default="",
        help="Optional comma-separated evaluation question IDs.",
    )
    args = parser.parse_args(argv)
    report_path = args.report or args.output_dir / "ablation_report.md"

    try:
        questions = load_eval_questions(args.questions)
        questions = select_questions(questions, args.question_ids)
        variants = load_ablation_variants(args.config_dir)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(f"Unable to load ablation inputs: {exc}")

    base_settings = get_settings()
    factory = variant_runner_factory or _default_variant_runner_factory
    runs = [
        _run_variant(
            variant=variant,
            questions=questions,
            questions_path=args.questions,
            output_dir=args.output_dir,
            base_settings=base_settings,
            runner_factory=factory,
        )
        for variant in variants
    ]
    payload = {
        "kind": "ablation_result",
        "phase": "P0b",
        "questions_path": str(args.questions),
        "question_count": len(questions),
        "question_ids": [question["id"] for question in questions],
        "config_dir": str(args.config_dir),
        "runs": runs,
    }
    write_json_atomic(args.output_dir / "ablation_result.json", payload)
    _write_canonical_artifacts(payload, args.output_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(format_ablation_report(payload), encoding="utf-8")
    return 0


def select_questions(
    questions: list[dict[str, Any]],
    raw_question_ids: str,
) -> list[dict[str, Any]]:
    """Filter questions by comma-separated IDs while preserving dataset order."""

    requested_ids = {
        item.strip()
        for item in raw_question_ids.split(",")
        if item.strip()
    }
    if not requested_ids:
        return questions

    known_ids = {question["id"] for question in questions}
    unknown_ids = sorted(requested_ids - known_ids)
    if unknown_ids:
        raise ValueError(f"Unknown question IDs: {', '.join(unknown_ids)}")
    return [
        question
        for question in questions
        if question["id"] in requested_ids
    ]


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a temporary file to avoid partial artifacts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def format_ablation_report(payload: dict[str, Any]) -> str:
    """Format the current ablation payload as Markdown."""

    lines = [
        "# P0b Ablation Report",
        "",
        (
            f"Questions: {payload.get('question_count', 'N/A')}. "
            "Only completed variants are used for trade-off statements."
        ),
        "",
        "| Method | Correctness | Context Relevance | Citation Accuracy | "
        "Fallback Accuracy | Unsupported Claims | Supported Claim Ratio | "
        "Avg Retry | Avg Latency | Errors | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in payload.get("runs", []):
        summary = run.get("summary", {})
        lines.append(
            (
                f"| {_display_method(run)} | "
                f"{_format_metric(summary.get('correctness_score'))} | "
                f"{_format_metric(summary.get('context_relevance_score'))} | "
                f"{_format_metric(summary.get('citation_hit_rate'))} | "
                f"{_format_metric(summary.get('fallback_accuracy'))} | "
                f"{_format_metric(summary.get('unsupported_claim_count'))} | "
                f"{_format_metric(summary.get('supported_claim_ratio'))} | "
                f"{_format_metric(summary.get('average_retry_count'))} | "
                f"{_format_metric(summary.get('average_latency'))} | "
                f"{_format_metric(summary.get('error_count'))} | "
                f"{run.get('status', 'incomplete')} |"
            )
        )

    lines.extend(_format_failed_case_analysis(payload.get("runs", [])))

    lines.extend(["", "## Observed Trade-offs", ""])
    tradeoffs = _build_observed_tradeoffs(payload.get("runs", []))
    if tradeoffs:
        lines.extend(f"- {tradeoff}" for tradeoff in tradeoffs)
    else:
        lines.append("- No adjacent completed variants are available for comparison.")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Correctness is a deterministic keyword and gold-answer overlap heuristic.",
            "- A single run does not provide confidence intervals or statistical significance.",
            "- Citation verification metrics are N/A when that capability is disabled.",
            "- Token usage and cost remain N/A unless the model client exposes reliable metadata.",
        ]
    )
    return "\n".join(lines)


def _run_variant(
    variant: AblationVariant,
    questions: list[dict[str, Any]],
    questions_path: Path,
    output_dir: Path,
    base_settings: Settings,
    runner_factory: VariantRunnerFactory,
) -> dict[str, Any]:
    variant_path = output_dir / "variants" / f"{variant.id}.json"
    resolved_settings = variant.apply_settings(base_settings)
    checkpoint = {
        "kind": "ablation_variant_result",
        "phase": "P0b",
        **variant.to_dict(),
        "status": "incomplete",
        "questions_path": str(questions_path),
        "question_count": len(questions),
        "runtime_config": build_runtime_config_snapshot(
            settings=resolved_settings,
            features=variant.features,
        ),
        "summary": {},
        "results": [],
        "error": None,
    }
    write_json_atomic(variant_path, checkpoint)

    try:
        runner = runner_factory(variant, base_settings)
        report = evaluate_questions(
            questions,
            run_agent_fn=runner,
            run_naive_fn=None,
        )
    except Exception as exc:
        checkpoint["error"] = _format_error(exc)
        write_json_atomic(variant_path, checkpoint)
        raise

    error_count = int(report["summary"].get("error_count", 0))
    completed = {
        **checkpoint,
        "status": "completed" if error_count == 0 else "completed_with_errors",
        "summary": report["summary"],
        "results": report["results"],
    }
    write_json_atomic(variant_path, completed)
    return completed


def _write_canonical_artifacts(
    payload: dict[str, Any],
    output_dir: Path,
) -> None:
    runs_by_id = {run["id"]: run for run in payload["runs"]}
    try:
        baseline_run = runs_by_id["v0_naive"]
        agentic_run = runs_by_id["v6_citation_verification"]
    except KeyError as exc:
        raise ValueError("Ablation matrix requires v0_naive and v6_citation_verification") from exc

    baseline_payload = _canonical_system_payload("naive_rag", baseline_run)
    agentic_payload = _canonical_system_payload("agentic_rag", agentic_run)
    comparison_payload = _derive_comparison_payload(
        baseline_run=baseline_run,
        agentic_run=agentic_run,
    )
    write_json_atomic(output_dir / "baseline_result.json", baseline_payload)
    write_json_atomic(output_dir / "agentic_result.json", agentic_payload)
    write_json_atomic(output_dir / "comparison_result.json", comparison_payload)


def _canonical_system_payload(
    system: str,
    run: dict[str, Any],
) -> dict[str, Any]:
    return {
        "system": system,
        "variant_id": run["id"],
        "status": run["status"],
        "runtime_config": run["runtime_config"],
        "summary": run["summary"],
        "results": run["results"],
    }


def _derive_comparison_payload(
    baseline_run: dict[str, Any],
    agentic_run: dict[str, Any],
) -> dict[str, Any]:
    baseline_by_id = {
        result["question_id"]: result
        for result in baseline_run["results"]
    }
    agentic_by_id = {
        result["question_id"]: result
        for result in agentic_run["results"]
    }
    if baseline_by_id.keys() != agentic_by_id.keys():
        raise ValueError("V0 and V6 results contain different question IDs")

    paired_results = [
        {
            "question_id": question_id,
            "question": baseline_by_id[question_id]["question"],
            "question_type": baseline_by_id[question_id]["question_type"],
            "naive": baseline_by_id[question_id],
            "agentic": agentic_by_id[question_id],
        }
        for question_id in baseline_by_id
    ]
    naive_summary = baseline_run["summary"]
    agentic_summary = agentic_run["summary"]
    return {
        "runtime_config": {
            "naive": baseline_run["runtime_config"],
            "agentic": agentic_run["runtime_config"],
        },
        "summary": {
            "mode": "comparison",
            "total_questions": len(paired_results),
            "naive": naive_summary,
            "agentic": agentic_summary,
            "comparison": _comparison_summary(naive_summary, agentic_summary),
        },
        "results": paired_results,
    }


def _comparison_summary(
    naive_summary: dict[str, Any],
    agentic_summary: dict[str, Any],
) -> dict[str, Any]:
    keys = [
        "source_hit_rate",
        "keyword_hit_rate",
        "citation_rate",
        "verification_rate",
        "fallback_correctness_rate",
        "average_latency",
    ]
    return {
        f"{system}_{key}": summary.get(key, "N/A")
        for key in keys
        for system, summary in (
            ("naive", naive_summary),
            ("agentic", agentic_summary),
        )
    }


def _default_variant_runner_factory(
    variant: AblationVariant,
    base_settings: Settings,
) -> EvaluationRunner:
    return create_variant_runner(variant, base_settings)


def _format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_failed_case_analysis(runs: list[dict[str, Any]]) -> list[str]:
    lines = [
        "",
        "## Failed Case Analysis",
        "",
        "| Method | "
        + " | ".join(FAILURE_COUNT_COLUMNS)
        + " |",
        "|---|" + "|".join("---:" for _ in FAILURE_COUNT_COLUMNS) + "|",
    ]
    for run in runs:
        counts = _failure_type_counts(run)
        cells = [
            _format_failure_count(counts.get(column, 0))
            for column in FAILURE_COUNT_COLUMNS
        ]
        lines.append(
            f"| {_escape_table_cell(_display_method(run))} | "
            + " | ".join(cells)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Representative Failed Cases",
            "",
            "| Method | Question ID | Type | Failure | Reason | Suggestion |",
            "|---|---|---|---|---|---|",
        ]
    )
    representatives = _representative_failed_cases(runs)
    if not representatives:
        lines.append("No failed cases recorded in completed runs.")
        return lines

    for item in representatives:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table_cell(item["method"]),
                    _escape_table_cell(item["question_id"]),
                    _escape_table_cell(item["question_type"]),
                    _escape_table_cell(item["failure_type"]),
                    _escape_table_cell(item["reason"]),
                    _escape_table_cell(item["suggestion"]),
                ]
            )
            + " |"
        )
    return lines


def _failure_type_counts(run: dict[str, Any]) -> dict[str, Any]:
    summary = run.get("summary")
    if not isinstance(summary, dict):
        return {}
    counts = summary.get("failure_type_counts")
    if not isinstance(counts, dict):
        return {}
    return counts


def _format_failure_count(value: Any) -> str:
    if _is_number(value):
        return str(int(value))
    return "0"


def _representative_failed_cases(runs: list[dict[str, Any]]) -> list[dict[str, str]]:
    representatives: list[dict[str, str]] = []
    for run in runs:
        results = run.get("results")
        if not isinstance(results, list):
            continue
        run_count = 0
        for result in results:
            if not isinstance(result, dict):
                continue
            analysis = result.get("failure_analysis")
            if not isinstance(analysis, dict):
                continue
            failure_type = str(analysis.get("failure_type", "")).strip()
            if not failure_type or failure_type == "no_failure":
                continue
            representatives.append(
                {
                    "method": _display_method(run),
                    "question_id": str(result.get("question_id", "")),
                    "question_type": str(result.get("question_type", "")),
                    "failure_type": failure_type,
                    "reason": str(analysis.get("reason", "")),
                    "suggestion": str(analysis.get("suggestion", "")),
                }
            )
            run_count += 1
            if run_count >= 3:
                break
    return representatives


def _escape_table_cell(value: Any) -> str:
    return " ".join(str(value).replace("|", "\\|").split())


def _display_method(run: dict[str, Any]) -> str:
    version = str(run.get("id", "")).split("_", 1)[0].upper()
    method = str(run.get("method", "")).strip()
    return f"{version} {method}".strip()


def _build_observed_tradeoffs(runs: list[dict[str, Any]]) -> list[str]:
    completed_runs = [
        run
        for run in runs
        if run.get("status") == "completed"
    ]
    tradeoffs: list[str] = []
    metrics = [
        ("correctness_score", "correctness", ""),
        ("context_relevance_score", "context relevance", ""),
        ("citation_hit_rate", "citation accuracy", ""),
        ("fallback_accuracy", "fallback accuracy", ""),
        ("average_retry_count", "average retry count", ""),
        ("average_latency", "average latency", "s"),
    ]
    for previous, current in zip(completed_runs, completed_runs[1:]):
        changes: list[str] = []
        previous_summary = previous.get("summary", {})
        current_summary = current.get("summary", {})
        for key, label, suffix in metrics:
            previous_value = previous_summary.get(key)
            current_value = current_summary.get(key)
            if not _is_number(previous_value) or not _is_number(current_value):
                continue
            delta = float(current_value) - float(previous_value)
            if abs(delta) < 0.00005:
                continue
            changes.append(f"{label} {delta:+.4f}{suffix}")
        if changes:
            tradeoffs.append(
                f"{_display_method(current)} vs {_display_method(previous)}: "
                + "; ".join(changes)
                + "."
            )
    return tradeoffs


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _format_error(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


if __name__ == "__main__":
    raise SystemExit(main())
