"""Lightweight ablation runner for P0a evaluation infrastructure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from agent.graph import run_agent
from evaluation.baselines import run_naive_rag
from evaluation.evaluate import evaluate_questions, load_eval_questions


DEFAULT_CONFIG_DIR = Path(__file__).with_name("configs")


def load_ablation_configs(
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
) -> list[dict[str, str]]:
    """Load simple key-value YAML ablation configs in filename order."""

    configs: list[dict[str, str]] = []
    for path in sorted(Path(config_dir).glob("*.yaml")):
        config = {
            "id": path.stem,
            "method": path.stem,
            "runner": "pending",
            "status": "pending",
        }
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(f"{path}:{line_number} missing ':' in config line")
            key, value = line.split(":", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"{path}:{line_number} missing config key")
            config[key] = value.strip()
        configs.append(config)

    return configs


def main(
    argv: list[str] | None = None,
    run_naive_fn: Callable[[str], dict[str, Any]] = run_naive_rag,
    run_agent_fn: Callable[[str], dict[str, Any]] = run_agent,
) -> int:
    """Run configured ablation variants and write JSON/Markdown artifacts."""

    parser = argparse.ArgumentParser(description="Run ablation evaluations.")
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
        help="Directory containing simple YAML ablation configs.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for ablation_result.json.",
    )
    parser.add_argument(
        "--report",
        default=None,
        type=Path,
        help="Path for Markdown ablation report. Defaults to output-dir/ablation_report.md.",
    )
    args = parser.parse_args(argv)
    report_path = args.report or args.output_dir / "ablation_report.md"

    try:
        questions = load_eval_questions(args.questions)
        configs = load_ablation_configs(args.config_dir)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(f"Unable to load ablation inputs: {exc}")

    payload = {
        "kind": "ablation_result",
        "phase": "P0a",
        "note": (
            "P0a provides evaluation infrastructure and reproducible artifacts. "
            "Final ablation numbers should be refreshed after P0b/P1/P2."
        ),
        "questions_path": str(args.questions),
        "config_dir": str(args.config_dir),
        "runs": [
            _run_config(config, questions, run_naive_fn, run_agent_fn)
            for config in configs
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ablation_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(format_ablation_report(payload), encoding="utf-8")
    return 0


def format_ablation_report(payload: dict[str, Any]) -> str:
    """Format ablation results as a Markdown report."""

    lines = [
        "# Ablation Report",
        "",
        (
            "P0a is evaluation infrastructure for reproducible artifacts. "
            "Final numbers should be refreshed after P0b/P1/P2 implement and "
            "stabilize the underlying algorithm toggles."
        ),
        "",
        "| Method | Correctness | Context Relevance | Citation Accuracy | "
        "Fallback Accuracy | Unsupported Claims | Avg Latency | Scope | "
        "Independent | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]

    for run in payload.get("runs", []):
        summary = run.get("summary", {})
        lines.append(
            (
                f"| {run.get('method', '')} | "
                f"{_format_metric(summary.get('correctness_score'))} | "
                f"{_format_metric(summary.get('context_relevance_score'))} | "
                f"{_format_metric(summary.get('citation_hit_rate'))} | "
                f"{_format_metric(summary.get('fallback_accuracy'))} | "
                f"{_format_metric(summary.get('unsupported_claim_count'))} | "
                f"{_format_metric(summary.get('average_latency'))} | "
                f"{run.get('runner_scope', '')} | "
                f"{run.get('independent_ablation', '')} | "
                f"{run.get('status', 'pending')} |"
            )
        )

    run_notes = [
        run
        for run in payload.get("runs", [])
        if run.get("notes") or run.get("runner_scope") or run.get("independent_ablation")
    ]
    if run_notes:
        lines.extend(["", "## Run Notes", ""])
        for run in run_notes:
            notes = run.get("notes", "")
            lines.append(
                (
                    f"- {run.get('method', run.get('id'))}: "
                    f"scope={run.get('runner_scope', '')}; "
                    f"independent_ablation={run.get('independent_ablation', '')}; "
                    f"{notes}"
                ).rstrip()
            )

    return "\n".join(lines)


def _run_config(
    config: dict[str, str],
    questions: list[dict[str, Any]],
    run_naive_fn: Callable[[str], dict[str, Any]],
    run_agent_fn: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    run = dict(config)
    if config.get("status") == "supported" and config.get("runner") == "naive":
        report = evaluate_questions(
            questions,
            run_agent_fn=run_naive_fn,
            run_naive_fn=None,
        )
        run.update(
            {
                "status": "completed",
                "summary": report["summary"],
                "results": report["results"],
            }
        )
        return run

    if config.get("status") == "supported" and config.get("runner") == "agentic":
        report = evaluate_questions(
            questions,
            run_agent_fn=run_agent_fn,
            run_naive_fn=None,
        )
        run.update(
            {
                "status": "completed",
                "summary": report["summary"],
                "results": report["results"],
            }
        )
        return run

    run["status"] = "pending"
    run.setdefault("summary", {})
    run.setdefault("results", [])
    run.setdefault(
        "notes",
        "Variant is not supported by the current P0a runner.",
    )
    return run


def _format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
