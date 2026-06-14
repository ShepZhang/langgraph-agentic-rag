"""Pure report-to-view transformations for the Evaluation Dashboard."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, cast

from evaluation.dashboard_models import (
    DiagnosticsSource,
    FailureCaseDetail,
    FailureCaseRow,
)


METRIC_KEYS = (
    "correctness_score",
    "context_relevance_score",
    "citation_hit_rate",
    "fallback_accuracy",
    "unsupported_claim_count",
    "average_latency",
    "average_retry_count",
)
SYSTEM_LABELS = {
    "naive": "Naive RAG",
    "agentic": "Agentic RAG",
}


def build_summary_rows(
    report: Mapping[str, Any],
    default_system: str = "agentic",
) -> list[list[Any]]:
    """Return table rows for single-system or comparison summaries."""

    summary = _mapping(report.get("summary"))
    if summary.get("mode") == "comparison":
        return [
            _metric_row("Naive RAG", _mapping(summary.get("naive"))),
            _metric_row("Agentic RAG", _mapping(summary.get("agentic"))),
        ]
    return [_metric_row(_system_label(default_system), summary)]


def build_ablation_summary_rows(payload: Mapping[str, Any]) -> list[list[Any]]:
    """Return one metric row per valid ablation variant."""

    rows: list[list[Any]] = []
    for run in _sequence(payload.get("runs")):
        if not isinstance(run, Mapping):
            continue
        run_id = str(run.get("id") or "unknown")
        method = str(run.get("method") or run_id)
        rows.append(_metric_row(f"{run_id} {method}", _mapping(run.get("summary"))))
    return rows


def build_failure_cases(
    report: Mapping[str, Any],
    default_system: str = "agentic",
    diagnostics_source: str = "stored",
) -> list[FailureCaseRow]:
    """Flatten failed rows from single-system or paired reports."""

    summary = _mapping(report.get("summary"))
    cases: list[FailureCaseRow] = []
    if summary.get("mode") == "comparison":
        for paired in _sequence(report.get("results")):
            if not isinstance(paired, Mapping):
                continue
            for system in ("naive", "agentic"):
                result = paired.get(system)
                if isinstance(result, Mapping):
                    case = _failure_case(result, system, diagnostics_source)
                    if case is not None:
                        cases.append(case)
        return cases

    for result in _sequence(report.get("results")):
        if not isinstance(result, Mapping):
            continue
        case = _failure_case(result, default_system, diagnostics_source)
        if case is not None:
            cases.append(case)
    return cases


def build_ablation_failure_cases(
    payload: Mapping[str, Any],
) -> list[FailureCaseRow]:
    """Flatten failed rows from all ablation variants."""

    cases: list[FailureCaseRow] = []
    for run in _sequence(payload.get("runs")):
        if not isinstance(run, Mapping):
            continue
        run_id = str(run.get("id") or "unknown")
        method = str(run.get("method") or run_id)
        for result in _sequence(run.get("results")):
            if not isinstance(result, Mapping):
                continue
            source = str(result.get("_diagnostics_source") or "stored")
            case = _failure_case(
                result,
                run_id,
                source,
                system_label=f"{run_id} {method}",
            )
            if case is not None:
                cases.append(case)
    return cases


def build_failure_count_rows(
    cases: Sequence[FailureCaseRow],
) -> list[list[Any]]:
    """Count visible failures by system and type."""

    counts = Counter(
        (case["system"], case["system_label"], case["failure_type"])
        for case in cases
    )
    return [
        [system_label, failure_type, count]
        for (_, system_label, failure_type), count in sorted(counts.items())
    ]


def filter_failure_cases(
    cases: Sequence[FailureCaseRow],
    system: str | None = None,
    failure_type: str | None = None,
) -> list[FailureCaseRow]:
    """Filter existing rows without rerunning evaluation."""

    return [
        case
        for case in cases
        if (not system or system == "all" or case["system"] == system)
        and (
            not failure_type
            or failure_type == "all"
            or case["failure_type"] == failure_type
        )
    ]


def get_failure_detail(
    cases: Sequence[FailureCaseRow],
    case_key: str | None,
) -> FailureCaseDetail:
    """Return a safe detail record for one selected failure."""

    for case in cases:
        if case["case_key"] == case_key:
            return {
                "case_key": case["case_key"],
                "title": (
                    f"{case['system_label']} / {case['question_id']} / "
                    f"{case['failure_type']}"
                ),
                "reason": case["reason"],
                "suggestion": case["suggestion"],
                "diagnostics_source": case["diagnostics_source"],
            }
    return {
        "case_key": "",
        "title": "No failed case selected",
        "reason": "",
        "suggestion": "",
        "diagnostics_source": "unavailable",
    }


def failure_cases_to_table(cases: Sequence[FailureCaseRow]) -> list[list[str]]:
    """Convert failure records to table rows."""

    return [
        [
            case["case_key"],
            case["system_label"],
            case["question_id"],
            case["question_type"],
            case["failure_type"],
            case["diagnostics_source"],
            case["question"],
        ]
        for case in cases
    ]


def get_runtime_config(
    payload: Mapping[str, Any],
    variant_id: str | None,
) -> dict[str, Any]:
    """Return one ablation variant's saved runtime configuration."""

    for run in _sequence(payload.get("runs")):
        if isinstance(run, Mapping) and run.get("id") == variant_id:
            return dict(_mapping(run.get("runtime_config")))
    return {}


def _metric_row(label: str, summary: Mapping[str, Any]) -> list[Any]:
    return [label, *[_metric_value(summary.get(key)) for key in METRIC_KEYS]]


def _metric_value(value: Any) -> Any:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return round(value, 4)
    return value


def _failure_case(
    result: Mapping[str, Any],
    system: str,
    diagnostics_source: str,
    system_label: str | None = None,
) -> FailureCaseRow | None:
    analysis = result.get("failure_analysis")
    if not isinstance(analysis, Mapping):
        return None
    failure_type = str(analysis.get("failure_type") or "")
    if not failure_type or failure_type == "no_failure":
        return None
    question_id = str(result.get("question_id") or analysis.get("question_id") or "")
    return {
        "case_key": f"{system}:{question_id}",
        "system": system,
        "system_label": system_label or _system_label(system),
        "question_id": question_id,
        "question_type": str(result.get("question_type") or "unspecified"),
        "question": str(result.get("question") or ""),
        "failure_type": failure_type,
        "reason": str(analysis.get("reason") or ""),
        "suggestion": str(analysis.get("suggestion") or ""),
        "diagnostics_source": _diagnostics_source(diagnostics_source),
    }


def _system_label(system: str) -> str:
    return SYSTEM_LABELS.get(system, system)


def _diagnostics_source(value: str) -> DiagnosticsSource:
    if value in {"stored", "derived", "unavailable"}:
        return cast(DiagnosticsSource, value)
    return "unavailable"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, list) else []
