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
        if not isinstance(run, Mapping) or not is_complete_ablation_run(run):
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
        return _assign_unique_case_keys(cases)

    for result in _sequence(report.get("results")):
        if not isinstance(result, Mapping):
            continue
        case = _failure_case(result, default_system, diagnostics_source)
        if case is not None:
            cases.append(case)
    return _assign_unique_case_keys(cases)


def build_ablation_failure_cases(
    payload: Mapping[str, Any],
) -> list[FailureCaseRow]:
    """Flatten failed rows from all ablation variants."""

    cases: list[FailureCaseRow] = []
    for run in _sequence(payload.get("runs")):
        if not isinstance(run, Mapping) or not is_complete_ablation_run(run):
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
    return _assign_unique_case_keys(cases)


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


def history_runs_to_table(rows: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    """Convert persisted history run records to dashboard table rows."""

    return [
        [
            _text(row.get("run_id")),
            _text(row.get("created_at")),
            _text(row.get("source")),
            _text(row.get("workspace_id")),
            _text(row.get("status")),
            _text(row.get("mode")),
            _text(row.get("evaluator_version")) or "legacy",
            row.get("schema_version")
            if row.get("schema_version") is not None
            else "legacy",
            _hash_prefix(row.get("prompt_manifest_hash")),
            row.get("question_count")
            if row.get("question_count") is not None
            else 0,
            _text(row.get("result_path")),
        ]
        for row in rows
    ]


def history_trends_to_table(rows: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    """Convert persisted history metric records to dashboard table rows."""

    return [
        [
            _text(row.get("created_at")),
            _text(row.get("run_id")),
            _text(row.get("system_label")) or _text(row.get("system_id")),
            _text(row.get("evaluator_version")) or "legacy",
            _hash_prefix(row.get("prompt_manifest_hash")),
            _text(row.get("metric_name")),
            _metric_value(row.get("metric_value")),
        ]
        for row in rows
    ]


def get_runtime_config(
    payload: Mapping[str, Any],
    variant_id: str | None,
) -> dict[str, Any]:
    """Return one ablation variant's saved runtime configuration."""

    for run in _sequence(payload.get("runs")):
        if (
            isinstance(run, Mapping)
            and is_complete_ablation_run(run)
            and run.get("id") == variant_id
        ):
            return dict(_mapping(run.get("runtime_config")))
    return {}


def is_complete_ablation_run(run: Mapping[str, Any]) -> bool:
    """Accept completed runs and legacy runs that do not store status."""

    if "status" not in run or run.get("status") is None:
        return True
    status = run.get("status")
    return isinstance(status, str) and status.strip() == "completed"


def _metric_row(label: str, summary: Mapping[str, Any]) -> list[Any]:
    return [label, *[_metric_value(summary.get(key)) for key in METRIC_KEYS]]


def _metric_value(value: Any) -> Any:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return round(value, 4)
    return value


def _hash_prefix(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    if len(value) <= 12:
        return value
    return value[:12]


def _failure_case(
    result: Mapping[str, Any],
    system: str,
    diagnostics_source: str,
    system_label: str | None = None,
) -> FailureCaseRow | None:
    source = _diagnostics_source(diagnostics_source)
    analysis = result.get("failure_analysis")
    failure_type = (
        str(analysis.get("failure_type") or "")
        if isinstance(analysis, Mapping)
        else ""
    )
    if failure_type == "no_failure":
        return None
    if failure_type:
        reason = str(analysis.get("reason") or "")
        suggestion = str(analysis.get("suggestion") or "")
        analysis_question_id = str(analysis.get("question_id") or "")
    else:
        failure_signals = _unavailable_failure_signals(result)
        if source != "unavailable" or not failure_signals:
            return None
        failure_type = "unclassified_failure"
        reason = (
            "Failure diagnostics are unavailable for this legacy result; "
            f"observed signal(s): {', '.join(failure_signals)}."
        )
        suggestion = (
            "Rerun the evaluation or load a newer artifact with stored "
            "failure diagnostics."
        )
        analysis_question_id = ""
    question_id = str(result.get("question_id") or analysis_question_id)
    return {
        "case_key": "",
        "system": system,
        "system_label": system_label or _system_label(system),
        "question_id": question_id,
        "question_type": str(result.get("question_type") or "unspecified"),
        "question": str(result.get("question") or ""),
        "failure_type": failure_type,
        "reason": reason,
        "suggestion": suggestion,
        "diagnostics_source": source,
    }


def _unavailable_failure_signals(result: Mapping[str, Any]) -> list[str]:
    signals: list[str] = []
    if result.get("correct") is False:
        signals.append("correct=false")
    if result.get("fallback_correct") is False:
        signals.append("fallback_correct=false")
    error = result.get("error")
    if isinstance(error, str) and error.strip():
        signals.append("error")
    return signals


def _assign_unique_case_keys(
    cases: list[FailureCaseRow],
) -> list[FailureCaseRow]:
    exact_keys = {
        f"{case['system']}:{case['question_id']}"
        for case in cases
        if case["question_id"]
    }
    occurrences: Counter[str] = Counter()
    used_keys: set[str] = set()

    for row_number, case in enumerate(cases, start=1):
        question_id = case["question_id"]
        if question_id:
            base_key = f"{case['system']}:{question_id}"
            occurrences[base_key] += 1
            occurrence = occurrences[base_key]
            if occurrence == 1:
                case_key = base_key
            else:
                case_key = f"{base_key}:{occurrence}"
                while case_key in exact_keys or case_key in used_keys:
                    occurrence += 1
                    case_key = f"{base_key}:{occurrence}"
                occurrences[base_key] = occurrence
        else:
            base_key = f"{case['system']}:row-{row_number}"
            case_key = base_key
            suffix = 1
            while case_key in exact_keys or case_key in used_keys:
                suffix += 1
                case_key = f"{base_key}:{suffix}"

        case["case_key"] = case_key
        used_keys.add(case_key)

    return cases


def _system_label(system: str) -> str:
    return SYSTEM_LABELS.get(system, system)


def _diagnostics_source(value: str) -> DiagnosticsSource:
    if value in {"stored", "derived", "unavailable"}:
        return cast(DiagnosticsSource, value)
    return "unavailable"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return value
    return []
