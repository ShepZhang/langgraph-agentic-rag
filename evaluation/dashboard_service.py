"""Application service for Evaluation Dashboard workflows."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent.graph import run_agent
from baseline.naive_rag import run_naive_rag
from evaluation.dashboard_formatters import (
    build_ablation_failure_cases,
    build_ablation_summary_rows,
    build_failure_cases,
    build_failure_count_rows,
    build_summary_rows,
    filter_failure_cases as filter_failure_case_rows,
    get_failure_detail as select_failure_detail,
    get_runtime_config as select_runtime_config,
)
from evaluation.dashboard_models import (
    DEFAULT_ABLATION_RESULT_PATH,
    AblationResultProvider,
    DashboardStatus,
    DashboardSystemMode,
    DashboardView,
    FailureCaseDetail,
    FailureCaseRow,
    QuestionOption,
)
from evaluation.evaluate import evaluate_questions, load_eval_questions
from evaluation.failure_analyzer import analyze_failure


QuestionLoader = Callable[[], list[dict[str, Any]]]
EvaluationRunner = Callable[..., dict[str, Any]]
IdFactory = Callable[[str], str]
_STORED_ANALYSIS_FIELDS = ("failure_type", "reason", "suggestion")
_DERIVABLE_RESULT_FIELDS = frozenset(
    (
        "question_id",
        "correct",
        "fallback_correct",
        "fallback_triggered",
        "answer_returned",
        "source_hit",
        "context_relevant",
        "citation_hit",
        "retrieved_documents",
        "relevant_documents",
        "citations",
        "error",
    )
)


class JsonAblationResultProvider:
    """Load a saved ablation result from a JSON file."""

    def __init__(
        self,
        path: str | Path = DEFAULT_ABLATION_RESULT_PATH,
    ) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        with self.path.open(encoding="utf-8") as result_file:
            payload = json.load(result_file)

        if not isinstance(payload, dict):
            raise ValueError("ablation result payload must be an object")
        if not isinstance(payload.get("runs"), list):
            raise ValueError("ablation result payload runs must be a list")
        return payload


class EvaluationDashboardService:
    """Coordinate dashboard evaluation requests and view formatting."""

    def __init__(
        self,
        question_loader: QuestionLoader = load_eval_questions,
        agentic_runner: EvaluationRunner = run_agent,
        naive_runner: EvaluationRunner = run_naive_rag,
        ablation_provider: AblationResultProvider | None = None,
        id_factory: IdFactory | None = None,
    ) -> None:
        self._question_loader = question_loader
        self._agentic_runner = agentic_runner
        self._naive_runner = naive_runner
        self._ablation_provider = (
            ablation_provider or JsonAblationResultProvider()
        )
        self._id_factory = id_factory or _default_id_factory

    def list_questions(self) -> list[QuestionOption]:
        """Return dashboard question choices in dataset order."""

        options: list[QuestionOption] = []
        for question in self._question_loader():
            question_id = str(question.get("id") or "")
            question_type = str(
                question.get("question_type") or "unspecified"
            )
            question_text = str(question.get("question") or "")
            options.append(
                {
                    "id": question_id,
                    "label": (
                        f"[{question_id}] {question_type} - {question_text}"
                    ),
                    "question_type": question_type,
                    "question": question_text,
                }
            )
        return options

    def run_quick_evaluation(
        self,
        question_ids: list[str],
        system_mode: DashboardSystemMode,
    ) -> DashboardView:
        """Run selected questions through the requested evaluation mode."""

        if not question_ids:
            return self._empty_view(
                status="failed",
                message="Select at least one question.",
            )
        if system_mode not in {"naive", "agentic", "comparison"}:
            return self._empty_view(
                status="failed",
                message=f"Unsupported system mode: {system_mode}",
            )

        try:
            questions = self._question_loader()
            requested_ids = set(question_ids)
            known_ids = {
                str(question.get("id") or "")
                for question in questions
            }
            unknown_ids = sorted(requested_ids - known_ids)
            if unknown_ids:
                return self._empty_view(
                    status="failed",
                    message=(
                        "Unknown question ID(s): "
                        + ", ".join(unknown_ids)
                    ),
                )

            selected_questions = [
                question
                for question in questions
                if str(question.get("id") or "") in requested_ids
            ]
            report, default_system = self._evaluate(
                selected_questions,
                system_mode,
            )
            failure_cases = build_failure_cases(
                report,
                default_system=default_system,
            )
            return {
                "status": "completed",
                "run_id": self._id_factory("quick"),
                "summary_rows": build_summary_rows(
                    report,
                    default_system=default_system,
                ),
                "failure_count_rows": build_failure_count_rows(
                    failure_cases
                ),
                "failure_cases": failure_cases,
                "raw_report": report,
                "message": (
                    "Quick evaluation completed for "
                    f"{len(selected_questions)} question(s) with "
                    f"{len(failure_cases)} failure(s)."
                ),
            }
        except Exception as exc:  # noqa: BLE001 - return UI-safe failure data.
            return self._empty_view(
                status="failed",
                message=(
                    "Quick evaluation failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

    def load_ablation_snapshot(self) -> DashboardView:
        """Load and compatibly enrich a saved ablation report in memory."""

        try:
            payload = self._ablation_provider.load()
            (
                enriched,
                skipped,
                unavailable_diagnostics,
                metadata_error,
            ) = _enrich_ablation_payload(
                payload,
                self._question_loader,
            )
            failure_cases = build_ablation_failure_cases(enriched)
            variant_count = len(enriched["runs"])
            failure_count = len(failure_cases)
            message = (
                f"Loaded {variant_count} ablation variant(s) with "
                f"{failure_count} failure(s)."
            )
            if skipped:
                message += (
                    " Snapshot is partial; "
                    f"skipped {skipped} malformed item(s)."
                )
            if unavailable_diagnostics:
                message += (
                    " Snapshot is degraded; "
                    "unavailable diagnostics: "
                    f"{unavailable_diagnostics}."
                )
                if metadata_error:
                    message += f" Metadata unavailable: {metadata_error}."
            return {
                "status": "completed",
                "run_id": self._id_factory("snapshot"),
                "summary_rows": build_ablation_summary_rows(enriched),
                "failure_count_rows": build_failure_count_rows(
                    failure_cases
                ),
                "failure_cases": failure_cases,
                "raw_report": enriched,
                "message": message,
            }
        except Exception as exc:  # noqa: BLE001 - return UI-safe failure data.
            return self._empty_view(
                status="unavailable",
                message=(
                    "Ablation snapshot unavailable: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

    def get_runtime_config(
        self,
        view: DashboardView,
        variant_id: str | None,
    ) -> dict[str, Any]:
        """Return the stored runtime configuration for one variant."""

        if not variant_id or variant_id == "all":
            return {}
        raw_report = view.get("raw_report") if isinstance(view, Mapping) else {}
        if not isinstance(raw_report, Mapping):
            return {}
        return deepcopy(select_runtime_config(raw_report, variant_id))

    def filter_failure_cases(
        self,
        view: DashboardView,
        system: str | None = None,
        failure_type: str | None = None,
    ) -> list[FailureCaseRow]:
        """Filter a completed view's failure rows."""

        return filter_failure_case_rows(
            view["failure_cases"],
            system=system,
            failure_type=failure_type,
        )

    def get_failure_detail(
        self,
        view: DashboardView,
        case_key: str | None,
    ) -> FailureCaseDetail:
        """Select one failure detail from a completed view."""

        return select_failure_detail(view["failure_cases"], case_key)

    def _evaluate(
        self,
        questions: list[dict[str, Any]],
        system_mode: DashboardSystemMode,
    ) -> tuple[dict[str, Any], str]:
        if system_mode == "naive":
            return (
                evaluate_questions(
                    questions,
                    run_agent_fn=self._naive_runner,
                    run_naive_fn=None,
                ),
                "naive",
            )
        if system_mode == "agentic":
            return (
                evaluate_questions(
                    questions,
                    run_agent_fn=self._agentic_runner,
                    run_naive_fn=None,
                ),
                "agentic",
            )
        return (
            evaluate_questions(
                questions,
                run_agent_fn=self._agentic_runner,
                run_naive_fn=self._naive_runner,
            ),
            "agentic",
        )

    def _empty_view(
        self,
        status: DashboardStatus,
        message: str,
    ) -> DashboardView:
        return {
            "status": status,
            "run_id": "",
            "summary_rows": [],
            "failure_count_rows": [],
            "failure_cases": [],
            "raw_report": {},
            "message": message,
        }


def _default_id_factory(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{prefix}-{timestamp}-{uuid4()}"


def _enrich_ablation_payload(
    payload: Mapping[str, Any],
    question_loader: QuestionLoader,
) -> tuple[dict[str, Any], int, int, str | None]:
    if not isinstance(payload, Mapping):
        raise ValueError("ablation result payload must be an object")

    enriched = deepcopy(dict(payload))
    runs = enriched.get("runs")
    if not isinstance(runs, list):
        raise ValueError("ablation result payload runs must be a list")

    valid_runs: list[dict[str, Any]] = []
    skipped = 0
    unavailable_diagnostics = 0
    questions_by_id: dict[str, dict[str, Any]] | None = None
    metadata_error: str | None = None

    def question_for_result(result: Mapping[str, Any]) -> dict[str, Any] | None:
        nonlocal questions_by_id, metadata_error
        question_id = _clean_text(result.get("question_id"))
        if not question_id or metadata_error is not None:
            return None
        if questions_by_id is None:
            try:
                questions_by_id = {
                    str(question.get("id") or ""): question
                    for question in question_loader()
                    if isinstance(question, dict)
                }
            except Exception as exc:  # noqa: BLE001 - degrade per row.
                metadata_error = f"{type(exc).__name__}: {exc}"
                questions_by_id = {}
                return None
        return questions_by_id.get(question_id)

    for raw_run in runs:
        if not isinstance(raw_run, dict):
            skipped += 1
            continue

        run = raw_run
        results = run.get("results")
        if not isinstance(results, list):
            run["results"] = []
            valid_runs.append(run)
            skipped += 1
            continue

        valid_results: list[dict[str, Any]] = []
        for raw_result in results:
            if not isinstance(raw_result, dict):
                skipped += 1
                continue

            result = raw_result
            analysis = result.get("failure_analysis")
            if _is_stored_failure_analysis(analysis):
                result["_diagnostics_source"] = "stored"
            elif not _is_derivable_legacy_result(result):
                result["_diagnostics_source"] = "unavailable"
                unavailable_diagnostics += 1
            else:
                question = question_for_result(result)
                if question is None:
                    result["_diagnostics_source"] = "unavailable"
                    unavailable_diagnostics += 1
                else:
                    result["failure_analysis"] = analyze_failure(
                        question,
                        result,
                    )
                    result["_diagnostics_source"] = "derived"
            valid_results.append(result)

        run["results"] = valid_results
        valid_runs.append(run)

    enriched["runs"] = valid_runs
    return enriched, skipped, unavailable_diagnostics, metadata_error


def _is_stored_failure_analysis(analysis: Any) -> bool:
    return isinstance(analysis, Mapping) and all(
        _clean_text(analysis.get(field))
        for field in _STORED_ANALYSIS_FIELDS
    )


def _is_derivable_legacy_result(result: Mapping[str, Any]) -> bool:
    return (
        _DERIVABLE_RESULT_FIELDS.issubset(result.keys())
        and bool(_clean_text(result.get("question_id")))
    )


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()
