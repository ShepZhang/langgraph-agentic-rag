"""Application service for Evaluation Dashboard workflows."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent.graph import run_agent
from baseline.naive_rag import run_naive_rag
from evaluation.dashboard_formatters import (
    build_failure_cases,
    build_failure_count_rows,
    build_summary_rows,
    filter_failure_cases as filter_failure_case_rows,
    get_failure_detail as select_failure_detail,
)
from evaluation.dashboard_models import (
    DEFAULT_ABLATION_RESULT_PATH,
    AblationResultProvider,
    DashboardStatus,
    DashboardView,
    FailureCaseDetail,
    FailureCaseRow,
    QuestionOption,
)
from evaluation.evaluate import evaluate_questions, load_eval_questions


QuestionLoader = Callable[[], list[dict[str, Any]]]
EvaluationRunner = Callable[..., dict[str, Any]]
IdFactory = Callable[[str], str]


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
        self._ablation_provider = ablation_provider
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
        system_mode: str,
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
            known_ids = {
                str(question.get("id") or "")
                for question in questions
            }
            unknown_ids = [
                question_id
                for question_id in question_ids
                if question_id not in known_ids
            ]
            if unknown_ids:
                return self._empty_view(
                    status="failed",
                    message=(
                        "Unknown question ID(s): "
                        + ", ".join(unknown_ids)
                    ),
                )

            selected_ids = set(question_ids)
            selected_questions = [
                question
                for question in questions
                if str(question.get("id") or "") in selected_ids
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
        system_mode: str,
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
