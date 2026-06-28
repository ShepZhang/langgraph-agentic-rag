"""Typed, JSON-ready contracts for the Evaluation Dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict


DashboardSystemMode = Literal["naive", "agentic", "comparison"]
DashboardStatus = Literal["completed", "failed", "unavailable"]
DiagnosticsSource = Literal["stored", "derived", "unavailable"]

SMOKE_QUESTION_IDS = ("q001", "q016", "q027", "q030", "q033")
DEFAULT_ABLATION_RESULT_PATH = Path("experiments/results/ablation_result.json")

METRIC_COLUMNS = [
    "System",
    "Correctness",
    "Context Relevance",
    "Citation Accuracy",
    "Fallback Accuracy",
    "Unsupported Claims",
    "Avg Latency (s)",
    "Avg Retry",
]
FAILURE_COUNT_COLUMNS = ["System", "Failure Type", "Count"]
FAILURE_CASE_COLUMNS = [
    "Case Key",
    "System",
    "Question ID",
    "Question Type",
    "Failure Type",
    "Diagnostics",
    "Question",
]
HISTORY_RUN_COLUMNS = [
    "Run ID",
    "Created At",
    "Source",
    "Workspace",
    "Status",
    "Mode",
    "Evaluator",
    "Schema",
    "Prompt Hash",
    "Questions",
    "Result Path",
]
HISTORY_TREND_COLUMNS = [
    "Created At",
    "Run ID",
    "System",
    "Evaluator",
    "Prompt Hash",
    "Metric",
    "Value",
]


class QuestionOption(TypedDict):
    id: str
    label: str
    question_type: str
    question: str


class FailureCaseRow(TypedDict):
    case_key: str
    system: str
    system_label: str
    question_id: str
    question_type: str
    question: str
    failure_type: str
    reason: str
    suggestion: str
    diagnostics_source: DiagnosticsSource


class FailureCaseDetail(TypedDict):
    case_key: str
    title: str
    reason: str
    suggestion: str
    diagnostics_source: DiagnosticsSource


class DashboardView(TypedDict):
    status: DashboardStatus
    run_id: str
    summary_rows: list[list[Any]]
    failure_count_rows: list[list[Any]]
    failure_cases: list[FailureCaseRow]
    raw_report: dict[str, Any]
    message: str


class HistoryDashboardView(TypedDict):
    status: DashboardStatus
    run_rows: list[list[Any]]
    trend_rows: list[list[Any]]
    metric_choices: list[str]
    message: str


class AblationRunHandle(TypedDict):
    run_id: str
    status: str


class AblationResultProvider(Protocol):
    def load(self) -> dict[str, Any]:
        """Load one saved ablation payload."""


class AblationRunner(Protocol):
    def run(
        self,
        question_ids: list[str],
        variant_ids: list[str],
    ) -> AblationRunHandle:
        """Start an ablation run and return its handle."""
