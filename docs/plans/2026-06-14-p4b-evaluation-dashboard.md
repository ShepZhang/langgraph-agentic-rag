# P4b Evaluation Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Gradio Evaluation Dashboard that runs small Naive/Agentic comparisons, reads existing V0-V6 ablation artifacts, and exposes metrics plus deterministic failed-case diagnostics.

**Architecture:** Keep `evaluation.evaluate` and `evaluation.failure_analyzer` as the metric and diagnostic sources of truth. Add typed dashboard contracts, pure report-to-view formatters, and an injected `EvaluationDashboardService`; Gradio calls that service directly and stores the last successful quick and snapshot views in separate `gr.State` values. The initial ablation view is read-only, while protocols reserve a later background runner.

**Tech Stack:** Python 3.12, TypedDict and Protocol, Gradio 6.15, pytest, existing evaluation and ablation JSON formats.

---

## File Map

### Create

- `evaluation/dashboard_models.py`: stable dashboard constants, typed row records, view model, artifact provider protocol, and future ablation runner protocol.
- `evaluation/dashboard_formatters.py`: pure summary, failure, filtering, detail, and runtime-config transformations.
- `evaluation/dashboard_service.py`: question selection, runner dispatch, run IDs, snapshot loading, legacy diagnostic enrichment, and normalized view construction.
- `tests/test_dashboard_service.py`: service, formatter, snapshot compatibility, filtering, and error-boundary tests.

### Modify

- `ui/gradio_app.py`: preserve Document QA helpers, add dashboard event adapters, split UI construction into two main tabs, and wire independent dashboard states.
- `tests/test_gradio_app.py`: preserve current tests and add event-helper plus application-structure coverage.
- `tests/test_fastapi_routes.py`: assert the displayed P4b application version.
- `README.md`: document the dashboard workflow, snapshot semantics, and full-live-ablation roadmap.
- `CHANGELOG.md`: add `v0.4.1-p4b`.
- `api/main.py`: bump displayed API version to `0.4.1-p4b`.

## Stable Contracts

The implementation must preserve these mode and state values:

```python
DashboardSystemMode = Literal["naive", "agentic", "comparison"]
DashboardStatus = Literal["completed", "failed", "unavailable"]
DiagnosticsSource = Literal["stored", "derived", "unavailable"]

SMOKE_QUESTION_IDS = ("q001", "q016", "q027", "q030", "q033")
DEFAULT_ABLATION_RESULT_PATH = Path("experiments/results/ablation_result.json")
```

The dashboard view remains JSON-ready:

```python
{
    "status": "completed",
    "run_id": "quick-20260614T120000000000-ab12cd34",
    "summary_rows": [],
    "failure_count_rows": [],
    "failure_cases": [],
    "raw_report": {},
    "message": "Evaluation completed for 5 question(s).",
}
```

No UI code recomputes evaluation metrics or calls `analyze_failure()` directly.

---

### Task 1: Dashboard Contracts and Pure Formatters

**Files:**
- Create: `evaluation/dashboard_models.py`
- Create: `evaluation/dashboard_formatters.py`
- Create: `tests/test_dashboard_service.py`

- [ ] **Step 1: Write failing formatter and contract tests**

Create `tests/test_dashboard_service.py` with the shared fixtures and first tests:

```python
from __future__ import annotations

import json
from pathlib import Path

from evaluation.dashboard_formatters import (
    build_failure_cases,
    build_failure_count_rows,
    build_summary_rows,
    filter_failure_cases,
    get_failure_detail,
)
from evaluation.dashboard_models import (
    METRIC_COLUMNS,
    SMOKE_QUESTION_IDS,
)


def _result(question_id: str, failure_type: str = "no_failure") -> dict:
    return {
        "question_id": question_id,
        "question_type": "single_doc",
        "question": f"Question {question_id}",
        "correct": failure_type == "no_failure",
        "fallback_correct": True,
        "failure_analysis": {
            "question_id": question_id,
            "failure_type": failure_type,
            "reason": f"Reason for {question_id}",
            "suggestion": f"Suggestion for {question_id}",
        },
    }


def _summary(**overrides) -> dict:
    summary = {
        "correctness_score": 0.8,
        "context_relevance_score": 0.9,
        "citation_hit_rate": 0.75,
        "fallback_accuracy": 1.0,
        "unsupported_claim_count": 0,
        "average_latency": 2.5,
        "average_retry_count": 0.4,
        "failure_type_counts": {
            "no_failure": 1,
            "retrieval_failure": 1,
        },
    }
    summary.update(overrides)
    return summary


def test_dashboard_constants_define_smoke_questions_and_metric_columns():
    assert SMOKE_QUESTION_IDS == ("q001", "q016", "q027", "q030", "q033")
    assert METRIC_COLUMNS == [
        "System",
        "Correctness",
        "Context Relevance",
        "Citation Accuracy",
        "Fallback Accuracy",
        "Unsupported Claims",
        "Avg Latency (s)",
        "Avg Retry",
    ]


def test_build_summary_rows_supports_single_and_comparison_reports():
    single = {"summary": _summary(), "results": [_result("q001")]}
    comparison = {
        "summary": {
            "mode": "comparison",
            "naive": _summary(correctness_score=0.5),
            "agentic": _summary(correctness_score=0.8),
        },
        "results": [],
    }

    assert build_summary_rows(single, default_system="agentic")[0][0] == "Agentic RAG"
    comparison_rows = build_summary_rows(comparison)
    assert [row[0] for row in comparison_rows] == ["Naive RAG", "Agentic RAG"]
    assert comparison_rows[0][1] == 0.5
    assert comparison_rows[1][1] == 0.8


def test_build_summary_rows_preserves_unavailable_metrics():
    report = {
        "summary": _summary(
            unsupported_claim_count=None,
            citation_hit_rate=None,
        ),
        "results": [],
    }

    row = build_summary_rows(report, default_system="naive")[0]

    assert row[3] == "N/A"
    assert row[5] == "N/A"


def test_failure_rows_can_be_counted_filtered_and_selected():
    report = {
        "summary": _summary(),
        "results": [
            _result("q001"),
            _result("q002", "retrieval_failure"),
        ],
    }

    cases = build_failure_cases(report, default_system="agentic")

    assert [case["case_key"] for case in cases] == ["agentic:q002"]
    assert cases[0]["diagnostics_source"] == "stored"
    assert build_failure_count_rows(cases) == [
        ["Agentic RAG", "retrieval_failure", 1],
    ]
    assert filter_failure_cases(cases, system="agentic") == cases
    assert filter_failure_cases(cases, failure_type="citation_failure") == []
    detail = get_failure_detail(cases, "agentic:q002")
    assert detail["reason"] == "Reason for q002"
    assert get_failure_detail(cases, None)["case_key"] == ""
```

- [ ] **Step 2: Run the tests and verify the imports fail**

Run:

```bash
../../.venv/bin/python -m pytest tests/test_dashboard_service.py -q
```

Expected: collection fails because `evaluation.dashboard_models` and
`evaluation.dashboard_formatters` do not exist.

- [ ] **Step 3: Implement the dashboard contracts**

Create `evaluation/dashboard_models.py`:

```python
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
```

- [ ] **Step 4: Implement pure report formatters**

Create `evaluation/dashboard_formatters.py`:

```python
"""Pure report-to-view transformations for the Evaluation Dashboard."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from evaluation.dashboard_models import FailureCaseDetail, FailureCaseRow


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
    """Convert failure records to Gradio Dataframe rows."""

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


def _diagnostics_source(value: str) -> str:
    return value if value in {"stored", "derived", "unavailable"} else "unavailable"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, list) else []
```

- [ ] **Step 5: Run formatter tests**

Run:

```bash
../../.venv/bin/python -m pytest tests/test_dashboard_service.py -q
```

Expected: all Task 1 tests pass.

- [ ] **Step 6: Commit contracts and formatters**

```bash
git add evaluation/dashboard_models.py evaluation/dashboard_formatters.py tests/test_dashboard_service.py
git commit -m "feat: add dashboard view contracts and formatters"
```

---

### Task 2: Quick Evaluation Service

**Files:**
- Create: `evaluation/dashboard_service.py`
- Modify: `tests/test_dashboard_service.py`

- [ ] **Step 1: Add failing service tests**

Append to `tests/test_dashboard_service.py`:

```python
import pytest

from evaluation.dashboard_service import EvaluationDashboardService


def _question(question_id: str) -> dict:
    return {
        "id": question_id,
        "question": f"Question {question_id}",
        "question_type": "single_doc",
        "gold_answer": "retrieval",
        "expected_sources": ["notes.md"],
        "expected_keywords": ["retrieval"],
        "answerable": True,
        "should_answer": True,
        "expected_behavior": "answer_with_citation",
        "chat_history": [],
        "requires_rewrite": False,
    }


def _runner(label: str, calls: list[tuple[str, str]]):
    def run(question: str, chat_history: list[dict]) -> dict:
        calls.append((label, question))
        return {
            "answer": "retrieval [1]",
            "citations": [{"source": "notes.md"}],
            "retrieved_documents": [{"source": "notes.md"}],
            "relevant_documents": [{"source": "notes.md"}],
            "retry_count": 0,
            "fallback_reason": "",
            "citation_verification_enabled": label == "agentic",
            "claim_verification_results": [],
        }

    return run


def test_service_lists_questions_and_preserves_dataset_order():
    questions = [_question("q002"), _question("q001")]
    service = EvaluationDashboardService(question_loader=lambda: questions)

    options = service.list_questions()

    assert [option["id"] for option in options] == ["q002", "q001"]
    assert options[0]["label"] == "[q002] single_doc - Question q002"


def test_repository_dashboard_exposes_all_36_evaluation_questions():
    options = EvaluationDashboardService().list_questions()

    assert len(options) == 36
    assert options[0]["id"] == "q001"
    assert options[-1]["id"] == "q036"


@pytest.mark.parametrize(
    ("mode", "expected_labels"),
    [
        ("naive", ["naive"]),
        ("agentic", ["agentic"]),
        ("comparison", ["naive", "agentic"]),
    ],
)
def test_service_dispatches_quick_evaluation_modes(mode, expected_labels):
    calls: list[tuple[str, str]] = []
    service = EvaluationDashboardService(
        question_loader=lambda: [_question("q001")],
        naive_runner=_runner("naive", calls),
        agentic_runner=_runner("agentic", calls),
        id_factory=lambda prefix: f"{prefix}-fixed",
    )

    view = service.run_quick_evaluation(["q001"], mode)

    assert view["status"] == "completed"
    assert view["run_id"] == "quick-fixed"
    assert [label for label, _ in calls] == expected_labels
    assert len(view["summary_rows"]) == len(expected_labels)


def test_service_rejects_empty_and_unknown_question_ids_before_runner_calls():
    calls: list[tuple[str, str]] = []
    service = EvaluationDashboardService(
        question_loader=lambda: [_question("q001")],
        agentic_runner=_runner("agentic", calls),
    )

    empty = service.run_quick_evaluation([], "agentic")
    unknown = service.run_quick_evaluation(["q999"], "agentic")

    assert empty["status"] == "failed"
    assert "Select at least one" in empty["message"]
    assert unknown["status"] == "failed"
    assert "q999" in unknown["message"]
    assert calls == []


def test_service_records_per_case_runner_error_as_tool_failure():
    def failing_runner(question: str, chat_history: list[dict]) -> dict:
        raise RuntimeError("synthetic runner failure")

    service = EvaluationDashboardService(
        question_loader=lambda: [_question("q001")],
        agentic_runner=failing_runner,
        id_factory=lambda prefix: f"{prefix}-fixed",
    )

    view = service.run_quick_evaluation(["q001"], "agentic")

    assert view["status"] == "completed"
    assert view["failure_cases"][0]["failure_type"] == "tool_failure"
    assert "synthetic runner failure" in view["failure_cases"][0]["reason"]


def test_service_returns_failed_view_for_whole_run_failure():
    def broken_loader():
        raise ValueError("evaluation dataset unavailable")

    service = EvaluationDashboardService(question_loader=broken_loader)

    view = service.run_quick_evaluation(["q001"], "agentic")

    assert view["status"] == "failed"
    assert view["raw_report"] == {}
    assert "evaluation dataset unavailable" in view["message"]
```

- [ ] **Step 2: Run the new service tests and verify failure**

Run:

```bash
../../.venv/bin/python -m pytest tests/test_dashboard_service.py -q
```

Expected: tests fail because `EvaluationDashboardService` is missing.

- [ ] **Step 3: Implement quick evaluation orchestration**

Create `evaluation/dashboard_service.py`:

```python
"""Application service for quick evaluation and saved ablation views."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
from typing import Any

from agent.graph import run_agent
from baseline.naive_rag import run_naive_rag
from evaluation.dashboard_formatters import (
    build_ablation_failure_cases,
    build_ablation_summary_rows,
    build_failure_cases,
    build_failure_count_rows,
    build_summary_rows,
    filter_failure_cases,
    get_failure_detail,
    get_runtime_config,
)
from evaluation.dashboard_models import (
    DEFAULT_ABLATION_RESULT_PATH,
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


class JsonAblationResultProvider:
    """Read one saved ablation result without mutating it."""

    def __init__(self, path: str | Path = DEFAULT_ABLATION_RESULT_PATH) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        with self.path.open(encoding="utf-8") as result_file:
            payload = json.load(result_file)
        if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
            raise ValueError("ablation artifact must contain a runs list")
        return payload


class EvaluationDashboardService:
    """Coordinate evaluation runners and normalize dashboard state."""

    def __init__(
        self,
        question_loader: QuestionLoader = load_eval_questions,
        agentic_runner: EvaluationRunner = run_agent,
        naive_runner: EvaluationRunner = run_naive_rag,
        ablation_provider: Any | None = None,
        id_factory: IdFactory | None = None,
    ) -> None:
        self.question_loader = question_loader
        self.agentic_runner = agentic_runner
        self.naive_runner = naive_runner
        self.ablation_provider = (
            ablation_provider or JsonAblationResultProvider()
        )
        self.id_factory = id_factory or _new_run_id

    def list_questions(self) -> list[QuestionOption]:
        questions = self.question_loader()
        return [
            {
                "id": str(question["id"]),
                "label": (
                    f"[{question['id']}] "
                    f"{question.get('question_type', 'unspecified')} - "
                    f"{question['question']}"
                ),
                "question_type": str(
                    question.get("question_type") or "unspecified"
                ),
                "question": str(question["question"]),
            }
            for question in questions
        ]

    def run_quick_evaluation(
        self,
        question_ids: list[str],
        system_mode: DashboardSystemMode,
    ) -> DashboardView:
        try:
            questions = self._select_questions(question_ids)
            if system_mode == "naive":
                report = evaluate_questions(
                    questions,
                    run_agent_fn=self.naive_runner,
                    run_naive_fn=None,
                )
                default_system = "naive"
            elif system_mode == "agentic":
                report = evaluate_questions(
                    questions,
                    run_agent_fn=self.agentic_runner,
                    run_naive_fn=None,
                )
                default_system = "agentic"
            elif system_mode == "comparison":
                report = evaluate_questions(
                    questions,
                    run_agent_fn=self.agentic_runner,
                    run_naive_fn=self.naive_runner,
                )
                default_system = "agentic"
            else:
                raise ValueError(f"Unsupported dashboard system mode: {system_mode}")
        except Exception as exc:
            return _empty_view("failed", f"Evaluation failed: {exc}")

        failure_cases = build_failure_cases(report, default_system)
        return {
            "status": "completed",
            "run_id": self.id_factory("quick"),
            "summary_rows": build_summary_rows(report, default_system),
            "failure_count_rows": build_failure_count_rows(failure_cases),
            "failure_cases": failure_cases,
            "raw_report": report,
            "message": (
                f"Evaluation completed for {len(questions)} question(s). "
                f"Failed cases: {len(failure_cases)}."
            ),
        }

    def filter_failure_cases(
        self,
        dashboard_view: DashboardView,
        system: str | None = None,
        failure_type: str | None = None,
    ) -> list[FailureCaseRow]:
        return filter_failure_cases(
            dashboard_view.get("failure_cases", []),
            system=system,
            failure_type=failure_type,
        )

    def get_failure_detail(
        self,
        dashboard_view: DashboardView,
        case_key: str | None,
    ) -> FailureCaseDetail:
        return get_failure_detail(
            dashboard_view.get("failure_cases", []),
            case_key,
        )

    def get_runtime_config(
        self,
        dashboard_view: DashboardView,
        variant_id: str | None,
    ) -> dict[str, Any]:
        return get_runtime_config(
            dashboard_view.get("raw_report", {}),
            variant_id,
        )

    def _select_questions(
        self,
        question_ids: list[str],
    ) -> list[dict[str, Any]]:
        if not question_ids:
            raise ValueError("Select at least one evaluation question.")
        questions = self.question_loader()
        requested = set(question_ids)
        known = {str(question["id"]) for question in questions}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"Unknown question IDs: {', '.join(unknown)}")
        return [
            question
            for question in questions
            if str(question["id"]) in requested
        ]


def _empty_view(status: str, message: str) -> DashboardView:
    return {
        "status": status,
        "run_id": "",
        "summary_rows": [],
        "failure_count_rows": [],
        "failure_cases": [],
        "raw_report": {},
        "message": message,
    }


def _new_run_id(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    return f"{prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"
```

The snapshot imports in this initial file are intentional; Task 3 completes
their use.

- [ ] **Step 4: Run quick service tests**

Run:

```bash
../../.venv/bin/python -m pytest tests/test_dashboard_service.py -q
```

Expected: all Task 1 and Task 2 tests pass.

- [ ] **Step 5: Commit the quick evaluation service**

```bash
git add evaluation/dashboard_service.py tests/test_dashboard_service.py
git commit -m "feat: add quick evaluation dashboard service"
```

---

### Task 3: Read-Only Ablation Snapshot and Legacy Diagnostics

**Files:**
- Modify: `evaluation/dashboard_service.py`
- Modify: `tests/test_dashboard_service.py`

- [ ] **Step 1: Add failing snapshot and compatibility tests**

Append to `tests/test_dashboard_service.py`:

```python
from evaluation.dashboard_service import JsonAblationResultProvider


class StaticProvider:
    def __init__(self, payload):
        self.payload = payload

    def load(self):
        return self.payload


def _ablation_payload(result: dict) -> dict:
    return {
        "kind": "ablation_result",
        "runs": [
            {
                "id": "v0_naive",
                "method": "Naive RAG",
                "status": "completed",
                "runtime_config": {"llm": {"model": "test-model"}},
                "summary": _summary(),
                "results": [result],
            }
        ],
    }


def test_load_ablation_snapshot_uses_stored_failure_analysis():
    payload = _ablation_payload(_result("q001", "retrieval_failure"))
    service = EvaluationDashboardService(
        question_loader=lambda: [_question("q001")],
        ablation_provider=StaticProvider(payload),
        id_factory=lambda prefix: f"{prefix}-fixed",
    )

    view = service.load_ablation_snapshot()

    assert view["status"] == "completed"
    assert view["run_id"] == "snapshot-fixed"
    assert view["failure_cases"][0]["diagnostics_source"] == "stored"
    assert service.get_runtime_config(view, "v0_naive") == {
        "llm": {"model": "test-model"}
    }


def test_load_ablation_snapshot_derives_missing_diagnostics_without_mutation():
    legacy_result = {
        **_result("q001", "retrieval_failure"),
        "failure_analysis": None,
        "correct": False,
        "source_hit": False,
        "context_relevant": False,
        "citation_hit": False,
        "fallback_correct": True,
        "fallback_triggered": False,
        "answer_returned": True,
        "retry_count": 0,
        "unsupported_claim_count": 0,
        "retrieved_documents": [{"source": "other.md"}],
        "relevant_documents": [],
        "citations": [],
        "error": None,
    }
    payload = _ablation_payload(legacy_result)
    original = json.loads(json.dumps(payload))
    service = EvaluationDashboardService(
        question_loader=lambda: [_question("q001")],
        ablation_provider=StaticProvider(payload),
    )

    view = service.load_ablation_snapshot()

    assert view["failure_cases"][0]["failure_type"] == "retrieval_failure"
    assert view["failure_cases"][0]["diagnostics_source"] == "derived"
    assert payload == original


def test_load_ablation_snapshot_keeps_valid_runs_when_one_run_is_malformed():
    valid = _ablation_payload(_result("q001", "retrieval_failure"))["runs"][0]
    payload = {"runs": ["invalid", valid]}
    service = EvaluationDashboardService(
        question_loader=lambda: [_question("q001")],
        ablation_provider=StaticProvider(payload),
    )

    view = service.load_ablation_snapshot()

    assert view["status"] == "completed"
    assert len(view["summary_rows"]) == 1
    assert "partial" in view["message"].lower()


def test_load_ablation_snapshot_returns_unavailable_for_missing_artifact(tmp_path):
    service = EvaluationDashboardService(
        ablation_provider=JsonAblationResultProvider(tmp_path / "missing.json"),
    )

    view = service.load_ablation_snapshot()

    assert view["status"] == "unavailable"
    assert "missing.json" in view["message"]


def test_json_provider_does_not_rewrite_artifact(tmp_path):
    path = tmp_path / "ablation.json"
    payload = _ablation_payload(_result("q001", "retrieval_failure"))
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    before = path.read_bytes()

    loaded = JsonAblationResultProvider(path).load()

    assert loaded == payload
    assert path.read_bytes() == before
```

- [ ] **Step 2: Run snapshot tests and verify failure**

Run:

```bash
../../.venv/bin/python -m pytest tests/test_dashboard_service.py -q
```

Expected: tests fail because `load_ablation_snapshot()` is missing.

- [ ] **Step 3: Implement snapshot loading and in-memory enrichment**

Add these methods to `EvaluationDashboardService`:

```python
    def load_ablation_snapshot(self) -> DashboardView:
        try:
            payload = self.ablation_provider.load()
            questions_by_id = {
                str(question["id"]): question
                for question in self.question_loader()
            }
            enriched, skipped = _enrich_ablation_payload(
                payload,
                questions_by_id,
            )
        except Exception as exc:
            return _empty_view(
                "unavailable",
                f"Ablation snapshot unavailable: {exc}",
            )

        failure_cases = build_ablation_failure_cases(enriched)
        message = (
            f"Loaded {len(enriched['runs'])} ablation variant(s). "
            f"Failed cases: {len(failure_cases)}."
        )
        if skipped:
            message += f" Snapshot is partial; skipped {skipped} malformed item(s)."
        return {
            "status": "completed",
            "run_id": self.id_factory("snapshot"),
            "summary_rows": build_ablation_summary_rows(enriched),
            "failure_count_rows": build_failure_count_rows(failure_cases),
            "failure_cases": failure_cases,
            "raw_report": enriched,
            "message": message,
        }
```

Add these module helpers below `_new_run_id()`:

```python
def _enrich_ablation_payload(
    payload: Mapping[str, Any],
    questions_by_id: Mapping[str, dict[str, Any]],
) -> tuple[dict[str, Any], int]:
    enriched = deepcopy(dict(payload))
    valid_runs: list[dict[str, Any]] = []
    skipped = 0
    runs = enriched.get("runs")
    if not isinstance(runs, list):
        raise ValueError("ablation artifact must contain a runs list")

    for raw_run in runs:
        if not isinstance(raw_run, dict):
            skipped += 1
            continue
        run = raw_run
        results = run.get("results")
        if not isinstance(results, list):
            run["results"] = []
            skipped += 1
            valid_runs.append(run)
            continue

        valid_results: list[dict[str, Any]] = []
        for raw_result in results:
            if not isinstance(raw_result, dict):
                skipped += 1
                continue
            result = raw_result
            analysis = result.get("failure_analysis")
            if isinstance(analysis, Mapping):
                result["_diagnostics_source"] = "stored"
            else:
                question_id = str(result.get("question_id") or "")
                question = questions_by_id.get(question_id)
                if question is None:
                    result["_diagnostics_source"] = "unavailable"
                else:
                    result["failure_analysis"] = analyze_failure(question, result)
                    result["_diagnostics_source"] = "derived"
            valid_results.append(result)
        run["results"] = valid_results
        valid_runs.append(run)

    enriched["runs"] = valid_runs
    return enriched, skipped
```

The `deepcopy()` is required: compatibility enrichment must never mutate the
provider payload or write the historical artifact.

- [ ] **Step 4: Run dashboard service tests**

Run:

```bash
../../.venv/bin/python -m pytest tests/test_dashboard_service.py -q
```

Expected: all dashboard service tests pass.

- [ ] **Step 5: Run existing evaluation and ablation regressions**

Run:

```bash
../../.venv/bin/python -m pytest \
  tests/test_evaluate.py \
  tests/test_failure_analyzer.py \
  tests/test_ablation.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit snapshot support**

```bash
git add evaluation/dashboard_service.py tests/test_dashboard_service.py
git commit -m "feat: add ablation snapshot dashboard provider"
```

---

### Task 4: Gradio Event Adapters

**Files:**
- Modify: `ui/gradio_app.py`
- Modify: `tests/test_gradio_app.py`

- [ ] **Step 1: Add failing event-adapter tests**

Extend imports in `tests/test_gradio_app.py`:

```python
from ui.gradio_app import (
    answer_question,
    build_document_index,
    filter_dashboard_failures,
    format_failure_detail,
    question_selection,
    run_dashboard_evaluation,
)
```

Append:

```python
class FakeDashboardService:
    def __init__(self, view):
        self.view = view
        self.run_calls = []
        self.filter_calls = []

    def run_quick_evaluation(self, question_ids, system_mode):
        self.run_calls.append((question_ids, system_mode))
        return self.view

    def filter_failure_cases(self, view, system=None, failure_type=None):
        self.filter_calls.append((view, system, failure_type))
        return view["failure_cases"]

    def get_failure_detail(self, view, case_key):
        for case in view["failure_cases"]:
            if case["case_key"] == case_key:
                return {
                    "case_key": case_key,
                    "title": "Agentic RAG / q001 / retrieval_failure",
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


def _dashboard_view(status="completed"):
    return {
        "status": status,
        "run_id": "quick-fixed",
        "summary_rows": [["Agentic RAG", 0.8, 0.9, 0.7, 1.0, 0, 2.5, 0.4]],
        "failure_count_rows": [["Agentic RAG", "retrieval_failure", 1]],
        "failure_cases": [
            {
                "case_key": "agentic:q001",
                "system": "agentic",
                "system_label": "Agentic RAG",
                "question_id": "q001",
                "question_type": "single_doc",
                "question": "Question q001",
                "failure_type": "retrieval_failure",
                "reason": "Expected source missing.",
                "suggestion": "Tune retrieval.",
                "diagnostics_source": "stored",
            }
        ],
        "raw_report": {},
        "message": "Evaluation completed.",
    }


def test_question_selection_returns_smoke_and_all_ids():
    options = [{"id": "q001"}, {"id": "q002"}, {"id": "q016"}]

    assert question_selection(options, "smoke") == ["q001", "q016"]
    assert question_selection(options, "all") == ["q001", "q002", "q016"]


def test_run_dashboard_evaluation_returns_visible_rows_and_state():
    view = _dashboard_view()
    service = FakeDashboardService(view)

    result = run_dashboard_evaluation(
        "Agentic RAG",
        ["q001"],
        {},
        service=service,
    )

    state, status, metrics, counts, cases, case_update = result
    assert state == view
    assert "completed" in status.lower()
    assert metrics == view["summary_rows"]
    assert counts == view["failure_count_rows"]
    assert cases[0][0] == "agentic:q001"
    assert case_update["choices"] == [
        ("q001 / retrieval_failure", "agentic:q001")
    ]
    assert service.run_calls == [(["q001"], "agentic")]


def test_failed_run_preserves_previous_successful_state():
    previous = _dashboard_view()
    failed = _dashboard_view(status="failed")
    failed["message"] = "Evaluation failed: unavailable"
    failed["summary_rows"] = []
    failed["failure_cases"] = []
    service = FakeDashboardService(failed)

    state, status, metrics, counts, cases, choices = run_dashboard_evaluation(
        "Agentic RAG",
        ["q001"],
        previous,
        service=service,
    )

    assert state == previous
    assert "failed" in status.lower()
    assert metrics == previous["summary_rows"]
    assert choices["choices"] == [
        ("q001 / retrieval_failure", "agentic:q001")
    ]


def test_filter_and_detail_helpers_do_not_run_evaluation_again():
    view = _dashboard_view()
    service = FakeDashboardService(view)

    counts, table, choices = filter_dashboard_failures(
        view,
        "agentic",
        "retrieval_failure",
        service=service,
    )
    detail = format_failure_detail(
        view,
        "agentic:q001",
        service=service,
    )

    assert table[0][0] == "agentic:q001"
    assert counts == [["Agentic RAG", "retrieval_failure", 1]]
    assert choices["choices"] == [
        ("q001 / retrieval_failure", "agentic:q001")
    ]
    assert "Expected source missing." in detail
    assert service.run_calls == []
    assert service.filter_calls
```

- [ ] **Step 2: Run UI helper tests and verify failure**

Run:

```bash
../../.venv/bin/python -m pytest tests/test_gradio_app.py -q
```

Expected: collection fails because the new helpers do not exist.

- [ ] **Step 3: Implement thin UI adapters**

Add these imports to `ui/gradio_app.py`:

```python
from collections.abc import Mapping

from evaluation.dashboard_formatters import (
    build_failure_count_rows,
    failure_cases_to_table,
)
from evaluation.dashboard_models import (
    FAILURE_CASE_COLUMNS,
    FAILURE_COUNT_COLUMNS,
    METRIC_COLUMNS,
    SMOKE_QUESTION_IDS,
)
from evaluation.dashboard_service import EvaluationDashboardService
```

Add these constants and helpers before `create_app()`:

```python
SYSTEM_MODE_VALUES = {
    "Naive RAG": "naive",
    "Agentic RAG": "agentic",
    "Compare Both": "comparison",
}


def question_selection(
    question_options: list[Mapping[str, Any]],
    selection: str,
) -> list[str]:
    """Return smoke or complete question IDs in dataset order."""

    all_ids = [str(option["id"]) for option in question_options]
    if selection == "all":
        return all_ids
    smoke = set(SMOKE_QUESTION_IDS)
    return [question_id for question_id in all_ids if question_id in smoke]


def run_dashboard_evaluation(
    system_label: str,
    question_ids: list[str] | None,
    previous_state: dict[str, Any] | None,
    service: EvaluationDashboardService | None = None,
) -> tuple[
    dict[str, Any],
    str,
    list,
    list,
    list,
    dict[str, Any],
]:
    """Run quick evaluation and preserve the last successful view on failure."""

    resolved_service = service or EvaluationDashboardService()
    mode = SYSTEM_MODE_VALUES.get(system_label, "comparison")
    view = resolved_service.run_quick_evaluation(list(question_ids or []), mode)
    active_view = (
        view
        if view["status"] == "completed"
        else dict(previous_state or view)
    )
    cases = active_view.get("failure_cases", [])
    return (
        active_view,
        view["message"],
        active_view.get("summary_rows", []),
        active_view.get("failure_count_rows", []),
        failure_cases_to_table(cases),
        gr.update(choices=_failure_choices(cases), value=None),
    )


def filter_dashboard_failures(
    dashboard_state: dict[str, Any] | None,
    system: str | None,
    failure_type: str | None,
    service: EvaluationDashboardService | None = None,
) -> tuple[list[list[Any]], list[list[str]], dict[str, Any]]:
    """Filter the current dashboard state without rerunning evaluation."""

    resolved_service = service or EvaluationDashboardService()
    state = dict(dashboard_state or {})
    cases = resolved_service.filter_failure_cases(
        state,
        system=system,
        failure_type=failure_type,
    )
    return (
        build_failure_count_rows(cases),
        failure_cases_to_table(cases),
        gr.update(choices=_failure_choices(cases), value=None),
    )


def format_failure_detail(
    dashboard_state: dict[str, Any] | None,
    case_key: str | None,
    service: EvaluationDashboardService | None = None,
) -> str:
    """Format one failure detail as concise Markdown."""

    resolved_service = service or EvaluationDashboardService()
    detail = resolved_service.get_failure_detail(
        dict(dashboard_state or {}),
        case_key,
    )
    if not detail["case_key"]:
        return "Select a failed case to inspect its diagnosis."
    return (
        f"### {detail['title']}\n\n"
        f"**Reason:** {detail['reason']}\n\n"
        f"**Suggestion:** {detail['suggestion']}\n\n"
        f"**Diagnostics source:** `{detail['diagnostics_source']}`"
    )


def _failure_choices(cases: list[Mapping[str, Any]]) -> list[tuple[str, str]]:
    return [
        (
            f"{case['question_id']} / {case['failure_type']}",
            str(case["case_key"]),
        )
        for case in cases
    ]
```

- [ ] **Step 4: Run Gradio helper tests**

Run:

```bash
../../.venv/bin/python -m pytest tests/test_gradio_app.py -q
```

Expected: existing Document QA tests and new event-adapter tests pass.

- [ ] **Step 5: Commit event adapters**

```bash
git add ui/gradio_app.py tests/test_gradio_app.py
git commit -m "feat: add evaluation dashboard event adapters"
```

---

### Task 5: Build the Gradio Evaluation Interface

**Files:**
- Modify: `ui/gradio_app.py`
- Modify: `tests/test_gradio_app.py`

- [ ] **Step 1: Add failing app-structure tests**

Append to `tests/test_gradio_app.py`:

```python
from ui.gradio_app import (
    format_variant_runtime_config,
    load_ablation_dashboard,
)


class FakeSnapshotService(FakeDashboardService):
    def load_ablation_snapshot(self):
        return self.view

    def get_runtime_config(self, view, variant_id):
        return view["raw_report"]["runs"][0]["runtime_config"]


def test_snapshot_helpers_return_dropdown_updates_and_runtime_config():
    view = _dashboard_view()
    view["run_id"] = "snapshot-fixed"
    view["raw_report"] = {
        "runs": [
            {
                "id": "v0_naive",
                "method": "Naive RAG",
                "runtime_config": {"llm": {"model": "test-model"}},
            }
        ]
    }
    service = FakeSnapshotService(view)

    result = load_ablation_dashboard({}, service=service)
    state, status, metrics, counts, cases, case_update, variant_update = result

    assert state == view
    assert "completed" in status.lower()
    assert metrics == view["summary_rows"]
    assert counts == view["failure_count_rows"]
    assert cases[0][0] == "agentic:q001"
    assert case_update["choices"][0][1] == "agentic:q001"
    assert variant_update["choices"] == [
        ("All variants", "all"),
        ("v0_naive Naive RAG", "v0_naive"),
    ]
    assert format_variant_runtime_config(
        view,
        "v0_naive",
        service=service,
    ) == {"llm": {"model": "test-model"}}


def test_snapshot_helper_preserves_previous_state_when_refresh_is_unavailable():
    previous = _dashboard_view()
    unavailable = _dashboard_view(status="unavailable")
    unavailable["message"] = "Ablation snapshot unavailable: missing artifact"
    unavailable["summary_rows"] = []
    unavailable["failure_cases"] = []
    service = FakeSnapshotService(unavailable)

    state, status, metrics, counts, cases, case_update, variant_update = (
        load_ablation_dashboard(previous, service=service)
    )

    assert state == previous
    assert "unavailable" in status.lower()
    assert metrics == previous["summary_rows"]
    assert cases[0][0] == "agentic:q001"
    assert case_update["choices"][0][1] == "agentic:q001"
    assert variant_update["choices"] == [("All variants", "all")]


def test_create_app_contains_document_and_evaluation_tabs():
    from ui.gradio_app import create_app

    app = create_app()
    config = app.get_config_file()
    labels = {
        component.get("props", {}).get("label")
        for component in config.get("components", [])
    }

    assert "Document QA" in labels
    assert "Evaluation" in labels
    assert "Quick Compare" in labels
    assert "Ablation Snapshot" in labels


def test_create_app_contains_dashboard_tables_and_filters():
    from ui.gradio_app import create_app

    app = create_app()
    config = app.get_config_file()
    labels = {
        component.get("props", {}).get("label")
        for component in config.get("components", [])
    }

    assert "Evaluation questions" in labels
    assert "Reliability metrics" in labels
    assert "Failure type counts" in labels
    assert "Failed cases" in labels
    assert "Failure case" in labels
```

- [ ] **Step 2: Run app-structure tests and verify failure**

Run:

```bash
../../.venv/bin/python -m pytest tests/test_gradio_app.py -q
```

Expected: the new label assertions fail against the current single-page app.

- [ ] **Step 3: Add snapshot UI adapters**

Add these functions to `ui/gradio_app.py`:

```python
def load_ablation_dashboard(
    previous_state: dict[str, Any] | None,
    service: EvaluationDashboardService | None = None,
) -> tuple[
    dict[str, Any],
    str,
    list,
    list,
    list,
    dict[str, Any],
    dict[str, Any],
]:
    """Load the saved ablation snapshot and preserve the last valid view."""

    resolved_service = service or EvaluationDashboardService()
    view = resolved_service.load_ablation_snapshot()
    active_view = (
        view
        if view["status"] == "completed"
        else dict(previous_state or view)
    )
    cases = active_view.get("failure_cases", [])
    runs = active_view.get("raw_report", {}).get("runs", [])
    variant_choices = [
        (f"{run.get('id')} {run.get('method')}", str(run.get("id")))
        for run in runs
        if isinstance(run, dict) and run.get("id")
    ]
    return (
        active_view,
        view["message"],
        active_view.get("summary_rows", []),
        active_view.get("failure_count_rows", []),
        failure_cases_to_table(cases),
        gr.update(choices=_failure_choices(cases), value=None),
        gr.update(
            choices=[("All variants", "all"), *variant_choices],
            value="all",
        ),
    )


def format_variant_runtime_config(
    dashboard_state: dict[str, Any] | None,
    variant_id: str | None,
    service: EvaluationDashboardService | None = None,
) -> dict[str, Any]:
    """Return saved runtime config for one selected ablation variant."""

    resolved_service = service or EvaluationDashboardService()
    return resolved_service.get_runtime_config(
        dict(dashboard_state or {}),
        variant_id,
    )
```

- [ ] **Step 4: Refactor `create_app()` into two tabs**

Replace the current `create_app()` body with a small composition root:

```python
def create_app() -> gr.Blocks:
    """Create the Gradio interface."""

    settings = get_settings()
    dashboard_service = EvaluationDashboardService()
    question_options = dashboard_service.list_questions()
    question_choices = [
        (option["label"], option["id"])
        for option in question_options
    ]
    smoke_ids = question_selection(question_options, "smoke")

    with gr.Blocks(
        title="Reliability-oriented Agentic RAG",
        css="""
        .dashboard-status { min-height: 2.5rem; }
        .dashboard-table { min-height: 12rem; }
        .dashboard-detail { min-height: 9rem; }
        """,
    ) as demo:
        gr.Markdown(
            "# Reliability-oriented Agentic RAG Document QA System"
        )
        with gr.Tabs():
            with gr.Tab("Document QA"):
                _build_document_qa_tab(settings)
            with gr.Tab("Evaluation"):
                _build_evaluation_tab(
                    dashboard_service,
                    question_options,
                    question_choices,
                    smoke_ids,
                )
    return demo
```

Move the current Document QA components and events unchanged into:

```python
def _build_document_qa_tab(settings: Any) -> None:
    """Build the existing upload, indexing, and QA workflow."""

    chat_history = gr.State([])
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Document Indexing")
            uploaded_files = gr.File(
                label="Upload documents",
                file_count="multiple",
                file_types=[".pdf", ".md", ".markdown", ".txt"],
            )
            build_button = gr.Button("Build Index", variant="primary")
            index_status = gr.Textbox(
                label="Index status",
                value="Upload documents, then build the index.",
                interactive=False,
            )
        with gr.Column():
            gr.Markdown("## Question Answering")
            question = gr.Textbox(label="Question", lines=3)
            ask_button = gr.Button("Ask", variant="primary")
            answer = gr.Textbox(label="Answer", lines=6, interactive=False)

    with gr.Row():
        citations = gr.JSON(label="Citations")
        retrieved_chunks = gr.JSON(label="Retrieved chunks")
    with gr.Row():
        rewritten_question = gr.Textbox(
            label="Rewritten question",
            interactive=False,
        )
        rewrite_count = gr.Number(
            label="Retry count",
            value=0,
            precision=0,
            interactive=False,
        )
    diagnostics = gr.Markdown(
        value=_format_diagnostics(0, 0, 0),
    )

    build_button.click(
        fn=build_document_index,
        inputs=uploaded_files,
        outputs=index_status,
    )
    ask_button.click(
        fn=answer_question,
        inputs=[question, chat_history],
        outputs=[
            answer,
            citations,
            retrieved_chunks,
            rewritten_question,
            rewrite_count,
            diagnostics,
            chat_history,
        ],
    )
    gr.Markdown(
        f"""
        **Current configuration**

        - LLM model: `{settings.openai_model}`
        - Embedding model: `{settings.embedding_model}`
        - Chroma path: `{settings.chroma_persist_dir}`
        - Top K: `{settings.top_k}`
        - Max retry count: `{settings.max_retry_count}`
        """
    )
```

Create `_build_evaluation_tab()` with two sub-tabs and separate states:

```python
def _build_evaluation_tab(
    service: EvaluationDashboardService,
    question_options: list[Mapping[str, Any]],
    question_choices: list[tuple[str, str]],
    smoke_ids: list[str],
) -> None:
    """Build quick evaluation and saved ablation views."""

    quick_state = gr.State({})
    snapshot_state = gr.State({})

    with gr.Tabs():
        with gr.Tab("Quick Compare"):
            with gr.Row():
                system_mode = gr.Radio(
                    choices=list(SYSTEM_MODE_VALUES),
                    value="Compare Both",
                    label="System mode",
                )
                questions = gr.Dropdown(
                    choices=question_choices,
                    value=smoke_ids,
                    multiselect=True,
                    label="Evaluation questions",
                    info="The 5-question smoke set is selected by default.",
                )
            with gr.Row():
                smoke_button = gr.Button("Select smoke set")
                all_button = gr.Button("Select all 36")
                run_button = gr.Button("Run Evaluation", variant="primary")
            quick_status = gr.Markdown(
                "Select a system mode and questions.",
                elem_classes="dashboard-status",
            )
            quick_metrics = gr.Dataframe(
                headers=METRIC_COLUMNS,
                value=[],
                label="Reliability metrics",
                interactive=False,
                wrap=True,
            )
            with gr.Row():
                quick_counts = gr.Dataframe(
                    headers=FAILURE_COUNT_COLUMNS,
                    value=[],
                    label="Failure type counts",
                    interactive=False,
                )
                with gr.Column():
                    quick_system_filter = gr.Dropdown(
                        choices=[
                            ("All systems", "all"),
                            ("Naive RAG", "naive"),
                            ("Agentic RAG", "agentic"),
                        ],
                        value="all",
                        label="Failure system",
                    )
                    quick_type_filter = gr.Dropdown(
                        choices=[
                            ("All failure types", "all"),
                            "retrieval_failure",
                            "reranking_failure",
                            "query_rewrite_failure",
                            "generation_failure",
                            "citation_failure",
                            "fallback_failure",
                            "tool_failure",
                        ],
                        value="all",
                        label="Failure type",
                    )
            quick_cases = gr.Dataframe(
                headers=FAILURE_CASE_COLUMNS,
                value=[],
                label="Failed cases",
                interactive=False,
                wrap=True,
                elem_classes="dashboard-table",
            )
            quick_case = gr.Dropdown(
                choices=[],
                label="Failure case",
            )
            quick_detail = gr.Markdown(
                "Select a failed case to inspect its diagnosis.",
                elem_classes="dashboard-detail",
            )

            smoke_button.click(
                fn=lambda: question_selection(question_options, "smoke"),
                outputs=questions,
                queue=False,
            )
            all_button.click(
                fn=lambda: question_selection(question_options, "all"),
                outputs=questions,
                queue=False,
            )
            run_event = run_button.click(
                fn=lambda: (
                    gr.update(interactive=False),
                    "Evaluation is running. A 36-question run may take several minutes.",
                ),
                outputs=[run_button, quick_status],
                queue=False,
            )
            run_event.then(
                fn=lambda mode, ids, state: run_dashboard_evaluation(
                    mode,
                    ids,
                    state,
                    service=service,
                ),
                inputs=[system_mode, questions, quick_state],
                outputs=[
                    quick_state,
                    quick_status,
                    quick_metrics,
                    quick_counts,
                    quick_cases,
                    quick_case,
                ],
            ).then(
                fn=lambda: gr.update(interactive=True),
                outputs=run_button,
                queue=False,
            )
            for component in (quick_system_filter, quick_type_filter):
                component.change(
                    fn=lambda state, system, failure: filter_dashboard_failures(
                        state,
                        system,
                        failure,
                        service=service,
                    ),
                    inputs=[
                        quick_state,
                        quick_system_filter,
                        quick_type_filter,
                    ],
                    outputs=[quick_counts, quick_cases, quick_case],
                    queue=False,
                )
            quick_case.change(
                fn=lambda state, key: format_failure_detail(
                    state,
                    key,
                    service=service,
                ),
                inputs=[quick_state, quick_case],
                outputs=quick_detail,
                queue=False,
            )

        with gr.Tab("Ablation Snapshot"):
            gr.Markdown(
                "This view reads the saved V0-V6 artifact and does not rerun models."
            )
            refresh_snapshot = gr.Button("Refresh Snapshot", variant="primary")
            snapshot_status = gr.Markdown(
                "Load the saved ablation artifact.",
                elem_classes="dashboard-status",
            )
            snapshot_metrics = gr.Dataframe(
                headers=METRIC_COLUMNS,
                value=[],
                label="Ablation reliability metrics",
                interactive=False,
                wrap=True,
            )
            snapshot_variant = gr.Dropdown(
                choices=[("All variants", "all")],
                value="all",
                label="Ablation variant",
            )
            snapshot_type_filter = gr.Dropdown(
                choices=[
                    ("All failure types", "all"),
                    "retrieval_failure",
                    "reranking_failure",
                    "query_rewrite_failure",
                    "generation_failure",
                    "citation_failure",
                    "fallback_failure",
                    "tool_failure",
                ],
                value="all",
                label="Ablation failure type",
            )
            snapshot_config = gr.JSON(label="Runtime configuration")
            snapshot_counts = gr.Dataframe(
                headers=FAILURE_COUNT_COLUMNS,
                value=[],
                label="Ablation failure type counts",
                interactive=False,
            )
            snapshot_cases = gr.Dataframe(
                headers=FAILURE_CASE_COLUMNS,
                value=[],
                label="Ablation failed cases",
                interactive=False,
                wrap=True,
                elem_classes="dashboard-table",
            )
            snapshot_case = gr.Dropdown(
                choices=[],
                label="Ablation failure case",
            )
            snapshot_detail = gr.Markdown(
                "Select a failed case to inspect stored or derived diagnostics.",
                elem_classes="dashboard-detail",
            )

            refresh_snapshot.click(
                fn=lambda state: load_ablation_dashboard(
                    state,
                    service=service,
                ),
                inputs=snapshot_state,
                outputs=[
                    snapshot_state,
                    snapshot_status,
                    snapshot_metrics,
                    snapshot_counts,
                    snapshot_cases,
                    snapshot_case,
                    snapshot_variant,
                ],
            )
            snapshot_variant.change(
                fn=lambda state, variant: format_variant_runtime_config(
                    state,
                    variant,
                    service=service,
                ),
                inputs=[snapshot_state, snapshot_variant],
                outputs=snapshot_config,
                queue=False,
            )
            for component in (snapshot_variant, snapshot_type_filter):
                component.change(
                    fn=lambda state, variant, failure: filter_dashboard_failures(
                        state,
                        variant,
                        failure,
                        service=service,
                    ),
                    inputs=[
                        snapshot_state,
                        snapshot_variant,
                        snapshot_type_filter,
                    ],
                    outputs=[
                        snapshot_counts,
                        snapshot_cases,
                        snapshot_case,
                    ],
                    queue=False,
                )
            snapshot_case.change(
                fn=lambda state, key: format_failure_detail(
                    state,
                    key,
                    service=service,
                ),
                inputs=[snapshot_state, snapshot_case],
                outputs=snapshot_detail,
                queue=False,
            )

            gr.Markdown(
                "Future upgrade: background live V0-V6 runs with progress, "
                "cancellation, checkpoint recovery, shared run IDs, and trace linkage."
            )
```

All dynamic Dropdown choice changes use
`gr.update(choices=choices, value=None)` at the event-adapter boundary.

- [ ] **Step 5: Run Gradio tests**

Run:

```bash
../../.venv/bin/python -m pytest tests/test_gradio_app.py -q
```

Expected: all Gradio tests pass.

- [ ] **Step 6: Run focused dashboard tests**

Run:

```bash
../../.venv/bin/python -m pytest \
  tests/test_dashboard_service.py \
  tests/test_gradio_app.py \
  -q
```

Expected: all focused tests pass.

- [ ] **Step 7: Commit the Evaluation tab**

```bash
git add ui/gradio_app.py tests/test_gradio_app.py
git commit -m "feat: add Gradio evaluation dashboard"
```

---

### Task 6: Documentation and Version Metadata

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `api/main.py`

- [ ] **Step 1: Add a failing version assertion**

Append to `tests/test_fastapi_routes.py`:

```python
def test_api_reports_p4b_version():
    from api.main import create_app

    assert create_app().version == "0.4.1-p4b"
```

- [ ] **Step 2: Run the version test and verify failure**

Run:

```bash
../../.venv/bin/python -m pytest \
  tests/test_fastapi_routes.py::test_api_reports_p4b_version \
  -q
```

Expected: fails because the API still reports `0.4.0-p4a`.

- [ ] **Step 3: Bump version and update release notes**

Change `api/main.py`:

```python
    app = FastAPI(
        title="Reliability-oriented Agentic RAG API",
        version="0.4.1-p4b",
    )
```

Add this section at the top of `CHANGELOG.md`:

```markdown
## v0.4.1-p4b - Evaluation Dashboard

Date: 2026-06-14

### Added

- Added a Gradio Evaluation tab with five-question smoke evaluation, manual
  36-question selection, and Naive/Agentic comparison modes.
- Added metric, failure-count, failed-case filtering, and selected-case views.
- Added a read-only V0-V6 ablation snapshot view with saved runtime configs.
- Added transparent in-memory failure diagnostics for pre-P4a artifacts,
  labeled as derived and never written back to the source artifact.

### Notes

- Quick evaluation is synchronous and may take several minutes for all
  36 questions.
- The ablation view reads existing artifacts; it does not run V0-V6 from the
  browser.
- Background ablation jobs, progress, cancellation, trace linkage, and
  historical trends remain roadmap items.
```

- [ ] **Step 4: Document dashboard usage and roadmap**

Update the Gradio usage section in `README.md` so it includes:

```markdown
The Gradio application contains two main tabs:

- `Document QA`: upload, index, ask questions, and inspect citations and
  retrieval diagnostics.
- `Evaluation`: run a five-question smoke evaluation or select any of the
  36 local evaluation questions.

`Evaluation > Quick Compare` supports Naive RAG, Agentic RAG, and paired
comparison on the same selected records. It shows correctness, context
relevance, citation accuracy, fallback accuracy, unsupported claims, average
latency, average retries, failure counts, and deterministic failed-case
diagnostics.

`Evaluation > Ablation Snapshot` reads
`experiments/results/ablation_result.json`. Historical artifacts without P4a
failure fields are enriched only in memory with the deterministic analyzer and
are labeled `derived`; the source JSON is not modified.
```

Move the interactive dashboard item from `Next Milestones` to `Completed Work`
and add these explicit future items:

```markdown
- Add `BackgroundAblationRunner` for live V0-V6 execution.
- Add background progress, cancellation, and checkpoint recovery.
- Share evaluation run IDs across Gradio and FastAPI.
- Link failed cases to `trace_id` and a node-level trace viewer.
- Add prompt versioning and prompt regression comparisons.
- Add historical evaluation runs and metric trend comparison.
```

- [ ] **Step 5: Run version and documentation checks**

Run:

```bash
../../.venv/bin/python -m pytest \
  tests/test_fastapi_routes.py::test_api_reports_p4b_version \
  -q
rg -n "v0.4.1-p4b|Evaluation Dashboard|BackgroundAblationRunner|derived" \
  README.md CHANGELOG.md api/main.py
git diff --check
```

Expected: the test passes, all required documentation terms are present, and
`git diff --check` prints no errors.

- [ ] **Step 6: Commit documentation and metadata**

```bash
git add README.md CHANGELOG.md api/main.py tests/test_fastapi_routes.py
git commit -m "docs: publish p4b evaluation dashboard"
```

---

### Task 7: Full Verification and Browser QA

**Files:**
- Modify only if verification exposes a defect in files already listed above.

- [ ] **Step 1: Run focused dashboard and regression tests**

Run:

```bash
../../.venv/bin/python -m pytest \
  tests/test_dashboard_service.py \
  tests/test_gradio_app.py \
  tests/test_evaluate.py \
  tests/test_failure_analyzer.py \
  tests/test_ablation.py \
  tests/test_fastapi_routes.py \
  -q
```

Expected: all focused and adjacent regression tests pass.

- [ ] **Step 2: Run the full repository suite**

Run:

```bash
../../.venv/bin/python -m pytest -q
```

Expected: all tests pass with no failures.

- [ ] **Step 3: Start the Gradio application**

Run from the worktree:

```bash
GRADIO_SERVER_NAME=127.0.0.1 \
GRADIO_SERVER_PORT=7861 \
../../.venv/bin/python app.py
```

Expected: Gradio reports a local URL at `http://127.0.0.1:7861`.

- [ ] **Step 4: Verify the UI in the in-app browser**

Use the Browser plugin to inspect `http://127.0.0.1:7861` at:

- desktop viewport around 1440 by 900
- narrow viewport around 390 by 844

Verify:

- `Document QA` still renders upload, indexing, QA, citations, and diagnostics
- `Evaluation` renders `Quick Compare` and `Ablation Snapshot`
- the smoke set contains q001, q016, q027, q030, and q033
- selecting all exposes 36 IDs
- snapshot refresh renders V0-V6 rows from the saved artifact
- derived diagnostics are visibly labeled
- long question, reason, suggestion, and error text wraps without overlap
- buttons, dropdowns, and tables remain usable at the narrow viewport

- [ ] **Step 5: Exercise a deterministic dashboard flow**

Before any paid model call, use injected fake runners through the focused tests
or a temporary REPL invocation to verify state transitions. Then, only when
the configured model and index are available, run the five-question smoke
comparison once and confirm:

- status changes from running to completed
- the run button becomes interactive again
- metric and failure rows render
- changing filters does not trigger another model run
- a failed-case selection renders reason, suggestion, and diagnostics source

- [ ] **Step 6: Stop the development server**

Stop the Gradio process and verify no required test or server process remains
running.

- [ ] **Step 7: Inspect final diff and history**

Run:

```bash
git status --short --branch
git log --oneline --decorate -8
git diff main...HEAD --stat
git diff --check main...HEAD
```

Expected:

- only intentional P4b files differ from `main`
- the worktree is clean
- all commits are scoped and readable
- no whitespace errors are reported

- [ ] **Step 8: Create a final verification commit only if needed**

If browser or full-suite verification required a corrective change:

```bash
git add evaluation ui tests README.md CHANGELOG.md api/main.py
git commit -m "fix: finalize p4b dashboard verification"
```

If no corrective change was needed, do not create an empty commit.

---

## Completion Checklist

- [ ] Quick evaluation supports Naive, Agentic, and Compare Both.
- [ ] The five-question smoke set is the default.
- [ ] All 36 questions can be selected.
- [ ] Evaluation metrics come from `evaluation.evaluate`.
- [ ] Failure diagnostics come from stored P4a results or transparent in-memory derivation.
- [ ] Filters operate on `gr.State` without rerunning models.
- [ ] Whole-run failures preserve the last successful dashboard state.
- [ ] Missing ablation artifacts do not break quick evaluation.
- [ ] Historical ablation JSON is never rewritten.
- [ ] Document QA behavior remains intact.
- [ ] Focused and full test suites pass.
- [ ] Desktop and narrow browser checks pass.
- [ ] README, changelog, roadmap, and API version reflect `v0.4.1-p4b`.
