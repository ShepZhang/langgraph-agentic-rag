# P4c Modular Evaluation Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the monolithic evaluator into typed, focused modules while preserving every existing public function, deterministic metric, JSON artifact, CLI behavior, Dashboard consumer, FastAPI route, and ablation workflow.

**Architecture:** Keep `evaluation/evaluate.py` as a compatibility facade. Move typed records, dataset normalization, deterministic scoring, runner adaptation, comparison, reporting, optional judge contracts, and atomic JSON storage into dedicated modules; convert typed records back to the existing dictionary contract at system boundaries.

**Tech Stack:** Python 3.12, dataclasses, `typing.Protocol`, JSON, pathlib, pytest, existing LangGraph runners and evaluation fixtures.

---

## File Map

### New Files

- `evaluation/schemas.py`: typed evaluation-domain records and compatibility serialization.
- `evaluation/dataset.py`: dataset loading, validation, and legacy-field normalization.
- `evaluation/metrics.py`: deterministic result scoring, summary metric registry, and aggregation.
- `evaluation/runners.py`: callable adaptation, timing, error capture, and single-question execution.
- `evaluation/comparison.py`: Naive/Agentic paired execution and comparison summaries.
- `evaluation/judges.py`: optional judge protocol, disabled default, and failure-safe judge invocation.
- `evaluation/reporting.py`: pure single-system and comparison text renderers.
- `evaluation/storage.py`: result-store protocol, atomic JSON store, and compatibility artifact writer.
- `tests/test_evaluation_schemas.py`
- `tests/test_evaluation_dataset.py`
- `tests/test_evaluation_metrics.py`
- `tests/test_evaluation_runners.py`
- `tests/test_evaluation_comparison.py`
- `tests/test_evaluation_judges.py`
- `tests/test_evaluation_reporting.py`
- `tests/test_evaluation_storage.py`

### Modified Files

- `evaluation/evaluate.py`: reduce to public compatibility wrappers and CLI assembly.
- `evaluation/runtime_config.py`: add non-breaking evaluator/schema version metadata.
- `tests/test_evaluate.py`: retain facade contract tests; update only additive metadata assertions.
- `README.md`: document the Approach B module boundaries and honest limitations.
- `docs/superpowers/plans/2026-06-14-p4c-modular-evaluation-framework.md`: mark completed tasks during execution.

### Explicitly Unchanged Consumers

- `api/services/evaluation.py`
- `evaluation/dashboard_service.py`
- `evaluation/matrix.py`
- `experiments/run_ablation.py`

These files continue using the public compatibility facade. The matrix also
imports `DEFAULT_EVAL_PATH`, `evaluate_single_system`, and `summarize_results`;
its existing tests prove those additional contracts.

## Execution Prerequisite

Before Task 1, use `superpowers:using-git-worktrees` to create an isolated
`codex/p4c-modular-evaluation` worktree and synchronize it with `main` commit
`e59e8cd`. Do not copy or delete the untracked root `.superpowers/` directory.

Run the baseline suite in the worktree:

```bash
.venv/bin/python -m pytest -q
```

Expected: all existing tests pass before P4c changes. After synchronizing with
the June 15 `main` baseline, the expected count is `408 passed`.

---

### Task 1: Add Typed Evaluation Domain Records

**Files:**
- Create: `evaluation/schemas.py`
- Create: `tests/test_evaluation_schemas.py`

- [ ] **Step 1: Write failing schema serialization tests**

Create `tests/test_evaluation_schemas.py` with focused tests:

```python
from evaluation.schemas import (
    ComparisonEvaluationSummary,
    EvaluationQuestion,
    EvaluationReport,
    EvaluationResult,
    EvaluationSummary,
    JudgeResult,
    PairedEvaluationResult,
)


def test_question_preserves_legacy_and_unknown_fields_in_compatibility_dict():
    question = EvaluationQuestion(
        id="q001",
        question="What is RAG?",
        question_type="single_doc",
        gold_answer="Retrieval augmented generation.",
        expected_sources=["notes.md"],
        expected_keywords=["retrieval"],
        source_match_mode="any",
        answerable=True,
        expected_behavior="answer_with_citation",
        chat_history=[],
        requires_rewrite=False,
        extra_fields={"expected_source": "notes.md", "difficulty": "easy"},
    )

    payload = question.to_compat_dict()

    assert payload["expected_source"] == "notes.md"
    assert payload["difficulty"] == "easy"
    assert payload["answerable"] is True
    assert payload["should_answer"] is True
    assert payload["expected_sources"] == ["notes.md"]
    assert payload["source_match_mode"] == "any"


def test_unavailable_metrics_serialize_as_none():
    summary = EvaluationSummary.empty()

    payload = summary.to_dict()

    assert payload["unsupported_claim_count"] is None
    assert payload["supported_claim_ratio"] is None
    assert payload["citation_verification_pass_rate"] is None


def test_single_and_comparison_reports_keep_established_shapes():
    result = EvaluationResult.empty(
        question_id="q001",
        question_type="single_doc",
        question="What is RAG?",
    )
    summary = EvaluationSummary.empty()
    single = EvaluationReport(summary=summary, results=[result])
    paired = PairedEvaluationResult(
        question="What is RAG?",
        requires_rewrite=False,
        naive=result,
        agentic=result,
    )
    comparison_summary = ComparisonEvaluationSummary(
        total_questions=1,
        naive=summary,
        agentic=summary,
        comparison={"naive_source_hit_rate": 0, "agentic_source_hit_rate": 0},
    )
    comparison = EvaluationReport(
        summary=comparison_summary,
        results=[paired],
    )

    assert single.to_dict()["results"][0]["question_id"] == "q001"
    assert comparison.to_dict()["summary"]["mode"] == "comparison"
    assert comparison.to_dict()["results"][0]["naive"]["question_id"] == "q001"


def test_judge_result_distinguishes_disabled_failed_and_completed_states():
    assert JudgeResult.disabled().status == "disabled"
    assert JudgeResult.failed("RuntimeError: unavailable").error
    assert JudgeResult.completed({"semantic_correctness": 0.8}).scores == {
        "semantic_correctness": 0.8
    }
```

- [ ] **Step 2: Run the tests and verify the module is missing**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_schemas.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'evaluation.schemas'`.

- [ ] **Step 3: Implement the typed records and compatibility serializers**

Create `evaluation/schemas.py` with frozen dataclasses and explicit external
serialization:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from agent.state import ChatMessage


JudgeStatus = Literal["disabled", "completed", "failed"]


@dataclass(frozen=True)
class EvaluationQuestion:
    id: str
    question: str
    question_type: str
    gold_answer: str
    expected_sources: list[str]
    expected_keywords: list[str]
    source_match_mode: Literal["any", "all"]
    answerable: bool
    expected_behavior: str
    chat_history: list[ChatMessage]
    requires_rewrite: bool
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def to_compat_dict(self) -> dict[str, Any]:
        payload = dict(self.extra_fields)
        payload.update(
            {
                "id": self.id,
                "question": self.question,
                "question_type": self.question_type,
                "gold_answer": self.gold_answer,
                "expected_sources": list(self.expected_sources),
                "expected_keywords": list(self.expected_keywords),
                "source_match_mode": self.source_match_mode,
                "answerable": self.answerable,
                "should_answer": self.answerable,
                "expected_behavior": self.expected_behavior,
                "chat_history": [dict(message) for message in self.chat_history],
                "requires_rewrite": self.requires_rewrite,
            }
        )
        return payload


@dataclass
class EvaluationResult:
    question_id: str
    question_type: str
    question: str
    chat_history_supplied: bool = False
    chat_history_used: bool = False
    answer_returned: bool = False
    fallback_triggered: bool = False
    fallback_correct: bool = False
    correct: bool = False
    context_relevant: bool = False
    citation_hit: bool = False
    citation_returned: bool = False
    is_verified: bool = False
    citation_verification_applicable: bool = False
    claim_count: int = 0
    unsupported_claim_count: int | None = None
    supported_claim_count: int | None = None
    total_claim_count: int | None = None
    source_hit: bool = False
    keyword_hit: bool = False
    citation_verification_passed: bool = False
    rewrite_triggered: bool = False
    retry_count: int = 0
    retrieved_doc_count: int = 0
    relevant_doc_count: int = 0
    token_usage: Any = None
    estimated_cost: float | None = None
    latency: float = 0
    error: str | None = None
    answer: str = ""
    citations: list[Any] = field(default_factory=list)
    claims: list[Any] = field(default_factory=list)
    claim_verification_results: list[Any] = field(default_factory=list)
    retrieved_documents: list[Any] = field(default_factory=list)
    relevant_documents: list[Any] = field(default_factory=list)
    failure_analysis: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(
        cls,
        question_id: str,
        question_type: str,
        question: str,
    ) -> EvaluationResult:
        return cls(
            question_id=question_id,
            question_type=question_type,
            question=question,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationSummary:
    total_questions: int = 0
    answer_rate: float = 0
    fallback_rate: float = 0
    citation_rate: float = 0
    source_hit_rate: float = 0
    keyword_hit_rate: float = 0
    fallback_correctness_rate: float = 0
    verification_rate: float = 0
    average_claim_count: float = 0
    correctness_score: float = 0
    context_relevance_score: float = 0
    citation_hit_rate: float = 0
    fallback_accuracy: float = 0
    unsupported_claim_count: int | None = None
    supported_claim_ratio: float | None = None
    citation_verification_pass_rate: float | None = None
    average_token_usage: float = 0
    estimated_cost: float = 0
    average_retry_count: float = 0
    average_retrieved_docs: float = 0
    average_relevant_docs: float = 0
    relevant_filtering_rate: float = 0
    average_latency: float = 0
    rewrite_triggered_count: int = 0
    error_count: int = 0
    failure_type_counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> EvaluationSummary:
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PairedEvaluationResult:
    question: str
    requires_rewrite: bool
    naive: EvaluationResult
    agentic: EvaluationResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "requires_rewrite": self.requires_rewrite,
            "naive": self.naive.to_dict(),
            "agentic": self.agentic.to_dict(),
        }


@dataclass
class ComparisonEvaluationSummary:
    total_questions: int
    naive: EvaluationSummary
    agentic: EvaluationSummary
    comparison: dict[str, Any]
    mode: str = "comparison"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "total_questions": self.total_questions,
            "naive": self.naive.to_dict(),
            "agentic": self.agentic.to_dict(),
            "comparison": dict(self.comparison),
        }


@dataclass
class EvaluationReport:
    summary: EvaluationSummary | ComparisonEvaluationSummary
    results: list[EvaluationResult] | list[PairedEvaluationResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(frozen=True)
class RuntimeMetadata:
    schema_version: int
    evaluator_version: str
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluator_version": self.evaluator_version,
            **self.config,
        }


@dataclass(frozen=True)
class JudgeResult:
    status: JudgeStatus
    scores: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    error: str | None = None

    @classmethod
    def disabled(cls) -> JudgeResult:
        return cls(status="disabled")

    @classmethod
    def completed(
        cls,
        scores: dict[str, float],
        reason: str = "",
    ) -> JudgeResult:
        return cls(status="completed", scores=dict(scores), reason=reason)

    @classmethod
    def failed(cls, error: str) -> JudgeResult:
        return cls(status="failed", error=error)
```

- [ ] **Step 4: Run focused and existing evaluator tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_schemas.py tests/test_evaluate.py -q
```

Expected: all tests pass; the new module has not changed runtime behavior.

- [ ] **Step 5: Commit the typed domain layer**

```bash
git add evaluation/schemas.py tests/test_evaluation_schemas.py
git commit -m "refactor: add typed evaluation domain records"
```

---

### Task 2: Extract Dataset Loading And Normalization

**Files:**
- Create: `evaluation/dataset.py`
- Create: `tests/test_evaluation_dataset.py`
- Modify: `evaluation/evaluate.py`

- [ ] **Step 1: Write focused dataset tests against the typed API**

Create `tests/test_evaluation_dataset.py`:

```python
import json

import pytest

from evaluation.dataset import load_questions, normalize_question


def test_load_questions_returns_typed_records_and_preserves_legacy_source(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question": "What is RAG?",
                    "expected_source": "notes.md",
                    "difficulty": "easy",
                }
            ]
        ),
        encoding="utf-8",
    )

    questions = load_questions(path)

    assert questions[0].id == "q001"
    assert questions[0].expected_sources == ["notes.md"]
    assert questions[0].extra_fields["expected_source"] == "notes.md"
    assert questions[0].extra_fields["difficulty"] == "easy"


def test_normalize_question_rejects_conflicting_answerability():
    with pytest.raises(ValueError, match="answerable and should_answer must match"):
        normalize_question(
            {
                "question": "Conflict?",
                "answerable": True,
                "should_answer": False,
            },
            index=0,
        )


def test_normalize_question_preserves_all_source_match_mode():
    question = normalize_question(
        {
            "question": "Compare both documents.",
            "expected_sources": ["product.md", "security.md"],
            "source_match_mode": "all",
        },
        index=0,
    )

    assert question.source_match_mode == "all"


def test_normalize_question_rejects_invalid_source_match_mode():
    with pytest.raises(ValueError, match="source_match_mode"):
        normalize_question(
            {"question": "Invalid?", "source_match_mode": "some"},
            index=0,
        )


@pytest.mark.parametrize(
    "chat_history",
    [["bad"], [{"role": "user"}], [{"role": "user", "content": 3}]],
)
def test_normalize_question_rejects_malformed_chat_history(chat_history):
    with pytest.raises(ValueError, match="chat_history"):
        normalize_question(
            {"question": "Bad history?", "chat_history": chat_history},
            index=0,
        )


def test_load_questions_rejects_non_list_root(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text('{"question": "not a list"}', encoding="utf-8")

    with pytest.raises(ValueError, match="evaluation questions must be a list"):
        load_questions(path)
```

- [ ] **Step 2: Verify the focused tests fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_dataset.py -q
```

Expected: collection fails because `evaluation.dataset` does not exist.

- [ ] **Step 3: Implement dataset normalization with the current validation semantics**

Create `evaluation/dataset.py` and move the current normalization behavior into
these public typed functions:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from agent.state import ChatMessage
from evaluation.schemas import EvaluationQuestion


DEFAULT_EVAL_PATH = Path(__file__).with_name("eval_questions.json")


def load_questions(
    path: str | Path = DEFAULT_EVAL_PATH,
) -> list[EvaluationQuestion]:
    with Path(path).open(encoding="utf-8") as question_file:
        records = json.load(question_file)
    if not isinstance(records, list):
        raise ValueError("evaluation questions must be a list")
    return [
        normalize_question(record, index)
        for index, record in enumerate(records)
    ]


def normalize_questions(records: list[dict[str, Any]]) -> list[EvaluationQuestion]:
    return [
        normalize_question(record, index)
        for index, record in enumerate(records)
    ]


def normalize_question(record: Any, index: int) -> EvaluationQuestion:
    if not isinstance(record, dict):
        raise ValueError(f"evaluation question at index {index} must be an object")
    question = record.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"evaluation question at index {index} requires question")

    requires_rewrite = record.get("requires_rewrite", False)
    if not isinstance(requires_rewrite, bool):
        raise ValueError("requires_rewrite must be a boolean")

    answerable = _get_answerable(record)
    expected_behavior = _normalize_expected_behavior(
        record.get("expected_behavior"),
        answerable,
    )
    known_fields = {
        "id",
        "question",
        "question_type",
        "gold_answer",
        "expected_sources",
        "expected_keywords",
        "source_match_mode",
        "answerable",
        "should_answer",
        "expected_behavior",
        "chat_history",
        "requires_rewrite",
    }
    return EvaluationQuestion(
        id=str(record.get("id") or f"q{index + 1:03d}"),
        question=question,
        question_type=str(record.get("question_type") or "unspecified").strip(),
        gold_answer=str(record.get("gold_answer") or "").strip(),
        expected_sources=_normalize_expected_sources(record),
        expected_keywords=_normalize_string_list(
            record.get("expected_keywords"),
            "expected_keywords",
        ),
        source_match_mode=_normalize_source_match_mode(
            record.get("source_match_mode", "any")
        ),
        answerable=answerable,
        expected_behavior=expected_behavior,
        chat_history=_normalize_chat_history(record.get("chat_history", [])),
        requires_rewrite=requires_rewrite,
        extra_fields={
            key: value for key, value in record.items() if key not in known_fields
        },
    )


def _normalize_expected_sources(record: dict[str, Any]) -> list[str]:
    if "expected_sources" in record:
        return _normalize_string_list(
            record.get("expected_sources"),
            "expected_sources",
        )
    return _normalize_string_list(record.get("expected_source"), "expected_source")


def _normalize_source_match_mode(value: Any) -> str:
    if value not in {"any", "all"}:
        raise ValueError("source_match_mode must be 'any' or 'all'")
    return value


def _get_answerable(record: dict[str, Any]) -> bool:
    has_answerable = "answerable" in record
    has_should_answer = "should_answer" in record
    if has_answerable:
        answerable = record.get("answerable")
        if not isinstance(answerable, bool):
            raise ValueError("answerable must be a boolean")
        if has_should_answer:
            should_answer = record.get("should_answer")
            if not isinstance(should_answer, bool):
                raise ValueError("should_answer must be a boolean")
            if answerable != should_answer:
                raise ValueError("answerable and should_answer must match")
        return answerable
    if has_should_answer:
        should_answer = record.get("should_answer")
        if not isinstance(should_answer, bool):
            raise ValueError("should_answer must be a boolean")
        return should_answer
    return True


def _normalize_expected_behavior(value: Any, answerable: bool) -> str:
    if value is None:
        return "answer_with_citation" if answerable else "fallback"
    if not isinstance(value, str):
        raise ValueError(
            "expected_behavior must be one of answer_with_citation or fallback"
        )
    normalized = value.strip()
    if normalized not in {"answer_with_citation", "fallback"}:
        raise ValueError(
            "expected_behavior must be one of answer_with_citation or fallback"
        )
    if answerable and normalized == "fallback":
        raise ValueError("expected_behavior must match answerable")
    if not answerable and normalized == "answer_with_citation":
        raise ValueError("expected_behavior must match answerable")
    return normalized


def _normalize_chat_history(value: Any) -> list[ChatMessage]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("chat_history must be a list")
    normalized: list[ChatMessage] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(
                "chat_history must contain dict entries with role and content"
            )
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not role.strip():
            raise ValueError("chat_history entries require string role and content")
        if not isinstance(content, str):
            raise ValueError("chat_history entries require string role and content")
        normalized.append(cast(ChatMessage, dict(item)))
    return normalized


def _normalize_string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return list(value)
        raise ValueError(f"{field_name} must contain only strings")
    raise ValueError(f"{field_name} must be a string or list of strings")
```

- [ ] **Step 4: Replace only the facade loader and normalization entrypoints**

In `evaluation/evaluate.py`, import `DEFAULT_EVAL_PATH`, `load_questions`, and
`normalize_questions`. Keep the public dictionary contract:

```python
from evaluation.dataset import (
    DEFAULT_EVAL_PATH,
    load_questions,
    normalize_questions,
)


def load_eval_questions(
    path: str | Path = DEFAULT_EVAL_PATH,
) -> list[dict[str, Any]]:
    return [question.to_compat_dict() for question in load_questions(path)]
```

At the beginning of `evaluate_questions`, replace the local normalization
comprehension with:

```python
normalized_questions = normalize_questions(questions)
normalized_question_dicts = [
    question.to_compat_dict() for question in normalized_questions
]
```

Continue passing `normalized_question_dicts` to the unchanged legacy evaluator
for this task. Remove the duplicated dataset-only helpers from
`evaluation/evaluate.py`.

- [ ] **Step 5: Run dataset and facade regression tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_evaluation_dataset.py \
  tests/test_evaluate.py -q
```

Expected: all tests pass with identical normalized dictionaries.

- [ ] **Step 6: Commit the dataset extraction**

```bash
git add evaluation/dataset.py evaluation/evaluate.py \
  tests/test_evaluation_dataset.py
git commit -m "refactor: extract evaluation dataset normalization"
```

---

### Task 3: Extract Deterministic Result Scoring

**Files:**
- Create: `evaluation/metrics.py`
- Create: `tests/test_evaluation_metrics.py`
- Modify: `evaluation/evaluate.py`

- [ ] **Step 1: Write failing per-question scoring tests**

Create the first section of `tests/test_evaluation_metrics.py`:

```python
import math

from evaluation.dataset import normalize_question
from evaluation.metrics import build_error_result, score_system_output


def test_score_system_output_preserves_reliability_fields():
    question = normalize_question(
        {
            "id": "q001",
            "question": "How does grading help?",
            "expected_keywords": ["evidence"],
            "expected_sources": ["notes.md"],
            "answerable": True,
        },
        index=0,
    )
    result = score_system_output(
        question,
        {
            "answer": "It checks evidence [1].",
            "citations": [{"source": "notes.md"}],
            "retrieved_documents": [{"source": "notes.md"}],
            "relevant_documents": [{"source": "notes.md"}],
            "claims": [{"claim": "checks evidence", "supported": True}],
            "retry_count": 1,
            "is_verified": True,
        },
    )

    assert result.correct is True
    assert result.context_relevant is True
    assert result.citation_hit is True
    assert result.supported_claim_count == 1
    assert result.unsupported_claim_count == 0
    assert result.rewrite_triggered is True


def test_score_system_output_ignores_invalid_optional_cost():
    question = normalize_question({"question": "Cost?"}, index=0)

    result = score_system_output(
        question,
        {"answer": "ok", "estimated_cost": math.inf},
    )

    assert result.estimated_cost is None


def test_score_system_output_honors_all_source_match_mode():
    question = normalize_question(
        {
            "question": "Compare both documents.",
            "expected_sources": ["product.md", "security.md"],
            "source_match_mode": "all",
        },
        index=0,
    )

    result = score_system_output(
        question,
        {
            "answer": "Combined answer.",
            "citations": [{"source": "product.md"}],
            "retrieved_documents": [
                {"source": "product.md"},
                {"source": "security.md"},
            ],
        },
    )

    assert result.source_hit is True
    assert result.citation_hit is False


def test_build_error_result_uses_question_identity_and_no_false_metrics():
    question = normalize_question(
        {"id": "q009", "question": "Broken?", "question_type": "unanswerable"},
        index=0,
    )

    result = build_error_result(question)

    assert result.question_id == "q009"
    assert result.correct is False
    assert result.unsupported_claim_count is None
```

- [ ] **Step 2: Verify the scoring tests fail**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_evaluation_metrics.py::test_score_system_output_preserves_reliability_fields \
  -q
```

Expected: collection fails because `evaluation.metrics` does not exist.

- [ ] **Step 3: Implement result scoring by moving the current helpers**

Create `evaluation/metrics.py` with:

```python
from __future__ import annotations

import math
import re
from typing import Any

from evaluation.failure_analyzer import analyze_failure
from evaluation.schemas import EvaluationQuestion, EvaluationResult


def score_system_output(
    question: EvaluationQuestion,
    system_result: dict[str, Any],
) -> EvaluationResult:
    if not isinstance(system_result, dict):
        raise ValueError("agent result must be a dictionary")
    answer = system_result.get("answer", "")
    if not isinstance(answer, str):
        raise ValueError("agent answer must be a string")

    citations = _safe_list(system_result.get("citations", []), "citations")
    claims = _safe_list(system_result.get("claims", []), "claims")
    verification_results = _safe_list(
        system_result.get("claim_verification_results", []),
        "claim_verification_results",
    )
    retrieved = _safe_list(
        system_result.get("retrieved_documents", []),
        "retrieved_documents",
    )
    relevant = _safe_list(
        system_result.get("relevant_documents", []),
        "relevant_documents",
    )
    retry_count = _safe_int(
        system_result.get(
            "retry_count",
            system_result.get("rewrite_count", 0),
        ),
        "retry_count",
    )
    fallback_reason = system_result.get("fallback_reason", "")
    if fallback_reason is None:
        fallback_reason = ""
    if not isinstance(fallback_reason, str):
        raise ValueError("fallback_reason must be a string")

    fallback_triggered = bool(fallback_reason.strip()) or is_fallback_answer(answer)
    answer_returned = bool(answer.strip()) and not fallback_triggered
    verification_applicable = bool(
        system_result.get("citation_verification_enabled", True)
    )
    legacy_claim_labels = (
        "citation_verification_enabled" not in system_result
        and not verification_results
    )
    records = claims if legacy_claim_labels else verification_results
    unsupported = (
        _count_unsupported_claims(records) if verification_applicable else None
    )
    supported = _count_supported_claims(records) if verification_applicable else None
    total = len(records) if verification_applicable else None

    return EvaluationResult(
        question_id=question.id,
        question_type=question.question_type,
        question=question.question,
        chat_history_supplied=bool(question.chat_history),
        chat_history_used=bool(system_result.get("chat_history_used", False)),
        answer_returned=answer_returned,
        fallback_triggered=fallback_triggered,
        fallback_correct=fallback_triggered is (not question.answerable),
        correct=_is_correct_answer(
            answer,
            answer_returned,
            question.expected_keywords,
            question.gold_answer,
            question.answerable,
        ),
        context_relevant=_has_expected_source(
            question.expected_sources,
            relevant,
            retrieved,
            question.source_match_mode,
        ),
        citation_hit=_has_expected_source(
            question.expected_sources,
            citations,
            [],
            question.source_match_mode,
        ),
        citation_returned=bool(citations),
        is_verified=bool(system_result.get("is_verified", False)),
        citation_verification_applicable=verification_applicable,
        claim_count=len(claims),
        unsupported_claim_count=unsupported,
        supported_claim_count=supported,
        total_claim_count=total,
        source_hit=(
            _has_expected_source(
                question.expected_sources,
                citations,
                [],
                question.source_match_mode,
            )
            or _has_expected_source(
                question.expected_sources,
                retrieved,
                [],
                question.source_match_mode,
            )
        ),
        keyword_hit=(
            answer_returned
            and _has_expected_keywords(answer, question.expected_keywords)
        ),
        citation_verification_passed=bool(
            system_result.get(
                "citation_verification_passed",
                system_result.get("is_verified", False),
            )
        ),
        rewrite_triggered=retry_count > 0,
        retry_count=retry_count,
        retrieved_doc_count=len(retrieved),
        relevant_doc_count=len(relevant),
        token_usage=system_result.get("token_usage"),
        estimated_cost=_safe_cost(system_result.get("estimated_cost")),
        answer=answer,
        citations=citations,
        claims=claims,
        claim_verification_results=verification_results,
        retrieved_documents=retrieved,
        relevant_documents=relevant,
    )


def build_error_result(question: EvaluationQuestion) -> EvaluationResult:
    result = EvaluationResult.empty(
        question_id=question.id,
        question_type=question.question_type,
        question=question.question,
    )
    result.chat_history_supplied = bool(question.chat_history)
    return result


def attach_failure_analysis(
    question: EvaluationQuestion,
    result: EvaluationResult,
) -> EvaluationResult:
    result.failure_analysis = analyze_failure(
        question.to_compat_dict(),
        result.to_dict(),
    )
    return result
```

Relocate these exact existing definitions byte-for-byte from
`evaluation/evaluate.py` into `evaluation/metrics.py`, then update their local
call sites to use the relocated definitions:

```text
_safe_list
_safe_int
_has_expected_source
_has_expected_keywords
_is_correct_answer
_has_gold_answer_overlap
_content_terms
_count_supported_claims
_count_unsupported_claims
_safe_cost
_extract_total_tokens
_is_fallback_answer
_rate
_average
```

- [ ] **Step 4: Route legacy success/error builders through typed scoring**

In `evaluation/evaluate.py`, keep `_build_success_result` and
`_build_error_result` temporarily as compatibility wrappers:

```python
from evaluation.dataset import normalize_question
from evaluation.metrics import (
    build_error_result as build_typed_error_result,
    score_system_output,
)


def _build_success_result(
    item: dict[str, Any],
    agent_result: dict[str, Any],
) -> dict[str, Any]:
    question = normalize_question(item, index=0)
    return score_system_output(question, agent_result).to_dict()


def _build_error_result(item: dict[str, Any]) -> dict[str, Any]:
    question = normalize_question(item, index=0)
    return build_typed_error_result(question).to_dict()
```

Remove only the duplicated scoring helpers after all references use the new
module.

- [ ] **Step 5: Run scoring and facade regression tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_evaluation_metrics.py \
  tests/test_evaluate.py -q
```

Expected: all tests pass, including malformed payload, token, cost, fallback,
claim, source-hit, and citation-hit cases.

- [ ] **Step 6: Commit result scoring extraction**

```bash
git add evaluation/metrics.py evaluation/evaluate.py \
  tests/test_evaluation_metrics.py
git commit -m "refactor: extract deterministic evaluation scoring"
```

---

### Task 4: Add The Summary Metric Registry And Aggregation

**Files:**
- Modify: `evaluation/metrics.py`
- Modify: `tests/test_evaluation_metrics.py`
- Modify: `evaluation/evaluate.py`

- [ ] **Step 1: Add failing registry and aggregation tests**

Append to `tests/test_evaluation_metrics.py`:

```python
from evaluation.metrics import DEFAULT_SUMMARY_METRICS, summarize_results
from evaluation.schemas import EvaluationResult


def test_summary_metric_registry_has_unique_stable_names():
    names = [metric.name for metric in DEFAULT_SUMMARY_METRICS]

    assert len(names) == len(set(names))
    assert "correctness_score" in names
    assert "context_relevance_score" in names
    assert "citation_hit_rate" in names
    assert "fallback_accuracy" in names


def test_summarize_results_matches_expected_denominators():
    question = normalize_question(
        {
            "question": "Supported?",
            "expected_sources": ["notes.md"],
            "expected_keywords": ["supported"],
        },
        index=0,
    )
    result = EvaluationResult.empty(
        question_id=question.id,
        question_type=question.question_type,
        question=question.question,
    )
    result.answer_returned = True
    result.correct = True
    result.context_relevant = True
    result.citation_hit = True
    result.source_hit = True
    result.keyword_hit = True

    summary = summarize_results([result], [question])

    assert summary.correctness_score == 1.0
    assert summary.context_relevance_score == 1.0
    assert summary.citation_hit_rate == 1.0
    assert summary.source_hit_rate == 1.0


def test_summarize_results_keeps_verification_metrics_unavailable():
    question = normalize_question({"question": "No verifier?"}, index=0)
    result = EvaluationResult.empty(
        question_id=question.id,
        question_type=question.question_type,
        question=question.question,
    )

    summary = summarize_results([result], [question])

    assert summary.unsupported_claim_count is None
    assert summary.supported_claim_ratio is None
    assert summary.citation_verification_pass_rate is None
```

- [ ] **Step 2: Run the new tests and verify missing symbols**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_evaluation_metrics.py::test_summary_metric_registry_has_unique_stable_names \
  -q
```

Expected: import fails because the registry and aggregation API are not
implemented.

- [ ] **Step 3: Implement the explicit registry and typed aggregation**

Add to `evaluation/metrics.py`:

```python
from collections.abc import Callable
from dataclasses import dataclass

from evaluation.failure_analyzer import summarize_failure_types
from evaluation.schemas import EvaluationSummary


@dataclass(frozen=True)
class SummaryMetric:
    name: str
    compute: Callable[
        [list[EvaluationResult], list[EvaluationQuestion]],
        int | float | None,
    ]


DEFAULT_SUMMARY_METRICS = (
    SummaryMetric(
        "correctness_score",
        lambda results, questions: _rate(
            sum(1 for result in results if result.correct),
            len(results),
        ),
    ),
    SummaryMetric(
        "context_relevance_score",
        lambda results, questions: _rate(
            sum(1 for result in results if result.context_relevant),
            sum(1 for question in questions if question.expected_sources),
        ),
    ),
    SummaryMetric(
        "citation_hit_rate",
        lambda results, questions: _rate(
            sum(1 for result in results if result.citation_hit),
            sum(1 for question in questions if question.expected_sources),
        ),
    ),
    SummaryMetric(
        "fallback_accuracy",
        lambda results, questions: _rate(
            sum(1 for result in results if result.fallback_correct),
            len(results),
        ),
    ),
)
```

Implement `summarize_results(results, questions) -> EvaluationSummary` by
moving the current `_summarize` calculations without changing denominator,
rounding, unavailable-value, token, cost, or failure-count behavior. Populate
the registered fields with:

```python
registered = {
    metric.name: metric.compute(results, questions)
    for metric in DEFAULT_SUMMARY_METRICS
}
```

Construct `EvaluationSummary` with all current fields. For failure analysis,
pass compatibility dictionaries:

```python
failure_type_counts=summarize_failure_types(
    [result.to_dict() for result in results]
)
```

For zero results, return `EvaluationSummary.empty()`.

- [ ] **Step 4: Replace legacy `_summarize` with a typed wrapper**

In `evaluation/evaluate.py`, temporarily retain the private wrapper needed by
comparison code:

```python
from evaluation.metrics import summarize_results


def _summarize(
    results: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    typed_questions = normalize_questions(questions)
    typed_results = [
        EvaluationResult(**result)
        for result in results
    ]
    return summarize_results(typed_results, typed_questions).to_dict()
```

Import `EvaluationResult` from `evaluation.schemas`.

- [ ] **Step 5: Run all metric and evaluator tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_evaluation_metrics.py \
  tests/test_evaluate.py -q
```

Expected: all tests pass with the exact pre-P4c summary dictionaries.

- [ ] **Step 6: Commit aggregation extraction**

```bash
git add evaluation/metrics.py evaluation/evaluate.py \
  tests/test_evaluation_metrics.py
git commit -m "refactor: add evaluation metric registry"
```

---

### Task 5: Extract Runner Adaptation And Single-Case Execution

**Files:**
- Create: `evaluation/runners.py`
- Create: `tests/test_evaluation_runners.py`
- Modify: `evaluation/evaluate.py`

- [ ] **Step 1: Write failing runner adapter tests**

Create `tests/test_evaluation_runners.py`:

```python
from evaluation.dataset import normalize_question
from evaluation.runners import CallableRunnerAdapter, evaluate_question


def test_callable_runner_adapter_supports_one_argument_functions():
    adapter = CallableRunnerAdapter(lambda question: {"answer": question})

    result = adapter.run("What?", [{"role": "user", "content": "Earlier"}])

    assert result == {"answer": "What?"}


def test_callable_runner_adapter_passes_history_to_two_argument_functions():
    observed = {}

    def runner(question, history):
        observed["question"] = question
        observed["history"] = history
        return {"answer": "ok", "chat_history_used": True}

    history = [{"role": "user", "content": "Earlier"}]
    result = CallableRunnerAdapter(runner).run("Follow-up?", history)

    assert result["chat_history_used"] is True
    assert observed == {"question": "Follow-up?", "history": history}


def test_evaluate_question_records_runner_errors_and_latency():
    question = normalize_question({"question": "Broken?"}, index=0)
    timer_values = iter([1.0, 1.25])

    def runner(question):
        raise RuntimeError("offline")

    result = evaluate_question(
        question,
        CallableRunnerAdapter(runner),
        timer=lambda: next(timer_values),
    )

    assert result.error == "RuntimeError: offline"
    assert result.latency == 0.25
    assert result.failure_analysis["failure_type"] == "tool_failure"
```

- [ ] **Step 2: Verify runner tests fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_runners.py -q
```

Expected: collection fails because `evaluation.runners` does not exist.

- [ ] **Step 3: Implement the runner protocol, adapter, and execution**

Create `evaluation/runners.py`:

```python
from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from typing import Any, Protocol

from agent.state import ChatMessage
from evaluation.metrics import (
    attach_failure_analysis,
    build_error_result,
    score_system_output,
)
from evaluation.schemas import EvaluationQuestion, EvaluationResult


class EvaluationRunner(Protocol):
    def run(
        self,
        question: str,
        chat_history: list[ChatMessage],
    ) -> dict[str, Any]:
        ...


class CallableRunnerAdapter:
    def __init__(self, runner: Callable[..., dict[str, Any]]) -> None:
        self._runner = runner

    def run(
        self,
        question: str,
        chat_history: list[ChatMessage],
    ) -> dict[str, Any]:
        try:
            parameters = inspect.signature(self._runner).parameters.values()
        except (TypeError, ValueError):
            return self._runner(question, chat_history)
        positional = [
            parameter
            for parameter in parameters
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ]
        accepts_varargs = any(
            parameter.kind is inspect.Parameter.VAR_POSITIONAL
            for parameter in parameters
        )
        if accepts_varargs or len(positional) >= 2:
            return self._runner(question, chat_history)
        return self._runner(question)


def evaluate_question(
    question: EvaluationQuestion,
    runner: EvaluationRunner,
    timer: Callable[[], float] = time.perf_counter,
) -> EvaluationResult:
    started_at = timer()
    try:
        raw_result = runner.run(question.question, question.chat_history)
        result = score_system_output(question, raw_result)
        error = None
    except Exception as exc:  # noqa: BLE001 - failures are evaluation data.
        result = build_error_result(question)
        error = _format_error(exc)
    result.latency = timer() - started_at
    result.error = error
    return attach_failure_analysis(question, result)


def _format_error(exc: Exception) -> str:
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__
```

- [ ] **Step 4: Route the facade's single-system evaluator through the adapter**

Update `_evaluate_single_system` in `evaluation/evaluate.py`:

```python
from evaluation.runners import CallableRunnerAdapter, evaluate_question


def _evaluate_single_system(
    item: dict[str, Any],
    runner: Callable[..., dict[str, Any]],
    timer: Callable[[], float],
) -> dict[str, Any]:
    question = normalize_question(item, index=0)
    return evaluate_question(
        question,
        CallableRunnerAdapter(runner),
        timer,
    ).to_dict()
```

Delete `_invoke_evaluation_runner` and `_format_error` from the facade after
all references use `evaluation.runners`.

- [ ] **Step 5: Run runner, evaluator, and dashboard service tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_evaluation_runners.py \
  tests/test_evaluate.py \
  tests/test_dashboard_service.py -q
```

Expected: all tests pass, including per-case errors remaining completed
dashboard runs.

- [ ] **Step 6: Commit runner extraction**

```bash
git add evaluation/runners.py evaluation/evaluate.py \
  tests/test_evaluation_runners.py
git commit -m "refactor: extract evaluation runner execution"
```

---

### Task 6: Extract Single-System And Comparison Orchestration

**Files:**
- Create: `evaluation/comparison.py`
- Create: `tests/test_evaluation_comparison.py`
- Modify: `evaluation/evaluate.py`

- [ ] **Step 1: Write failing typed orchestration tests**

Create `tests/test_evaluation_comparison.py`:

```python
from evaluation.comparison import evaluate_comparison, evaluate_single_system
from evaluation.dataset import normalize_questions
from evaluation.runners import CallableRunnerAdapter


def test_evaluate_single_system_returns_typed_report():
    questions = normalize_questions(
        [{"id": "q001", "question": "What?", "expected_sources": []}]
    )

    report = evaluate_single_system(
        questions,
        CallableRunnerAdapter(lambda question: {"answer": "ok"}),
        timer=iter([0.0, 0.1]).__next__,
    )

    assert report.summary.total_questions == 1
    assert report.results[0].question_id == "q001"


def test_evaluate_comparison_preserves_dataset_order_and_paired_shape():
    questions = normalize_questions(
        [
            {"id": "q002", "question": "Second?"},
            {"id": "q001", "question": "First?", "requires_rewrite": True},
        ]
    )
    timer_values = iter([0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4])

    report = evaluate_comparison(
        questions,
        agentic_runner=CallableRunnerAdapter(
            lambda question: {"answer": f"agentic:{question}"}
        ),
        naive_runner=CallableRunnerAdapter(
            lambda question: {"answer": f"naive:{question}"}
        ),
        timer=lambda: next(timer_values),
    )
    payload = report.to_dict()

    assert [item["question"] for item in payload["results"]] == [
        "Second?",
        "First?",
    ]
    assert payload["results"][1]["requires_rewrite"] is True
    assert payload["summary"]["mode"] == "comparison"
```

- [ ] **Step 2: Verify orchestration tests fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_comparison.py -q
```

Expected: collection fails because `evaluation.comparison` does not exist.

- [ ] **Step 3: Implement typed orchestration**

Create `evaluation/comparison.py`:

```python
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from evaluation.metrics import summarize_results
from evaluation.runners import EvaluationRunner, evaluate_question
from evaluation.schemas import (
    ComparisonEvaluationSummary,
    EvaluationQuestion,
    EvaluationReport,
    EvaluationSummary,
    PairedEvaluationResult,
)


def evaluate_single_system(
    questions: list[EvaluationQuestion],
    runner: EvaluationRunner,
    timer: Callable[[], float] = time.perf_counter,
) -> EvaluationReport:
    results = [
        evaluate_question(question, runner, timer)
        for question in questions
    ]
    return EvaluationReport(
        summary=summarize_results(results, questions),
        results=results,
    )


def evaluate_comparison(
    questions: list[EvaluationQuestion],
    agentic_runner: EvaluationRunner,
    naive_runner: EvaluationRunner,
    timer: Callable[[], float] = time.perf_counter,
) -> EvaluationReport:
    paired: list[PairedEvaluationResult] = []
    naive_results = []
    agentic_results = []
    for question in questions:
        naive = evaluate_question(question, naive_runner, timer)
        agentic = evaluate_question(question, agentic_runner, timer)
        naive_results.append(naive)
        agentic_results.append(agentic)
        paired.append(
            PairedEvaluationResult(
                question=question.question,
                requires_rewrite=question.requires_rewrite,
                naive=naive,
                agentic=agentic,
            )
        )
    naive_summary = summarize_results(naive_results, questions)
    agentic_summary = summarize_results(agentic_results, questions)
    return EvaluationReport(
        summary=ComparisonEvaluationSummary(
            total_questions=len(questions),
            naive=naive_summary,
            agentic=agentic_summary,
            comparison=build_comparison_summary(
                naive_summary,
                agentic_summary,
            ),
        ),
        results=paired,
    )


def build_comparison_summary(
    naive: EvaluationSummary,
    agentic: EvaluationSummary,
) -> dict[str, Any]:
    return {
        "naive_source_hit_rate": naive.source_hit_rate,
        "agentic_source_hit_rate": agentic.source_hit_rate,
        "naive_keyword_hit_rate": naive.keyword_hit_rate,
        "agentic_keyword_hit_rate": agentic.keyword_hit_rate,
        "naive_citation_rate": naive.citation_rate,
        "agentic_citation_rate": agentic.citation_rate,
        "naive_verification_rate": naive.verification_rate,
        "agentic_verification_rate": agentic.verification_rate,
        "naive_fallback_correctness_rate": naive.fallback_correctness_rate,
        "agentic_fallback_correctness_rate": agentic.fallback_correctness_rate,
        "naive_average_latency": naive.average_latency,
        "agentic_average_latency": agentic.average_latency,
    }
```

- [ ] **Step 4: Make `evaluate_questions` delegate to typed orchestration**

Replace the body of `evaluate_questions` in `evaluation/evaluate.py`:

```python
def evaluate_questions(
    questions: list[dict[str, Any]],
    run_agent_fn: Callable[[str], dict[str, Any]] = run_agent,
    run_naive_fn: Callable[[str], dict[str, Any]] | None = None,
    timer: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    typed_questions = normalize_questions(questions)
    agentic_runner = CallableRunnerAdapter(run_agent_fn)
    if run_naive_fn is not None:
        return evaluate_comparison(
            typed_questions,
            agentic_runner=agentic_runner,
            naive_runner=CallableRunnerAdapter(run_naive_fn),
            timer=timer,
        ).to_dict()
    return evaluate_single_system(
        typed_questions,
        runner=agentic_runner,
        timer=timer,
    ).to_dict()
```

Remove `_evaluate_comparison`, `_evaluate_single_system`, `_summarize`, and
`_build_comparison_summary` from the facade after all tests pass.

- [ ] **Step 5: Run comparison, facade, dashboard, and ablation tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_evaluation_comparison.py \
  tests/test_evaluate.py \
  tests/test_dashboard_service.py \
  tests/test_ablation.py -q
```

Expected: all tests pass; ablation and dashboard still import the facade.

- [ ] **Step 6: Commit orchestration extraction**

```bash
git add evaluation/comparison.py evaluation/evaluate.py \
  tests/test_evaluation_comparison.py
git commit -m "refactor: extract evaluation comparison orchestration"
```

---

### Task 7: Extract Report Rendering

**Files:**
- Create: `evaluation/reporting.py`
- Create: `tests/test_evaluation_reporting.py`
- Modify: `evaluation/evaluate.py`

- [ ] **Step 1: Write failing pure renderer tests**

Create `tests/test_evaluation_reporting.py`:

```python
from evaluation.reporting import format_evaluation_report


def test_format_single_system_report_keeps_question_diagnostics():
    report = {
        "summary": {"total_questions": 1, "correctness_score": 1.0},
        "results": [
            {
                "question": "What is RAG?",
                "answer_returned": True,
                "fallback_triggered": False,
                "citation_returned": True,
                "source_hit": True,
                "keyword_hit": True,
                "rewrite_triggered": False,
                "retry_count": 0,
                "retrieved_doc_count": 2,
                "relevant_doc_count": 1,
                "latency": 0.25,
                "error": None,
            }
        ],
    }

    text = format_evaluation_report(report)

    assert "Evaluation Report" in text
    assert "correctness_score: 1.0" in text
    assert "retrieved=2" in text


def test_format_comparison_report_keeps_metric_table():
    report = {
        "summary": {
            "mode": "comparison",
            "naive": {"source_hit_rate": 0.5},
            "agentic": {"source_hit_rate": 1.0},
        },
        "results": [],
    }

    text = format_evaluation_report(report)

    assert "| Metric | Naive RAG | Agentic RAG |" in text
    assert "| Source Hit Rate | 0.5 | 1.0 |" in text
```

- [ ] **Step 2: Verify renderer tests fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_reporting.py -q
```

Expected: collection fails because `evaluation.reporting` does not exist.

- [ ] **Step 3: Implement pure report rendering**

Create `evaluation/reporting.py`. Move the existing `format_report`,
`_format_comparison_report`, and `_format_bool` logic into:

```python
from __future__ import annotations

from typing import Any


def format_evaluation_report(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    if summary.get("mode") == "comparison":
        return _format_comparison_report(report)
    lines = ["Evaluation Report", "", "Summary"]
    for key, value in summary.items():
        lines.append(f"{key}: {value}")
    lines.extend(["", "Questions"])
    for index, result in enumerate(report.get("results", []), start=1):
        lines.append(
            (
                f"{index}. {result.get('question', '')} | "
                f"answer={_format_bool(result.get('answer_returned'))} | "
                f"fallback={_format_bool(result.get('fallback_triggered'))} | "
                f"citation={_format_bool(result.get('citation_returned'))} | "
                f"source_hit={_format_bool(result.get('source_hit'))} | "
                f"keyword_hit={_format_bool(result.get('keyword_hit'))} | "
                f"rewrite={_format_bool(result.get('rewrite_triggered'))} | "
                f"retry_count={result.get('retry_count', 0)} | "
                f"retrieved={result.get('retrieved_doc_count', 0)} | "
                f"relevant={result.get('relevant_doc_count', 0)} | "
                f"latency={float(result.get('latency', 0)):.4f}s | "
                f"error={result.get('error') or ''}"
            )
        )
    return "\n".join(lines)


def _format_bool(value: Any) -> str:
    return "true" if value else "false"
```

Relocate the exact existing `_format_comparison_report` definition from
`evaluation/evaluate.py` into `evaluation/reporting.py`. Keep its table labels,
field lookups, `N/A` defaults, and per-question row formatting byte-for-byte;
its `_format_bool` reference resolves to the new local helper above.

- [ ] **Step 4: Make the facade delegate report formatting**

In `evaluation/evaluate.py`:

```python
from evaluation.reporting import format_evaluation_report


def format_report(report: dict[str, Any]) -> str:
    return format_evaluation_report(report)
```

Remove the old comparison and boolean formatting helpers from the facade.

- [ ] **Step 5: Run renderer and facade report tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_evaluation_reporting.py \
  tests/test_evaluate.py -q
```

Expected: all tests pass with identical terminal output.

- [ ] **Step 6: Commit reporting extraction**

```bash
git add evaluation/reporting.py evaluation/evaluate.py \
  tests/test_evaluation_reporting.py
git commit -m "refactor: extract evaluation report rendering"
```

---

### Task 8: Add Optional Judge Contracts

**Files:**
- Create: `evaluation/judges.py`
- Create: `tests/test_evaluation_judges.py`

- [ ] **Step 1: Write failing judge contract tests**

Create `tests/test_evaluation_judges.py`:

```python
from evaluation.dataset import normalize_question
from evaluation.judges import DisabledJudge, invoke_judge
from evaluation.schemas import EvaluationResult


def _result():
    return EvaluationResult.empty("q001", "single_doc", "What?")


def test_disabled_judge_performs_no_scoring():
    question = normalize_question({"question": "What?"}, index=0)

    result = invoke_judge(DisabledJudge(), question, _result())

    assert result.status == "disabled"
    assert result.scores == {}
    assert result.error is None


def test_invoke_judge_records_failure_without_raising():
    class FailingJudge:
        def evaluate(self, question, result):
            raise RuntimeError("judge unavailable")

    question = normalize_question({"question": "What?"}, index=0)

    result = invoke_judge(FailingJudge(), question, _result())

    assert result.status == "failed"
    assert result.error == "RuntimeError: judge unavailable"
```

- [ ] **Step 2: Verify judge tests fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_judges.py -q
```

Expected: collection fails because `evaluation.judges` does not exist.

- [ ] **Step 3: Implement the protocol and conservative default**

Create `evaluation/judges.py`:

```python
from __future__ import annotations

from typing import Protocol

from evaluation.schemas import (
    EvaluationQuestion,
    EvaluationResult,
    JudgeResult,
)


class Judge(Protocol):
    def evaluate(
        self,
        question: EvaluationQuestion,
        result: EvaluationResult,
    ) -> JudgeResult:
        ...


class DisabledJudge:
    def evaluate(
        self,
        question: EvaluationQuestion,
        result: EvaluationResult,
    ) -> JudgeResult:
        return JudgeResult.disabled()


def invoke_judge(
    judge: Judge,
    question: EvaluationQuestion,
    result: EvaluationResult,
) -> JudgeResult:
    try:
        return judge.evaluate(question, result)
    except Exception as exc:  # noqa: BLE001 - optional judge failure is diagnostic.
        message = str(exc)
        error = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
        return JudgeResult.failed(error)
```

Do not call a network model and do not add judge fields to compatibility
reports in P4c. The protocol is intentionally ready for a later
`EvaluationEngine`.

- [ ] **Step 4: Run judge and schema tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_evaluation_judges.py \
  tests/test_evaluation_schemas.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit judge contracts**

```bash
git add evaluation/judges.py tests/test_evaluation_judges.py
git commit -m "feat: add optional evaluation judge contract"
```

---

### Task 9: Add Atomic Result Storage And Compatibility Artifacts

**Files:**
- Create: `evaluation/storage.py`
- Create: `tests/test_evaluation_storage.py`
- Modify: `evaluation/evaluate.py`
- Modify: `evaluation/runtime_config.py`
- Modify: `tests/test_evaluate.py`

- [ ] **Step 1: Write failing result-store tests**

Create `tests/test_evaluation_storage.py`:

```python
import json

from evaluation.storage import JsonResultStore, write_compatibility_artifacts


def test_json_result_store_saves_and_loads_utf8_payload(tmp_path):
    store = JsonResultStore(tmp_path)

    saved_path = store.save("run_001", {"answer": "可靠回答"})

    assert saved_path == str(tmp_path / "run_001.json")
    assert store.load("run_001") == {"answer": "可靠回答"}
    assert not (tmp_path / "run_001.json.tmp").exists()


def test_json_result_store_returns_none_for_missing_run(tmp_path):
    assert JsonResultStore(tmp_path).load("missing") is None


def test_compatibility_writer_keeps_comparison_artifact_names(tmp_path):
    report = {
        "summary": {
            "mode": "comparison",
            "naive": {"correctness_score": 0.5},
            "agentic": {"correctness_score": 1.0},
        },
        "results": [
            {
                "question": "What?",
                "naive": {"question_id": "q001"},
                "agentic": {"question_id": "q001"},
            }
        ],
    }

    write_compatibility_artifacts(
        report,
        tmp_path,
        runtime_config={"schema_version": 1, "evaluator_version": "p4c"},
    )

    assert (tmp_path / "baseline_result.json").exists()
    assert (tmp_path / "agentic_result.json").exists()
    assert (tmp_path / "comparison_result.json").exists()
    comparison = json.loads(
        (tmp_path / "comparison_result.json").read_text(encoding="utf-8")
    )
    assert comparison["runtime_config"]["schema_version"] == 1
```

- [ ] **Step 2: Verify storage tests fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_storage.py -q
```

Expected: collection fails because `evaluation.storage` does not exist.

- [ ] **Step 3: Implement the result-store protocol and JSON backend**

Create `evaluation/storage.py`:

```python
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol


class ResultStore(Protocol):
    def save(self, run_id: str, payload: Mapping[str, Any]) -> str:
        ...

    def load(self, run_id: str) -> dict[str, Any] | None:
        ...


class JsonResultStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def save(self, run_id: str, payload: Mapping[str, Any]) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{run_id}.json"
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(
            json.dumps(
                dict(payload),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)
        return str(path)

    def load(self, run_id: str) -> dict[str, Any] | None:
        path = self.root / f"{run_id}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"stored evaluation result {run_id} must be an object")
        return payload


def write_compatibility_artifacts(
    report: dict[str, Any],
    output_dir: str | Path,
    runtime_config: dict[str, Any],
) -> None:
    store = JsonResultStore(output_dir)
    summary = report.get("summary", {})
    if summary.get("mode") == "comparison":
        paired_results = report.get("results", [])
        store.save(
            "baseline_result",
            {
                "system": "naive_rag",
                "runtime_config": runtime_config,
                "summary": summary.get("naive", {}),
                "results": [
                    paired.get("naive", {})
                    for paired in paired_results
                ],
            },
        )
        store.save(
            "agentic_result",
            {
                "system": "agentic_rag",
                "runtime_config": runtime_config,
                "summary": summary.get("agentic", {}),
                "results": [
                    paired.get("agentic", {})
                    for paired in paired_results
                ],
            },
        )
        store.save(
            "comparison_result",
            {"runtime_config": runtime_config, **report},
        )
        return
    store.save(
        "agentic_result",
        {
            "system": "agentic_rag",
            "runtime_config": runtime_config,
            "summary": report.get("summary", {}),
            "results": report.get("results", []),
        },
    )
```

- [ ] **Step 4: Add additive evaluator metadata**

At the top of `evaluation/runtime_config.py`, define:

```python
from evaluation.schemas import RuntimeMetadata


EVALUATION_SCHEMA_VERSION = 1
EVALUATOR_VERSION = "p4c"
```

Move the existing sanitized configuration body into a typed builder:

```python
def build_runtime_metadata(
    settings: Settings | None = None,
    features: AgentFeatureFlags | None = None,
) -> RuntimeMetadata:
    resolved = settings or get_settings()
    resolved_features = features or AgentFeatureFlags()
    return RuntimeMetadata(
        schema_version=EVALUATION_SCHEMA_VERSION,
        evaluator_version=EVALUATOR_VERSION,
        config={
            "agent_features": resolved_features.to_dict(),
            "llm": {
                "provider": resolved.llm_provider,
                "model": resolved.effective_llm_model,
                "temperature": resolved.temperature,
            },
            "retriever": {
                "top_k": resolved.top_k,
                "hybrid_retrieval_enabled": resolved.hybrid_retrieval_enabled,
                "dense_top_k": resolved.dense_top_k,
                "bm25_top_k": resolved.bm25_top_k,
                "fusion_top_k": resolved.fusion_top_k,
            },
            "reranker": {
                "enabled": resolved.reranker_enabled,
                "model": resolved.reranker_model,
                "top_n": resolved.reranker_top_n,
                "candidate_top_k": resolved.reranker_candidate_top_k,
            },
            "vectorstore": {
                "collection_name": resolved.chroma_collection_name,
            },
        },
    )


def build_runtime_config_snapshot(
    settings: Settings | None = None,
    features: AgentFeatureFlags | None = None,
) -> dict[str, Any]:
    return build_runtime_metadata(settings=settings, features=features).to_dict()
```

Do not add secrets, base URLs, or local persistence paths.

- [ ] **Step 5: Delegate the compatibility artifact writer**

Replace `write_evaluation_artifacts` in `evaluation/evaluate.py`:

```python
from evaluation.storage import write_compatibility_artifacts


def write_evaluation_artifacts(
    report: dict[str, Any],
    output_dir: str | Path,
) -> None:
    write_compatibility_artifacts(
        report,
        output_dir,
        runtime_config=build_runtime_config_snapshot(),
    )
```

Delete `_write_json` from the facade.

Add assertions to the existing artifact tests in `tests/test_evaluate.py`:

```python
assert comparison_payload["runtime_config"]["schema_version"] == 1
assert comparison_payload["runtime_config"]["evaluator_version"] == "p4c"
```

- [ ] **Step 6: Run storage, facade, ablation, and dashboard tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_evaluation_storage.py \
  tests/test_evaluate.py \
  tests/test_ablation.py \
  tests/test_dashboard_service.py -q
```

Expected: all tests pass; saved artifacts retain their old names and layouts.

- [ ] **Step 7: Commit storage and metadata**

```bash
git add evaluation/storage.py evaluation/runtime_config.py \
  evaluation/evaluate.py tests/test_evaluation_storage.py \
  tests/test_evaluate.py
git commit -m "refactor: add atomic evaluation result storage"
```

---

### Task 10: Finish The Compatibility Facade

**Files:**
- Modify: `evaluation/evaluate.py`
- Modify: `tests/test_evaluate.py`

- [ ] **Step 1: Add a facade ownership regression test**

Append to `tests/test_evaluate.py`:

```python
def test_evaluate_module_remains_public_compatibility_facade():
    from evaluation import evaluate

    assert callable(evaluate.load_eval_questions)
    assert callable(evaluate.evaluate_questions)
    assert callable(evaluate.evaluate_single_system)
    assert callable(evaluate.summarize_results)
    assert callable(evaluate.format_report)
    assert callable(evaluate.write_evaluation_artifacts)
    assert callable(evaluate.main)
```

- [ ] **Step 2: Run the facade test before cleanup**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_evaluate.py::test_evaluate_module_remains_public_compatibility_facade \
  -q
```

Expected: PASS before cleanup, establishing the public contract.

- [ ] **Step 3: Reduce `evaluation/evaluate.py` to assembly and wrappers**

After prior tasks have moved implementation details, the facade must contain
only:

```python
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent.graph import run_agent
from evaluation.baselines import run_naive_rag
from evaluation.comparison import (
    evaluate_comparison,
    evaluate_single_system as evaluate_typed_system,
)
from evaluation.dataset import DEFAULT_EVAL_PATH, load_questions, normalize_questions
from evaluation.metrics import summarize_results as summarize_typed_results
from evaluation.reporting import format_evaluation_report
from evaluation.runners import CallableRunnerAdapter, evaluate_question
from evaluation.runtime_config import build_runtime_config_snapshot
from evaluation.schemas import EvaluationResult
from evaluation.storage import write_compatibility_artifacts


def load_eval_questions(
    path: str | Path = DEFAULT_EVAL_PATH,
) -> list[dict[str, Any]]:
    return [question.to_compat_dict() for question in load_questions(path)]


def evaluate_questions(
    questions: list[dict[str, Any]],
    run_agent_fn: Callable[[str], dict[str, Any]] = run_agent,
    run_naive_fn: Callable[[str], dict[str, Any]] | None = None,
    timer: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    typed_questions = normalize_questions(questions)
    agentic = CallableRunnerAdapter(run_agent_fn)
    if run_naive_fn is not None:
        return evaluate_comparison(
            typed_questions,
            agentic_runner=agentic,
            naive_runner=CallableRunnerAdapter(run_naive_fn),
            timer=timer,
        ).to_dict()
    return evaluate_typed_system(
        typed_questions,
        runner=agentic,
        timer=timer,
    ).to_dict()


def evaluate_single_system(
    item: dict[str, Any],
    runner: Callable[[str], dict[str, Any]],
    timer: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    question = normalize_questions([item])[0]
    return evaluate_question(
        question,
        CallableRunnerAdapter(runner),
        timer,
    ).to_dict()


def summarize_results(
    results: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    typed_questions = normalize_questions(questions)
    typed_results = [EvaluationResult(**result) for result in results]
    return summarize_typed_results(typed_results, typed_questions).to_dict()


def format_report(report: dict[str, Any]) -> str:
    return format_evaluation_report(report)


def write_evaluation_artifacts(
    report: dict[str, Any],
    output_dir: str | Path,
) -> None:
    write_compatibility_artifacts(
        report,
        output_dir,
        runtime_config=build_runtime_config_snapshot(),
    )
```

Retain the existing `main` parser behavior and `if __name__ == "__main__"`
entrypoint unchanged. Remove all private metric, normalization, rendering,
comparison, and storage helpers.

- [ ] **Step 4: Verify facade size and public regression tests**

Run:

```bash
wc -l evaluation/evaluate.py
.venv/bin/python -m pytest tests/test_evaluate.py -q
```

Expected: the facade is focused and approximately 150 lines; all legacy
contract tests pass.

- [ ] **Step 5: Run direct consumer regression tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_dashboard_service.py \
  tests/test_fastapi_routes.py \
  tests/test_evaluation_matrix.py \
  tests/test_ablation.py \
  tests/test_baseline.py \
  tests/test_baselines.py -q
```

Expected: all tests pass without changing consumer imports.

- [ ] **Step 6: Commit the final facade cleanup**

```bash
git add evaluation/evaluate.py tests/test_evaluate.py
git commit -m "refactor: reduce evaluator to compatibility facade"
```

---

### Task 11: Document P4c And Verify The Full Project

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-06-14-p4c-modular-evaluation-framework.md`

- [ ] **Step 1: Update the README completed work and architecture**

Add a concise evaluation architecture section:

```markdown
### Modular Evaluation Framework

The P4c evaluator keeps the stable `evaluation.evaluate` API while separating
dataset validation, runner adaptation, deterministic metrics, comparison,
reporting, optional judges, and result storage into focused modules. Internal
dataclasses provide typed records; JSON adapters preserve the existing
Dashboard, FastAPI, CLI, and ablation artifact contracts.

The default evaluator remains deterministic and offline-testable. The `Judge`
and `ResultStore` protocols are extension boundaries; a DeepSeek semantic
judge and SQLite historical-run store remain roadmap work rather than claimed
features.
```

Move the modular Approach B evaluator from `Next Milestones` into
`Completed Work`. Preserve these roadmap items:

```markdown
- Implement a configurable DeepSeek semantic correctness and groundedness judge.
- Add SQLite-backed historical evaluation runs and trend comparison.
- Introduce an `EvaluationEngine` that composes runners, metrics, judges, and stores.
- Add prompt version tracking and prompt regression checks.
```

- [ ] **Step 2: Run formatting and import checks**

Run:

```bash
.venv/bin/python -m compileall evaluation
.venv/bin/python -c "from evaluation.evaluate import evaluate_questions, load_eval_questions, format_report; print(len(load_eval_questions()))"
```

Expected: compilation succeeds and the import smoke test prints `36`.

- [ ] **Step 3: Run the complete test suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all project tests pass. Record the exact test count in the execution
notes before claiming completion.

- [ ] **Step 4: Run CLI compatibility smoke tests without live model calls**

Run the injected-runner CLI contract through pytest:

```bash
.venv/bin/python -m pytest \
  tests/test_evaluate.py::test_main_prints_report_with_injected_runner \
  tests/test_evaluate.py::test_main_writes_comparison_artifacts \
  tests/test_evaluate.py::test_main_writes_single_system_agentic_artifact_schema \
  -q
```

Expected: three tests pass and artifact schemas include additive P4c metadata.

- [ ] **Step 5: Inspect final diff and ensure no unrelated files are included**

Run:

```bash
git status --short
git diff --check
git diff --stat main...HEAD
```

Expected: only P4c evaluation, tests, README, and plan files are changed. The
root `.superpowers/` directory remains untouched and untracked.

- [ ] **Step 6: Mark the implementation plan complete and commit documentation**

Mark all completed plan checkboxes, then run:

```bash
git add README.md \
  docs/superpowers/plans/2026-06-14-p4c-modular-evaluation-framework.md
git commit -m "docs: publish p4c modular evaluation framework"
```

- [ ] **Step 7: Request code review before integration**

Invoke `superpowers:requesting-code-review` against the P4c branch. Address
confirmed correctness, compatibility, or test findings before presenting
merge options.

- [ ] **Step 8: Finish the development branch**

After review and fresh verification, invoke
`superpowers:finishing-a-development-branch`. Offer merge, pull request, keep,
or cleanup choices. Create tag `v0.4.2-p4c` only after the user explicitly
chooses integration and the integration succeeds.

## Final Verification Matrix

| Concern | Verification |
|---|---|
| Dataset compatibility | `test_evaluation_dataset.py`, legacy loader tests |
| Deterministic metric parity | `test_evaluation_metrics.py`, `test_evaluate.py` |
| History-aware runners | `test_evaluation_runners.py` |
| Naive/Agentic pairing | `test_evaluation_comparison.py` |
| Report compatibility | `test_evaluation_reporting.py`, CLI tests |
| Atomic artifacts | `test_evaluation_storage.py` |
| Optional judge isolation | `test_evaluation_judges.py` |
| Dashboard compatibility | `test_dashboard_service.py` |
| FastAPI compatibility | `test_fastapi_routes.py` |
| Historical matrix compatibility | `test_evaluation_matrix.py` |
| Ablation compatibility | `test_ablation.py` |
| Whole-project regression | `.venv/bin/python -m pytest -q` |
