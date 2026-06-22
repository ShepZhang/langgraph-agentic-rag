"""Tests for typed evaluation orchestration."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from agent.state import ChatMessage
import evaluation.comparison as comparison
from evaluation.dataset import normalize_questions
from evaluation.judges import DisabledJudge, Judge
from evaluation.runners import EvaluationRunner
from evaluation.schemas import (
    ComparisonEvaluationSummary,
    EvaluationReport,
    EvaluationResult,
    EvaluationSummary,
    JudgeResult,
    PairedEvaluationResult,
)


class StaticRunner:
    def __init__(
        self,
        answers: dict[str, dict[str, object]],
        calls: list[tuple[str, str]],
        label: str,
    ) -> None:
        self._answers = answers
        self._calls = calls
        self._label = label

    def run(
        self,
        question: str,
        chat_history: list[ChatMessage],
    ) -> dict[str, object]:
        self._calls.append((self._label, question))
        return self._answers[question]


def test_evaluate_single_system_returns_typed_report() -> None:
    questions = normalize_questions(
        [
            {
                "id": "q001",
                "question": "What does RAG use?",
                "expected_keywords": ["evidence"],
                "expected_sources": ["notes.md"],
            }
        ]
    )
    calls: list[tuple[str, str]] = []
    runner: EvaluationRunner = StaticRunner(
        {
            "What does RAG use?": {
                "answer": "RAG uses evidence.",
                "citations": [{"source": "notes.md"}],
                "retrieved_documents": [{"source": "notes.md"}],
                "relevant_documents": [{"source": "notes.md"}],
            }
        },
        calls,
        "agentic",
    )
    timer_values: Iterator[float] = iter([1.0, 1.25])

    report = comparison.evaluate_single_system(
        questions,
        runner,
        timer=lambda: next(timer_values),
    )

    assert isinstance(report, EvaluationReport)
    assert isinstance(report.summary, EvaluationSummary)
    assert isinstance(report.results[0], EvaluationResult)
    assert report.summary.total_questions == 1
    assert report.summary.source_hit_rate == 1.0
    assert report.results[0].question_id == "q001"
    assert report.results[0].latency == 0.25
    assert calls == [("agentic", "What does RAG use?")]
    assert report.results[0].judge == JudgeResult.disabled()
    assert report.results[0].failure_analysis["failure_type"] == "no_failure"


def test_evaluate_single_system_uses_disabled_judge_by_default(monkeypatch) -> None:
    questions = normalize_questions([{"question": "What does RAG use?"}])
    seen_judges: list[Judge] = []

    def fake_invoke_judge(
        judge: Judge,
        question: Any,
        result: EvaluationResult,
    ) -> JudgeResult:
        seen_judges.append(judge)
        return JudgeResult.disabled()

    monkeypatch.setattr(comparison, "invoke_judge", fake_invoke_judge)

    report = comparison.evaluate_single_system(
        questions,
        StaticRunner(
            {"What does RAG use?": {"answer": "RAG uses evidence."}},
            [],
            "agentic",
        ),
        timer=lambda: 0.0,
    )

    assert len(seen_judges) == 1
    assert isinstance(seen_judges[0], DisabledJudge)
    assert report.results[0].judge == JudgeResult.disabled()
    assert report.results[0].failure_analysis["failure_type"] == "no_failure"


def test_evaluate_single_system_invokes_judge_once_before_failure_analysis(
    monkeypatch,
) -> None:
    questions = normalize_questions(
        [{"question": "What does RAG use?", "expected_sources": ["notes.md"]}]
    )
    events: list[str] = []

    class RecordingJudge:
        def evaluate(self, question: Any, result: EvaluationResult) -> JudgeResult:
            events.append(f"judge:{question.question}:{result.answer}")
            return JudgeResult.completed(
                {"semantic_correctness": 0.8},
                reason="Judge completed.",
            )

    def fake_attach_failure_analysis(question: Any, result: EvaluationResult) -> EvaluationResult:
        events.append(f"analysis:{result.judge.status}:{result.judge.reason}")
        result.failure_analysis = {"failure_type": "no_failure", "reason": "ok"}
        return result

    monkeypatch.setattr(comparison, "attach_failure_analysis", fake_attach_failure_analysis)

    report = comparison.evaluate_single_system(
        questions,
        StaticRunner(
            {"What does RAG use?": {"answer": "RAG uses evidence."}},
            [],
            "agentic",
        ),
        timer=lambda: 0.0,
        judge=RecordingJudge(),
    )

    assert events == [
        "judge:What does RAG use?:RAG uses evidence.",
        "analysis:completed:Judge completed.",
    ]
    assert report.results[0].judge == JudgeResult.completed(
        {"semantic_correctness": 0.8},
        reason="Judge completed.",
    )
    assert report.results[0].failure_analysis == {
        "failure_type": "no_failure",
        "reason": "ok",
    }


def test_evaluate_single_system_preserves_failed_judge_and_deterministic_fields(
    monkeypatch,
) -> None:
    questions = normalize_questions(
        [{"question": "What does RAG use?", "expected_sources": ["notes.md"]}]
    )

    def fake_attach_failure_analysis(question: Any, result: EvaluationResult) -> EvaluationResult:
        result.failure_analysis = {
            "failure_type": "retrieval_failure",
            "reason": "missing evidence",
        }
        return result

    monkeypatch.setattr(comparison, "attach_failure_analysis", fake_attach_failure_analysis)

    report = comparison.evaluate_single_system(
        questions,
        StaticRunner(
            {
                "What does RAG use?": {
                    "answer": "RAG uses evidence.",
                    "citations": [{"source": "notes.md"}],
                    "retrieved_documents": [{"source": "notes.md"}],
                    "relevant_documents": [{"source": "notes.md"}],
                }
            },
            [],
            "agentic",
        ),
        timer=lambda: 0.0,
        judge=DisabledJudge(),
    )

    result = report.results[0]
    assert result.answer_returned is True
    assert result.source_hit is True
    assert result.error is None
    assert result.judge == JudgeResult.disabled()
    assert result.failure_analysis == {
        "failure_type": "retrieval_failure",
        "reason": "missing evidence",
    }


def test_evaluate_comparison_preserves_dataset_order_and_paired_shape() -> None:
    questions = normalize_questions(
        [
            {
                "id": "q001",
                "question": "First?",
                "expected_keywords": ["first"],
                "expected_sources": ["first.md"],
                "requires_rewrite": True,
            },
            {
                "id": "q002",
                "question": "Second?",
                "expected_keywords": ["second"],
                "expected_sources": ["second.md"],
            },
        ]
    )
    calls: list[tuple[str, str]] = []
    naive_runner: EvaluationRunner = StaticRunner(
        {
            "First?": {"answer": "first"},
            "Second?": {"answer": "second"},
        },
        calls,
        "naive",
    )
    agentic_runner: EvaluationRunner = StaticRunner(
        {
            "First?": {
                "answer": "first",
                "citations": [{"source": "first.md"}],
            },
            "Second?": {
                "answer": "second",
                "citations": [{"source": "second.md"}],
            },
        },
        calls,
        "agentic",
    )
    timer_values: Iterator[float] = iter([0.0, 0.1, 0.1, 0.3, 0.3, 0.6, 0.6, 1.0])

    report = comparison.evaluate_comparison(
        questions,
        agentic_runner=agentic_runner,
        naive_runner=naive_runner,
        timer=lambda: next(timer_values),
    )

    assert calls == [
        ("naive", "First?"),
        ("agentic", "First?"),
        ("naive", "Second?"),
        ("agentic", "Second?"),
    ]
    assert isinstance(report, EvaluationReport)
    assert isinstance(report.summary, ComparisonEvaluationSummary)
    assert all(isinstance(result, PairedEvaluationResult) for result in report.results)
    assert report.summary.mode == "comparison"
    assert report.summary.total_questions == 2
    assert [result.question for result in report.results] == ["First?", "Second?"]
    assert report.results[0].requires_rewrite is True
    assert report.results[1].requires_rewrite is False
    assert [result.naive.question_id for result in report.results] == ["q001", "q002"]
    assert [result.agentic.question_id for result in report.results] == [
        "q001",
        "q002",
    ]
    assert set(report.summary.comparison) == {
        "naive_source_hit_rate",
        "agentic_source_hit_rate",
        "naive_keyword_hit_rate",
        "agentic_keyword_hit_rate",
        "naive_citation_rate",
        "agentic_citation_rate",
        "naive_verification_rate",
        "agentic_verification_rate",
        "naive_fallback_correctness_rate",
        "agentic_fallback_correctness_rate",
        "naive_average_latency",
        "agentic_average_latency",
        "naive_average_semantic_correctness",
        "agentic_average_semantic_correctness",
        "naive_average_groundedness",
        "agentic_average_groundedness",
        "naive_judge_completion_rate",
        "agentic_judge_completion_rate",
    }
    assert report.to_dict()["results"][0].keys() == {
        "question",
        "requires_rewrite",
        "naive",
        "agentic",
    }


def test_evaluate_comparison_invokes_judge_once_per_system_in_existing_order() -> None:
    questions = normalize_questions(
        [
            {"question": "First?"},
            {"question": "Second?"},
        ]
    )
    judge_calls: list[tuple[str, str]] = []

    class RecordingJudge:
        def evaluate(self, question: Any, result: EvaluationResult) -> JudgeResult:
            judge_calls.append((question.question, result.answer))
            return JudgeResult.completed(
                {"semantic_correctness": 0.5},
                reason=f"judged:{result.answer}",
            )

    report = comparison.evaluate_comparison(
        questions,
        agentic_runner=StaticRunner(
            {"First?": {"answer": "agentic-first"}, "Second?": {"answer": "agentic-second"}},
            [],
            "agentic",
        ),
        naive_runner=StaticRunner(
            {"First?": {"answer": "naive-first"}, "Second?": {"answer": "naive-second"}},
            [],
            "naive",
        ),
        timer=lambda: 0.0,
        judge=RecordingJudge(),
    )

    assert judge_calls == [
        ("First?", "naive-first"),
        ("First?", "agentic-first"),
        ("Second?", "naive-second"),
        ("Second?", "agentic-second"),
    ]
    assert report.results[0].naive.judge.status == "completed"
    assert report.results[0].agentic.judge.status == "completed"
    assert report.results[1].naive.judge.status == "completed"
    assert report.results[1].agentic.judge.status == "completed"


@pytest.mark.parametrize(
    ("runner", "expected_error"),
    [
        (
            StaticRunner({}, [], "agentic"),
            "RuntimeError: offline",
        ),
        (
            StaticRunner({"Broken?": {"answer": "x", "retry_count": "many"}}, [], "agentic"),
            "ValueError: retry_count must be an integer",
        ),
    ],
)
def test_evaluate_single_system_skips_judge_when_system_result_unavailable(
    monkeypatch,
    runner: EvaluationRunner,
    expected_error: str,
) -> None:
    questions = normalize_questions([{"question": "Broken?", "expected_sources": ["notes.md"]}])

    class BrokenRunner:
        def run(self, question: str, chat_history: list[ChatMessage]) -> dict[str, object]:
            if isinstance(runner, StaticRunner):
                payload = runner._answers.get(question)
                if payload is None:
                    raise RuntimeError("offline")
                return payload
            raise AssertionError("unexpected runner")

    def fail_invoke_judge(*args: Any, **kwargs: Any) -> JudgeResult:
        raise AssertionError("judge should not be invoked")

    monkeypatch.setattr(comparison, "invoke_judge", fail_invoke_judge)

    report = comparison.evaluate_single_system(
        questions,
        BrokenRunner(),
        timer=lambda: 0.0,
        judge=DisabledJudge(),
    )

    result = report.results[0]
    assert result.error == expected_error
    assert result.answer_returned is False
    assert result.source_hit is False
    assert result.judge == JudgeResult.failed(
        comparison.SYSTEM_RESULT_UNAVAILABLE_ERROR
    )
    assert result.failure_analysis["failure_type"] == "tool_failure"
