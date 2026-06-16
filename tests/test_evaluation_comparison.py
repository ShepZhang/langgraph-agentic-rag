"""Tests for typed evaluation orchestration."""

from __future__ import annotations

from collections.abc import Iterator

from agent.state import ChatMessage
from evaluation.comparison import evaluate_comparison, evaluate_single_system
from evaluation.dataset import normalize_questions
from evaluation.runners import EvaluationRunner
from evaluation.schemas import (
    ComparisonEvaluationSummary,
    EvaluationReport,
    EvaluationResult,
    EvaluationSummary,
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

    report = evaluate_single_system(
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

    report = evaluate_comparison(
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
    }
    assert report.to_dict()["results"][0].keys() == {
        "question",
        "requires_rewrite",
        "naive",
        "agentic",
    }
