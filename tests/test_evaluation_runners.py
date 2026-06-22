"""Tests for evaluation runner adaptation and single-question execution."""

from __future__ import annotations

from collections.abc import Iterator

from agent.state import ChatMessage
from evaluation.dataset import normalize_question
from evaluation.runners import CallableRunnerAdapter, evaluate_question


def test_callable_runner_adapter_invokes_one_argument_runner() -> None:
    calls: list[str] = []

    def runner(question: str) -> dict[str, object]:
        calls.append(question)
        return {"answer": "retrieval augmented generation"}

    adapter = CallableRunnerAdapter(runner)

    assert adapter.run("What is RAG?", []) == {
        "answer": "retrieval augmented generation"
    }
    assert calls == ["What is RAG?"]


def test_callable_runner_adapter_passes_history_to_two_argument_runner() -> None:
    history: list[ChatMessage] = [
        {"role": "user", "content": "Tell me about RAG."},
        {"role": "assistant", "content": "RAG retrieves evidence."},
    ]
    calls: list[tuple[str, list[ChatMessage]]] = []

    def runner(
        question: str,
        chat_history: list[ChatMessage],
    ) -> dict[str, object]:
        calls.append((question, chat_history))
        return {"answer": "It uses prior turns.", "chat_history_used": True}

    adapter = CallableRunnerAdapter(runner)

    assert adapter.run("What did we discuss?", history) == {
        "answer": "It uses prior turns.",
        "chat_history_used": True,
    }
    assert calls == [("What did we discuss?", history)]


def test_evaluate_question_records_runner_errors_latency_and_empty_failure_analysis() -> None:
    question = normalize_question(
        {
            "id": "q-error",
            "question": "What is unavailable?",
            "expected_sources": ["notes.md"],
            "chat_history": [{"role": "user", "content": "Use the notes."}],
        },
        0,
    )
    timer_values = iter([10.0, 12.5])

    def fake_timer() -> float:
        return next(timer_values)

    class OfflineRunner:
        def run(
            self,
            _question: str,
            _chat_history: list[ChatMessage],
        ) -> dict[str, object]:
            raise RuntimeError("offline")

    result = evaluate_question(question, OfflineRunner(), timer=fake_timer)

    assert result.question_id == "q-error"
    assert result.question == "What is unavailable?"
    assert result.chat_history_supplied is True
    assert result.answer_returned is False
    assert result.latency == 2.5
    assert result.error == "RuntimeError: offline"
    assert result.failure_analysis == {}


def test_evaluate_question_scores_successful_runner_output() -> None:
    question = normalize_question(
        {
            "id": "q-success",
            "question": "How does RAG answer?",
            "expected_keywords": ["evidence"],
            "expected_sources": ["notes.md"],
        },
        0,
    )
    timer_values: Iterator[float] = iter([1.0, 1.25])

    def fake_timer() -> float:
        return next(timer_values)

    class SuccessfulRunner:
        def run(
            self,
            _question: str,
            _chat_history: list[ChatMessage],
        ) -> dict[str, object]:
            return {
                "answer": "RAG answers with evidence.",
                "citations": [{"source": "notes.md"}],
                "retrieved_documents": [{"source": "notes.md"}],
                "relevant_documents": [{"source": "notes.md"}],
            }

    result = evaluate_question(question, SuccessfulRunner(), timer=fake_timer)

    assert result.answer_returned is True
    assert result.keyword_hit is True
    assert result.source_hit is True
    assert result.latency == 0.25
    assert result.error is None
    assert result.failure_analysis == {}
