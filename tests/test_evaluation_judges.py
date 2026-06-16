"""Tests for optional evaluation judge contracts."""

from __future__ import annotations

from evaluation.judges import DisabledJudge, invoke_judge
from evaluation.schemas import EvaluationQuestion, EvaluationResult


def _question() -> EvaluationQuestion:
    return EvaluationQuestion(
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
    )


def _result() -> EvaluationResult:
    return EvaluationResult.empty(
        question_id="q001",
        question_type="single_doc",
        question="What is RAG?",
    )


def test_disabled_judge_performs_no_scoring():
    judge = DisabledJudge()

    result = judge.evaluate(_question(), _result())

    assert result.status == "disabled"
    assert result.scores == {}
    assert result.reason == ""
    assert result.error is None


def test_invoke_judge_records_failure_without_raising():
    class BrokenJudge:
        def evaluate(
            self,
            question: EvaluationQuestion,
            result: EvaluationResult,
        ):
            raise RuntimeError("judge unavailable")

    result = invoke_judge(BrokenJudge(), _question(), _result())

    assert result.status == "failed"
    assert result.error == "RuntimeError: judge unavailable"
    assert result.scores == {}
