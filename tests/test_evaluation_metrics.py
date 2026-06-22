"""Tests for deterministic per-question evaluation scoring."""

from __future__ import annotations

from evaluation.dataset import normalize_question
from evaluation.metrics import (
    DEFAULT_SUMMARY_METRICS,
    build_error_result,
    score_system_output,
    summarize_results,
)
from evaluation.schemas import EvaluationQuestion, EvaluationResult, JudgeResult


def _question(**overrides: object) -> EvaluationQuestion:
    defaults = {
        "id": "q-metrics",
        "question": "How does citation verification work?",
        "question_type": "single_doc",
        "gold_answer": "It checks claims against cited evidence.",
        "expected_sources": ["agentic_rag_notes.md"],
        "expected_keywords": ["claims", "evidence"],
        "source_match_mode": "any",
        "answerable": True,
        "expected_behavior": "answer_with_citation",
        "chat_history": [],
        "requires_rewrite": False,
    }
    defaults.update(overrides)
    return EvaluationQuestion(**defaults)  # type: ignore[arg-type]


def test_score_system_output_preserves_reliability_fields():
    question = _question(chat_history=[{"role": "user", "content": "Prior turn"}])
    result = score_system_output(
        question,
        {
            "answer": "It checks claims against cited evidence.",
            "citations": [{"source": "agentic_rag_notes.md"}],
            "claims": [{"claim": "Claim verification checks evidence."}],
            "claim_verification_results": [
                {"claim": "Claim verification checks evidence.", "supported": True}
            ],
            "retrieved_documents": [{"source": "agentic_rag_notes.md"}],
            "relevant_documents": [{"source": "agentic_rag_notes.md"}],
            "chat_history_used": True,
            "is_verified": True,
            "citation_verification_passed": True,
            "retry_count": 2,
            "token_usage": {"total_tokens": 42},
            "estimated_cost": 0.000123,
        },
    )

    assert result.question_id == "q-metrics"
    assert result.chat_history_supplied is True
    assert result.chat_history_used is True
    assert result.answer_returned is True
    assert result.fallback_triggered is False
    assert result.context_relevant is True
    assert result.citation_hit is True
    assert result.source_hit is True
    assert result.keyword_hit is True
    assert result.is_verified is True
    assert result.citation_verification_applicable is True
    assert result.citation_verification_passed is True
    assert result.claim_count == 1
    assert result.supported_claim_count == 1
    assert result.unsupported_claim_count == 0
    assert result.total_claim_count == 1
    assert result.rewrite_triggered is True
    assert result.retry_count == 2
    assert result.token_usage == {"total_tokens": 42}
    assert result.estimated_cost == 0.000123


def test_score_system_output_ignores_invalid_optional_cost():
    result = score_system_output(
        _question(),
        {
            "answer": "It checks claims against cited evidence.",
            "estimated_cost": "not-a-number",
        },
    )

    assert result.estimated_cost is None


def test_score_system_output_honors_all_source_match_mode():
    question = _question(
        expected_sources=["agentic_rag_notes.md", "pipeline.md"],
        source_match_mode="all",
    )

    partial = score_system_output(
        question,
        {
            "answer": "It checks claims against cited evidence.",
            "citations": [{"source": "agentic_rag_notes.md"}],
            "retrieved_documents": [{"source": "agentic_rag_notes.md"}],
            "relevant_documents": [{"source": "agentic_rag_notes.md"}],
        },
    )
    complete = score_system_output(
        question,
        {
            "answer": "It checks claims against cited evidence.",
            "citations": [
                {"source": "agentic_rag_notes.md"},
                {"source": "pipeline.md"},
            ],
            "retrieved_documents": [
                {"source": "agentic_rag_notes.md"},
                {"source": "pipeline.md"},
            ],
            "relevant_documents": [
                {"source": "agentic_rag_notes.md"},
                {"source": "pipeline.md"},
            ],
        },
    )

    assert partial.citation_hit is False
    assert partial.context_relevant is False
    assert partial.source_hit is False
    assert complete.citation_hit is True
    assert complete.context_relevant is True
    assert complete.source_hit is True


def test_build_error_result_uses_question_identity_and_no_false_metrics():
    question = _question(
        id="q-error",
        question="What failed?",
        question_type="unanswerable",
        answerable=False,
        expected_behavior="fallback",
        chat_history=[{"role": "user", "content": "Earlier"}],
    )

    result = build_error_result(question)

    assert result.question_id == "q-error"
    assert result.question == "What failed?"
    assert result.question_type == "unanswerable"
    assert result.chat_history_supplied is True
    assert result.chat_history_used is False
    assert result.answer_returned is False
    assert result.fallback_triggered is False
    assert result.fallback_correct is False
    assert result.correct is False
    assert result.context_relevant is False
    assert result.citation_hit is False
    assert result.citation_verification_applicable is False
    assert result.unsupported_claim_count is None
    assert result.supported_claim_count is None
    assert result.total_claim_count is None
    assert result.source_hit is False
    assert result.keyword_hit is False
    assert result.estimated_cost is None


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


def test_summarize_results_computes_judge_completion_counts():
    """completed/failed counts; attempted excludes disabled."""
    question = normalize_question({"question": "Judge test?"}, index=0)

    r_completed = EvaluationResult.empty(
        question_id="q1", question_type="single_doc", question="Judge test?"
    )
    r_completed.answer_returned = True
    r_completed.judge = JudgeResult.completed(
        {"semantic_correctness": 0.75, "groundedness": 0.5},
        reason="ok",
    )

    r_failed = EvaluationResult.empty(
        question_id="q2", question_type="single_doc", question="Judge test?"
    )
    r_failed.answer_returned = True
    r_failed.judge = JudgeResult.failed("some error")

    r_disabled = EvaluationResult.empty(
        question_id="q3", question_type="single_doc", question="Judge test?"
    )
    r_disabled.answer_returned = True
    r_disabled.judge = JudgeResult.disabled()

    summary = summarize_results(
        [r_completed, r_failed, r_disabled],
        [question, question, question],
    )

    assert summary.judge_completed_count == 1
    assert summary.judge_failed_count == 1
    assert summary.judge_completion_rate == 0.5  # 1/2 attempted, rounded 4
    assert summary.average_semantic_correctness == 0.75
    assert summary.average_groundedness == 0.5
    assert summary.groundedness_applicable_count == 1


def test_summarize_results_judge_metrics_none_when_all_disabled_or_empty():
    """Disabled or empty judge results → all judge metrics are None."""
    question = normalize_question({"question": "No judge?"}, index=0)

    r_disabled = EvaluationResult.empty(
        question_id="q1", question_type="single_doc", question="No judge?"
    )
    r_disabled.answer_returned = True
    r_disabled.judge = JudgeResult.disabled()

    summary = summarize_results([r_disabled], [question])

    assert summary.judge_completed_count == 0
    assert summary.judge_failed_count == 0
    assert summary.judge_completion_rate is None
    assert summary.average_semantic_correctness is None
    assert summary.average_groundedness is None
    assert summary.groundedness_applicable_count == 0

    # Empty results list
    summary_empty = summarize_results([], [])
    assert summary_empty.judge_completion_rate is None
    assert summary_empty.average_semantic_correctness is None
    assert summary_empty.average_groundedness is None
    assert summary_empty.groundedness_applicable_count == 0


def test_summarize_results_ignores_invalid_judge_scores():
    """String, NaN, Inf, and bool scores are ignored for both semantic and
    groundedness; only finite numeric values contribute to count and average."""
    question = normalize_question({"question": "Invalid scores?"}, index=0)

    r_str = EvaluationResult.empty(
        question_id="q1", question_type="single_doc", question="Invalid scores?"
    )
    r_str.answer_returned = True
    r_str.judge = JudgeResult.completed(
        {"semantic_correctness": "not-a-number", "groundedness": 0.5},
        reason="string semantic, valid grounded",
    )

    r_nan = EvaluationResult.empty(
        question_id="q2", question_type="single_doc", question="Invalid scores?"
    )
    r_nan.answer_returned = True
    r_nan.judge = JudgeResult.completed(
        {"semantic_correctness": float("nan"), "groundedness": float("nan")},
        reason="nan semantic, nan grounded",
    )

    r_inf = EvaluationResult.empty(
        question_id="q3", question_type="single_doc", question="Invalid scores?"
    )
    r_inf.answer_returned = True
    r_inf.judge = JudgeResult.completed(
        {"semantic_correctness": float("inf"), "groundedness": float("inf")},
        reason="inf semantic, inf grounded",
    )

    r_bool = EvaluationResult.empty(
        question_id="q4", question_type="single_doc", question="Invalid scores?"
    )
    r_bool.answer_returned = True
    r_bool.judge = JudgeResult.completed(
        {"semantic_correctness": True, "groundedness": True},
        reason="bool semantic, bool grounded",
    )

    r_str_grounded = EvaluationResult.empty(
        question_id="q5", question_type="single_doc", question="Invalid scores?"
    )
    r_str_grounded.answer_returned = True
    r_str_grounded.judge = JudgeResult.completed(
        {"semantic_correctness": 0.5, "groundedness": "bad_value"},
        reason="valid semantic, string grounded",
    )

    r_finite = EvaluationResult.empty(
        question_id="q6", question_type="single_doc", question="Invalid scores?"
    )
    r_finite.answer_returned = True
    r_finite.judge = JudgeResult.completed(
        {"semantic_correctness": 0.75, "groundedness": 0.3},
        reason="valid",
    )

    summary = summarize_results(
        [r_str, r_nan, r_inf, r_bool, r_str_grounded, r_finite],
        [question] * 6,
    )

    # Semantic: only r_str_grounded (0.5) and r_finite (0.75) are finite
    assert summary.average_semantic_correctness == 0.625  # (0.5 + 0.75) / 2

    # Groundedness: r_str=0.5 (valid), r_finite=0.3 (valid).
    # r_nan=NaN (ignored), r_inf=Inf (ignored), r_bool=True (ignored),
    # r_str_grounded="bad_value" (ignored).
    # Average: (0.5 + 0.3) / 2 = 0.4
    assert summary.average_groundedness == 0.4
    assert summary.groundedness_applicable_count == 2


def test_summarize_results_ignores_overflowing_judge_score():
    question = normalize_question({"question": "Overflowing score?"}, index=0)
    result = EvaluationResult.empty(
        question_id=question.id,
        question_type=question.question_type,
        question=question.question,
    )
    result.answer_returned = True
    result.judge = JudgeResult.completed(
        {"semantic_correctness": 10**309, "groundedness": 0.5},
        reason="overflowing semantic score",
    )

    summary = summarize_results([result], [question])

    assert summary.average_semantic_correctness is None
    assert summary.average_groundedness == 0.5
    assert summary.groundedness_applicable_count == 1


def test_summarize_results_ignores_judge_scores_outside_normalized_range():
    question = normalize_question({"question": "Out-of-range scores?"}, index=0)
    high = EvaluationResult.empty(
        question_id="q-high",
        question_type=question.question_type,
        question=question.question,
    )
    high.judge = JudgeResult.completed(
        {"semantic_correctness": 4.0, "groundedness": 0.75},
        reason="raw score leaked into normalized field",
    )
    low = EvaluationResult.empty(
        question_id="q-low",
        question_type=question.question_type,
        question=question.question,
    )
    low.judge = JudgeResult.completed(
        {"semantic_correctness": 0.5, "groundedness": -1.0},
        reason="negative normalized score",
    )

    summary = summarize_results([high, low], [question, question])

    assert summary.average_semantic_correctness == 0.5
    assert summary.average_groundedness == 0.75
    assert summary.groundedness_applicable_count == 1


def test_summarize_results_groundedness_excludes_none_scores():
    """Groundedness None scores are excluded from count and average."""
    question = normalize_question({"question": "None groundedness?"}, index=0)

    r_none = EvaluationResult.empty(
        question_id="q1", question_type="single_doc", question="None groundedness?"
    )
    r_none.answer_returned = True
    r_none.judge = JudgeResult.completed(
        {"semantic_correctness": 0.5, "groundedness": None},
        reason="fallback case",
    )

    r_scored = EvaluationResult.empty(
        question_id="q2", question_type="single_doc", question="None groundedness?"
    )
    r_scored.answer_returned = True
    r_scored.judge = JudgeResult.completed(
        {"semantic_correctness": 0.75, "groundedness": 1.0},
        reason="full support",
    )

    summary = summarize_results([r_none, r_scored], [question, question])

    assert summary.groundedness_applicable_count == 1  # only r_scored
    assert summary.average_groundedness == 1.0  # only r_scored's score
    assert summary.average_semantic_correctness == 0.625  # (0.5 + 0.75) / 2
