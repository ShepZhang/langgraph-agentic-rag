"""Tests for deterministic per-question evaluation scoring."""

from __future__ import annotations

from evaluation.metrics import build_error_result, score_system_output
from evaluation.schemas import EvaluationQuestion


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
