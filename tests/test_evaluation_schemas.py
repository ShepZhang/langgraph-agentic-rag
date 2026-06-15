"""Tests for typed evaluation-domain records."""

from __future__ import annotations

from evaluation.schemas import (
    ComparisonEvaluationSummary,
    EvaluationQuestion,
    EvaluationReport,
    EvaluationResult,
    EvaluationSummary,
    JudgeResult,
    PairedEvaluationResult,
    RuntimeMetadata,
)


def test_question_preserves_legacy_and_unknown_fields_in_compatibility_dict():
    expected_sources = ["notes.md"]
    expected_keywords = ["retrieval"]
    chat_history = [{"role": "user", "content": "Explain retrieval."}]
    question = EvaluationQuestion(
        id="q001",
        question="What is RAG?",
        question_type="single_doc",
        gold_answer="Retrieval augmented generation.",
        expected_sources=expected_sources,
        expected_keywords=expected_keywords,
        source_match_mode="any",
        answerable=True,
        expected_behavior="answer_with_citation",
        chat_history=chat_history,
        requires_rewrite=False,
        extra_fields={
            "expected_source": "notes.md",
            "difficulty": "easy",
            "answerable": False,
            "source_match_mode": "all",
        },
    )

    payload = question.to_compat_dict()

    assert payload["expected_source"] == "notes.md"
    assert payload["difficulty"] == "easy"
    assert payload["answerable"] is True
    assert payload["should_answer"] is True
    assert payload["expected_sources"] == ["notes.md"]
    assert payload["source_match_mode"] == "any"
    assert payload["expected_sources"] is not expected_sources
    assert payload["expected_keywords"] is not expected_keywords
    assert payload["chat_history"] is not chat_history
    assert payload["chat_history"][0] is not chat_history[0]


def test_unavailable_metrics_serialize_as_none():
    summary = EvaluationSummary.empty()

    payload = summary.to_dict()

    assert payload["unsupported_claim_count"] is None
    assert payload["supported_claim_ratio"] is None
    assert payload["citation_verification_pass_rate"] is None
    assert payload["total_questions"] == 0
    assert payload["failure_type_counts"] == {}


def test_empty_result_covers_current_single_question_result_shape():
    result = EvaluationResult.empty(
        question_id="q001",
        question_type="single_doc",
        question="What is RAG?",
    )

    assert set(result.to_dict()) == {
        "question_id",
        "question_type",
        "question",
        "chat_history_supplied",
        "chat_history_used",
        "answer_returned",
        "fallback_triggered",
        "fallback_correct",
        "correct",
        "context_relevant",
        "citation_hit",
        "citation_returned",
        "is_verified",
        "citation_verification_applicable",
        "claim_count",
        "unsupported_claim_count",
        "supported_claim_count",
        "total_claim_count",
        "source_hit",
        "keyword_hit",
        "citation_verification_passed",
        "rewrite_triggered",
        "retry_count",
        "retrieved_doc_count",
        "relevant_doc_count",
        "token_usage",
        "estimated_cost",
        "latency",
        "error",
        "answer",
        "citations",
        "claims",
        "claim_verification_results",
        "retrieved_documents",
        "relevant_documents",
        "failure_analysis",
    }


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

    assert set(single.to_dict()) == {"summary", "results"}
    assert single.to_dict()["results"][0]["question_id"] == "q001"
    assert comparison.to_dict()["summary"]["mode"] == "comparison"
    assert comparison.to_dict()["results"][0]["naive"]["question_id"] == "q001"
    assert comparison.to_dict()["results"][0]["agentic"]["question_id"] == "q001"


def test_runtime_metadata_puts_versions_first_and_expands_config():
    metadata = RuntimeMetadata(
        schema_version=1,
        evaluator_version="p4c",
        config={"llm": {"model": "test-model"}},
    )

    payload = metadata.to_dict()

    assert list(payload) == ["schema_version", "evaluator_version", "llm"]
    assert payload == {
        "schema_version": 1,
        "evaluator_version": "p4c",
        "llm": {"model": "test-model"},
    }


def test_judge_result_distinguishes_disabled_failed_and_completed_states():
    disabled = JudgeResult.disabled()
    failed = JudgeResult.failed("RuntimeError: unavailable")
    completed = JudgeResult.completed(
        {"semantic_correctness": 0.8},
        reason="The answer matches the reference.",
    )

    assert disabled.status == "disabled"
    assert disabled.scores == {}
    assert disabled.error is None
    assert failed.status == "failed"
    assert failed.error == "RuntimeError: unavailable"
    assert completed.status == "completed"
    assert completed.scores == {"semantic_correctness": 0.8}
    assert completed.reason == "The answer matches the reference."
