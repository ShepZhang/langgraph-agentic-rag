"""Tests for typed evaluation-domain records."""

from __future__ import annotations

from typing import Any, cast, get_type_hints

import pytest

from agent.state import Citation, RetrievedDocument
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


def test_question_copies_nested_inputs_and_serialized_payloads():
    expected_sources = ["notes.md"]
    extra_fields = {"metadata": {"tags": ["baseline"]}}
    question = EvaluationQuestion(
        id="q001",
        question="What is RAG?",
        question_type="single_doc",
        gold_answer="Retrieval augmented generation.",
        expected_sources=expected_sources,
        expected_keywords=["retrieval"],
        source_match_mode="any",
        answerable=True,
        expected_behavior="answer_with_citation",
        chat_history=[{"role": "user", "content": "Explain retrieval."}],
        requires_rewrite=False,
        extra_fields=extra_fields,
    )

    expected_sources.append("mutated.md")
    extra_fields["metadata"]["tags"].append("mutated")
    payload = question.to_compat_dict()
    payload["expected_sources"].append("serialized.md")
    payload["metadata"]["tags"].append("serialized")

    assert question.to_compat_dict()["expected_sources"] == ["notes.md"]
    assert question.to_compat_dict()["metadata"] == {"tags": ["baseline"]}


def test_unavailable_metrics_serialize_as_none():
    summary = EvaluationSummary.empty()

    payload = summary.to_dict()

    assert payload["unsupported_claim_count"] is None
    assert payload["supported_claim_ratio"] is None
    assert payload["citation_verification_pass_rate"] is None
    assert payload["total_questions"] == 0
    assert payload["failure_type_counts"] == {}


def test_summary_copies_failure_counts_input_and_serialized_payload():
    failure_type_counts = {"no_failure": 1}
    summary = EvaluationSummary(failure_type_counts=failure_type_counts)

    failure_type_counts["no_failure"] = 2
    payload = summary.to_dict()
    payload["failure_type_counts"]["no_failure"] = 3

    assert summary.to_dict()["failure_type_counts"] == {"no_failure": 1}


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


def test_result_copies_structured_inputs_and_serialized_payloads():
    citations = [{"source": "notes.md", "snippet": "Evidence."}]
    retrieved_documents = [{"source": "notes.md", "matched_queries": ["rag"]}]
    failure_analysis = {"failure_type": "no_failure"}
    result = EvaluationResult(
        question_id="q001",
        question_type="single_doc",
        question="What is RAG?",
        citations=citations,
        retrieved_documents=retrieved_documents,
        relevant_documents=retrieved_documents,
        failure_analysis=failure_analysis,
    )

    citations[0]["source"] = "mutated.md"
    retrieved_documents[0]["matched_queries"].append("mutated")
    failure_analysis["failure_type"] = "tool_failure"
    payload = result.to_dict()
    payload["citations"][0]["source"] = "serialized.md"
    payload["retrieved_documents"][0]["matched_queries"].append("serialized")
    payload["failure_analysis"]["failure_type"] = "citation_failure"

    fresh_payload = result.to_dict()
    assert fresh_payload["citations"][0]["source"] == "notes.md"
    assert fresh_payload["retrieved_documents"][0]["matched_queries"] == ["rag"]
    assert fresh_payload["failure_analysis"]["failure_type"] == "no_failure"


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


def test_comparison_summary_copies_nested_comparison_data():
    comparison_data = {
        "source_hit": {"naive": 0.5, "agentic": 1.0},
    }
    comparison = ComparisonEvaluationSummary(
        total_questions=1,
        naive=EvaluationSummary.empty(),
        agentic=EvaluationSummary.empty(),
        comparison=comparison_data,
    )

    comparison_data["source_hit"]["naive"] = 0.0
    payload = comparison.to_dict()
    payload["comparison"]["source_hit"]["agentic"] = 0.0

    assert comparison.to_dict()["comparison"]["source_hit"] == {
        "naive": 0.5,
        "agentic": 1.0,
    }


def test_comparison_summary_copies_nested_system_summaries():
    naive = EvaluationSummary(total_questions=1, answer_rate=0.5)
    agentic = EvaluationSummary(total_questions=1, answer_rate=1.0)
    comparison = ComparisonEvaluationSummary(
        total_questions=1,
        naive=naive,
        agentic=agentic,
        comparison={},
    )

    naive.answer_rate = 0.0
    agentic.answer_rate = 0.0

    payload = comparison.to_dict()
    assert payload["naive"]["answer_rate"] == 0.5
    assert payload["agentic"]["answer_rate"] == 1.0


def test_report_rejects_single_summary_with_paired_results():
    result = EvaluationResult.empty("q001", "single_doc", "What is RAG?")
    paired = PairedEvaluationResult(
        question="What is RAG?",
        requires_rewrite=False,
        naive=result,
        agentic=result,
    )

    with pytest.raises(ValueError, match="single-system"):
        EvaluationReport(summary=EvaluationSummary.empty(), results=[paired])


def test_report_rejects_comparison_summary_with_single_results():
    result = EvaluationResult.empty("q001", "single_doc", "What is RAG?")
    summary = ComparisonEvaluationSummary(
        total_questions=1,
        naive=EvaluationSummary.empty(),
        agentic=EvaluationSummary.empty(),
        comparison={},
    )

    with pytest.raises(ValueError, match="comparison"):
        EvaluationReport(summary=summary, results=[result])


def test_report_rejects_unknown_summary_type_immediately():
    result = EvaluationResult.empty("q001", "single_doc", "What is RAG?")

    with pytest.raises(ValueError, match="evaluation summary"):
        EvaluationReport(summary=cast(Any, object()), results=[result])


def test_report_copies_results_input_list():
    result = EvaluationResult.empty("q001", "single_doc", "What is RAG?")
    results = [result]
    report = EvaluationReport(
        summary=EvaluationSummary.empty(),
        results=results,
    )

    results.append(EvaluationResult.empty("q002", "single_doc", "What is retrieval?"))
    result.answer = "mutated"

    assert len(report.to_dict()["results"]) == 1
    assert report.to_dict()["results"][0]["answer"] == ""


def test_single_report_copies_summary_snapshot():
    summary = EvaluationSummary(
        total_questions=1,
        answer_rate=1.0,
        failure_type_counts={"no_failure": 1},
    )
    report = EvaluationReport(summary=summary, results=[])

    summary.answer_rate = 0.0
    summary.failure_type_counts["no_failure"] = 2

    assert report.to_dict()["summary"]["answer_rate"] == 1.0
    assert report.to_dict()["summary"]["failure_type_counts"] == {
        "no_failure": 1,
    }


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


def test_runtime_metadata_protects_versions_and_copies_nested_config():
    config = {
        "schema_version": 999,
        "evaluator_version": "wrong",
        "llm": {"model": "test-model"},
    }
    metadata = RuntimeMetadata(
        schema_version=1,
        evaluator_version="p4c",
        config=config,
    )

    config["llm"]["model"] = "mutated-input"
    payload = metadata.to_dict()
    payload["llm"]["model"] = "mutated-payload"

    assert metadata.to_dict() == {
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


def test_judge_result_copies_input_scores():
    scores = {"semantic_correctness": 0.8}

    result = JudgeResult(status="completed", scores=scores)
    scores["semantic_correctness"] = 0.1

    assert result.scores == {"semantic_correctness": 0.8}


def test_result_uses_existing_document_contract_types():
    hints = get_type_hints(EvaluationResult)

    assert hints["citations"] == list[Citation]
    assert hints["claims"] == list[dict[str, object]]
    assert hints["claim_verification_results"] == list[dict[str, object]]
    assert hints["retrieved_documents"] == list[RetrievedDocument]
    assert hints["relevant_documents"] == list[RetrievedDocument]
    assert hints["failure_analysis"] == dict[str, str]
