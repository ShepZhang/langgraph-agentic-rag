"""Tests for deterministic failed-case attribution."""

from __future__ import annotations

from evaluation.failure_analyzer import analyze_failure, summarize_failure_types


def _question(**overrides):
    question = {
        "id": "q001",
        "question": "How does Agentic RAG use evidence?",
        "answerable": True,
        "expected_sources": ["docs/agentic_rag_notes.md"],
        "requires_rewrite": False,
        "question_type": "single_doc",
    }
    question.update(overrides)
    return question


def _result(**overrides):
    result = {
        "question_id": "q001",
        "correct": True,
        "fallback_correct": True,
        "source_hit": True,
        "context_relevant": True,
        "citation_hit": True,
        "error": None,
        "fallback_triggered": False,
        "answer_returned": True,
        "retry_count": 0,
        "unsupported_claim_count": 0,
        "retrieved_documents": [
            {"source": "/workspace/docs/agentic_rag_notes.md"},
        ],
        "relevant_documents": [
            {"metadata": {"file_path": "/workspace/docs/agentic_rag_notes.md"}},
        ],
        "citations": [
            {"source": "agentic_rag_notes.md"},
        ],
    }
    result.update(overrides)
    return result


def test_successful_case_returns_no_failure():
    analysis = analyze_failure(_question(), _result())

    assert analysis == {
        "question_id": "q001",
        "failure_type": "no_failure",
        "reason": "The case satisfied correctness, fallback, and evidence checks.",
        "suggestion": "No action required.",
    }


def test_error_is_attributed_to_tool_failure_with_trace_suggestion():
    analysis = analyze_failure(
        _question(),
        _result(
            correct=False,
            source_hit=False,
            error="RuntimeError: vector store unavailable",
        ),
    )

    assert analysis["failure_type"] == "tool_failure"
    assert "vector store unavailable" in analysis["reason"]
    assert "trace" in analysis["suggestion"].lower()


def test_answerable_case_that_falls_back_is_fallback_failure():
    analysis = analyze_failure(
        _question(answerable=True),
        _result(
            correct=False,
            fallback_correct=False,
            fallback_triggered=True,
            answer_returned=False,
        ),
    )

    assert analysis["failure_type"] == "fallback_failure"


def test_unanswerable_case_that_answers_is_fallback_failure():
    analysis = analyze_failure(
        _question(answerable=False, expected_sources=[]),
        _result(
            correct=False,
            fallback_correct=False,
            fallback_triggered=False,
            answer_returned=True,
            source_hit=False,
            context_relevant=False,
            citation_hit=False,
            retrieved_documents=[],
            relevant_documents=[],
            citations=[],
        ),
    )

    assert analysis["failure_type"] == "fallback_failure"


def test_query_rewrite_failure_precedes_retrieval_failure_when_no_retry_or_source_hit():
    analysis = analyze_failure(
        _question(requires_rewrite=True, question_type="follow_up"),
        _result(
            correct=False,
            source_hit=False,
            context_relevant=False,
            citation_hit=False,
            retry_count=0,
            retrieved_documents=[],
            relevant_documents=[],
            citations=[],
        ),
    )

    assert analysis["failure_type"] == "query_rewrite_failure"
    assert "standalone" in analysis["suggestion"].lower()


def test_missing_expected_source_in_candidates_is_retrieval_failure():
    analysis = analyze_failure(
        _question(expected_sources=["docs/policy.md"]),
        _result(
            correct=False,
            source_hit=False,
            context_relevant=False,
            citation_hit=False,
            retrieved_documents=[
                {"metadata": {"source": "/workspace/docs/other.md"}},
            ],
            relevant_documents=[],
            citations=[],
        ),
    )

    assert analysis["failure_type"] == "retrieval_failure"
    assert "Expected source" in analysis["reason"]


def test_expected_source_in_candidates_but_not_relevant_or_cited_is_reranking_failure():
    analysis = analyze_failure(
        _question(expected_sources=["policy.md"]),
        _result(
            correct=False,
            source_hit=True,
            context_relevant=False,
            citation_hit=False,
            retrieved_documents=[
                {"document_id": "/workspace/docs/policy.md"},
            ],
            relevant_documents=[
                {"source": "/workspace/docs/other.md"},
            ],
            citations=[],
        ),
    )

    assert analysis["failure_type"] == "reranking_failure"


def test_unsupported_claims_are_citation_failure():
    analysis = analyze_failure(
        _question(),
        _result(
            correct=False,
            unsupported_claim_count=2,
        ),
    )

    assert analysis["failure_type"] == "citation_failure"


def test_evidence_hit_but_incorrect_answer_is_generation_failure():
    analysis = analyze_failure(
        _question(expected_sources=["agentic_rag"]),
        _result(
            correct=False,
            source_hit=True,
            context_relevant=True,
            citation_hit=True,
            retrieved_documents=[
                {"metadata": {"file_path": "/workspace/docs/agentic_rag_notes.md"}},
            ],
            relevant_documents=[
                {"source": "/workspace/docs/agentic_rag_notes.md"},
            ],
            citations=[
                {"source": "/workspace/docs/agentic_rag_notes.md"},
            ],
        ),
    )

    assert analysis["failure_type"] == "generation_failure"


def test_summarize_failure_types_counts_existing_analysis_and_missing_as_tool_failure():
    summary = summarize_failure_types(
        [
            {"failure_analysis": {"failure_type": "no_failure"}},
            {"failure_analysis": {"failure_type": "retrieval_failure"}},
            {"failure_analysis": {"failure_type": "retrieval_failure"}},
            {"failure_analysis": {"failure_type": "tool_failure"}},
            {},
        ]
    )

    assert summary == {
        "no_failure": 1,
        "retrieval_failure": 2,
        "tool_failure": 2,
    }
