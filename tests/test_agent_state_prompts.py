"""Tests for Agent state and prompt formatting."""

from __future__ import annotations

from agent.prompts import (
    ANSWER_GENERATION_PROMPT,
    ANSWER_REVISION_PROMPT,
    CLAIM_VERIFICATION_PROMPT,
    CITATION_VERIFICATION_PROMPT,
    CLAIM_EXTRACTION_PROMPT,
    QUERY_REWRITE_PROMPT,
    RETRY_QUERY_REWRITE_PROMPT,
    RETRIEVAL_GRADING_PROMPT,
    format_chat_history,
    format_documents,
)
from agent.state import create_initial_state


def test_create_initial_state_sets_defaults():
    state = create_initial_state("What is RAG?")

    assert state["question"] == "What is RAG?"
    assert state["current_query"] == ""
    assert state["rewritten_question"] == ""
    assert state["standalone_question"] == ""
    assert state["query_transform"] == {}
    assert state["query_transform_strategy"] == ""
    assert state["query_transform_reason"] == ""
    assert state["expanded_queries"] == []
    assert state["sub_questions"] == []
    assert state["retrieval_queries"] == []
    assert state["multi_query_used"] is False
    assert state["multi_query_result_count"] == 0
    assert state["chat_history"] == []
    assert state["previous_queries"] == []
    assert state["documents"] == []
    assert state["relevant_documents"] == []
    assert state["document_grades"] == []
    assert state["relevant_document_count"] == 0
    assert state["partial_document_count"] == 0
    assert state["max_relevance_confidence"] == 0.0
    assert state["partial_relevance_recovery"] == {
        "triggered": False,
        "action": "none",
        "reason": "",
        "partial_document_indices": [],
    }
    assert state["grading_reason"] == ""
    assert state["answer"] == ""
    assert state["citations"] == []
    assert state["draft_answer"] == ""
    assert state["used_citation_indices"] == []
    assert state["cited_documents"] == []
    assert state["claims"] == []
    assert state["claim_verification"] == {}
    assert state["claim_verification_results"] == []
    assert state["unsupported_claims"] == []
    assert state["claim_verification_reason"] == ""
    assert state["citation_verification_passed"] is False
    assert state["citation_revision_count"] == 0
    assert state["max_citation_revision_count"] == 1
    assert state["citation_verification_skipped"] is False
    assert state["is_verified"] is False
    assert state["rewrite_count"] == 0
    assert state["retry_count"] == 0
    assert state["retrieval_attempt"] == 0
    assert state["max_retry_count"] == 2
    assert state["is_relevant"] is False
    assert state["route"] == ""
    assert state["fallback_reason"] == ""


def test_create_initial_state_preserves_chat_history():
    history = [{"role": "user", "content": "Tell me about LangGraph"}]

    state = create_initial_state("How does it help?", chat_history=history)

    assert state["chat_history"] == history


def test_format_chat_history_handles_empty_and_nonempty_history():
    assert format_chat_history([]) == "No prior chat history."

    formatted = format_chat_history(
        [
            {"role": "user", "content": "What is RAG?"},
            {"role": "assistant", "content": "Retrieval augmented generation."},
        ]
    )

    assert "user: What is RAG?" in formatted
    assert "assistant: Retrieval augmented generation." in formatted


def test_format_documents_includes_metadata_and_content():
    docs = [
        {
            "content": "Chunk text",
            "source": "paper.pdf",
            "page": 2,
            "chunk_id": "paper.pdf:p2:c1",
            "score": 0.87,
            "rerank_score": 0.93,
        }
    ]

    formatted = format_documents(docs)

    assert "[1]" in formatted
    assert "source=paper.pdf" in formatted
    assert "page=2" in formatted
    assert "chunk_id=paper.pdf:p2:c1" in formatted
    assert "score=0.87" in formatted
    assert "rerank_score=0.93" in formatted
    assert "Chunk text" in formatted


def test_prompts_contain_required_guardrails():
    assert "standalone" in QUERY_REWRITE_PROMPT.lower()
    assert "previous retrieval query" in RETRY_QUERY_REWRITE_PROMPT.lower()
    assert "avoid repeating" in RETRY_QUERY_REWRITE_PROMPT.lower()
    assert "Partial relevance recovery" in RETRY_QUERY_REWRITE_PROMPT
    assert "retrieved chunks" in ANSWER_GENERATION_PROMPT.lower()
    assert "Original user question" in ANSWER_GENERATION_PROMPT
    assert "Retrieval query" in ANSWER_GENERATION_PROMPT
    assert "answer the original user question" in ANSWER_GENERATION_PROMPT
    assert "used_citation_indices" in ANSWER_GENERATION_PROMPT
    assert "citation markers in answer must exactly match" in ANSWER_GENERATION_PROMPT
    assert "weak retrieval evidence" in ANSWER_GENERATION_PROMPT
    assert "citation safety fallback" in ANSWER_GENERATION_PROMPT
    assert "claim_id" in CLAIM_EXTRACTION_PROMPT
    assert "cited_chunk_ids" in CLAIM_EXTRACTION_PROMPT
    assert "verification_label" in CITATION_VERIFICATION_PROMPT
    assert "partially_supported" in CITATION_VERIFICATION_PROMPT
    assert "unsupported" in ANSWER_REVISION_PROMPT.lower()
    assert "used_citation_indices" in ANSWER_REVISION_PROMPT
    assert "Original user question" in RETRIEVAL_GRADING_PROMPT
    assert "Retrieval query" in RETRIEVAL_GRADING_PROMPT
    assert "original user question" in RETRIEVAL_GRADING_PROMPT
    assert "json" in RETRIEVAL_GRADING_PROMPT.lower()
    assert "document_index" in RETRIEVAL_GRADING_PROMPT
    assert "partially_relevant" in RETRIEVAL_GRADING_PROMPT
    assert "confidence" in RETRIEVAL_GRADING_PROMPT
    assert "keyword" in RETRIEVAL_GRADING_PROMPT.lower()
    assert "claim-level citation verifier" in CLAIM_VERIFICATION_PROMPT.lower()
    assert "Original user question" in CLAIM_VERIFICATION_PROMPT
    assert "Answer to verify" in CLAIM_VERIFICATION_PROMPT
    assert "Selected citation chunks" in CLAIM_VERIFICATION_PROMPT
    assert "verified" in CLAIM_VERIFICATION_PROMPT
