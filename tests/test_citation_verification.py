"""Tests for claim-level citation verification parsing."""

from __future__ import annotations

from agent.citation_verification import (
    build_claim_verification_summary,
    parse_citation_verification_response,
    parse_claim_extraction_response,
)


def test_parse_claim_extraction_response_keeps_valid_cited_chunk_ids():
    result = parse_claim_extraction_response(
        (
            '{"claims": ['
            '{"claim_id": "c001", "claim": "Agentic RAG uses grading.", '
            '"cited_chunk_ids": ["chunk-1", "missing"]}'
            '], "reason": "one claim"}'
        ),
        valid_chunk_ids=["chunk-1"],
    )

    assert result == {
        "claims": [
            {
                "claim_id": "c001",
                "claim": "Agentic RAG uses grading.",
                "cited_chunk_ids": ["chunk-1"],
            }
        ],
        "reason": "one claim",
    }


def test_parse_claim_extraction_response_drops_invalid_claims():
    result = parse_claim_extraction_response(
        (
            '{"claims": ['
            '{"claim_id": "", "claim": "Blank id.", "cited_chunk_ids": ["chunk-1"]},'
            '{"claim_id": "c002", "claim": "", "cited_chunk_ids": ["chunk-1"]},'
            '{"claim_id": "c003", "claim": "Valid claim.", "cited_chunk_ids": ["chunk-2"]}'
            '], "reason": ""}'
        ),
        valid_chunk_ids=["chunk-1"],
    )

    assert result == {
        "claims": [
            {
                "claim_id": "c003",
                "claim": "Valid claim.",
                "cited_chunk_ids": [],
            }
        ],
        "reason": "No claim extraction reason provided.",
    }


def test_parse_claim_extraction_response_returns_none_for_invalid_json():
    assert (
        parse_claim_extraction_response("not json", valid_chunk_ids=["chunk-1"])
        is None
    )


def test_parse_citation_verification_response_marks_invalid_confidence_unsupported():
    result = parse_citation_verification_response(
        (
            '{"results": ['
            '{"claim_id": "c001", "claim": "A", "cited_chunk_ids": ["chunk-1"], '
            '"verification_label": "SUPPORTED", "confidence": 2, "reason": "ok"},'
            '{"claim_id": "c002", "claim": "B", "cited_chunk_ids": ["chunk-2"], '
            '"verification_label": "unknown", "confidence": -1, "reason": "bad"}'
            '], "reason": "checked"}'
        ),
        valid_chunk_ids=["chunk-1", "chunk-2"],
    )

    assert result is not None
    assert result["results"][0] == {
        "claim_id": "c001",
        "claim": "A",
        "cited_chunk_ids": ["chunk-1"],
        "verification_label": "unsupported",
        "confidence": 0.0,
        "reason": "ok",
    }
    assert result["results"][1] == {
        "claim_id": "c002",
        "claim": "B",
        "cited_chunk_ids": ["chunk-2"],
        "verification_label": "unsupported",
        "confidence": 0.0,
        "reason": "bad",
    }
    assert result["reason"] == "checked"


def test_parse_citation_verification_response_ignores_invalid_claim_records():
    result = parse_citation_verification_response(
        (
            '{"results": ['
            '{"claim_id": "", "claim": "bad", "cited_chunk_ids": ["chunk-1"], '
            '"verification_label": "supported", "confidence": 0.9, "reason": "bad"},'
            '{"claim_id": "c002", "claim": "Valid", "cited_chunk_ids": ["missing"], '
            '"verification_label": "supported", "confidence": 0.9, "reason": ""}'
            '], "reason": ""}'
        ),
        valid_chunk_ids=["chunk-1"],
    )

    assert result == {
        "results": [
            {
                "claim_id": "c002",
                "claim": "Valid",
                "cited_chunk_ids": [],
                "verification_label": "supported",
                "confidence": 0.9,
                "reason": "No citation verification reason provided.",
            }
        ],
        "reason": "No citation verification reason provided.",
    }


def test_parse_citation_verification_response_returns_none_for_invalid_json():
    assert (
        parse_citation_verification_response("not json", valid_chunk_ids=["chunk-1"])
        is None
    )


def test_build_claim_verification_summary_counts_unsupported_claims():
    results = [
        {
            "claim_id": "c001",
            "claim": "A",
            "cited_chunk_ids": ["chunk-1"],
            "verification_label": "supported",
            "confidence": 0.9,
            "reason": "ok",
        },
        {
            "claim_id": "c002",
            "claim": "B",
            "cited_chunk_ids": [],
            "verification_label": "unsupported",
            "confidence": 0.2,
            "reason": "no support",
        },
    ]

    summary = build_claim_verification_summary(results, reason="checked")

    assert summary == {
        "verified": False,
        "results": results,
        "reason": "checked",
        "unsupported_claims": [results[1]],
    }


def test_build_claim_verification_summary_requires_all_supported():
    results = [
        {
            "claim_id": "c001",
            "claim": "A",
            "cited_chunk_ids": ["chunk-1"],
            "verification_label": "supported",
            "confidence": 0.9,
            "reason": "ok",
        },
        {
            "claim_id": "c002",
            "claim": "B",
            "cited_chunk_ids": ["chunk-1"],
            "verification_label": "partially_supported",
            "confidence": 0.6,
            "reason": "too broad",
        },
    ]

    summary = build_claim_verification_summary(results, reason="")

    assert summary["verified"] is False
    assert summary["unsupported_claims"] == [results[1]]
    assert summary["reason"] == "One or more claims are unsupported."
