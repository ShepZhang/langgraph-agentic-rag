"""Tests for structured retrieval grading parsing."""

from __future__ import annotations

from agent.retrieval_grading import parse_retrieval_grading_response


def test_parse_structured_retrieval_grading_response():
    raw = (
        '{"grades": ['
        '{"document_index": 1, "relevance": "relevant", "confidence": 0.91, '
        '"reason": "Directly answers."},'
        '{"document_index": 2, "relevance": "partially_relevant", "confidence": 0.52, '
        '"reason": "Mentions the topic only."},'
        '{"document_index": 3, "relevance": "irrelevant", "confidence": 0.2, '
        '"reason": "Wrong topic."}'
        '], "reason": "One chunk directly answers."}'
    )

    result = parse_retrieval_grading_response(raw, document_count=3)

    assert result == {
        "grades": [
            {
                "document_index": 1,
                "relevance": "relevant",
                "confidence": 0.91,
                "reason": "Directly answers.",
            },
            {
                "document_index": 2,
                "relevance": "partially_relevant",
                "confidence": 0.52,
                "reason": "Mentions the topic only.",
            },
            {
                "document_index": 3,
                "relevance": "irrelevant",
                "confidence": 0.2,
                "reason": "Wrong topic.",
            },
        ],
        "relevant_indices": [1],
        "partially_relevant_indices": [2],
        "reason": "One chunk directly answers.",
    }


def test_parse_retrieval_grading_reads_fenced_structured_json():
    raw = (
        "```json\n"
        '{"grades": [{"document_index": 1, "relevance": "relevant", '
        '"confidence": 1.0, "reason": "Direct."}], "reason": "ok"}'
        "\n```"
    )

    result = parse_retrieval_grading_response(raw, document_count=1)

    assert result["relevant_indices"] == [1]
    assert result["grades"][0]["confidence"] == 1.0


def test_parse_retrieval_grading_response_keeps_legacy_schema():
    raw = '{"relevant": true, "relevant_indices": [2], "reason": "Chunk 2 answers."}'

    result = parse_retrieval_grading_response(raw, document_count=3)

    assert result["relevant_indices"] == [2]
    assert result["partially_relevant_indices"] == []
    assert result["grades"] == [
        {
            "document_index": 2,
            "relevance": "relevant",
            "confidence": 1.0,
            "reason": "Chunk 2 answers.",
        }
    ]
    assert result["reason"] == "Chunk 2 answers."


def test_parse_retrieval_grading_clamps_confidence_and_invalid_labels():
    raw = (
        '{"grades": ['
        '{"document_index": 1, "relevance": "highly_relevant", "confidence": 2, '
        '"reason": ""},'
        '{"document_index": 2, "relevance": "partially_relevant", "confidence": -1, '
        '"reason": "weak"},'
        '{"document_index": 99, "relevance": "relevant", "confidence": 0.9, '
        '"reason": "out of range"}'
        '], "reason": ""}'
    )

    result = parse_retrieval_grading_response(raw, document_count=2)

    assert result["grades"] == [
        {
            "document_index": 1,
            "relevance": "irrelevant",
            "confidence": 1.0,
            "reason": "No grading reason provided.",
        },
        {
            "document_index": 2,
            "relevance": "partially_relevant",
            "confidence": 0.0,
            "reason": "weak",
        },
    ]
    assert result["relevant_indices"] == []
    assert result["partially_relevant_indices"] == [2]
    assert result["reason"] == "No grading reason provided."


def test_parse_retrieval_grading_invalid_json_returns_empty_result():
    result = parse_retrieval_grading_response("not json", document_count=2)

    assert result == {
        "grades": [],
        "relevant_indices": [],
        "partially_relevant_indices": [],
        "reason": "Could not parse retrieval grading JSON.",
    }
