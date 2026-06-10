"""Tests for multi-query retrieval helpers."""

from __future__ import annotations

from agent.multi_query import build_retrieval_queries, merge_retrieved_documents


def test_build_retrieval_queries_uses_expanded_queries_for_multi_query_only():
    queries = build_retrieval_queries(
        current_query="Agentic RAG advantages",
        expanded_queries=[
            "Agentic RAG benefits",
            "",
            "Agentic RAG advantages",
            "reliability controls",
        ],
        strategy="multi_query",
    )

    assert queries == [
        "Agentic RAG advantages",
        "Agentic RAG benefits",
        "reliability controls",
    ]


def test_build_retrieval_queries_keeps_single_query_for_rewrite_and_decomposition():
    assert build_retrieval_queries(
        current_query="Compare Agentic RAG and naive RAG",
        expanded_queries=["Agentic RAG benefits"],
        strategy="rewrite",
    ) == ["Compare Agentic RAG and naive RAG"]
    assert build_retrieval_queries(
        current_query="Compare Agentic RAG and naive RAG",
        expanded_queries=["Agentic RAG benefits"],
        strategy="decomposition",
    ) == ["Compare Agentic RAG and naive RAG"]


def test_build_retrieval_queries_falls_back_when_current_query_is_blank():
    assert build_retrieval_queries(
        current_query=" ",
        expanded_queries=["Agentic RAG benefits"],
        strategy="multi_query",
    ) == ["Agentic RAG benefits"]


def test_merge_retrieved_documents_deduplicates_by_chunk_id_and_tracks_queries():
    query_results = [
        (
            "Agentic RAG advantages",
            [
                {
                    "content": "Agentic RAG adds grading.",
                    "source": "notes.md",
                    "chunk_id": "notes:c1",
                    "score": 0.9,
                },
                {
                    "content": "Hybrid retrieval combines signals.",
                    "source": "retrieval.md",
                    "chunk_id": "retrieval:c2",
                    "score": 0.8,
                },
            ],
        ),
        (
            "reliability controls",
            [
                {
                    "content": "Agentic RAG adds grading.",
                    "source": "notes.md",
                    "chunk_id": "notes:c1",
                    "score": 0.7,
                },
                {
                    "content": "Fallback handles missing evidence.",
                    "source": "notes.md",
                    "chunk_id": "notes:c3",
                    "score": 0.6,
                },
            ],
        ),
    ]

    merged = merge_retrieved_documents(query_results)

    assert [doc["chunk_id"] for doc in merged] == [
        "notes:c1",
        "retrieval:c2",
        "notes:c3",
    ]
    assert merged[0]["matched_queries"] == [
        "Agentic RAG advantages",
        "reliability controls",
    ]
    assert merged[0]["retrieval_query_count"] == 2
    assert merged[0]["multi_query_rank"] == 1
    assert merged[1]["matched_queries"] == ["Agentic RAG advantages"]
    assert merged[2]["matched_queries"] == ["reliability controls"]


def test_merge_retrieved_documents_uses_fallback_key_without_chunk_id():
    query_results = [
        (
            "first",
            [{"content": "same", "source": "a.md", "page": 1, "score": 0.9}],
        ),
        (
            "second",
            [{"content": "same", "source": "a.md", "page": 1, "score": 0.4}],
        ),
    ]

    merged = merge_retrieved_documents(query_results)

    assert len(merged) == 1
    assert merged[0]["matched_queries"] == ["first", "second"]
    assert merged[0]["retrieval_query_count"] == 2
