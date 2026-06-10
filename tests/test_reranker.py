"""Tests for optional cross-encoder reranking."""

from __future__ import annotations

from langchain_core.documents import Document

from rag.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.predicted_pairs = None

    def predict(self, pairs):
        self.predicted_pairs = pairs
        return self.scores


def test_cross_encoder_reranker_sorts_by_rerank_score():
    model = FakeCrossEncoder([0.2, 0.9, 0.5])
    reranker = CrossEncoderReranker("fake-model", model=model)
    docs = [
        (Document(page_content="first", metadata={"source": "a.md"}), 0.8),
        (Document(page_content="second", metadata={"source": "b.md"}), 0.7),
        (Document(page_content="third", metadata={"source": "c.md"}), 0.6),
    ]

    reranked = reranker.rerank("question", docs, top_k=2)

    assert model.predicted_pairs == [
        ("question", "first"),
        ("question", "second"),
        ("question", "third"),
    ]
    assert [doc.metadata["source"] for doc, _score, _rerank_score in reranked] == [
        "b.md",
        "c.md",
    ]
    assert [rerank_score for _doc, _score, rerank_score in reranked] == [0.9, 0.5]


def test_cross_encoder_reranker_returns_structured_records():
    model = FakeCrossEncoder([0.2, 0.9])
    reranker = CrossEncoderReranker("fake-model", model=model)
    docs = [
        (
            Document(
                page_content="first",
                metadata={"source": "a.md", "chunk_id": "a-1"},
            ),
            0.8,
        ),
        (
            Document(
                page_content="second",
                metadata={
                    "source": "b.md",
                    "chunk_id": "b-1",
                    "document_id": "doc-b",
                },
            ),
            0.7,
        ),
    ]

    records = reranker.rerank_as_records("question", docs, top_k=2)

    assert records == [
        {
            "document_id": "doc-b",
            "chunk_id": "b-1",
            "content": "second",
            "metadata": {
                "source": "b.md",
                "chunk_id": "b-1",
                "document_id": "doc-b",
            },
            "vector_score": 0.7,
            "rerank_score": 0.9,
            "rank": 1,
        },
        {
            "document_id": None,
            "chunk_id": "a-1",
            "content": "first",
            "metadata": {"source": "a.md", "chunk_id": "a-1"},
            "vector_score": 0.8,
            "rerank_score": 0.2,
            "rank": 2,
        },
    ]


def test_cross_encoder_reranker_returns_empty_for_no_candidates():
    reranker = CrossEncoderReranker("fake-model", model=FakeCrossEncoder([]))

    assert reranker.rerank("question", [], top_k=3) == []
    assert reranker.rerank("question", [], top_k=0) == []
    assert reranker.rerank_as_records("question", [], top_k=3) == []
