"""Tests for lightweight BM25 sparse retrieval."""

from __future__ import annotations

from langchain_core.documents import Document

from rag.bm25_retriever import BM25Retriever


def test_bm25_retriever_ranks_exact_keyword_matches_first():
    docs = [
        Document(
            page_content="Dense retrieval method finds semantic neighbors.",
            metadata={"source": "dense.md", "chunk_id": "dense-1"},
        ),
        Document(
            page_content="BM25 sparse retrieval matches exact terms and source filenames.",
            metadata={"source": "bm25.md", "chunk_id": "bm25-1"},
        ),
        Document(
            page_content="Rerankers score query and chunk pairs.",
            metadata={"source": "reranker.md", "chunk_id": "reranker-1"},
        ),
    ]
    retriever = BM25Retriever(docs)

    results = retriever.retrieve("Which method matches exact BM25 filenames?", top_k=2)

    assert [doc.metadata["source"] for doc, _score in results] == [
        "bm25.md",
        "dense.md",
    ]
    assert results[0][1] > results[1][1]


def test_bm25_retriever_returns_empty_for_empty_inputs():
    retriever = BM25Retriever([])

    assert retriever.retrieve("BM25", top_k=3) == []
    assert retriever.retrieve("BM25", top_k=0) == []
    assert BM25Retriever([Document(page_content="", metadata={})]).retrieve("", top_k=3) == []


def test_bm25_retriever_omits_zero_score_documents():
    retriever = BM25Retriever(
        [
            Document(page_content="dense vectors", metadata={"source": "dense.md"}),
            Document(page_content="reranker precision", metadata={"source": "rerank.md"}),
        ]
    )

    assert retriever.retrieve("payroll credentials", top_k=2) == []
