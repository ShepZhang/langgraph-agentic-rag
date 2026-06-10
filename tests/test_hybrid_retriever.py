"""Tests for dense + BM25 hybrid retrieval."""

from __future__ import annotations

from dataclasses import replace

from langchain_core.documents import Document

from config import get_settings
from rag.hybrid_retriever import HybridRetriever


class FakeVectorStoreManager:
    def __init__(
        self,
        dense_results: list[tuple[Document, float | None]],
        corpus: list[Document],
    ) -> None:
        self.dense_results = dense_results
        self.corpus = corpus
        self.similarity_calls: list[tuple[str, int | None]] = []
        self.corpus_calls = 0

    def similarity_search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[tuple[Document, float | None]]:
        self.similarity_calls.append((query, top_k))
        return self.dense_results[:top_k]

    def get_all_documents(self) -> list[Document]:
        self.corpus_calls += 1
        return self.corpus


def test_hybrid_retriever_fuses_dense_and_bm25_results():
    dense_only = Document(
        page_content="Semantic evidence for agent workflow reliability.",
        metadata={"source": "dense.md", "chunk_id": "dense-1"},
    )
    dense_overlap = Document(
        page_content="Query transformation helps clarify ambiguous questions.",
        metadata={"source": "query.md", "chunk_id": "query-1"},
    )
    bm25_overlap = Document(
        page_content="Query transformation helps clarify ambiguous user questions.",
        metadata={"source": "query.md", "chunk_id": "query-1"},
    )
    bm25_only = Document(
        page_content="BM25 exact keyword retrieval handles questions identifiers.",
        metadata={"source": "bm25.md", "chunk_id": "bm25-1"},
    )
    settings = replace(get_settings(), dense_top_k=2, bm25_top_k=2, fusion_top_k=3)
    manager = FakeVectorStoreManager(
        dense_results=[(dense_only, 0.8), (dense_overlap, 0.7)],
        corpus=[bm25_overlap, bm25_only],
    )

    results = HybridRetriever(manager, settings=settings).retrieve(
        "How does query transformation help ambiguous questions?",
        top_k=3,
    )

    assert manager.similarity_calls == [
        ("How does query transformation help ambiguous questions?", 2)
    ]
    assert manager.corpus_calls == 1
    assert [doc.metadata["chunk_id"] for doc, _score in results] == [
        "query-1",
        "dense-1",
        "bm25-1",
    ]
    assert results[0][0].metadata["dense_rank"] == 2
    assert results[0][0].metadata["bm25_rank"] == 1
    assert results[0][0].metadata["fusion_score"] == results[0][1]


def test_hybrid_retriever_falls_back_to_dense_when_sparse_corpus_is_empty():
    dense_doc = Document(
        page_content="Dense result",
        metadata={"source": "dense.md", "chunk_id": "dense-1"},
    )
    settings = replace(get_settings(), dense_top_k=5, bm25_top_k=5, fusion_top_k=5)
    manager = FakeVectorStoreManager(dense_results=[(dense_doc, 0.42)], corpus=[])

    results = HybridRetriever(manager, settings=settings).retrieve("question", top_k=2)

    assert manager.similarity_calls == [("question", 5)]
    assert [doc.metadata["chunk_id"] for doc, _score in results] == ["dense-1"]
    assert results[0][0].metadata["dense_rank"] == 1


def test_hybrid_retriever_rejects_zero_top_k():
    settings = replace(get_settings(), dense_top_k=2, bm25_top_k=2, fusion_top_k=2)
    manager = FakeVectorStoreManager(dense_results=[], corpus=[])

    results = HybridRetriever(manager, settings=settings).retrieve("question", top_k=0)

    assert results == []
    assert manager.similarity_calls == []
