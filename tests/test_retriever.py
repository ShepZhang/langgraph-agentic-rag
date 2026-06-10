"""Tests for retriever result normalization."""

from __future__ import annotations

from dataclasses import replace

from langchain_core.documents import Document

from config import get_settings
from rag.retriever import Retriever, retrieve


class FakeVectorStoreManager:
    def __init__(self):
        self.calls = []

    def similarity_search(self, query, top_k=None):
        self.calls.append((query, top_k))
        return [
            (
                Document(
                    page_content="Relevant context",
                    metadata={"source": "notes.md", "page": None, "chunk_id": "notes.md:pNA:c1"},
                ),
                0.91,
            ),
            (
                Document(
                    page_content="More context",
                    metadata={"source": "paper.pdf", "page": 2, "chunk_id": "paper.pdf:p2:c3"},
                ),
                None,
            ),
        ]


class FakeReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, documents, top_k):
        self.calls.append((query, documents, top_k))
        return [
            (documents[2][0], documents[2][1], 0.99),
            (documents[0][0], documents[0][1], 0.77),
        ][:top_k]


class FakeHybridRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, query, top_k=None):
        self.calls.append((query, top_k))
        return self.results[:top_k]


def test_retriever_returns_normalized_chunks():
    manager = FakeVectorStoreManager()
    retriever = Retriever(vectorstore_manager=manager)

    chunks = retriever.retrieve("What is RAG?", top_k=2)

    assert manager.calls == [("What is RAG?", 2)]
    assert chunks == [
        {
            "content": "Relevant context",
            "source": "notes.md",
            "page": None,
            "chunk_id": "notes.md:pNA:c1",
            "score": 0.91,
        },
        {
            "content": "More context",
            "source": "paper.pdf",
            "page": 2,
            "chunk_id": "paper.pdf:p2:c3",
            "score": None,
        },
    ]


def test_retriever_does_not_load_reranker_when_disabled(monkeypatch):
    def fail_if_called(settings):
        raise AssertionError("reranker should not be loaded")

    monkeypatch.setattr("rag.retriever.get_reranker", fail_if_called)
    settings = replace(get_settings(), reranker_enabled=False)
    retriever = Retriever(
        vectorstore_manager=FakeVectorStoreManager(),
        settings=settings,
    )

    chunks = retriever.retrieve("What is RAG?", top_k=1)

    assert chunks[0]["source"] == "notes.md"


def test_retriever_fetches_candidate_top_k_and_reranks_when_enabled():
    class CandidateManager:
        def __init__(self):
            self.calls = []

        def similarity_search(self, query, top_k=None):
            self.calls.append((query, top_k))
            return [
                (
                    Document(page_content="weak context", metadata={"source": "weak.md"}),
                    0.4,
                ),
                (
                    Document(page_content="medium context", metadata={"source": "medium.md"}),
                    0.5,
                ),
                (
                    Document(page_content="best context", metadata={"source": "best.md"}),
                    0.6,
                ),
            ]

    settings = replace(
        get_settings(),
        top_k=2,
        reranker_enabled=True,
        reranker_candidate_top_k=3,
    )
    manager = CandidateManager()
    reranker = FakeReranker()
    retriever = Retriever(
        vectorstore_manager=manager,
        reranker=reranker,
        settings=settings,
    )

    chunks = retriever.retrieve("question")

    assert manager.calls == [("question", 3)]
    assert reranker.calls[0][0] == "question"
    assert reranker.calls[0][2] == 2
    assert [chunk["source"] for chunk in chunks] == ["best.md", "weak.md"]
    assert [chunk["score"] for chunk in chunks] == [0.6, 0.4]
    assert [chunk["rerank_score"] for chunk in chunks] == [0.99, 0.77]


def test_retriever_uses_hybrid_retrieval_when_enabled():
    manager = FakeVectorStoreManager()
    hybrid = FakeHybridRetriever(
        [
            (
                Document(
                    page_content="Hybrid context",
                    metadata={
                        "source": "hybrid.md",
                        "chunk_id": "hybrid-1",
                        "fusion_score": 0.03,
                    },
                ),
                0.03,
            )
        ]
    )
    settings = replace(
        get_settings(),
        hybrid_retrieval_enabled=True,
        reranker_enabled=False,
        top_k=2,
    )
    retriever = Retriever(
        vectorstore_manager=manager,
        hybrid_retriever=hybrid,
        settings=settings,
    )

    chunks = retriever.retrieve("question", top_k=1)

    assert manager.calls == []
    assert hybrid.calls == [("question", 1)]
    assert chunks == [
        {
            "content": "Hybrid context",
            "source": "hybrid.md",
            "page": None,
            "chunk_id": "hybrid-1",
            "score": 0.03,
        }
    ]


def test_retriever_reranks_hybrid_candidate_pool_when_enabled():
    hybrid_results = [
        (
            Document(page_content=f"context {index}", metadata={"source": f"{index}.md"}),
            float(index),
        )
        for index in range(5)
    ]
    hybrid = FakeHybridRetriever(hybrid_results)
    reranker = FakeReranker()
    settings = replace(
        get_settings(),
        hybrid_retrieval_enabled=True,
        reranker_enabled=True,
        top_k=2,
        reranker_candidate_top_k=3,
        fusion_top_k=5,
    )
    retriever = Retriever(
        vectorstore_manager=FakeVectorStoreManager(),
        hybrid_retriever=hybrid,
        reranker=reranker,
        settings=settings,
    )

    chunks = retriever.retrieve("question")

    assert hybrid.calls == [("question", 5)]
    assert reranker.calls[0][0] == "question"
    assert reranker.calls[0][2] == 2
    assert [chunk["source"] for chunk in chunks] == ["2.md", "0.md"]
    assert [chunk["rerank_score"] for chunk in chunks] == [0.99, 0.77]


def test_module_retrieve_uses_injected_manager(monkeypatch):
    manager = FakeVectorStoreManager()
    monkeypatch.setattr("rag.retriever.get_vectorstore_manager", lambda: manager)

    chunks = retrieve("question", top_k=1)

    assert manager.calls == [("question", 1)]
    assert chunks[0]["source"] == "notes.md"
