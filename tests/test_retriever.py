"""Tests for retriever result normalization."""

from __future__ import annotations

from langchain_core.documents import Document

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


def test_module_retrieve_uses_injected_manager(monkeypatch):
    manager = FakeVectorStoreManager()
    monkeypatch.setattr("rag.retriever.get_vectorstore_manager", lambda: manager)

    chunks = retrieve("question", top_k=1)

    assert manager.calls == [("question", 1)]
    assert chunks[0]["source"] == "notes.md"
