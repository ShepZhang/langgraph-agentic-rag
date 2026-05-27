"""Retriever wrapper that returns normalized chunks for agents and UI."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.documents import Document

from rag.vectorstore import get_vectorstore_manager


class RetrievedChunk(TypedDict):
    """Normalized retrieved chunk shape."""

    content: str
    source: str | None
    page: int | None
    chunk_id: str | None
    score: float | None


class Retriever:
    """Project-level retriever over the configured vector store."""

    def __init__(self, vectorstore_manager: Any | None = None) -> None:
        self.vectorstore_manager = vectorstore_manager or get_vectorstore_manager()

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Retrieve and normalize relevant chunks."""

        results = self.vectorstore_manager.similarity_search(query, top_k=top_k)
        return [_normalize_result(document, score) for document, score in results]


def retrieve(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """Retrieve chunks using the default vector store manager."""

    return Retriever().retrieve(query, top_k=top_k)


def _normalize_result(document: Document, score: float | None) -> RetrievedChunk:
    metadata = document.metadata or {}
    return {
        "content": document.page_content,
        "source": metadata.get("source"),
        "page": metadata.get("page"),
        "chunk_id": metadata.get("chunk_id"),
        "score": score,
    }
