"""Retriever wrapper that returns normalized chunks for agents and UI."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.documents import Document

from config import Settings, get_settings
from rag.reranker import get_reranker
from rag.vectorstore import get_vectorstore_manager


class RetrievedChunk(TypedDict, total=False):
    """Normalized retrieved chunk shape."""

    content: str
    source: str | None
    page: int | None
    chunk_id: str | None
    score: float | None
    rerank_score: float | None


class Retriever:
    """Project-level retriever over the configured vector store."""

    def __init__(
        self,
        vectorstore_manager: Any | None = None,
        reranker: Any | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.vectorstore_manager = vectorstore_manager or get_vectorstore_manager()
        if reranker is not None:
            self.reranker = reranker
        elif self.settings.reranker_enabled:
            self.reranker = get_reranker(self.settings)
        else:
            self.reranker = None

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Retrieve and normalize relevant chunks."""

        final_top_k = top_k if top_k is not None else self.settings.top_k
        candidate_top_k = final_top_k
        if self.settings.reranker_enabled:
            candidate_top_k = max(final_top_k, self.settings.reranker_candidate_top_k)

        results = self.vectorstore_manager.similarity_search(
            query,
            top_k=candidate_top_k,
        )
        if not self.reranker:
            return [_normalize_result(document, score) for document, score in results]

        reranked = self.reranker.rerank(query, results, top_k=final_top_k)
        return [
            _normalize_result(document, score, rerank_score=rerank_score)
            for document, score, rerank_score in reranked
        ]


def retrieve(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """Retrieve chunks using the default vector store manager."""

    return Retriever().retrieve(query, top_k=top_k)


def _normalize_result(
    document: Document,
    score: float | None,
    rerank_score: float | None = None,
) -> RetrievedChunk:
    metadata = document.metadata or {}
    chunk: RetrievedChunk = {
        "content": document.page_content,
        "source": metadata.get("source"),
        "page": metadata.get("page"),
        "chunk_id": metadata.get("chunk_id"),
        "score": score,
    }
    if rerank_score is not None:
        chunk["rerank_score"] = rerank_score
    return chunk
