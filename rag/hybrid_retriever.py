"""Hybrid dense + sparse retrieval pipeline."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from config import Settings, get_settings
from rag.bm25_retriever import BM25Retriever
from rag.fusion import reciprocal_rank_fusion


ScoredDocument = tuple[Document, float | None]


class HybridRetriever:
    """Fuse dense vector retrieval with BM25 sparse retrieval using RRF."""

    def __init__(
        self,
        vectorstore_manager: Any,
        settings: Settings | None = None,
    ) -> None:
        self.vectorstore_manager = vectorstore_manager
        self.settings = settings or get_settings()

    def retrieve(self, query: str, top_k: int | None = None) -> list[ScoredDocument]:
        """Return fused dense + BM25 candidates for a query."""

        fusion_top_k = top_k if top_k is not None else self.settings.fusion_top_k
        if fusion_top_k <= 0:
            return []

        dense_results = self.vectorstore_manager.similarity_search(
            query,
            top_k=self.settings.dense_top_k,
        )
        bm25_results = self._retrieve_bm25(query)
        return reciprocal_rank_fusion(
            [
                ("dense", dense_results),
                ("bm25", bm25_results),
            ],
            top_k=fusion_top_k,
        )

    def _retrieve_bm25(self, query: str) -> list[ScoredDocument]:
        corpus = self.vectorstore_manager.get_all_documents()
        if not corpus:
            return []
        return BM25Retriever(corpus).retrieve(query, top_k=self.settings.bm25_top_k)
