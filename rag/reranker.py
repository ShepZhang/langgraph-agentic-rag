"""Optional cross-encoder reranking for retrieved document candidates."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from config import Settings, get_settings

ScoredDocument = tuple[Document, float | None]
RerankedDocument = tuple[Document, float | None, float]


class CrossEncoderReranker:
    """Rerank retrieved candidates with a sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.model = model or _load_cross_encoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[ScoredDocument],
        top_k: int,
    ) -> list[RerankedDocument]:
        """Return candidates sorted by cross-encoder relevance score."""

        if not documents or top_k <= 0:
            return []

        pairs = [(query, document.page_content) for document, _score in documents]
        raw_scores = self.model.predict(pairs)
        reranked = [
            (document, vector_score, float(rerank_score))
            for (document, vector_score), rerank_score in zip(
                documents,
                raw_scores,
                strict=True,
            )
        ]
        return sorted(reranked, key=lambda item: item[2], reverse=True)[:top_k]


def get_reranker(settings: Settings | None = None) -> CrossEncoderReranker:
    """Create the configured reranker."""

    resolved_settings = settings or get_settings()
    return CrossEncoderReranker(model_name=resolved_settings.reranker_model)


def _load_cross_encoder(model_name: str) -> Any:
    """Load CrossEncoder lazily so disabled reranking has no model cost."""

    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)
