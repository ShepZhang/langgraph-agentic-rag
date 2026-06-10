"""Reciprocal Rank Fusion utilities for hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document


ScoredDocument = tuple[Document, float | None]
NamedRankedList = tuple[str, list[ScoredDocument]]


@dataclass
class _FusedCandidate:
    document: Document
    score: float
    ranks: dict[str, int]


def reciprocal_rank_fusion(
    ranked_lists: Iterable[NamedRankedList],
    top_k: int,
    rank_constant: int = 60,
) -> list[tuple[Document, float]]:
    """Fuse ranked document lists with Reciprocal Rank Fusion."""

    if top_k <= 0:
        return []

    fused: dict[str, _FusedCandidate] = {}
    for list_name, documents in ranked_lists:
        for rank, (document, _score) in enumerate(documents, start=1):
            document_key = _document_key(document)
            rank_score = 1 / (rank_constant + rank)
            if document_key not in fused:
                fused[document_key] = _FusedCandidate(
                    document=document,
                    score=0.0,
                    ranks={},
                )
            fused[document_key].score += rank_score
            fused[document_key].ranks[f"{list_name}_rank"] = rank

    ordered_candidates = sorted(
        fused.values(),
        key=lambda candidate: candidate.score,
        reverse=True,
    )
    return [
        (_copy_document_with_fusion_metadata(candidate), candidate.score)
        for candidate in ordered_candidates[:top_k]
    ]


def _document_key(document: Document) -> str:
    metadata = document.metadata or {}
    chunk_id = metadata.get("chunk_id")
    if chunk_id:
        return f"chunk:{chunk_id}"

    source = metadata.get("source", "")
    page = metadata.get("page", "")
    return f"fallback:{source}:{page}:{document.page_content}"


def _copy_document_with_fusion_metadata(candidate: _FusedCandidate) -> Document:
    metadata = dict(candidate.document.metadata or {})
    metadata["fusion_score"] = candidate.score
    metadata.update(candidate.ranks)
    return Document(page_content=candidate.document.page_content, metadata=metadata)
