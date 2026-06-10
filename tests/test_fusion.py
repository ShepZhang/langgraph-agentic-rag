"""Tests for reciprocal rank fusion."""

from __future__ import annotations

from langchain_core.documents import Document

from rag.fusion import reciprocal_rank_fusion


def test_rrf_deduplicates_chunks_and_combines_rank_scores():
    dense_only = Document(
        page_content="semantic match",
        metadata={"source": "dense.md", "chunk_id": "dense-1"},
    )
    overlap_dense = Document(
        page_content="hybrid match",
        metadata={"source": "shared.md", "chunk_id": "shared-1"},
    )
    overlap_sparse = Document(
        page_content="hybrid match duplicate",
        metadata={"source": "shared.md", "chunk_id": "shared-1"},
    )

    fused = reciprocal_rank_fusion(
        [
            ("dense", [(dense_only, 0.9), (overlap_dense, 0.8)]),
            ("bm25", [(overlap_sparse, 7.0)]),
        ],
        top_k=2,
        rank_constant=10,
    )

    assert [doc.metadata["chunk_id"] for doc, _score in fused] == [
        "shared-1",
        "dense-1",
    ]
    assert fused[0][1] > fused[1][1]
    assert fused[0][0].metadata["fusion_score"] == fused[0][1]
    assert fused[0][0].metadata["dense_rank"] == 2
    assert fused[0][0].metadata["bm25_rank"] == 1


def test_rrf_respects_top_k_and_ignores_empty_lists():
    docs = [
        Document(page_content=f"doc {index}", metadata={"chunk_id": f"c{index}"})
        for index in range(3)
    ]

    fused = reciprocal_rank_fusion(
        [
            ("empty", []),
            ("dense", [(docs[0], 0.3), (docs[1], 0.2), (docs[2], 0.1)]),
        ],
        top_k=2,
    )

    assert [doc.metadata["chunk_id"] for doc, _score in fused] == ["c0", "c1"]
    assert reciprocal_rank_fusion([("dense", [(docs[0], 0.3)])], top_k=0) == []
