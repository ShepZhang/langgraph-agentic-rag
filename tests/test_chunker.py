"""Tests for document chunking."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag.chunker import split_documents


def test_split_documents_preserves_metadata_and_adds_chunk_ids():
    docs = [
        Document(
            page_content="Alpha beta gamma. " * 20,
            metadata={
                "source": "notes.md",
                "page": None,
                "source_path": "/tmp/notes.md",
                "file_hash": "abc123",
            },
        )
    ]

    chunks = split_documents(docs, chunk_size=80, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "notes.md" for chunk in chunks)
    assert all(chunk.metadata["page"] is None for chunk in chunks)
    assert all(chunk.metadata["source_path"] == "/tmp/notes.md" for chunk in chunks)
    assert all(chunk.metadata["file_hash"] == "abc123" for chunk in chunks)
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == [
        f"notes.md:pNA:c{index}" for index in range(1, len(chunks) + 1)
    ]


def test_split_documents_uses_page_number_in_chunk_id():
    docs = [
        Document(
            page_content="One two three. " * 12,
            metadata={"source": "paper.pdf", "page": 4},
        )
    ]

    chunks = split_documents(docs, chunk_size=60, chunk_overlap=5)

    assert chunks[0].metadata["chunk_id"] == "paper.pdf:p4:c1"


def test_split_documents_respects_explicit_zero_chunk_size():
    docs = [
        Document(
            page_content="Short text.",
            metadata={"source": "notes.md", "page": None},
        )
    ]

    with pytest.raises(ValueError):
        split_documents(docs, chunk_size=0, chunk_overlap=0)


def test_split_documents_counts_chunks_independently_per_source_page():
    docs = [
        Document(
            page_content="Alpha beta gamma. " * 10,
            metadata={"source": "shared.pdf", "page": 1},
        ),
        Document(
            page_content="Delta epsilon zeta. " * 10,
            metadata={"source": "shared.pdf", "page": 2},
        ),
    ]

    chunks = split_documents(docs, chunk_size=50, chunk_overlap=5)
    chunk_ids_by_page = {
        page: [
            chunk.metadata["chunk_id"]
            for chunk in chunks
            if chunk.metadata["page"] == page
        ]
        for page in (1, 2)
    }

    assert chunk_ids_by_page[1] == [
        f"shared.pdf:p1:c{index}"
        for index in range(1, len(chunk_ids_by_page[1]) + 1)
    ]
    assert chunk_ids_by_page[2] == [
        f"shared.pdf:p2:c{index}"
        for index in range(1, len(chunk_ids_by_page[2]) + 1)
    ]
    assert chunk_ids_by_page[1]
    assert chunk_ids_by_page[2]
