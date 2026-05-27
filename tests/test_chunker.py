"""Tests for document chunking."""

from __future__ import annotations

from langchain_core.documents import Document

from rag.chunker import split_documents


def test_split_documents_preserves_metadata_and_adds_chunk_ids():
    docs = [
        Document(
            page_content="Alpha beta gamma. " * 20,
            metadata={"source": "notes.md", "page": None},
        )
    ]

    chunks = split_documents(docs, chunk_size=80, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "notes.md" for chunk in chunks)
    assert all(chunk.metadata["page"] is None for chunk in chunks)
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
