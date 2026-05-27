"""Text splitting utilities for RAG indexing."""

from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split documents into chunks while preserving source metadata."""

    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size if chunk_size is not None else settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
    )

    chunks = splitter.split_documents(documents)
    counters: defaultdict[tuple[str, str], int] = defaultdict(int)

    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        page = chunk.metadata.get("page")
        page_label = f"p{page}" if page is not None else "pNA"
        key = (source, page_label)
        counters[key] += 1
        chunk.metadata["chunk_id"] = f"{source}:{page_label}:c{counters[key]}"

    return chunks
