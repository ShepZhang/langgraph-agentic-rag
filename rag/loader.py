"""Document loading utilities for PDF, Markdown, and TXT files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol

from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}


class UnsupportedFileTypeError(ValueError):
    """Raised when a document extension is not supported."""


class PdfReaderProtocol(Protocol):
    """Minimal PDF reader protocol used for dependency injection in tests."""

    pages: list[object]


def load_documents(file_paths: Iterable[str | Path]) -> list[Document]:
    """Load supported files into LangChain Documents."""

    documents: list[Document] = []
    for file_path in file_paths:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            documents.extend(load_pdf_document(path))
        elif suffix in {".md", ".markdown", ".txt"}:
            documents.append(load_text_document(path))
        else:
            raise UnsupportedFileTypeError(
                f"Unsupported file type {suffix!r}. Supported types: "
                f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

    return documents


def load_text_document(file_path: str | Path) -> Document:
    """Load a Markdown or TXT file as a single document."""

    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return Document(
        page_content=text,
        metadata={"source": path.name, "page": None},
    )


def load_pdf_document(
    file_path: str | Path,
    reader_cls: type[PdfReaderProtocol] | None = None,
) -> list[Document]:
    """Load a PDF file as one document per non-empty page."""

    path = Path(file_path)
    if reader_cls is None:
        from pypdf import PdfReader

        reader_cls = PdfReader

    reader = reader_cls(path)
    documents: list[Document] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "page": page_number},
            )
        )

    return documents
