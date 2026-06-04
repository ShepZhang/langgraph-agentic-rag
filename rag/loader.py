"""Document loading utilities for PDF, Markdown, and TXT files."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable, Protocol

from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}
logger = logging.getLogger(__name__)


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
    logger.info("Loaded text document source=%s", path.name)
    return Document(
        page_content=text,
        metadata=_base_metadata(path, page=None),
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
    base_metadata = _base_metadata(path, page=None)

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={**base_metadata, "page": page_number},
            )
        )

    logger.info("Loaded PDF document source=%s pages=%s", path.name, len(documents))
    return documents


def _base_metadata(path: Path, page: int | None) -> dict[str, object]:
    """Build metadata shared by loaded document pages."""

    resolved_path = path.resolve()
    return {
        "source": path.name,
        "source_path": str(resolved_path),
        "file_hash": _file_hash(resolved_path),
        "page": page,
    }


def _file_hash(path: Path) -> str:
    """Return a stable SHA-256 hash for a source file."""

    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
