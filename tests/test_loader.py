"""Tests for document loading."""

from __future__ import annotations

import pytest

from rag.loader import UnsupportedFileTypeError, load_documents, load_pdf_document


def test_load_text_and_markdown_documents_preserves_metadata(tmp_path):
    txt_path = tmp_path / "notes.txt"
    md_path = tmp_path / "guide.md"
    txt_path.write_text("plain text content", encoding="utf-8")
    md_path.write_text("# Guide\n\nmarkdown content", encoding="utf-8")

    docs = load_documents([txt_path, md_path])

    assert [doc.page_content for doc in docs] == [
        "plain text content",
        "# Guide\n\nmarkdown content",
    ]
    assert docs[0].metadata["source"] == "notes.txt"
    assert docs[0].metadata["page"] is None
    assert docs[0].metadata["source_path"] == str(txt_path.resolve())
    assert len(docs[0].metadata["file_hash"]) == 64
    assert docs[1].metadata["source"] == "guide.md"
    assert docs[1].metadata["page"] is None
    assert docs[1].metadata["source_path"] == str(md_path.resolve())
    assert len(docs[1].metadata["file_hash"]) == 64


def test_load_pdf_document_with_injected_reader_preserves_page_metadata(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")

    class FakePage:
        def __init__(self, text: str):
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class FakeReader:
        def __init__(self, _path):
            self.pages = [FakePage("page one"), FakePage(""), FakePage("page three")]

    docs = load_pdf_document(pdf_path, reader_cls=FakeReader)

    assert [doc.page_content for doc in docs] == ["page one", "page three"]
    assert docs[0].metadata["source"] == "paper.pdf"
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["source_path"] == str(pdf_path.resolve())
    assert len(docs[0].metadata["file_hash"]) == 64
    assert docs[1].metadata["source"] == "paper.pdf"
    assert docs[1].metadata["page"] == 3
    assert docs[1].metadata["source_path"] == str(pdf_path.resolve())
    assert docs[0].metadata["file_hash"] == docs[1].metadata["file_hash"]


def test_load_documents_rejects_unsupported_file_type(tmp_path):
    csv_path = tmp_path / "table.csv"
    csv_path.write_text("a,b\n1,2", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError, match="Unsupported file type"):
        load_documents([csv_path])
