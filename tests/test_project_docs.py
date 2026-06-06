"""Tests for the portfolio sample document corpus."""

from pathlib import Path

from rag.loader import load_documents


SAMPLE_DOCS = [
    "sample_docs/agentic_rag_notes.md",
    "sample_docs/employee_handbook.md",
    "sample_docs/product_specs.md",
    "sample_docs/security_policy.md",
]


def test_portfolio_sample_corpus_files_exist_and_load():
    paths = [Path(path) for path in SAMPLE_DOCS]
    assert all(path.exists() for path in paths)

    documents = load_documents(paths)

    assert len(documents) == 4
    assert {doc.metadata["source"] for doc in documents} == {
        "agentic_rag_notes.md",
        "employee_handbook.md",
        "product_specs.md",
        "security_policy.md",
    }
    assert all(len(doc.metadata["file_hash"]) == 64 for doc in documents)


def test_sample_corpus_contains_expected_benchmark_facts():
    combined = "\n".join(
        Path(path).read_text(encoding="utf-8") for path in SAMPLE_DOCS
    )

    for fact in [
        "20 days of paid time off",
        "three remote-work days per week",
        "25 MB",
        "PDF, Markdown, and TXT",
        "four hours",
        "managed secrets vault",
    ]:
        assert fact in combined
