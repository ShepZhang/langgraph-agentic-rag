"""Tests for the portfolio sample document corpus."""

from pathlib import Path

from rag.loader import load_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOCS = [
    PROJECT_ROOT / "sample_docs/agentic_rag_notes.md",
    PROJECT_ROOT / "sample_docs/employee_handbook.md",
    PROJECT_ROOT / "sample_docs/product_specs.md",
    PROJECT_ROOT / "sample_docs/security_policy.md",
]

EXPECTED_FACTS_BY_SOURCE = {
    "employee_handbook.md": [
        "Full-time employees may work remotely up to three days per week.",
        (
            "Employees should coordinate their three remote-work days per week "
            "with their manager and team so that customer coverage and planned "
            "collaboration remain available."
        ),
        "Core collaboration hours are 10:00 AM to 3:00 PM Pacific Time.",
        "Full-time employees receive 20 days of paid time off per calendar year.",
        (
            "Requests for 5 or more consecutive days of paid time off should be "
            "submitted at least 10 business days before the first day of leave."
        ),
        "Each full-time employee has an annual learning budget of USD 1,500.",
        "Purchases require manager approval and an itemized receipt.",
        (
            "Travel costing above USD 500 requires preapproval from the "
            "employee's manager."
        ),
        (
            "Expense reports and itemized receipts must be submitted within 30 "
            "days after the expense is incurred or the trip ends."
        ),
    ],
    "product_specs.md": [
        (
            "Atlas is a private document QA app for asking grounded questions "
            "about a user-provided document collection."
        ),
        "Atlas supports PDF, Markdown, and TXT files.",
        "The maximum upload size is 25 MB per file.",
        (
            "The demo build replaces the active collection whenever the user "
            "builds an index."
        ),
        (
            "The lower-level vectorstore API supports incremental indexing with "
            "deterministic chunk IDs, allowing callers to avoid duplicate chunks "
            "across repeated indexing operations."
        ),
        "Atlas uses Chroma to retrieve vector candidates.",
        (
            "An optional cross-encoder can rerank those candidates before "
            "LangGraph performs retrieval grading and selects the evidence used "
            "for answer generation."
        ),
        (
            "Normal answers must contain citation markers that match the "
            "returned citation indices."
        ),
        (
            "Atlas also performs lightweight claim verification against the "
            "selected evidence."
        ),
        "Atlas has no formal production SLA.",
        (
            "It is intended for local demo and evaluation use, not as a "
            "production service commitment."
        ),
    ],
    "security_policy.md": [
        (
            "Multi-factor authentication (MFA) is required for company email, "
            "source control, cloud administration, production systems, the "
            "managed secrets vault, and any other service that supports "
            "confidential company or customer information."
        ),
        (
            "Production access is granted just in time (JIT) and expires after "
            "four hours."
        ),
        (
            "Every request must include a ticket reference and approval from "
            "the on-call engineering lead before access is activated."
        ),
        "Secrets must be stored in the managed secrets vault.",
        (
            "Passwords, API keys, tokens, certificates, and other credentials "
            "must never be stored in source control or shared documents."
        ),
        "Privileged access is reviewed quarterly.",
        (
            "Access that is no longer required must be removed within one "
            "business day after a role change, termination, or review finding "
            "establishes that the access is unnecessary."
        ),
        (
            "Suspected credential exposure or unauthorized access must be "
            "reported to the security incident channel within 30 minutes of "
            "discovery."
        ),
    ],
}


def test_portfolio_sample_corpus_files_exist_and_load():
    assert all(path.exists() for path in SAMPLE_DOCS)

    documents = load_documents(SAMPLE_DOCS)

    assert len(documents) == 4
    assert {doc.metadata["source"] for doc in documents} == {
        "agentic_rag_notes.md",
        "employee_handbook.md",
        "product_specs.md",
        "security_policy.md",
    }
    assert all(len(doc.metadata["file_hash"]) == 64 for doc in documents)


def test_sample_corpus_contains_source_specific_benchmark_facts():
    paths_by_name = {path.name: path for path in SAMPLE_DOCS}

    for source, expected_facts in EXPECTED_FACTS_BY_SOURCE.items():
        document_text = " ".join(
            paths_by_name[source].read_text(encoding="utf-8").split()
        )

        for fact in expected_facts:
            assert fact in document_text
