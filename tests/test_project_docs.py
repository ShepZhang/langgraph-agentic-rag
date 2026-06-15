"""Tests for the portfolio sample document corpus."""

import json
from pathlib import Path

from PIL import Image

from evaluation.evaluate import load_eval_questions
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


def test_portfolio_documentation_files_exist():
    documentation_files = [
        PROJECT_ROOT / "docs/evaluation.md",
        PROJECT_ROOT / "docs/demo.md",
        PROJECT_ROOT / "assets/architecture.png",
    ]

    assert all(path.exists() for path in documentation_files)


def test_architecture_diagram_has_portfolio_dimensions():
    with Image.open(PROJECT_ROOT / "assets/architecture.png") as image:
        width, height = image.size

    assert width >= 1600
    assert height >= 1000


def test_demo_contains_required_scenarios():
    demo_text = (PROJECT_ROOT / "docs/demo.md").read_text(encoding="utf-8")

    for heading in [
        "Direct Answer",
        "Contextual Follow-Up",
        "Query Rewrite",
        "Correct Fallback",
        "Reranker",
        "Citation Safety",
    ]:
        assert heading in demo_text


def test_demo_excludes_secrets_and_includes_reproducibility_commands():
    demo_text = (PROJECT_ROOT / "docs/demo.md").read_text(encoding="utf-8")

    for forbidden_text in ["OPENAI_API_KEY=", "sk-", "Bearer"]:
        assert forbidden_text not in demo_text

    for required_text in [
        ".venv/bin/python -m pip install -r requirements.txt",
        ".venv/bin/python app.py",
        ".venv/bin/python -c",
        ".venv/bin/python -m evaluation.matrix",
        "python app.py",
        "sample_docs",
        "RERANKER_ENABLED=true",
        "RERANKER_ENABLED=false",
        "sentence-transformers/all-MiniLM-L6-v2",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "chat LLM",
    ]:
        assert required_text in demo_text


def test_readme_links_portfolio_materials_and_interview_topics():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/evaluation.md" in readme
    assert "docs/demo.md" in readme
    assert "## Interview Talking Points" in readme
    assert "Reranker vs retrieval grading" in readme
    assert "Original question vs retrieval query" in readme
    assert "Citation-aware generation vs claim verification" in readme
    assert "| Source Hit Rate | 0.6 | 0.8 |" not in readme
    assert (
        "DeepSeek Evaluation Benchmark" in readme
        or "Agentic + Reranker" in readme
    )
    assert readme.count("```") % 2 == 0
    assert (
        "If the LLM config or vector index is missing, evaluation records errors "
        "per question and still prints a report."
    ) not in readme


def test_evaluation_docs_describe_retry_trigger_count_accurately():
    evaluation_text = (PROJECT_ROOT / "docs/evaluation.md").read_text(
        encoding="utf-8"
    )

    assert "triggered seven rewrites" not in evaluation_text
    assert "seven questions triggered at least one retry rewrite" in evaluation_text


def test_readme_deepseek_summary_matches_saved_result_artifact():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    result_path = PROJECT_ROOT / "evaluation/results/deepseek_matrix_2026-06-07.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    variants = result["summary"]["variants"]

    metric_rows = [
        ("Retrieval Source Hit Rate", "source_hit_rate"),
        ("Keyword Hit Rate", "keyword_hit_rate"),
        ("Citation Rate", "citation_rate"),
        ("Claim Verification Rate", "verification_rate"),
        ("Fallback Correctness", "fallback_correctness_rate"),
        ("Average Latency", "average_latency"),
    ]

    for label, metric_key in metric_rows:
        expected_row = (
            f"| {label} | {variants['naive'][metric_key]} | "
            f"{variants['agentic'][metric_key]} | "
            f"{variants['agentic_reranker'][metric_key]} |"
        )
        assert expected_row in readme


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


def test_portfolio_evaluation_set_has_balanced_question_groups():
    evaluation_path = PROJECT_ROOT / "evaluation/eval_questions.json"
    questions = load_eval_questions(evaluation_path)

    assert len(questions) == 34
    assert sum(item["should_answer"] for item in questions) >= 20
    assert sum(not item["should_answer"] for item in questions) >= 6
    assert sum(item["requires_rewrite"] for item in questions) >= 6
    assert sum(item["source_match_mode"] == "all" for item in questions) >= 2

    observed_sources = {
        source
        for item in questions
        for source in item["expected_sources"]
    }
    assert observed_sources == {
        "agentic_rag_notes.md",
        "employee_handbook.md",
        "product_specs.md",
        "security_policy.md",
    }

    questions_by_text = {item["question"]: item for item in questions}
    assert questions_by_text["Which file types does Atlas support?"][
        "expected_keywords"
    ] == ["PDF", "Markdown", "TXT"]
    assert questions_by_text["Which systems require multi-factor authentication?"][
        "expected_keywords"
    ] == ["company email", "production systems"]
    assert questions_by_text[
        "What notice is required for five or more consecutive days of leave?"
    ]["expected_keywords"] == [
        "5 or more consecutive days",
        "10 business days",
    ]
    assert questions_by_text["How long does temporary elevated access last?"][
        "requires_rewrite"
    ] is True
    assert questions_by_text["What upload limit does the document tool have?"][
        "requires_rewrite"
    ] is True


def test_answerable_evaluation_keywords_exist_in_expected_sources():
    evaluation_path = PROJECT_ROOT / "evaluation/eval_questions.json"
    questions = load_eval_questions(evaluation_path)
    source_texts = {
        path.name: " ".join(path.read_text(encoding="utf-8").split()).lower()
        for path in SAMPLE_DOCS
    }

    for item in questions:
        if not item["should_answer"]:
            continue

        expected_texts = [
            source_texts[source]
            for source in item["expected_sources"]
        ]
        for keyword in item["expected_keywords"]:
            assert any(
                keyword.lower() in source_text
                for source_text in expected_texts
            ), f"{item['question']}: {keyword}"
