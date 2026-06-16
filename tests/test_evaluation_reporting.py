"""Tests for evaluation report rendering."""

from __future__ import annotations

from evaluation.reporting import format_evaluation_report


def test_single_system_report_keeps_question_diagnostics() -> None:
    report = {
        "summary": {
            "total_questions": 1,
            "source_hit_rate": 1.0,
        },
        "results": [
            {
                "question": "What is Agentic RAG?",
                "answer_returned": True,
                "fallback_triggered": False,
                "citation_returned": True,
                "source_hit": True,
                "keyword_hit": True,
                "rewrite_triggered": False,
                "retry_count": 1,
                "retrieved_doc_count": 2,
                "relevant_doc_count": 1,
                "latency": 0.125,
                "error": None,
            }
        ],
    }

    rendered = format_evaluation_report(report)

    assert "Evaluation Report" in rendered
    assert "source_hit_rate: 1.0" in rendered
    assert "retrieved=2" in rendered


def test_comparison_report_keeps_metric_table() -> None:
    report = {
        "summary": {
            "mode": "comparison",
            "naive": {
                "source_hit_rate": 0.5,
            },
            "agentic": {
                "source_hit_rate": 1.0,
            },
        },
        "results": [],
    }

    rendered = format_evaluation_report(report)

    assert "| Metric | Naive RAG | Agentic RAG |" in rendered
    assert "| Source Hit Rate | 0.5 | 1.0 |" in rendered
