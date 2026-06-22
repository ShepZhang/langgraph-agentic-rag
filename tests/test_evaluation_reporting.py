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


def test_comparison_report_shows_judge_metrics_none_as_na() -> None:
    """Judge metrics with None values render as N/A."""
    report = {
        "summary": {
            "mode": "comparison",
            "naive": {
                "source_hit_rate": 0.5,
                "judge_completion_rate": None,
                "average_semantic_correctness": None,
                "average_groundedness": None,
            },
            "agentic": {
                "source_hit_rate": 1.0,
                "judge_completion_rate": 0.8,
                "average_semantic_correctness": 0.75,
                "average_groundedness": 0.5,
            },
        },
        "results": [],
    }

    rendered = format_evaluation_report(report)

    assert "| Judge Completion Rate | N/A | 0.8 |" in rendered
    assert "| Semantic Correctness | N/A | 0.75 |" in rendered
    assert "| Groundedness | N/A | 0.5 |" in rendered


def test_single_system_report_renders_judge_metrics() -> None:
    """Single-system report renders judge_completed_count, judge_failed_count,
    judge_completion_rate, average_semantic_correctness, average_groundedness."""
    report = {
        "summary": {
            "total_questions": 5,
            "judge_completed_count": 3,
            "judge_failed_count": 1,
            "judge_completion_rate": 0.75,
            "average_semantic_correctness": 0.85,
            "average_groundedness": 0.6,
        },
        "results": [],
    }

    rendered = format_evaluation_report(report)

    assert "judge_completed_count: 3" in rendered
    assert "judge_failed_count: 1" in rendered
    assert "judge_completion_rate: 0.75" in rendered
    assert "average_semantic_correctness: 0.85" in rendered
    assert "average_groundedness: 0.6" in rendered


def test_comparison_report_preserves_zero_metrics() -> None:
    """Zero-valued judge metrics are preserved (not replaced with N/A)."""
    report = {
        "summary": {
            "mode": "comparison",
            "naive": {
                "source_hit_rate": 0.0,
                "judge_completion_rate": 0.0,
                "average_semantic_correctness": 0.0,
                "average_groundedness": 0.0,
            },
            "agentic": {
                "source_hit_rate": 0.0,
                "judge_completion_rate": 0.0,
                "average_semantic_correctness": 0.0,
                "average_groundedness": 0.0,
            },
        },
        "results": [],
    }

    rendered = format_evaluation_report(report)

    assert "| Judge Completion Rate | 0.0 | 0.0 |" in rendered
    assert "| Semantic Correctness | 0.0 | 0.0 |" in rendered
    assert "| Groundedness | 0.0 | 0.0 |" in rendered
