"""Tests for dashboard contracts and pure report formatters."""

from __future__ import annotations

from pathlib import Path

from evaluation.dashboard_formatters import (
    build_ablation_failure_cases,
    build_ablation_summary_rows,
    build_failure_cases,
    build_failure_count_rows,
    build_summary_rows,
    failure_cases_to_table,
    filter_failure_cases,
    get_failure_detail,
    get_runtime_config,
)
from evaluation.dashboard_models import (
    DEFAULT_ABLATION_RESULT_PATH,
    FAILURE_CASE_COLUMNS,
    FAILURE_COUNT_COLUMNS,
    METRIC_COLUMNS,
    SMOKE_QUESTION_IDS,
)


def _result(question_id: str, failure_type: str = "no_failure") -> dict:
    return {
        "question_id": question_id,
        "question_type": "single_doc",
        "question": f"Question {question_id}",
        "correct": failure_type == "no_failure",
        "fallback_correct": True,
        "failure_analysis": {
            "question_id": question_id,
            "failure_type": failure_type,
            "reason": f"Reason for {question_id}",
            "suggestion": f"Suggestion for {question_id}",
        },
    }


def _summary(**overrides) -> dict:
    summary = {
        "correctness_score": 0.812345,
        "context_relevance_score": 0.9,
        "citation_hit_rate": 0.75,
        "fallback_accuracy": 1.0,
        "unsupported_claim_count": 0,
        "average_latency": 2.55555,
        "average_retry_count": 0.4,
        "failure_type_counts": {
            "no_failure": 1,
            "retrieval_failure": 1,
        },
    }
    summary.update(overrides)
    return summary


def test_dashboard_constants_define_exact_table_contracts():
    assert SMOKE_QUESTION_IDS == ("q001", "q016", "q027", "q030", "q033")
    assert DEFAULT_ABLATION_RESULT_PATH == Path(
        "experiments/results/ablation_result.json"
    )
    assert METRIC_COLUMNS == [
        "System",
        "Correctness",
        "Context Relevance",
        "Citation Accuracy",
        "Fallback Accuracy",
        "Unsupported Claims",
        "Avg Latency (s)",
        "Avg Retry",
    ]
    assert FAILURE_COUNT_COLUMNS == ["System", "Failure Type", "Count"]
    assert FAILURE_CASE_COLUMNS == [
        "Case Key",
        "System",
        "Question ID",
        "Question Type",
        "Failure Type",
        "Diagnostics",
        "Question",
    ]


def test_build_summary_rows_supports_single_and_comparison_reports():
    single = {"summary": _summary(), "results": [_result("q001")]}
    comparison = {
        "summary": {
            "mode": "comparison",
            "naive": _summary(correctness_score=0.5),
            "agentic": _summary(correctness_score=0.8),
        },
        "results": [],
    }

    assert build_summary_rows(single, default_system="agentic") == [
        ["Agentic RAG", 0.8123, 0.9, 0.75, 1.0, 0, 2.5556, 0.4]
    ]
    comparison_rows = build_summary_rows(comparison)
    assert [row[0] for row in comparison_rows] == ["Naive RAG", "Agentic RAG"]
    assert comparison_rows[0][1] == 0.5
    assert comparison_rows[1][1] == 0.8


def test_build_summary_rows_preserves_unavailable_metrics():
    report = {
        "summary": _summary(
            unsupported_claim_count=None,
            citation_hit_rate=None,
        ),
        "results": [],
    }

    row = build_summary_rows(report, default_system="naive")[0]

    assert row[0] == "Naive RAG"
    assert row[3] == "N/A"
    assert row[5] == "N/A"


def test_build_failure_cases_keeps_only_failures_with_stored_diagnostics():
    report = {
        "summary": _summary(),
        "results": [
            _result("q001"),
            _result("q002", "retrieval_failure"),
        ],
    }

    cases = build_failure_cases(report, default_system="agentic")

    assert cases == [
        {
            "case_key": "agentic:q002",
            "system": "agentic",
            "system_label": "Agentic RAG",
            "question_id": "q002",
            "question_type": "single_doc",
            "question": "Question q002",
            "failure_type": "retrieval_failure",
            "reason": "Reason for q002",
            "suggestion": "Suggestion for q002",
            "diagnostics_source": "stored",
        }
    ]


def test_build_failure_cases_flattens_comparison_in_system_order():
    report = {
        "summary": {"mode": "comparison"},
        "results": [
            {
                "naive": _result("q003", "citation_failure"),
                "agentic": _result("q003", "generation_failure"),
            }
        ],
    }

    cases = build_failure_cases(report)

    assert [case["case_key"] for case in cases] == [
        "naive:q003",
        "agentic:q003",
    ]
    assert [case["system_label"] for case in cases] == [
        "Naive RAG",
        "Agentic RAG",
    ]


def test_failure_rows_can_be_counted_filtered_and_selected():
    report = {
        "summary": {"mode": "comparison"},
        "results": [
            {
                "naive": _result("q001", "retrieval_failure"),
                "agentic": _result("q001", "citation_failure"),
            },
            {
                "naive": _result("q002", "retrieval_failure"),
                "agentic": _result("q002"),
            },
        ],
    }
    cases = build_failure_cases(report)

    assert build_failure_count_rows(cases) == [
        ["Agentic RAG", "citation_failure", 1],
        ["Naive RAG", "retrieval_failure", 2],
    ]
    assert filter_failure_cases(cases, system=None) == cases
    assert filter_failure_cases(cases, system="all", failure_type="all") == cases
    assert [
        case["case_key"]
        for case in filter_failure_cases(cases, system="naive")
    ] == ["naive:q001", "naive:q002"]
    assert [
        case["case_key"]
        for case in filter_failure_cases(
            cases,
            failure_type="citation_failure",
        )
    ] == ["agentic:q001"]

    detail = get_failure_detail(cases, "agentic:q001")
    assert detail == {
        "case_key": "agentic:q001",
        "title": "Agentic RAG / q001 / citation_failure",
        "reason": "Reason for q001",
        "suggestion": "Suggestion for q001",
        "diagnostics_source": "stored",
    }
    assert get_failure_detail(cases, None) == {
        "case_key": "",
        "title": "No failed case selected",
        "reason": "",
        "suggestion": "",
        "diagnostics_source": "unavailable",
    }


def test_ablation_formatters_runtime_config_and_table_conversion():
    payload = {
        "runs": [
            {
                "id": "v0_naive",
                "method": "Naive RAG",
                "status": "completed",
                "summary": _summary(correctness_score=0.5),
                "runtime_config": {"retriever": {"top_k": 4}},
                "results": [
                    {
                        **_result("q004", "retrieval_failure"),
                        "_diagnostics_source": "derived",
                    },
                    _result("q005"),
                ],
            },
            {
                "id": "v1_query_rewrite",
                "method": "+ Query Transformation",
                "status": "completed",
                "summary": _summary(
                    correctness_score=0.75,
                    unsupported_claim_count=None,
                ),
                "runtime_config": {"agent_features": {"rewrite": True}},
                "results": [
                    _result("q006", "generation_failure"),
                ],
            },
            "invalid-run",
        ]
    }

    summary_rows = build_ablation_summary_rows(payload)
    cases = build_ablation_failure_cases(payload)

    assert [row[0] for row in summary_rows] == [
        "v0_naive Naive RAG",
        "v1_query_rewrite + Query Transformation",
    ]
    assert summary_rows[0][1] == 0.5
    assert summary_rows[1][5] == "N/A"
    assert [case["case_key"] for case in cases] == [
        "v0_naive:q004",
        "v1_query_rewrite:q006",
    ]
    assert [case["diagnostics_source"] for case in cases] == [
        "derived",
        "stored",
    ]
    assert failure_cases_to_table(cases) == [
        [
            "v0_naive:q004",
            "v0_naive Naive RAG",
            "q004",
            "single_doc",
            "retrieval_failure",
            "derived",
            "Question q004",
        ],
        [
            "v1_query_rewrite:q006",
            "v1_query_rewrite + Query Transformation",
            "q006",
            "single_doc",
            "generation_failure",
            "stored",
            "Question q006",
        ],
    ]
    assert get_runtime_config(payload, "v0_naive") == {
        "retriever": {"top_k": 4}
    }
    assert get_runtime_config(payload, "missing") == {}
    assert get_runtime_config(payload, None) == {}
