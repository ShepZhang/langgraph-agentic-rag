"""Tests for dashboard contracts and pure report formatters."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

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
from evaluation.dashboard_service import (
    EvaluationDashboardService,
    JsonAblationResultProvider,
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


class StaticAblationProvider:
    def __init__(self, payload):
        self.payload = payload

    def load(self):
        return self.payload


def _ablation_payload(result: dict) -> dict:
    return {
        "kind": "ablation_result",
        "runs": [
            {
                "id": "v0_naive",
                "method": "Naive RAG",
                "status": "completed",
                "runtime_config": {"llm": {"model": "test-model"}},
                "summary": _summary(),
                "results": [result],
            }
        ],
    }


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


def test_build_failure_cases_assigns_stable_unique_keys_to_duplicate_and_empty_ids():
    duplicate_first = _result("dup", "retrieval_failure")
    duplicate_first["failure_analysis"]["reason"] = "First duplicate"
    duplicate_second = _result("dup", "citation_failure")
    duplicate_second["failure_analysis"]["reason"] = "Second duplicate"
    empty_first = _result("", "generation_failure")
    empty_first["failure_analysis"]["reason"] = "First empty ID"
    empty_second = _result("", "tool_failure")
    empty_second["failure_analysis"]["reason"] = "Second empty ID"
    report = {
        "summary": _summary(),
        "results": [
            duplicate_first,
            duplicate_second,
            empty_first,
            empty_second,
        ],
    }

    cases = build_failure_cases(report, default_system="agentic")

    assert [case["case_key"] for case in cases] == [
        "agentic:dup",
        "agentic:dup:2",
        "agentic:row-3",
        "agentic:row-4",
    ]
    assert [
        get_failure_detail(cases, case_key)["reason"]
        for case_key in (
            "agentic:dup",
            "agentic:dup:2",
            "agentic:row-3",
            "agentic:row-4",
        )
    ] == [
        "First duplicate",
        "Second duplicate",
        "First empty ID",
        "Second empty ID",
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


def test_ablation_failure_cases_assign_stable_unique_keys_per_full_output():
    duplicate_first = _result("dup", "retrieval_failure")
    duplicate_first["failure_analysis"]["reason"] = "Ablation duplicate one"
    duplicate_second = _result("dup", "citation_failure")
    duplicate_second["failure_analysis"]["reason"] = "Ablation duplicate two"
    empty_id = _result("", "tool_failure")
    empty_id["failure_analysis"]["reason"] = "Ablation empty ID"
    payload = {
        "runs": [
            {
                "id": "v1",
                "method": "Variant One",
                "results": (
                    duplicate_first,
                    duplicate_second,
                    empty_id,
                ),
            }
        ]
    }

    cases = build_ablation_failure_cases(payload)

    assert [case["case_key"] for case in cases] == [
        "v1:dup",
        "v1:dup:2",
        "v1:row-3",
    ]
    assert get_failure_detail(cases, "v1:dup:2")["reason"] == (
        "Ablation duplicate two"
    )
    assert get_failure_detail(cases, "v1:row-3")["reason"] == (
        "Ablation empty ID"
    )


def test_formatter_sequences_accept_tuples_without_expanding_strings():
    tuple_report = {
        "summary": _summary(),
        "results": (_result("q007", "retrieval_failure"),),
    }
    tuple_payload = {
        "runs": (
            {
                "id": "v2",
                "method": "Variant Two",
                "summary": _summary(),
            },
        )
    }

    assert [case["case_key"] for case in build_failure_cases(tuple_report)] == [
        "agentic:q007"
    ]
    assert [row[0] for row in build_ablation_summary_rows(tuple_payload)] == [
        "v2 Variant Two"
    ]
    assert build_failure_cases({"summary": _summary(), "results": "q007"}) == []
    assert build_failure_cases({"summary": _summary(), "results": b"q007"}) == []
    assert build_failure_cases(
        {"summary": _summary(), "results": bytearray(b"q007")}
    ) == []
    assert build_ablation_summary_rows({"runs": "v2"}) == []


def _question(question_id: str) -> dict:
    return {
        "id": question_id,
        "question_type": "single_doc",
        "question": f"Question {question_id}",
        "expected_sources": ["notes.md"],
        "answerable": True,
    }


def _runner_result(question: str) -> dict:
    return {
        "answer": f"Grounded answer for {question} [1].",
        "citations": [{"source": "notes.md"}],
        "retrieved_documents": [{"source": "notes.md"}],
        "relevant_documents": [{"source": "notes.md"}],
    }


def test_json_ablation_result_provider_loads_utf8_json_and_uses_default_path(
    tmp_path,
):
    result_path = tmp_path / "ablation.json"
    payload = {"runs": [{"id": "v1", "method": "\u65b9\u6cd5"}]}
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    before = result_path.read_bytes()

    provider = JsonAblationResultProvider(result_path)

    assert provider.path == result_path
    assert provider.load() == payload
    assert result_path.read_bytes() == before
    assert JsonAblationResultProvider().path == DEFAULT_ABLATION_RESULT_PATH


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "object"),
        ({"runs": {}}, "runs"),
        ({}, "runs"),
    ],
)
def test_json_ablation_result_provider_validates_payload_shape(
    tmp_path,
    payload,
    message,
):
    result_path = tmp_path / "invalid.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        JsonAblationResultProvider(result_path).load()


def test_list_questions_preserves_dataset_order_and_exact_labels():
    service = EvaluationDashboardService(
        question_loader=lambda: [_question("q002"), _question("q001")],
    )

    assert service.list_questions() == [
        {
            "id": "q002",
            "label": "[q002] single_doc - Question q002",
            "question_type": "single_doc",
            "question": "Question q002",
        },
        {
            "id": "q001",
            "label": "[q001] single_doc - Question q001",
            "question_type": "single_doc",
            "question": "Question q001",
        },
    ]


def test_list_questions_loads_all_repository_questions_in_order():
    options = EvaluationDashboardService().list_questions()

    assert len(options) == 36
    assert options[0]["id"] == "q001"
    assert options[-1]["id"] == "q036"


def test_load_ablation_snapshot_uses_stored_analysis_and_runtime_config():
    stored_result = {
        **_result("q001", "citation_failure"),
        "_diagnostics_source": "derived",
    }
    payload = _ablation_payload(stored_result)
    service = EvaluationDashboardService(
        question_loader=lambda: [_question("q001")],
        ablation_provider=StaticAblationProvider(payload),
        id_factory=lambda prefix: f"{prefix}-fixed",
    )

    view = service.load_ablation_snapshot()

    assert view["status"] == "completed"
    assert view["run_id"] == "snapshot-fixed"
    assert len(view["summary_rows"]) == 1
    assert view["failure_cases"][0]["diagnostics_source"] == "stored"
    assert view["failure_count_rows"] == [
        ["v0_naive Naive RAG", "citation_failure", 1]
    ]
    assert "1 ablation variant" in view["message"]
    assert "1 failure" in view["message"]
    assert service.get_runtime_config(view, "v0_naive") == {
        "llm": {"model": "test-model"}
    }
    assert service.get_runtime_config(view, "missing") == {}
    assert service.get_runtime_config(view, None) == {}
    assert service.get_runtime_config(view, "all") == {}


def test_load_ablation_snapshot_does_not_require_metadata_for_stored_analysis():
    def failing_loader():
        raise RuntimeError("question metadata unavailable")

    payload = _ablation_payload(_result("q001", "citation_failure"))
    service = EvaluationDashboardService(
        question_loader=failing_loader,
        ablation_provider=StaticAblationProvider(payload),
        id_factory=lambda prefix: f"{prefix}-stored-only",
    )

    view = service.load_ablation_snapshot()

    assert view["status"] == "completed"
    assert view["run_id"] == "snapshot-stored-only"
    assert view["failure_cases"][0]["diagnostics_source"] == "stored"
    assert "question metadata unavailable" not in view["message"]


def test_load_ablation_snapshot_derives_legacy_diagnostics_without_mutation():
    legacy_result = {
        "question_id": "q001",
        "question_type": "single_doc",
        "question": "Question q001",
        "correct": False,
        "source_hit": False,
        "context_relevant": False,
        "citation_hit": False,
        "fallback_correct": True,
        "fallback_triggered": False,
        "answer_returned": True,
        "retry_count": 0,
        "unsupported_claim_count": 0,
        "retrieved_documents": [{"source": "other.md"}],
        "relevant_documents": [],
        "citations": [],
        "error": None,
    }
    payload = _ablation_payload(legacy_result)
    payload["runs"][0]["summary"].pop("failure_type_counts")
    original = deepcopy(payload)

    def unexpected_runner(_question_text):
        raise AssertionError("snapshot loading must not rerun a benchmark")

    service = EvaluationDashboardService(
        question_loader=lambda: [_question("q001")],
        agentic_runner=unexpected_runner,
        naive_runner=unexpected_runner,
        ablation_provider=StaticAblationProvider(payload),
        id_factory=lambda prefix: f"{prefix}-legacy",
    )

    view = service.load_ablation_snapshot()

    assert view["status"] == "completed"
    assert view["failure_cases"][0]["failure_type"] == "retrieval_failure"
    assert view["failure_cases"][0]["diagnostics_source"] == "derived"
    assert view["failure_count_rows"] == [
        ["v0_naive Naive RAG", "retrieval_failure", 1]
    ]
    assert view["raw_report"]["runs"][0]["results"][0][
        "_diagnostics_source"
    ] == "derived"
    assert payload == original


def test_load_ablation_snapshot_marks_missing_metadata_diagnostics_unavailable():
    result = {
        "question_id": "q999",
        "question_type": "single_doc",
        "question": "Unknown question",
        "correct": False,
        "source_hit": False,
        "context_relevant": False,
        "citation_hit": False,
        "fallback_correct": True,
        "fallback_triggered": False,
        "answer_returned": True,
        "retrieved_documents": [{"source": "other.md"}],
        "relevant_documents": [],
        "citations": [],
        "error": None,
    }
    payload = _ablation_payload(result)
    original = deepcopy(payload)
    service = EvaluationDashboardService(
        question_loader=lambda: [_question("q001")],
        ablation_provider=StaticAblationProvider(payload),
        id_factory=lambda prefix: f"{prefix}-missing-metadata",
    )

    view = service.load_ablation_snapshot()

    enriched_result = view["raw_report"]["runs"][0]["results"][0]
    assert view["status"] == "completed"
    assert len(view["summary_rows"]) == 1
    assert view["failure_cases"] == []
    assert view["failure_count_rows"] == []
    assert enriched_result["_diagnostics_source"] == "unavailable"
    assert "failure_analysis" not in enriched_result
    assert "unavailable diagnostics: 1" in view["message"].lower()
    assert payload == original


def test_load_ablation_snapshot_rejects_incomplete_stored_analysis():
    result = {
        "question_id": "q001",
        "question_type": "single_doc",
        "question": "Question q001",
        "correct": False,
        "failure_analysis": {},
    }
    payload = _ablation_payload(result)
    service = EvaluationDashboardService(
        question_loader=lambda: [_question("q001")],
        ablation_provider=StaticAblationProvider(payload),
        id_factory=lambda prefix: f"{prefix}-invalid-analysis",
    )

    view = service.load_ablation_snapshot()

    enriched_result = view["raw_report"]["runs"][0]["results"][0]
    assert view["status"] == "completed"
    assert view["failure_cases"] == []
    assert view["failure_count_rows"] == []
    assert enriched_result["_diagnostics_source"] == "unavailable"
    assert enriched_result["failure_analysis"] == {}
    assert "unavailable diagnostics: 1" in view["message"].lower()


def test_load_ablation_snapshot_does_not_derive_incomplete_legacy_result():
    result = {
        "question_id": "q001",
        "question_type": "single_doc",
        "question": "Question q001",
        "correct": True,
    }
    payload = _ablation_payload(result)
    service = EvaluationDashboardService(
        question_loader=lambda: [_question("q001")],
        ablation_provider=StaticAblationProvider(payload),
        id_factory=lambda prefix: f"{prefix}-incomplete-legacy",
    )

    view = service.load_ablation_snapshot()

    enriched_result = view["raw_report"]["runs"][0]["results"][0]
    assert view["status"] == "completed"
    assert view["failure_cases"] == []
    assert enriched_result["_diagnostics_source"] == "unavailable"
    assert "failure_analysis" not in enriched_result
    assert "retrieval_failure" not in view["message"]
    assert "unavailable diagnostics: 1" in view["message"].lower()


def test_load_ablation_snapshot_degrades_when_metadata_unavailable_for_legacy():
    legacy_result = {
        "question_id": "q001",
        "question_type": "single_doc",
        "question": "Question q001",
        "correct": False,
        "source_hit": False,
        "context_relevant": False,
        "citation_hit": False,
        "fallback_correct": True,
        "fallback_triggered": False,
        "answer_returned": True,
        "retrieved_documents": [{"source": "other.md"}],
        "relevant_documents": [],
        "citations": [],
        "error": None,
    }
    stored_result = _result("q002", "citation_failure")
    payload = {
        "runs": [
            {
                **_ablation_payload(stored_result)["runs"][0],
                "results": [stored_result, legacy_result],
            }
        ]
    }

    def failing_loader():
        raise RuntimeError("question metadata unavailable")

    service = EvaluationDashboardService(
        question_loader=failing_loader,
        ablation_provider=StaticAblationProvider(payload),
        id_factory=lambda prefix: f"{prefix}-degraded",
    )

    view = service.load_ablation_snapshot()

    enriched_results = view["raw_report"]["runs"][0]["results"]
    assert view["status"] == "completed"
    assert [case["question_id"] for case in view["failure_cases"]] == ["q002"]
    assert enriched_results[0]["_diagnostics_source"] == "stored"
    assert enriched_results[1]["_diagnostics_source"] == "unavailable"
    assert "failure_analysis" not in enriched_results[1]
    assert "degraded" in view["message"].lower()
    assert "unavailable diagnostics: 1" in view["message"].lower()


def test_load_ablation_snapshot_keeps_metric_rows_and_reports_partial_payload():
    valid_run = _ablation_payload(
        _result("q001", "retrieval_failure")
    )["runs"][0]
    missing_results_run = {
        "id": "v1_missing_results",
        "method": "Missing Results",
        "summary": _summary(correctness_score=0.25),
        "results": {"not": "a list"},
    }
    valid_run["results"].insert(0, "invalid-result")
    payload = {
        "runs": [
            "invalid-run",
            missing_results_run,
            valid_run,
        ]
    }
    service = EvaluationDashboardService(
        question_loader=lambda: [_question("q001")],
        ablation_provider=StaticAblationProvider(payload),
        id_factory=lambda prefix: f"{prefix}-partial",
    )

    view = service.load_ablation_snapshot()

    assert view["status"] == "completed"
    assert [row[0] for row in view["summary_rows"]] == [
        "v1_missing_results Missing Results",
        "v0_naive Naive RAG",
    ]
    assert view["raw_report"]["runs"][0]["results"] == []
    assert [case["case_key"] for case in view["failure_cases"]] == [
        "v0_naive:q001"
    ]
    assert "partial" in view["message"].lower()
    assert "skipped 3" in view["message"].lower()


def test_load_ablation_snapshot_returns_unavailable_for_missing_artifact(
    tmp_path,
):
    missing_path = tmp_path / "missing-ablation.json"
    service = EvaluationDashboardService(
        ablation_provider=JsonAblationResultProvider(missing_path),
    )

    view = service.load_ablation_snapshot()

    assert view["status"] == "unavailable"
    assert view["run_id"] == ""
    assert view["summary_rows"] == []
    assert view["failure_count_rows"] == []
    assert view["failure_cases"] == []
    assert view["raw_report"] == {}
    assert missing_path.name in view["message"]


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"runs": {}},
    ],
)
def test_load_ablation_snapshot_returns_unavailable_for_malformed_top_level(
    tmp_path,
    payload,
):
    result_path = tmp_path / "malformed-ablation.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    service = EvaluationDashboardService(
        ablation_provider=JsonAblationResultProvider(result_path),
    )

    view = service.load_ablation_snapshot()

    assert view["status"] == "unavailable"
    assert view["run_id"] == ""
    assert view["raw_report"] == {}
    assert "ablation" in view["message"].lower()


def test_load_ablation_snapshot_does_not_load_metadata_for_empty_runs():
    def failing_loader():
        raise RuntimeError("question metadata unavailable")

    service = EvaluationDashboardService(
        question_loader=failing_loader,
        ablation_provider=StaticAblationProvider({"runs": []}),
    )

    view = service.load_ablation_snapshot()

    assert view["status"] == "completed"
    assert view["run_id"].startswith("snapshot-")
    assert view["raw_report"] == {"runs": []}
    assert "question metadata unavailable" not in view["message"]


def test_load_ablation_snapshot_does_not_rewrite_json_artifact(tmp_path):
    result_path = tmp_path / "ablation.json"
    payload = _ablation_payload(_result("q001", "retrieval_failure"))
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    before = result_path.read_bytes()
    service = EvaluationDashboardService(
        question_loader=lambda: [_question("q001")],
        ablation_provider=JsonAblationResultProvider(result_path),
    )

    view = service.load_ablation_snapshot()

    assert view["status"] == "completed"
    assert result_path.read_bytes() == before


def test_load_ablation_snapshot_contains_id_factory_errors():
    def failing_id_factory(prefix):
        raise RuntimeError(f"{prefix} id generation failed")

    service = EvaluationDashboardService(
        question_loader=lambda: [_question("q001")],
        ablation_provider=StaticAblationProvider({"runs": []}),
        id_factory=failing_id_factory,
    )

    view = service.load_ablation_snapshot()

    assert view["status"] == "unavailable"
    assert view["run_id"] == ""
    assert view["raw_report"] == {}
    assert "snapshot id generation failed" in view["message"]


def test_load_ablation_snapshot_contains_formatter_errors(monkeypatch):
    def failing_formatter(_payload):
        raise RuntimeError("snapshot formatting failed")

    monkeypatch.setattr(
        "evaluation.dashboard_service.build_ablation_summary_rows",
        failing_formatter,
    )
    service = EvaluationDashboardService(
        question_loader=lambda: [],
        ablation_provider=StaticAblationProvider({"runs": []}),
    )

    view = service.load_ablation_snapshot()

    assert view["status"] == "unavailable"
    assert view["run_id"] == ""
    assert view["raw_report"] == {}
    assert "snapshot formatting failed" in view["message"]


def test_service_runtime_config_is_safe_for_empty_views_and_deep_copied():
    payload = _ablation_payload(_result("q001", "retrieval_failure"))
    service = EvaluationDashboardService(
        question_loader=lambda: [_question("q001")],
        ablation_provider=StaticAblationProvider(payload),
    )
    view = service.load_ablation_snapshot()

    runtime_config = service.get_runtime_config(view, "v0_naive")
    runtime_config["llm"]["model"] = "mutated"

    assert service.get_runtime_config({}, "v0_naive") == {}
    assert service.get_runtime_config(
        {
            "status": "completed",
            "run_id": "empty",
            "summary_rows": [],
            "failure_count_rows": [],
            "failure_cases": [],
            "raw_report": {},
            "message": "",
        },
        "v0_naive",
    ) == {}
    assert view["raw_report"]["runs"][0]["runtime_config"]["llm"][
        "model"
    ] == "test-model"


@pytest.mark.parametrize(
    ("system_mode", "expected_calls", "expected_labels"),
    [
        (
            "naive",
            [("naive", "Question q001"), ("naive", "Question q002")],
            ["Naive RAG"],
        ),
        (
            "agentic",
            [("agentic", "Question q001"), ("agentic", "Question q002")],
            ["Agentic RAG"],
        ),
        (
            "comparison",
            [
                ("naive", "Question q001"),
                ("agentic", "Question q001"),
                ("naive", "Question q002"),
                ("agentic", "Question q002"),
            ],
            ["Naive RAG", "Agentic RAG"],
        ),
    ],
)
def test_run_quick_evaluation_dispatches_runners_and_builds_summary_rows(
    system_mode,
    expected_calls,
    expected_labels,
):
    calls = []
    id_calls = []

    def agentic_runner(question):
        calls.append(("agentic", question))
        return _runner_result(question)

    def naive_runner(question):
        calls.append(("naive", question))
        return _runner_result(question)

    def id_factory(prefix):
        id_calls.append(prefix)
        return f"{prefix}-id"

    service = EvaluationDashboardService(
        question_loader=lambda: [_question("q001"), _question("q002")],
        agentic_runner=agentic_runner,
        naive_runner=naive_runner,
        id_factory=id_factory,
    )

    view = service.run_quick_evaluation(["q002", "q001"], system_mode)

    assert view["status"] == "completed"
    assert view["run_id"] == "quick-id"
    assert id_calls == ["quick"]
    assert calls == expected_calls
    assert [row[0] for row in view["summary_rows"]] == expected_labels
    assert [
        result.get("question_id")
        or result["naive"]["question_id"]
        for result in view["raw_report"]["results"]
    ] == ["q001", "q002"]
    assert "2 question" in view["message"]
    assert "0 failure" in view["message"]


@pytest.mark.parametrize(
    ("question_ids", "system_mode", "message"),
    [
        ([], "agentic", "Select at least one"),
        (["q001", "q999"], "agentic", "q999"),
        (["q001"], "unsupported", "unsupported"),
    ],
)
def test_run_quick_evaluation_rejects_invalid_requests_before_runner_calls(
    question_ids,
    system_mode,
    message,
):
    runner_calls = []

    def runner(question):
        runner_calls.append(question)
        return _runner_result(question)

    service = EvaluationDashboardService(
        question_loader=lambda: [_question("q001")],
        agentic_runner=runner,
        naive_runner=runner,
        id_factory=lambda prefix: f"{prefix}-failed",
    )

    view = service.run_quick_evaluation(question_ids, system_mode)

    assert view["status"] == "failed"
    assert view["run_id"] == ""
    assert message in view["message"]
    assert runner_calls == []
    assert view["summary_rows"] == []
    assert view["failure_count_rows"] == []
    assert view["failure_cases"] == []
    assert view["raw_report"] == {}
    json.dumps(view)


def test_run_quick_evaluation_sorts_and_deduplicates_unknown_ids():
    service = EvaluationDashboardService(
        question_loader=lambda: [_question("q001")],
    )

    view = service.run_quick_evaluation(
        ["q999", "q998", "q999"],
        "agentic",
    )

    assert view["status"] == "failed"
    assert view["message"].count("q999") == 1
    assert view["message"].index("q998") < view["message"].index("q999")


def test_run_quick_evaluation_keeps_per_case_runner_errors_completed():
    def failing_runner(question):
        raise RuntimeError(f"runner failed for {question}")

    service = EvaluationDashboardService(
        question_loader=lambda: [_question("q001")],
        agentic_runner=failing_runner,
        id_factory=lambda prefix: f"{prefix}-error-case",
    )

    view = service.run_quick_evaluation(["q001"], "agentic")

    assert view["status"] == "completed"
    assert view["run_id"] == "quick-error-case"
    assert [case["failure_type"] for case in view["failure_cases"]] == [
        "tool_failure"
    ]
    assert view["failure_count_rows"] == [
        ["Agentic RAG", "tool_failure", 1]
    ]
    assert "1 question" in view["message"]
    assert "1 failure" in view["message"]


def test_run_quick_evaluation_returns_failed_view_for_loader_errors():
    def failing_loader():
        raise RuntimeError("dataset unavailable")

    service = EvaluationDashboardService(
        question_loader=failing_loader,
        id_factory=lambda prefix: f"{prefix}-loader-error",
    )

    view = service.run_quick_evaluation(["q001"], "agentic")

    assert view == {
        "status": "failed",
        "run_id": "",
        "summary_rows": [],
        "failure_count_rows": [],
        "failure_cases": [],
        "raw_report": {},
        "message": "Quick evaluation failed: RuntimeError: dataset unavailable",
    }


def test_run_quick_evaluation_contains_id_factory_errors():
    def failing_id_factory(prefix):
        raise RuntimeError("id generation failed")

    service = EvaluationDashboardService(
        question_loader=lambda: [_question("q001")],
        agentic_runner=_runner_result,
        id_factory=failing_id_factory,
    )

    view = service.run_quick_evaluation(["q001"], "agentic")

    assert view["status"] == "failed"
    assert view["run_id"] == ""
    assert view["raw_report"] == {}
    assert "id generation failed" in view["message"]


def test_service_filters_and_selects_failure_cases_without_mutating_view():
    cases = build_failure_cases(
        {
            "summary": {"mode": "comparison"},
            "results": [
                {
                    "naive": _result("q001", "retrieval_failure"),
                    "agentic": _result("q001", "citation_failure"),
                }
            ],
        }
    )
    view = {
        "status": "completed",
        "run_id": "quick-1",
        "summary_rows": [],
        "failure_count_rows": build_failure_count_rows(cases),
        "failure_cases": cases,
        "raw_report": {},
        "message": "",
    }
    original = deepcopy(view)
    service = EvaluationDashboardService()

    filtered = service.filter_failure_cases(
        view,
        system="agentic",
        failure_type="citation_failure",
    )
    detail = service.get_failure_detail(view, "agentic:q001")

    assert [case["case_key"] for case in filtered] == ["agentic:q001"]
    assert detail["reason"] == "Reason for q001"
    assert view == original


def test_default_id_factory_generates_unique_prefixed_run_ids():
    service = EvaluationDashboardService(
        question_loader=lambda: [_question("q001")],
        agentic_runner=_runner_result,
    )

    first = service.run_quick_evaluation(["q001"], "agentic")
    second = service.run_quick_evaluation(["q001"], "agentic")

    assert first["run_id"].startswith("quick-")
    assert second["run_id"].startswith("quick-")
    assert first["run_id"] != second["run_id"]
