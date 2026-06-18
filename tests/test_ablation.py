"""Tests for lightweight ablation experiment runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.run_ablation import format_ablation_report, main, select_questions
from experiments.variants import (
    AblationVariant,
    create_variant_runner,
    load_ablation_variants,
    validate_cumulative_variants,
)
from agent.features import AgentFeatureFlags
from config import get_settings
from evaluation.runtime_config import build_runtime_config_snapshot


CONFIG_DIR = Path("experiments/configs")


def _report_section(report: str, start: str, end: str) -> str:
    return report.split(start, 1)[1].split(end, 1)[0]


def test_repository_ablation_configs_define_distinct_cumulative_variants():
    variants = load_ablation_variants(CONFIG_DIR)

    assert [variant.id for variant in variants] == [
        "v0_naive",
        "v1_query_rewrite",
        "v2_retrieval_grading",
        "v3_retry_fallback",
        "v4_hybrid_retrieval",
        "v5_reranker",
        "v6_citation_verification",
    ]
    assert variants[0].runner == "naive"
    assert variants[3].features.conditional_retry_enabled is True
    assert variants[4].settings_overrides == {
        "hybrid_retrieval_enabled": True,
        "reranker_enabled": False,
    }
    assert variants[5].settings_overrides == {
        "hybrid_retrieval_enabled": True,
        "reranker_enabled": True,
    }
    assert variants[6].features.citation_verification_enabled is True

    validate_cumulative_variants(variants)


def test_validate_cumulative_variants_rejects_duplicate_effective_configs():
    duplicate = AblationVariant(
        id="v2_duplicate",
        method="duplicate",
        runner="agentic",
        features=AgentFeatureFlags(
            query_transformation_enabled=True,
            retrieval_grading_enabled=False,
            conditional_retry_enabled=False,
            citation_verification_enabled=False,
        ),
        settings_overrides={
            "hybrid_retrieval_enabled": False,
            "reranker_enabled": False,
        },
    )

    with pytest.raises(ValueError, match="duplicate effective"):
        validate_cumulative_variants(
            [
                AblationVariant(
                    id="v1_query_rewrite",
                    method="query rewrite",
                    runner="agentic",
                    features=duplicate.features,
                    settings_overrides=duplicate.settings_overrides,
                ),
                duplicate,
            ]
        )


def test_runtime_config_snapshot_includes_agent_feature_flags():
    features = AgentFeatureFlags(
        query_transformation_enabled=True,
        retrieval_grading_enabled=False,
        conditional_retry_enabled=False,
        citation_verification_enabled=False,
    )

    snapshot = build_runtime_config_snapshot(features=features)

    assert snapshot["agent_features"] == features.to_dict()


def test_runtime_config_snapshot_includes_safe_active_prompt_manifest():
    snapshot = build_runtime_config_snapshot()

    assert snapshot["schema_version"] == 2
    assert snapshot["evaluator_version"] == "p4d"
    assert set(snapshot["prompts"]) == {
        "agent.query_transform",
        "agent.retry_query_rewrite",
        "agent.retrieval_grading",
        "agent.answer_generation",
        "agent.claim_extraction",
        "agent.citation_verification",
        "agent.answer_revision",
        "tool.document_summary",
    }
    assert all(
        set(metadata) == {"version", "fingerprint"}
        for metadata in snapshot["prompts"].values()
    )
    assert "template" not in json.dumps(snapshot["prompts"])


def test_create_variant_runner_injects_variant_settings_and_features():
    variant = load_ablation_variants(CONFIG_DIR)[4]
    captured = {}

    def retriever_factory(settings):
        captured["retriever_settings"] = settings
        return lambda query: []

    def agent_runner(
        question,
        chat_history,
        *,
        settings,
        features,
        retriever_fn,
    ):
        captured.update(
            settings=settings,
            features=features,
            retriever_fn=retriever_fn,
            history=chat_history,
        )
        return {"answer": ""}

    runner = create_variant_runner(
        variant,
        base_settings=get_settings(),
        retriever_factory=retriever_factory,
        agent_runner=agent_runner,
    )
    history = [{"role": "user", "content": "Context"}]

    runner("Question?", history)

    assert captured["settings"].hybrid_retrieval_enabled is True
    assert captured["settings"].reranker_enabled is False
    assert captured["features"] == variant.features
    assert captured["retriever_settings"] == captured["settings"]
    assert captured["history"] == history


def test_ablation_main_checkpoints_variants_and_derives_comparison_without_reruns(
    tmp_path,
    monkeypatch,
):
    questions_path = tmp_path / "questions.json"
    output_dir = tmp_path / "artifacts"
    monkeypatch.setenv("OPENAI_API_KEY", "secret-key")
    questions_path.write_text(
        json.dumps(
            [
                {
                    "id": "q001",
                    "question": "What is Agentic RAG?",
                    "expected_keywords": ["retrieval"],
                    "expected_sources": ["notes.md"],
                }
            ]
        ),
        encoding="utf-8",
    )
    calls = []

    def runner_factory(variant, base_settings):
        def run(question, chat_history):
            calls.append(variant.id)
            if variant.id == "v1_query_rewrite":
                raise RuntimeError("synthetic variant error")
            verification_enabled = variant.features.citation_verification_enabled
            return {
                "answer": f"{variant.id} retrieval answer [1].",
                "citations": [{"source": "notes.md"}],
                "retrieved_documents": [{"source": "notes.md"}],
                "relevant_documents": [{"source": "notes.md"}],
                "claims": [{"claim_id": "c1"}] if verification_enabled else [],
                "claim_verification_results": (
                    [{"claim_id": "c1", "verification_label": "supported"}]
                    if verification_enabled
                    else []
                ),
                "citation_verification_enabled": verification_enabled,
                "citation_verification_passed": verification_enabled,
                "is_verified": verification_enabled,
            }

        return run

    exit_code = main(
        [
            "--questions",
            str(questions_path),
            "--config-dir",
            str(CONFIG_DIR),
            "--output-dir",
            str(output_dir),
        ],
        variant_runner_factory=runner_factory,
    )

    result_path = output_dir / "ablation_result.json"
    report_path = output_dir / "ablation_report.md"
    baseline_path = output_dir / "baseline_result.json"
    agentic_path = output_dir / "agentic_result.json"
    comparison_path = output_dir / "comparison_result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    agentic = json.loads(agentic_path.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")

    assert exit_code == 0
    assert result_path.exists()
    assert report_path.exists()
    assert baseline_path.exists()
    assert agentic_path.exists()
    assert comparison_path.exists()
    assert "secret-key" not in json.dumps(payload, ensure_ascii=False)
    assert [run["id"] for run in payload["runs"]] == [
        variant.id for variant in load_ablation_variants(CONFIG_DIR)
    ]
    assert payload["runs"][0]["status"] == "completed"
    assert payload["runs"][1]["status"] == "completed_with_errors"
    assert payload["runs"][-1]["status"] == "completed"
    assert len(calls) == 7
    assert calls.count("v0_naive") == 1
    assert calls.count("v6_citation_verification") == 1
    assert baseline["results"] == payload["runs"][0]["results"]
    assert agentic["results"] == payload["runs"][-1]["results"]
    assert comparison["summary"]["mode"] == "comparison"
    assert comparison["results"][0]["question_id"] == "q001"
    for variant in load_ablation_variants(CONFIG_DIR):
        assert (output_dir / "variants" / f"{variant.id}.json").exists()
    assert "Naive RAG" in report
    assert "Claim-level Citation Verification" in report


def test_select_questions_preserves_dataset_order_and_rejects_unknown_ids():
    questions = [
        {"id": "q001", "question": "one"},
        {"id": "q002", "question": "two"},
        {"id": "q003", "question": "three"},
    ]

    selected = select_questions(questions, "q003,q001")

    assert [question["id"] for question in selected] == ["q001", "q003"]
    with pytest.raises(ValueError, match="q999"):
        select_questions(questions, "q999")


def test_format_ablation_report_includes_failed_case_analysis():
    payload = {
        "runs": [
            {
                "id": "v0_naive",
                "method": "Naive RAG",
                "status": "completed",
                "summary": {
                    "failure_type_counts": {
                        "no_failure": 1,
                        "retrieval_failure": 1,
                    },
                },
                "results": [
                    {
                        "question_id": "q001",
                        "question_type": "single_doc",
                        "failure_analysis": {
                            "failure_type": "retrieval_failure",
                            "reason": "Expected source | missing\ncheck retriever",
                            "suggestion": "Improve retrieval.",
                        },
                    },
                    {
                        "question_id": "q002",
                        "question_type": "single_doc",
                        "failure_analysis": {
                            "failure_type": "no_failure",
                            "reason": "No action required.",
                            "suggestion": "No action required.",
                        },
                    },
                ],
            },
            {
                "id": "v6_citation_verification",
                "method": "+ Claim-level Citation Verification",
                "status": "completed",
                "summary": {
                    "failure_type_counts": {
                        "no_failure": 1,
                        "citation_failure": 1,
                    },
                },
                "results": [
                    {
                        "question_id": "q003",
                        "question_type": "multi_doc",
                        "failure_analysis": {
                            "failure_type": "citation_failure",
                            "reason": "Cited source is unsupported.",
                            "suggestion": "Tighten citation selection.",
                        },
                    },
                ],
            },
        ],
    }

    report = format_ablation_report(payload)

    assert "## Failed Case Analysis" in report
    assert "## Representative Failed Cases" in report
    assert "retrieval_failure" in report
    assert "citation_failure" in report
    assert (
        "| V0 Naive RAG | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |"
        in report
    )
    assert (
        "| V6 + Claim-level Citation Verification | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |"
        in report
    )
    assert "q001" in report
    assert "Expected source \\| missing check retriever" in report
    representative_cases = _report_section(
        report,
        "## Representative Failed Cases",
        "## Observed Trade-offs",
    )
    assert "q002" not in representative_cases
    assert "no_failure" not in representative_cases


def test_format_ablation_report_limits_representative_failed_cases_per_run():
    payload = {
        "runs": [
            {
                "id": "v0_naive",
                "method": "Naive RAG",
                "status": "completed",
                "summary": {
                    "failure_type_counts": {
                        "retrieval_failure": 4,
                    },
                },
                "results": [
                    {
                        "question_id": f"q00{index}",
                        "question_type": "single_doc",
                        "failure_analysis": {
                            "failure_type": "retrieval_failure",
                            "reason": f"Missing source {index}.",
                            "suggestion": "Tune retriever.",
                        },
                    }
                    for index in range(1, 5)
                ],
            },
            {
                "id": "v6_citation_verification",
                "method": "+ Claim-level Citation Verification",
                "status": "completed",
                "summary": {
                    "failure_type_counts": {
                        "citation_failure": 1,
                    },
                },
                "results": [
                    {
                        "question_id": "q101",
                        "question_type": "multi_doc",
                        "failure_analysis": {
                            "failure_type": "citation_failure",
                            "reason": "Citation mismatch.",
                            "suggestion": "Verify citations.",
                        },
                    }
                ],
            },
        ],
    }

    report = format_ablation_report(payload)

    representative_cases = _report_section(
        report,
        "## Representative Failed Cases",
        "## Observed Trade-offs",
    )
    assert "q001" in representative_cases
    assert "q002" in representative_cases
    assert "q003" in representative_cases
    assert "q004" not in representative_cases
    assert "q101" in representative_cases


def test_format_ablation_report_handles_missing_failure_analysis_inputs():
    payload = {
        "runs": [
            {
                "id": "v0_naive",
                "method": "Naive RAG",
                "status": "incomplete",
                "summary": None,
                "results": None,
            },
            {
                "id": "v6_citation_verification",
                "method": "+ Claim-level Citation Verification",
                "status": "completed",
                "results": [
                    {
                        "question_id": "q001",
                        "question_type": "single_doc",
                        "failure_analysis": None,
                    }
                ],
            },
            None,
        ],
    }

    report = format_ablation_report(payload)

    assert (
        "| V0 Naive RAG | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | incomplete |"
        in report
    )
    assert (
        "| V0 Naive RAG | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |"
        in report
    )
    assert "No failed cases recorded in completed runs." in report


def test_format_ablation_report_tolerates_missing_completed_run_summaries():
    payload = {
        "runs": [
            {
                "id": "v0_naive",
                "method": "Naive RAG",
                "status": "completed",
                "summary": None,
            },
            {
                "id": "v1_query_rewrite",
                "method": "+ Query Rewrite",
                "status": "completed",
                "summary": None,
            },
        ],
    }

    report = format_ablation_report(payload)

    assert "## Observed Trade-offs" in report
    assert "- No adjacent completed variants are available for comparison." in report


def test_format_ablation_report_uses_observed_metrics_and_explicit_limitations():
    payload = {
        "runs": [
            {
                "id": "v0_naive",
                "method": "Naive RAG",
                "status": "completed",
                "summary": {
                    "correctness_score": 0.5,
                    "context_relevance_score": 0.5,
                    "citation_hit_rate": 0.5,
                    "fallback_accuracy": 0.5,
                    "unsupported_claim_count": None,
                    "supported_claim_ratio": None,
                    "average_retry_count": 0.0,
                    "average_latency": 1.0,
                    "error_count": 0,
                },
            },
            {
                "id": "v6_citation_verification",
                "method": "+ Claim-level Citation Verification",
                "status": "completed",
                "summary": {
                    "correctness_score": 0.75,
                    "context_relevance_score": 0.75,
                    "citation_hit_rate": 0.75,
                    "fallback_accuracy": 0.75,
                    "unsupported_claim_count": 1,
                    "supported_claim_ratio": 0.8,
                    "average_retry_count": 0.5,
                    "average_latency": 2.0,
                    "error_count": 0,
                },
            },
        ]
    }

    report = format_ablation_report(payload)

    assert "| V0 Naive RAG |" in report
    assert "| V6 + Claim-level Citation Verification |" in report
    assert "## Observed Trade-offs" in report
    assert "correctness" in report.lower()
    assert "latency" in report.lower()
    assert "## Limitations" in report
    assert "N/A" in report
