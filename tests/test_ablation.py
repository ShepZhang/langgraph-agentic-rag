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
from evaluation.judge_config import EvaluationJudgeSettings
from evaluation.runtime_config import (
    build_runtime_config_snapshot,
    build_runtime_metadata,
)


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

    assert snapshot["schema_version"] == 3
    assert snapshot["evaluator_version"] == "p5a"
    assert set(snapshot["prompts"]) == {
        "agent.query_transform",
        "agent.retry_query_rewrite",
        "agent.retrieval_grading",
        "agent.answer_generation",
        "agent.claim_extraction",
        "agent.citation_verification",
        "agent.answer_revision",
        "evaluation.semantic_judge",
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
    comparison_metrics = comparison["summary"]["comparison"]
    expected_judge_metrics = {
        "naive_average_semantic_correctness": baseline["summary"][
            "average_semantic_correctness"
        ],
        "agentic_average_semantic_correctness": agentic["summary"][
            "average_semantic_correctness"
        ],
        "naive_average_groundedness": baseline["summary"]["average_groundedness"],
        "agentic_average_groundedness": agentic["summary"][
            "average_groundedness"
        ],
        "naive_judge_completion_rate": baseline["summary"][
            "judge_completion_rate"
        ],
        "agentic_judge_completion_rate": agentic["summary"][
            "judge_completion_rate"
        ],
    }
    assert {
        key: comparison_metrics[key]
        for key in expected_judge_metrics
    } == expected_judge_metrics
    assert set(expected_judge_metrics.values()) == {None}
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
        "| V0 Naive RAG | N/A | N/A | N/A | N/A | N/A | N/A | "
        "N/A | N/A | N/A | N/A | N/A | N/A | incomplete |"
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
                    "average_semantic_correctness": None,
                    "average_groundedness": None,
                    "judge_completion_rate": None,
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
                    "average_semantic_correctness": 0.875,
                    "average_groundedness": 0.92,
                    "judge_completion_rate": 1.0,
                },
            },
        ]
    }

    report = format_ablation_report(payload)

    assert "| V0 Naive RAG |" in report
    assert "| V6 + Claim-level Citation Verification |" in report
    assert "Semantic Correctness" in report
    assert "Groundedness" in report
    assert "Judge Completion" in report
    # Parse cells by header index to assert exact formatted judge values
    header = next(
        line for line in report.splitlines() if line.startswith("| Method |")
    )
    header_cols = [col.strip() for col in header.split("|") if col.strip()]
    sem_idx = header_cols.index("Semantic Correctness")
    ground_idx = header_cols.index("Groundedness")
    judge_idx = header_cols.index("Judge Completion")

    v6_line = next(
        line
        for line in report.splitlines()
        if line.startswith("| V6 + Claim-level Citation Verification |")
    )
    v6_cells = [cell.strip() for cell in v6_line.split("|") if cell.strip()]
    assert v6_cells[sem_idx] == "0.8750"
    assert v6_cells[ground_idx] == "0.9200"
    assert v6_cells[judge_idx] == "1.0000"

    v0_line = next(
        line for line in report.splitlines() if line.startswith("| V0 Naive RAG |")
    )
    v0_cells = [cell.strip() for cell in v0_line.split("|") if cell.strip()]
    assert v0_cells[sem_idx] == "N/A"
    assert v0_cells[ground_idx] == "N/A"
    assert v0_cells[judge_idx] == "N/A"

    assert "## Observed Trade-offs" in report
    assert "correctness" in report.lower()
    assert "latency" in report.lower()
    assert "## Limitations" in report
    assert "semantic judge scores are model-based signals" in report.lower()
    assert (
        "enabling judge adds one model call per successful system result"
        in report.lower()
    )
    assert "N/A" in report


def test_format_ablation_report_renders_none_semantic_judge_values_as_na():
    """None judge metrics render as N/A in the ablation summary table."""
    payload = {
        "runs": [
            {
                "id": "v0_naive",
                "method": "Naive RAG",
                "status": "completed",
                "summary": {
                    "correctness_score": 0.5,
                    "average_semantic_correctness": None,
                    "average_groundedness": None,
                    "judge_completion_rate": None,
                },
            },
        ]
    }

    report = format_ablation_report(payload)

    # Header/separator parity
    header = next(
        line for line in report.splitlines() if line.startswith("| Method |")
    )
    separator = next(
        line
        for line in report.splitlines()
        if line.startswith("|---") and "---:" in line
    )
    header_cols = [col.strip() for col in header.split("|") if col.strip()]
    sep_cols = [col.strip() for col in separator.split("|") if col.strip()]
    assert len(header_cols) == len(sep_cols), (
        f"header={len(header_cols)} != separator={len(sep_cols)}"
    )
    assert "Semantic Correctness" in header_cols
    assert "Groundedness" in header_cols
    assert "Judge Completion" in header_cols

    # Parse data row cells and assert judge columns are exactly N/A by header index
    sem_idx = header_cols.index("Semantic Correctness")
    ground_idx = header_cols.index("Groundedness")
    judge_idx = header_cols.index("Judge Completion")

    data_line = next(
        line for line in report.splitlines() if line.startswith("| V0 Naive RAG |")
    )
    data_cells = [cell.strip() for cell in data_line.split("|") if cell.strip()]
    assert data_cells[sem_idx] == "N/A"
    assert data_cells[ground_idx] == "N/A"
    assert data_cells[judge_idx] == "N/A"


def test_runtime_config_includes_safe_judge_metadata_disabled():
    """Default disabled judge → metadata includes enabled=False, no secrets."""
    metadata = build_runtime_metadata()

    config = metadata.to_dict()
    assert "judge" in config
    assert config["judge"] == {
        "enabled": False,
        "provider": "openai_compatible",
        "model": None,
        "temperature": 0.0,
    }
    # Prove secrets are absent from the serialized config
    assert "api_key" not in config["judge"]
    assert "base_url" not in config["judge"]
    serialized = str(config)
    assert "api_key" not in serialized
    assert "base_url" not in serialized


def test_runtime_config_redacts_judge_credentials_when_enabled():
    """Enabled judge → metadata includes enabled=True, temperature, but redacts secrets."""
    settings = EvaluationJudgeSettings(
        enabled=True,
        api_key="sk-secret-key-12345",
        base_url="https://judge.example.com/v1",
        model="deepseek-v4",
        temperature=0.0,
    )
    metadata = build_runtime_metadata(judge_settings=settings)

    config = metadata.to_dict()
    assert config["judge"]["enabled"] is True
    assert config["judge"]["provider"] == "openai_compatible"
    assert config["judge"]["model"] == "deepseek-v4"
    assert config["judge"]["temperature"] == 0.0
    # Secrets must be redacted
    assert "api_key" not in config["judge"]
    assert "base_url" not in config["judge"]
    assert "sk-secret-key-12345" not in str(config)
    assert "judge.example.com" not in str(config)


def test_runtime_config_judge_settings_cached_in_snapshot():
    """build_runtime_config_snapshot forwards judge_settings to metadata."""
    settings = EvaluationJudgeSettings(
        enabled=True,
        api_key="sk-redacted-key",
        base_url="https://judge.example.com/v1",
        model="deepseek-v4",
        temperature=0.0,
    )
    snapshot = build_runtime_config_snapshot(judge_settings=settings)

    assert snapshot["judge"]["enabled"] is True
    assert snapshot["judge"]["model"] == "deepseek-v4"
    assert snapshot["judge"]["temperature"] == 0.0
    assert "sk-redacted" not in str(snapshot)
    assert "judge.example.com" not in str(snapshot)
    assert "api_key" not in snapshot["judge"]
    assert "base_url" not in snapshot["judge"]


def test_runtime_config_sanitizes_injected_judge_metadata():
    snapshot = build_runtime_config_snapshot(
        judge_metadata={
            "enabled": True,
            "provider": " injected ",
            "model": " custom-judge ",
            "temperature": 10**309,
            "api_key": "secret-key",
            "base_url": "https://secret.example/v1",
        }
    )

    assert snapshot["judge"] == {
        "enabled": True,
        "provider": "injected",
        "model": "custom-judge",
        "temperature": None,
    }
    assert "secret-key" not in str(snapshot)
    assert "secret.example" not in str(snapshot)


def test_build_runtime_metadata_calls_load_judge_settings_once_when_omitted(
    monkeypatch,
):
    """load_evaluation_judge_settings is called exactly once when judge_settings
    is omitted, and not called when judge_settings is injected."""
    from evaluation import runtime_config as rc_module

    call_count = 0

    def fake_load():
        nonlocal call_count
        call_count += 1
        return EvaluationJudgeSettings(
            enabled=False,
            api_key="",
            base_url="",
            model="",
            temperature=0.0,
        )

    monkeypatch.setattr(rc_module, "load_evaluation_judge_settings", fake_load)

    # Omitted → load_evaluation_judge_settings called exactly once
    build_runtime_metadata()
    assert call_count == 1

    # Injected → load_evaluation_judge_settings NOT called (call count stays 1)
    settings = EvaluationJudgeSettings(
        enabled=True,
        api_key="sk-key-123",
        base_url="https://example.com/v1",
        model="deepseek-v4",
        temperature=0.0,
    )
    build_runtime_metadata(judge_settings=settings)
    assert call_count == 1
