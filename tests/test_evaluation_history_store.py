from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest

from evaluation.history_service import EvaluationHistoryService
from evaluation.history_store import (
    EVALUATION_HISTORY_DB_SCHEMA_VERSION,
    HistoryRecord,
    HistoryStore,
    MetricRecord,
    compute_prompt_manifest_hash,
    extract_history_record,
    import_history_artifact,
)


def _runtime_config() -> dict:
    return {
        "schema_version": 4,
        "evaluator_version": "p5b",
        "prompts": {
            "agent.answer_generation": {
                "version": "v1",
                "fingerprint": "sha256:a",
            },
            "evaluation.semantic_judge": {
                "version": "v1",
                "fingerprint": "sha256:b",
            },
        },
    }


def test_history_store_initializes_schema_idempotently(tmp_path):
    db_path = tmp_path / "history.sqlite3"
    store = HistoryStore(db_path)

    store.initialize()
    store.initialize()

    with sqlite3.connect(db_path) as connection:
        meta = dict(connection.execute("SELECT key, value FROM evaluation_history_meta"))
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert meta["schema_version"] == str(EVALUATION_HISTORY_DB_SCHEMA_VERSION)
    assert {
        "evaluation_history_meta",
        "evaluation_runs",
        "evaluation_system_metrics",
        "evaluation_failure_counts",
    }.issubset(tables)


def test_history_store_inserts_lists_and_queries_trends(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    record = HistoryRecord(
        run_id="eval_1",
        created_at="2026-06-27T12:00:00.000000Z",
        source="cli",
        workspace_id=None,
        status="completed",
        mode="single",
        schema_version=4,
        evaluator_version="p5b",
        result_path="results/agentic_result.json",
        question_count=1,
        question_ids=["q001"],
        runtime_config=_runtime_config(),
        prompt_manifest=_runtime_config()["prompts"],
        prompt_manifest_hash=compute_prompt_manifest_hash(
            _runtime_config()["prompts"]
        ),
        summary={"total_questions": 1},
        metrics=[
            MetricRecord(
                system_id="agentic",
                system_label="Agentic RAG",
                metric_name="correctness_score",
                metric_value=0.75,
                metric_text=None,
            )
        ],
        failure_counts={"agentic": {"generation_failure": 1}},
    )

    store.save_record(record)
    store.save_record(record)

    runs = store.list_runs(limit=5)
    trends = store.query_trends(metric="correctness_score", system=None, limit=5)

    assert [run["run_id"] for run in runs] == ["eval_1"]
    assert runs[0]["evaluator_version"] == "p5b"
    assert runs[0]["prompt_manifest_hash"].startswith("sha256:")
    assert trends == [
        {
            "created_at": "2026-06-27T12:00:00.000000Z",
            "run_id": "eval_1",
            "system_id": "agentic",
            "system_label": "Agentic RAG",
            "evaluator_version": "p5b",
            "prompt_manifest_hash": record.prompt_manifest_hash,
            "metric_name": "correctness_score",
            "metric_value": 0.75,
        }
    ]


def test_history_store_rejects_invalid_run_ids(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    record = HistoryRecord(
        run_id="../escape",
        created_at="2026-06-27T12:00:00.000000Z",
        source="cli",
        workspace_id=None,
        status="completed",
        mode="single",
        schema_version=4,
        evaluator_version="p5b",
        result_path=None,
        question_count=0,
        question_ids=[],
        runtime_config={},
        prompt_manifest={},
        prompt_manifest_hash="",
        summary={},
        metrics=[],
        failure_counts={},
    )

    with pytest.raises(ValueError, match="run_id"):
        store.save_record(record)


def test_prompt_manifest_hash_is_canonical():
    left = {
        "b": {"version": "v1", "fingerprint": "sha256:b"},
        "a": {"version": "v1", "fingerprint": "sha256:a"},
    }
    right = {
        "a": {"fingerprint": "sha256:a", "version": "v1"},
        "b": {"fingerprint": "sha256:b", "version": "v1"},
    }

    assert compute_prompt_manifest_hash(left) == compute_prompt_manifest_hash(right)
    assert compute_prompt_manifest_hash({}) == ""


def test_history_extraction_sanitizes_runtime_metadata_before_sqlite_persistence(
    tmp_path,
):
    store = HistoryStore(tmp_path / "history.sqlite3")
    payload = {
        "runtime_config": {
            "schema_version": 4,
            "evaluator_version": "p5b",
            "api_key": "sk-unsafe-runtime",
            "secret": "unsafe-secret",
            "token": "unsafe-token",
            "llm": {
                "provider": "openai_compatible",
                "model": "safe-model",
                "temperature": 0.0,
                "api_key": "sk-unsafe-llm",
            },
            "prompts": {
                "agent.answer_generation": {
                    "version": "v1",
                    "fingerprint": "sha256:safe",
                    "template": "unsafe full prompt template",
                    "rendered_prompt": "unsafe rendered prompt",
                    "api_key": "sk-unsafe-prompt",
                }
            },
        },
        "summary": {
            "total_questions": 1,
            "correctness_score": 1.0,
            "template": "unsafe summary template",
            "secret": "unsafe summary secret",
        },
        "results": [{"question_id": "q001"}],
    }

    record = extract_history_record(
        payload,
        run_id="eval_sanitized",
        created_at="2026-06-27T12:00:00.000000Z",
        source="import",
        result_path="unsafe.json",
    )
    store.save_record(record)

    with sqlite3.connect(store.db_path) as connection:
        runtime_config_json, prompt_manifest_json, summary_json = connection.execute(
            """
            SELECT runtime_config_json, prompt_manifest_json, summary_json
            FROM evaluation_runs
            WHERE run_id = ?
            """,
            ("eval_sanitized",),
        ).fetchone()

    persisted = "\n".join(
        [runtime_config_json, prompt_manifest_json, summary_json]
    )
    for forbidden in (
        "unsafe full prompt template",
        "unsafe rendered prompt",
        "sk-unsafe-runtime",
        "sk-unsafe-llm",
        "sk-unsafe-prompt",
        "unsafe-secret",
        "unsafe-token",
        "unsafe summary template",
        "unsafe summary secret",
    ):
        assert forbidden not in persisted

    assert json.loads(prompt_manifest_json) == {
        "agent.answer_generation": {
            "fingerprint": "sha256:safe",
            "version": "v1",
        }
    }
    assert json.loads(runtime_config_json)["llm"] == {
        "model": "safe-model",
        "provider": "openai_compatible",
        "temperature": 0.0,
    }


def test_history_store_rejects_unsupported_metric_names(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")

    with pytest.raises(ValueError, match="Unsupported history metric"):
        store.query_trends(metric="not_a_metric", system=None, limit=5)


def test_extract_single_system_report_records_agentic_metrics():
    payload = {
        "system": "agentic_rag",
        "runtime_config": _runtime_config(),
        "summary": {
            "total_questions": 1,
            "correctness_score": 1.0,
            "average_latency": 0.25,
            "failure_type_counts": {"no_failure": 1},
        },
        "results": [{"question_id": "q001"}],
    }

    record = extract_history_record(
        payload,
        run_id="eval_single",
        created_at="2026-06-27T12:00:00.000000Z",
        source="cli",
        result_path="agentic_result.json",
    )

    assert record.mode == "single"
    assert record.schema_version == 4
    assert record.evaluator_version == "p5b"
    assert record.question_count == 1
    assert record.question_ids == ["q001"]
    assert [(m.system_id, m.metric_name, m.metric_value) for m in record.metrics] == [
        ("agentic", "correctness_score", 1.0),
        ("agentic", "average_latency", 0.25),
    ]
    assert record.failure_counts == {"agentic": {"no_failure": 1}}


def test_extract_history_record_requires_keyword_metadata_arguments():
    with pytest.raises(TypeError):
        extract_history_record({}, "run", "date", "cli")


def test_extract_comparison_report_records_naive_and_agentic_metrics():
    payload = {
        "runtime_config": _runtime_config(),
        "summary": {
            "mode": "comparison",
            "total_questions": 1,
            "naive": {
                "total_questions": 1,
                "correctness_score": 0.5,
                "failure_type_counts": {"retrieval_failure": 1},
            },
            "agentic": {
                "total_questions": 1,
                "correctness_score": 0.75,
                "judge_completion_rate": 1.0,
                "failure_type_counts": {"no_failure": 1},
            },
        },
        "results": [{"naive": {"question_id": "q001"}, "agentic": {"question_id": "q001"}}],
    }

    record = extract_history_record(
        payload,
        run_id="eval_comparison",
        created_at="2026-06-27T12:00:00.000000Z",
        source="cli",
        result_path="comparison_result.json",
    )

    metrics = {
        (metric.system_id, metric.metric_name): metric.metric_value
        for metric in record.metrics
    }
    assert record.mode == "comparison"
    assert metrics[("naive", "correctness_score")] == 0.5
    assert metrics[("agentic", "correctness_score")] == 0.75
    assert metrics[("agentic", "judge_completion_rate")] == 1.0
    assert record.failure_counts["naive"] == {"retrieval_failure": 1}


def test_extract_legacy_matrix_report_uses_legacy_metadata():
    payload = {
        "summary": {
            "mode": "matrix",
            "total_questions": 1,
            "variants": {
                "naive": {"correctness_score": 0.2},
                "agentic": {"correctness_score": 0.6},
                "agentic_reranker": {"correctness_score": 0.7},
            },
        },
        "results": [{"question": "What?"}],
    }

    record = extract_history_record(
        payload,
        run_id="legacy_matrix",
        created_at="2026-06-27T12:00:00.000000Z",
        source="matrix",
        result_path="matrix.json",
    )

    assert record.mode == "matrix"
    assert record.schema_version is None
    assert record.evaluator_version is None
    assert record.prompt_manifest_hash == ""
    assert {metric.system_id for metric in record.metrics} == {
        "naive",
        "agentic",
        "agentic_reranker",
    }


def test_extract_ablation_payload_records_completed_variants():
    payload = {
        "kind": "ablation_result",
        "question_ids": ["q001"],
        "runs": [
            {
                "id": "v0_naive",
                "method": "Naive RAG",
                "status": "completed",
                "runtime_config": _runtime_config(),
                "summary": {"correctness_score": 0.3},
                "results": [{"question_id": "q001"}],
            },
            {
                "id": "v1_query_rewrite",
                "method": "+ Query Transformation",
                "status": "incomplete",
                "summary": {"correctness_score": 0.4},
                "results": [{"question_id": "q001"}],
            },
        ],
    }

    record = extract_history_record(
        payload,
        run_id="ablation_1",
        created_at="2026-06-27T12:00:00.000000Z",
        source="ablation",
        result_path="ablation_result.json",
    )

    assert record.mode == "ablation"
    assert record.question_ids == ["q001"]
    assert [(m.system_id, m.system_label, m.metric_name) for m in record.metrics] == [
        ("v0_naive", "v0_naive Naive RAG", "correctness_score")
    ]


def test_extract_ablation_fallback_questions_ignore_incomplete_runs():
    payload = {
        "kind": "ablation_result",
        "runs": [
            {
                "id": "v1_query_rewrite",
                "method": "+ Query Transformation",
                "status": "incomplete",
                "summary": {"correctness_score": 0.4},
                "results": [{"question_id": "q_bad"}],
            },
            {
                "id": "v0_naive",
                "method": "Naive RAG",
                "status": "completed",
                "runtime_config": _runtime_config(),
                "summary": {"correctness_score": 0.3},
                "results": [{"question_id": "q_good"}],
            },
        ],
    }

    record = extract_history_record(
        payload,
        run_id="ablation_fallback",
        created_at="2026-06-27T12:00:00.000000Z",
        source="ablation",
    )

    assert record.question_ids == ["q_good"]
    assert record.question_count == 1


def test_extract_api_wrapper_uses_nested_report_and_workspace():
    payload = {
        "run_id": "eval_api",
        "workspace_id": "workspace_1",
        "status": "completed",
        "result_path": "data/evaluation_runs/eval_api.json",
        "summary": {"total_questions": 1},
        "report": {
            "runtime_config": _runtime_config(),
            "summary": {"total_questions": 1, "correctness_score": 0.9},
            "results": [{"question_id": "q001"}],
        },
    }

    record = extract_history_record(
        payload,
        run_id="eval_api",
        created_at="2026-06-27T12:00:00.000000Z",
        source="api",
        result_path=payload["result_path"],
    )

    assert record.workspace_id == "workspace_1"
    assert record.status == "completed"
    assert record.mode == "single"
    assert record.metrics[0].system_id == "agentic"


def test_import_history_artifact_generates_stable_id(tmp_path):
    path = tmp_path / "agentic_result.json"
    path.write_text(
        json.dumps(
            {
                "system": "agentic_rag",
                "runtime_config": _runtime_config(),
                "summary": {"total_questions": 1, "correctness_score": 1.0},
                "results": [{"question_id": "q001"}],
            }
        ),
        encoding="utf-8",
    )
    store = HistoryStore(tmp_path / "history.sqlite3")

    first = import_history_artifact(path, store=store)
    second = import_history_artifact(path, store=store)

    assert first["status"] == "stored"
    assert second["status"] == "stored"
    assert first["run_id"] == second["run_id"]
    assert store.list_runs()[0]["run_id"] == first["run_id"]


def test_history_service_disabled_returns_disabled(tmp_path):
    settings = SimpleNamespace(
        evaluation_history_enabled=False,
        evaluation_history_db=tmp_path / "history.sqlite3",
    )
    service = EvaluationHistoryService(settings=settings)

    status = service.record_payload(
        {
            "runtime_config": _runtime_config(),
            "summary": {"total_questions": 1, "correctness_score": 1.0},
            "results": [{"question_id": "q001"}],
        },
        source="cli",
        result_path="agentic_result.json",
    )

    assert status == {"status": "disabled", "run_id": None, "error": None}
    assert not settings.evaluation_history_db.exists()


def test_history_service_stores_payload_and_isolates_failures(tmp_path, monkeypatch):
    settings = SimpleNamespace(
        evaluation_history_enabled=True,
        evaluation_history_db=tmp_path / "history.sqlite3",
    )
    service = EvaluationHistoryService(settings=settings)

    stored = service.record_payload(
        {
            "runtime_config": _runtime_config(),
            "summary": {"total_questions": 1, "correctness_score": 1.0},
            "results": [{"question_id": "q001"}],
        },
        source="cli",
        result_path="agentic_result.json",
        run_id="eval_history_service",
    )

    assert stored["status"] == "stored"
    assert stored["run_id"] == "eval_history_service"
    assert service.list_runs(limit=1)[0]["run_id"] == "eval_history_service"

    def fail_save(_record):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(service._store, "save_record", fail_save)
    failed = service.record_payload(
        {"summary": {}, "results": []},
        source="cli",
        result_path="broken.json",
        run_id="eval_failed",
    )

    assert failed["status"] == "failed"
    assert failed["run_id"] == "eval_failed"
    assert "database is locked" in str(failed["error"])
