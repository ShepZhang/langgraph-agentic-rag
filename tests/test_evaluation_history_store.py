from __future__ import annotations

import sqlite3

import pytest

from evaluation.history_store import (
    EVALUATION_HISTORY_DB_SCHEMA_VERSION,
    HistoryRecord,
    HistoryStore,
    MetricRecord,
    compute_prompt_manifest_hash,
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


def test_history_store_rejects_unsupported_metric_names(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")

    with pytest.raises(ValueError, match="Unsupported history metric"):
        store.query_trends(metric="not_a_metric", system=None, limit=5)
