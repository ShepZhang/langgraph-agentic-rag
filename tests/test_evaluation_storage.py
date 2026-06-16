from __future__ import annotations

import json

from evaluation.runtime_config import build_runtime_config_snapshot
from evaluation.storage import JsonResultStore, write_compatibility_artifacts


def test_json_result_store_saves_loads_utf8_payload_and_removes_tmp(tmp_path):
    store = JsonResultStore(tmp_path)
    payload = {"answer": "Agentic RAG 使用检索", "score": 1}

    path = store.save("run-1", payload)

    assert path == str(tmp_path / "run-1.json")
    assert store.load("run-1") == payload
    assert json.loads((tmp_path / "run-1.json").read_text(encoding="utf-8")) == payload
    assert not list(tmp_path.glob("*.tmp"))


def test_json_result_store_missing_load_returns_none(tmp_path):
    store = JsonResultStore(tmp_path)

    assert store.load("missing") is None


def test_compatibility_writer_keeps_comparison_artifact_names_and_metadata(tmp_path):
    runtime_config = build_runtime_config_snapshot()
    report = {
        "summary": {
            "mode": "comparison",
            "naive": {"total_questions": 1},
            "agentic": {"total_questions": 1},
        },
        "results": [
            {
                "naive": {"question": "What is Agentic RAG?"},
                "agentic": {"question": "What is Agentic RAG?"},
            }
        ],
    }

    write_compatibility_artifacts(report, tmp_path, runtime_config=runtime_config)

    baseline_path = tmp_path / "baseline_result.json"
    agentic_path = tmp_path / "agentic_result.json"
    comparison_path = tmp_path / "comparison_result.json"
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    agentic_payload = json.loads(agentic_path.read_text(encoding="utf-8"))
    comparison_payload = json.loads(comparison_path.read_text(encoding="utf-8"))

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "agentic_result.json",
        "baseline_result.json",
        "comparison_result.json",
    ]
    assert baseline_payload == {
        "system": "naive_rag",
        "runtime_config": runtime_config,
        "summary": {"total_questions": 1},
        "results": [{"question": "What is Agentic RAG?"}],
    }
    assert agentic_payload == {
        "system": "agentic_rag",
        "runtime_config": runtime_config,
        "summary": {"total_questions": 1},
        "results": [{"question": "What is Agentic RAG?"}],
    }
    assert comparison_payload == {"runtime_config": runtime_config, **report}
    assert comparison_payload["runtime_config"]["schema_version"] == 1
    assert comparison_payload["runtime_config"]["evaluator_version"] == "p4c"
