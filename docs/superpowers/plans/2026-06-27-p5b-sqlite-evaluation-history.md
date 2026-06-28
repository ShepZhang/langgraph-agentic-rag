# P5b SQLite Historical Evaluation + Trend Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SQLite-backed historical evaluation storage and read-only trend views while preserving existing JSON artifacts.

**Architecture:** Keep JSON artifacts as the full compatibility payload and add SQLite as a sidecar index. A pure `evaluation.history_store` module owns SQLite schema, extraction, import, and queries; `evaluation.history_service` isolates settings and write failures; CLI, FastAPI, and Dashboard call the service through narrow boundaries.

**Tech Stack:** Python standard library `sqlite3`, dataclasses/TypedDict, existing FastAPI/Pydantic schemas, existing Gradio dashboard tables, pytest, ruff, compileall.

---

## File Structure

Create:

- `evaluation/history_store.py` — SQLite schema, normalized records, prompt manifest hashing, payload extraction, insert/list/trend/import helpers.
- `evaluation/history_service.py` — settings-aware facade, disabled/failed/stored statuses, API/Dashboard-friendly reads.
- `tests/test_evaluation_history_store.py` — focused unit tests for schema, extraction, insert, query, legacy payloads, hashing, and importer.
- `docs/superpowers/plans/2026-06-27-p5b-sqlite-evaluation-history.md` — this implementation plan.

Modify:

- `config.py` — add `evaluation_history_enabled` and `evaluation_history_db`.
- `.env.example` — document P5b history settings.
- `evaluation/runtime_config.py` — advance runtime schema to `4` and evaluator to `p5b`.
- `evaluation/evaluate.py` — record CLI JSON artifacts into history after JSON writes succeed.
- `api/services/evaluation.py` — record API evaluation wrappers into history and expose list/trend methods.
- `api/schemas.py` — add history response models.
- `api/routes/evaluation.py` — add `/evaluation/history` and `/evaluation/history/trends` before `/{run_id}`.
- `evaluation/dashboard_models.py` — add history table contracts and typed views.
- `evaluation/dashboard_formatters.py` — add pure history row formatters.
- `evaluation/dashboard_service.py` — add read-only history snapshot/trend methods.
- `ui/gradio_app.py` — add a minimal `History Trends` Evaluation sub-tab.
- `tests/test_config.py` — cover new config defaults and validation.
- `tests/test_evaluation_storage.py` — preserve JSON compatibility and assert schema/evaluator version bump.
- `tests/test_evaluate.py` — cover history write after CLI JSON artifact write and disabled behavior.
- `tests/test_fastapi_routes.py` — cover new history routes and route ordering.
- `tests/test_dashboard_service.py` — cover history service views and formatters.
- `tests/test_gradio_app.py` — cover history UI helper output and tab wiring at the function level.
- `README.md`, `CHANGELOG.md`, `docs/github_release_checklist.md` — document P5b behavior, limits, and verification.

Do not modify:

- Agent graph behavior
- Judge scoring behavior
- existing JSON artifact filenames
- background evaluation behavior
- trace drill-down behavior

---

### Task 1: Add P5b Configuration And Runtime Metadata Version

**Files:**
- Modify: `config.py`
- Modify: `.env.example`
- Modify: `evaluation/runtime_config.py`
- Test: `tests/test_config.py`
- Test: `tests/test_evaluation_storage.py`

- [ ] **Step 1: Write failing config tests**

Add tests to `tests/test_config.py`:

```python
def test_settings_loads_evaluation_history_defaults(monkeypatch):
    for name in (
        "EVALUATION_HISTORY_ENABLED",
        "EVALUATION_HISTORY_DB",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = get_settings()

    assert settings.evaluation_history_enabled is True
    assert settings.evaluation_history_db == Path("./data/evaluation_history.sqlite3")


def test_settings_loads_evaluation_history_overrides(monkeypatch, tmp_path):
    db_path = tmp_path / "history.sqlite3"
    monkeypatch.setenv("EVALUATION_HISTORY_ENABLED", "false")
    monkeypatch.setenv("EVALUATION_HISTORY_DB", str(db_path))

    settings = get_settings()

    assert settings.evaluation_history_enabled is False
    assert settings.evaluation_history_db == db_path


def test_settings_rejects_empty_evaluation_history_db(monkeypatch):
    monkeypatch.setenv("EVALUATION_HISTORY_DB", "   ")

    with pytest.raises(ValueError, match="EVALUATION_HISTORY_DB"):
        get_settings()
```

If `tests/test_config.py` does not already import `Path`, `pytest`, and
`get_settings`, add:

```python
from pathlib import Path

import pytest

from config import get_settings
```

- [ ] **Step 2: Write failing runtime version assertion**

Update `tests/test_evaluation_storage.py` in
`test_compatibility_writer_keeps_comparison_artifact_names_and_metadata`:

```python
assert comparison_payload["runtime_config"]["schema_version"] == 4
assert comparison_payload["runtime_config"]["evaluator_version"] == "p5b"
```

- [ ] **Step 3: Run focused tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_config.py tests/test_evaluation_storage.py -q
```

Expected before implementation:

- config tests fail because `Settings` has no history fields
- storage test fails because runtime metadata still reports schema `3` and `p5a`

- [ ] **Step 4: Implement config fields**

In `config.py`, add fields to `Settings` after `evaluation_run_dir`:

```python
    evaluation_history_enabled: bool
    evaluation_history_db: Path
```

In `Settings.__post_init__`, after the `evaluation_run_dir` validation, add:

```python
        if not str(self.evaluation_history_db).strip():
            raise ValueError("EVALUATION_HISTORY_DB must not be empty")
```

In `get_settings()`, after `evaluation_run_dir=...`, add:

```python
        evaluation_history_enabled=_get_bool("EVALUATION_HISTORY_ENABLED", True),
        evaluation_history_db=Path(
            os.getenv("EVALUATION_HISTORY_DB", "./data/evaluation_history.sqlite3")
        ),
```

- [ ] **Step 5: Advance runtime metadata**

In `evaluation/runtime_config.py`, change:

```python
EVALUATION_SCHEMA_VERSION = 4
EVALUATOR_VERSION = "p5b"
```

Do not add local DB paths to `build_runtime_metadata()`.

- [ ] **Step 6: Update `.env.example`**

Append these settings under `EVALUATION_RUN_DIR`:

```dotenv

# SQLite sidecar history for evaluation runs.
EVALUATION_HISTORY_ENABLED=true
EVALUATION_HISTORY_DB=./data/evaluation_history.sqlite3
```

- [ ] **Step 7: Run focused tests and verify pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_config.py tests/test_evaluation_storage.py -q
```

Expected after implementation:

- all selected tests pass
- comparison artifacts still contain the existing JSON shape with schema `4` and evaluator `p5b`

- [ ] **Step 8: Commit Task 1**

Run:

```bash
git add config.py .env.example evaluation/runtime_config.py tests/test_config.py tests/test_evaluation_storage.py
git commit -m "feat: configure p5b evaluation history"
```

---

### Task 2: Implement SQLite History Store Schema, Records, And Queries

**Files:**
- Create: `evaluation/history_store.py`
- Create: `tests/test_evaluation_history_store.py`

- [ ] **Step 1: Write failing store schema and query tests**

Create `tests/test_evaluation_history_store.py` with:

```python
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

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
```

- [ ] **Step 2: Run store tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_history_store.py -q
```

Expected before implementation:

- import fails because `evaluation.history_store` does not exist

- [ ] **Step 3: Implement dataclasses, constants, and schema**

Create `evaluation/history_store.py` with these public constants and dataclasses:

```python
"""SQLite sidecar history store for evaluation runs."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EVALUATION_HISTORY_DB_SCHEMA_VERSION = 1
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
HISTORY_METRIC_NAMES = (
    "correctness_score",
    "context_relevance_score",
    "citation_hit_rate",
    "fallback_accuracy",
    "unsupported_claim_count",
    "average_latency",
    "average_retry_count",
    "error_count",
    "average_semantic_correctness",
    "average_groundedness",
    "judge_completion_rate",
)


@dataclass(frozen=True)
class MetricRecord:
    system_id: str
    system_label: str
    metric_name: str
    metric_value: float | None
    metric_text: str | None = None


@dataclass(frozen=True)
class HistoryRecord:
    run_id: str
    created_at: str
    source: str
    workspace_id: str | None
    status: str
    mode: str
    schema_version: int | None
    evaluator_version: str | None
    result_path: str | None
    question_count: int
    question_ids: list[str] = field(default_factory=list)
    runtime_config: dict[str, Any] = field(default_factory=dict)
    prompt_manifest: dict[str, Any] = field(default_factory=dict)
    prompt_manifest_hash: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    metrics: list[MetricRecord] = field(default_factory=list)
    failure_counts: dict[str, dict[str, int]] = field(default_factory=dict)
```

Then add:

```python
def compute_prompt_manifest_hash(manifest: Mapping[str, Any]) -> str:
    if not manifest:
        return ""
    canonical = json.dumps(
        dict(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _validate_run_id(run_id: str) -> None:
    if (
        not run_id
        or run_id in {".", ".."}
        or _RUN_ID_PATTERN.fullmatch(run_id) is None
    ):
        raise ValueError("run_id must be a safe file stem")
```

Implement `HistoryStore.initialize()` and `_connect()`:

```python
class HistoryStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS evaluation_history_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    workspace_id TEXT,
                    status TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    schema_version INTEGER,
                    evaluator_version TEXT,
                    result_path TEXT,
                    question_count INTEGER NOT NULL DEFAULT 0,
                    question_ids_json TEXT NOT NULL DEFAULT '[]',
                    runtime_config_json TEXT NOT NULL DEFAULT '{}',
                    prompt_manifest_json TEXT NOT NULL DEFAULT '{}',
                    prompt_manifest_hash TEXT NOT NULL DEFAULT '',
                    summary_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS evaluation_system_metrics (
                    run_id TEXT NOT NULL,
                    system_id TEXT NOT NULL,
                    system_label TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    metric_text TEXT,
                    PRIMARY KEY (run_id, system_id, metric_name),
                    FOREIGN KEY (run_id)
                        REFERENCES evaluation_runs(run_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS evaluation_failure_counts (
                    run_id TEXT NOT NULL,
                    system_id TEXT NOT NULL,
                    failure_type TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    PRIMARY KEY (run_id, system_id, failure_type),
                    FOREIGN KEY (run_id)
                        REFERENCES evaluation_runs(run_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_evaluation_runs_created_at
                ON evaluation_runs(created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_evaluation_runs_evaluator_version
                ON evaluation_runs(evaluator_version);

                CREATE INDEX IF NOT EXISTS idx_evaluation_runs_prompt_manifest_hash
                ON evaluation_runs(prompt_manifest_hash);

                CREATE INDEX IF NOT EXISTS idx_evaluation_metrics_name_system
                ON evaluation_system_metrics(metric_name, system_id);
                """
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO evaluation_history_meta (key, value)
                VALUES ('schema_version', ?)
                """,
                (str(EVALUATION_HISTORY_DB_SCHEMA_VERSION),),
            )
```

- [ ] **Step 4: Implement save/list/trend methods**

Add these methods to `HistoryStore`:

```python
    def save_record(self, record: HistoryRecord) -> str:
        _validate_run_id(record.run_id)
        self.initialize()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO evaluation_runs (
                    run_id,
                    created_at,
                    source,
                    workspace_id,
                    status,
                    mode,
                    schema_version,
                    evaluator_version,
                    result_path,
                    question_count,
                    question_ids_json,
                    runtime_config_json,
                    prompt_manifest_json,
                    prompt_manifest_hash,
                    summary_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.created_at,
                    record.source,
                    record.workspace_id,
                    record.status,
                    record.mode,
                    record.schema_version,
                    record.evaluator_version,
                    record.result_path,
                    record.question_count,
                    _json_text(record.question_ids),
                    _json_text(record.runtime_config),
                    _json_text(record.prompt_manifest),
                    record.prompt_manifest_hash,
                    _json_text(record.summary),
                ),
            )
            connection.execute(
                "DELETE FROM evaluation_system_metrics WHERE run_id = ?",
                (record.run_id,),
            )
            connection.execute(
                "DELETE FROM evaluation_failure_counts WHERE run_id = ?",
                (record.run_id,),
            )
            connection.executemany(
                """
                INSERT INTO evaluation_system_metrics (
                    run_id,
                    system_id,
                    system_label,
                    metric_name,
                    metric_value,
                    metric_text
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.run_id,
                        metric.system_id,
                        metric.system_label,
                        metric.metric_name,
                        metric.metric_value,
                        metric.metric_text,
                    )
                    for metric in record.metrics
                ],
            )
            failure_rows = [
                (record.run_id, system_id, failure_type, count)
                for system_id, counts in record.failure_counts.items()
                for failure_type, count in counts.items()
            ]
            connection.executemany(
                """
                INSERT INTO evaluation_failure_counts (
                    run_id,
                    system_id,
                    failure_type,
                    count
                )
                VALUES (?, ?, ?, ?)
                """,
                failure_rows,
            )
        return record.run_id

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        self.initialize()
        safe_limit = _safe_limit(limit)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    run_id,
                    created_at,
                    source,
                    workspace_id,
                    status,
                    mode,
                    schema_version,
                    evaluator_version,
                    result_path,
                    question_count,
                    prompt_manifest_hash
                FROM evaluation_runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def query_trends(
        self,
        metric: str,
        system: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if metric not in HISTORY_METRIC_NAMES:
            raise ValueError(f"Unsupported history metric: {metric}")
        self.initialize()
        safe_limit = _safe_limit(limit)
        parameters: list[Any] = [metric]
        system_filter = ""
        if system:
            system_filter = "AND metrics.system_id = ?"
            parameters.append(system)
        parameters.append(safe_limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    runs.created_at,
                    runs.run_id,
                    metrics.system_id,
                    metrics.system_label,
                    COALESCE(runs.evaluator_version, 'legacy') AS evaluator_version,
                    runs.prompt_manifest_hash,
                    metrics.metric_name,
                    metrics.metric_value
                FROM evaluation_system_metrics AS metrics
                JOIN evaluation_runs AS runs
                    ON runs.run_id = metrics.run_id
                WHERE metrics.metric_name = ?
                {system_filter}
                ORDER BY runs.created_at DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [dict(row) for row in reversed(rows)]
```

Add `_safe_limit()` near helpers:

```python
def _safe_limit(limit: int) -> int:
    return max(1, min(int(limit), 200))
```

- [ ] **Step 5: Run store tests and verify pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_history_store.py -q
```

Expected after implementation:

- all selected tests pass

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add evaluation/history_store.py tests/test_evaluation_history_store.py
git commit -m "feat: add sqlite evaluation history store"
```

---

### Task 3: Extract History Records From Evaluation Payload Shapes

**Files:**
- Modify: `evaluation/history_store.py`
- Modify: `tests/test_evaluation_history_store.py`

- [ ] **Step 1: Write failing extraction tests**

Append these tests to `tests/test_evaluation_history_store.py`:

```python
from evaluation.history_store import (
    extract_history_record,
    import_history_artifact,
)


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
```

- [ ] **Step 2: Run extraction tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_history_store.py -q
```

Expected before implementation:

- import or attribute failures for `extract_history_record` and `import_history_artifact`

- [ ] **Step 3: Implement extraction helpers**

In `evaluation/history_store.py`, add:

```python
def extract_history_record(
    payload: Mapping[str, Any],
    *,
    run_id: str,
    created_at: str,
    source: str,
    result_path: str | None = None,
) -> HistoryRecord:
    materialized = dict(payload)
    report = _nested_report(materialized)
    runtime_config = _runtime_config(materialized, report)
    prompt_manifest = _mapping(runtime_config.get("prompts"))
    summary = _mapping(report.get("summary"))
    mode = _mode_for_payload(materialized, report, summary)
    system_summaries = _system_summaries(materialized, report, summary, mode)
    question_ids = _question_ids(materialized, report)
    prompt_hash = compute_prompt_manifest_hash(prompt_manifest)

    return HistoryRecord(
        run_id=run_id,
        created_at=created_at,
        source=source,
        workspace_id=_optional_string(materialized.get("workspace_id")),
        status=str(materialized.get("status") or "completed"),
        mode=mode,
        schema_version=_optional_int(runtime_config.get("schema_version")),
        evaluator_version=_optional_string(runtime_config.get("evaluator_version")),
        result_path=result_path or _optional_string(materialized.get("result_path")),
        question_count=_question_count(summary, report, question_ids),
        question_ids=question_ids,
        runtime_config=dict(runtime_config),
        prompt_manifest=dict(prompt_manifest),
        prompt_manifest_hash=prompt_hash,
        summary=dict(summary),
        metrics=_metric_records(system_summaries),
        failure_counts=_failure_counts(system_summaries),
    )
```

Add the helper functions below it:

```python
def _nested_report(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    report = payload.get("report")
    return report if isinstance(report, Mapping) else payload


def _runtime_config(
    payload: Mapping[str, Any],
    report: Mapping[str, Any],
) -> Mapping[str, Any]:
    runtime_config = report.get("runtime_config")
    if isinstance(runtime_config, Mapping):
        return runtime_config
    runtime_config = payload.get("runtime_config")
    if isinstance(runtime_config, Mapping):
        return runtime_config
    if _is_ablation_payload(payload):
        for run in _sequence(payload.get("runs")):
            if isinstance(run, Mapping) and isinstance(run.get("runtime_config"), Mapping):
                return run["runtime_config"]
    return {}


def _mode_for_payload(
    payload: Mapping[str, Any],
    report: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> str:
    if _is_ablation_payload(payload):
        return "ablation"
    mode = summary.get("mode")
    if mode == "comparison":
        return "comparison"
    if mode == "matrix":
        return "matrix"
    return "single"


def _system_summaries(
    payload: Mapping[str, Any],
    report: Mapping[str, Any],
    summary: Mapping[str, Any],
    mode: str,
) -> list[tuple[str, str, Mapping[str, Any]]]:
    if mode == "comparison":
        return [
            ("naive", "Naive RAG", _mapping(summary.get("naive"))),
            ("agentic", "Agentic RAG", _mapping(summary.get("agentic"))),
        ]
    if mode == "matrix":
        variants = _mapping(summary.get("variants"))
        return [
            (str(system_id), _system_label(str(system_id)), _mapping(system_summary))
            for system_id, system_summary in variants.items()
            if isinstance(system_summary, Mapping)
        ]
    if mode == "ablation":
        rows: list[tuple[str, str, Mapping[str, Any]]] = []
        for run in _sequence(payload.get("runs")):
            if not isinstance(run, Mapping):
                continue
            status = str(run.get("status") or "completed")
            if status not in {"completed", "completed_with_errors"}:
                continue
            run_id = str(run.get("id") or "unknown")
            method = str(run.get("method") or run_id)
            rows.append((run_id, f"{run_id} {method}", _mapping(run.get("summary"))))
        return rows
    system = str(report.get("system") or payload.get("system") or "agentic_rag")
    system_id = _system_id(system)
    return [(system_id, _system_label(system_id), summary)]


def _metric_records(
    system_summaries: Sequence[tuple[str, str, Mapping[str, Any]]],
) -> list[MetricRecord]:
    records: list[MetricRecord] = []
    for system_id, label, summary in system_summaries:
        for metric_name in HISTORY_METRIC_NAMES:
            if metric_name not in summary:
                continue
            value = summary.get(metric_name)
            if isinstance(value, int | float) and not isinstance(value, bool):
                records.append(
                    MetricRecord(
                        system_id=system_id,
                        system_label=label,
                        metric_name=metric_name,
                        metric_value=float(value),
                        metric_text=None,
                    )
                )
            elif value is not None:
                records.append(
                    MetricRecord(
                        system_id=system_id,
                        system_label=label,
                        metric_name=metric_name,
                        metric_value=None,
                        metric_text=str(value),
                    )
                )
    return records
```

Add `_failure_counts`, `_question_ids`, and small coercion helpers:

```python
def _failure_counts(
    system_summaries: Sequence[tuple[str, str, Mapping[str, Any]]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for system_id, _, summary in system_summaries:
        raw_counts = summary.get("failure_type_counts")
        if not isinstance(raw_counts, Mapping):
            continue
        counts[system_id] = {
            str(failure_type): int(count)
            for failure_type, count in raw_counts.items()
            if isinstance(count, int) and count >= 0
        }
    return counts


def _question_ids(payload: Mapping[str, Any], report: Mapping[str, Any]) -> list[str]:
    raw_question_ids = payload.get("question_ids")
    if isinstance(raw_question_ids, Sequence) and not isinstance(raw_question_ids, str):
        return [str(item) for item in raw_question_ids if str(item)]
    question_ids: list[str] = []
    for result in _sequence(report.get("results")):
        if not isinstance(result, Mapping):
            continue
        question_id = result.get("question_id")
        if question_id is None:
            for system in ("naive", "agentic"):
                nested = result.get(system)
                if isinstance(nested, Mapping):
                    question_id = nested.get("question_id")
                    break
        if question_id is not None and str(question_id):
            question_ids.append(str(question_id))
    return question_ids


def _question_count(
    summary: Mapping[str, Any],
    report: Mapping[str, Any],
    question_ids: Sequence[str],
) -> int:
    total_questions = summary.get("total_questions")
    if isinstance(total_questions, int) and total_questions >= 0:
        return total_questions
    if question_ids:
        return len(question_ids)
    return len(_sequence(report.get("results")))


def _system_id(system: str) -> str:
    if system == "agentic_rag":
        return "agentic"
    if system == "naive_rag":
        return "naive"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", system).strip("_") or "unknown"


def _system_label(system_id: str) -> str:
    labels = {
        "agentic": "Agentic RAG",
        "naive": "Naive RAG",
        "agentic_reranker": "Agentic + Reranker",
    }
    return labels.get(system_id, system_id)


def _is_ablation_payload(payload: Mapping[str, Any]) -> bool:
    return payload.get("kind") == "ablation_result" and isinstance(payload.get("runs"), list)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return []


def _optional_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None
```

- [ ] **Step 4: Implement importer**

Add to `evaluation/history_store.py`:

```python
def import_history_artifact(
    path: str | Path,
    *,
    store: HistoryStore,
    source: str = "import",
) -> dict[str, str | None]:
    artifact_path = Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return {
            "status": "failed",
            "run_id": None,
            "error": "artifact payload must be a JSON object",
        }
    run_id = _stable_import_run_id(payload, artifact_path)
    record = extract_history_record(
        payload,
        run_id=run_id,
        created_at=_utc_now(),
        source=source,
        result_path=str(artifact_path),
    )
    store.save_record(record)
    return {"status": "stored", "run_id": run_id, "error": None}


def _stable_import_run_id(payload: Mapping[str, Any], path: Path) -> str:
    canonical = json.dumps(
        {"path": str(path), "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"hist_{digest}"


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
```

- [ ] **Step 5: Run extraction tests and verify pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_history_store.py -q
```

Expected after implementation:

- all selected tests pass

- [ ] **Step 6: Commit Task 3**

Run:

```bash
git add evaluation/history_store.py tests/test_evaluation_history_store.py
git commit -m "feat: normalize evaluation history payloads"
```

---

### Task 4: Add History Service And Wire CLI Artifact Writes

**Files:**
- Create: `evaluation/history_service.py`
- Modify: `evaluation/evaluate.py`
- Modify: `tests/test_evaluate.py`
- Modify: `tests/test_evaluation_storage.py`

- [ ] **Step 1: Write failing history service tests in `tests/test_evaluate.py`**

Add:

```python
def test_write_evaluation_artifacts_records_history_after_json_write(tmp_path, monkeypatch):
    calls = []

    class SpyHistoryService:
        def record_payload(self, payload, *, source, result_path, run_id=None):
            calls.append(
                {
                    "payload": payload,
                    "source": source,
                    "result_path": result_path,
                    "run_id": run_id,
                }
            )
            return {"status": "stored", "run_id": "eval_spy", "error": None}

    monkeypatch.setattr(
        evaluator,
        "EvaluationHistoryService",
        lambda: SpyHistoryService(),
    )
    report = {
        "runtime_config": evaluator.build_runtime_config_snapshot(),
        "summary": {"total_questions": 1, "correctness_score": 1.0},
        "results": [{"question_id": "q001"}],
    }

    evaluator.write_evaluation_artifacts(report, tmp_path)

    assert (tmp_path / "agentic_result.json").exists()
    assert calls == [
        {
            "payload": report,
            "source": "cli",
            "result_path": str(tmp_path / "agentic_result.json"),
            "run_id": None,
        }
    ]


def test_write_evaluation_artifacts_records_comparison_artifact_path(tmp_path, monkeypatch):
    calls = []

    class SpyHistoryService:
        def record_payload(self, payload, *, source, result_path, run_id=None):
            calls.append((source, result_path))
            return {"status": "stored", "run_id": "eval_spy", "error": None}

    monkeypatch.setattr(
        evaluator,
        "EvaluationHistoryService",
        lambda: SpyHistoryService(),
    )
    report = {
        "runtime_config": evaluator.build_runtime_config_snapshot(),
        "summary": {
            "mode": "comparison",
            "naive": {"total_questions": 1},
            "agentic": {"total_questions": 1},
        },
        "results": [{"naive": {"question_id": "q001"}, "agentic": {"question_id": "q001"}}],
    }

    evaluator.write_evaluation_artifacts(report, tmp_path)

    assert calls == [("cli", str(tmp_path / "comparison_result.json"))]
```

- [ ] **Step 2: Add direct service tests in `tests/test_evaluation_history_store.py`**

Append:

```python
from types import SimpleNamespace

from evaluation.history_service import EvaluationHistoryService


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
```

- [ ] **Step 3: Run focused tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluate.py tests/test_evaluation_history_store.py -q
```

Expected before implementation:

- imports fail for `evaluation.history_service`
- `evaluation.evaluate` has no `EvaluationHistoryService` attribute

- [ ] **Step 4: Implement `evaluation/history_service.py`**

Create:

```python
"""Settings-aware facade for evaluation history persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from config import Settings, get_settings
from evaluation.history_store import (
    HISTORY_METRIC_NAMES,
    HistoryStore,
    extract_history_record,
    import_history_artifact,
)


class EvaluationHistoryService:
    def __init__(
        self,
        settings: Settings | Any | None = None,
        store: HistoryStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._store = store or HistoryStore(self.settings.evaluation_history_db)

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.settings, "evaluation_history_enabled", True))

    def record_payload(
        self,
        payload: dict[str, Any],
        *,
        source: str,
        result_path: str | Path | None,
        run_id: str | None = None,
    ) -> dict[str, str | None]:
        resolved_run_id = run_id or _new_run_id()
        if not self.enabled:
            return {"status": "disabled", "run_id": None, "error": None}
        try:
            record = extract_history_record(
                payload,
                run_id=resolved_run_id,
                created_at=_utc_now(),
                source=source,
                result_path=str(result_path) if result_path is not None else None,
            )
            self._store.save_record(record)
        except Exception as exc:  # noqa: BLE001 - sidecar write must be isolated.
            return {
                "status": "failed",
                "run_id": resolved_run_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {"status": "stored", "run_id": resolved_run_id, "error": None}

    def import_artifact(
        self,
        path: str | Path,
        *,
        source: str = "import",
    ) -> dict[str, str | None]:
        if not self.enabled:
            return {"status": "disabled", "run_id": None, "error": None}
        try:
            return import_history_artifact(path, store=self._store, source=source)
        except Exception as exc:  # noqa: BLE001 - importer reports safe status.
            return {"status": "failed", "run_id": None, "error": f"{type(exc).__name__}: {exc}"}

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        return self._store.list_runs(limit=limit)

    def query_trends(
        self,
        metric: str = "correctness_score",
        system: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        return self._store.query_trends(metric=metric, system=system, limit=limit)

    def metric_names(self) -> tuple[str, ...]:
        return HISTORY_METRIC_NAMES


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _new_run_id() -> str:
    return f"eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:8]}"
```

- [ ] **Step 5: Wire `evaluation/evaluate.py`**

Add import:

```python
from evaluation.history_service import EvaluationHistoryService
```

Modify `write_evaluation_artifacts()`:

```python
def write_evaluation_artifacts(report: dict[str, Any], output_dir: str | Path) -> None:
    """Write structured JSON artifacts for downstream evaluation analysis."""

    output_path = Path(output_dir)
    runtime_config = report.get("runtime_config")
    if not isinstance(runtime_config, dict):
        runtime_config = build_runtime_config_snapshot()
    write_compatibility_artifacts(
        report,
        output_path,
        runtime_config=runtime_config,
    )
    result_path = (
        output_path / "comparison_result.json"
        if report.get("summary", {}).get("mode") == "comparison"
        else output_path / "agentic_result.json"
    )
    EvaluationHistoryService().record_payload(
        report,
        source="cli",
        result_path=str(result_path),
    )
```

- [ ] **Step 6: Run focused tests and verify pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluate.py tests/test_evaluation_history_store.py tests/test_evaluation_storage.py -q
```

Expected after implementation:

- selected tests pass
- JSON artifact tests remain compatible

- [ ] **Step 7: Commit Task 4**

Run:

```bash
git add evaluation/history_service.py evaluation/evaluate.py tests/test_evaluate.py tests/test_evaluation_history_store.py tests/test_evaluation_storage.py
git commit -m "feat: record evaluation artifacts in history"
```

---

### Task 5: Expose Evaluation History Through FastAPI

**Files:**
- Modify: `api/services/evaluation.py`
- Modify: `api/schemas.py`
- Modify: `api/routes/evaluation.py`
- Modify: `tests/test_fastapi_routes.py`

- [ ] **Step 1: Write failing FastAPI tests**

Update `FakeEvaluationService` in `tests/test_fastapi_routes.py`:

```python
    def list_history_runs(self, limit=20):
        return [
            {
                "run_id": "eval_1",
                "created_at": "2026-06-27T12:00:00.000000Z",
                "source": "api",
                "workspace_id": "workspace_1",
                "status": "completed",
                "mode": "comparison",
                "schema_version": 4,
                "evaluator_version": "p5b",
                "prompt_manifest_hash": "sha256:abc",
                "question_count": 1,
                "result_path": "data/evaluation_runs/eval_1.json",
            }
        ]

    def query_history_trends(self, metric="correctness_score", system=None, limit=20):
        return [
            {
                "created_at": "2026-06-27T12:00:00.000000Z",
                "run_id": "eval_1",
                "system_id": system or "agentic",
                "system_label": "Agentic RAG",
                "evaluator_version": "p5b",
                "prompt_manifest_hash": "sha256:abc",
                "metric_name": metric,
                "metric_value": 0.75,
            }
        ]
```

Add tests:

```python
def test_evaluation_history_routes_list_runs_and_trends_before_run_id_capture():
    client = create_test_client()

    history_response = client.get("/evaluation/history", params={"limit": 5})
    trends_response = client.get(
        "/evaluation/history/trends",
        params={"metric": "correctness_score", "system": "agentic", "limit": 5},
    )

    assert history_response.status_code == 200
    assert history_response.json()["runs"][0]["run_id"] == "eval_1"
    assert history_response.json()["runs"][0]["evaluator_version"] == "p5b"
    assert trends_response.status_code == 200
    assert trends_response.json() == {
        "metric": "correctness_score",
        "system": "agentic",
        "rows": [
            {
                "created_at": "2026-06-27T12:00:00.000000Z",
                "run_id": "eval_1",
                "system_id": "agentic",
                "system_label": "Agentic RAG",
                "evaluator_version": "p5b",
                "prompt_manifest_hash": "sha256:abc",
                "metric_name": "correctness_score",
                "metric_value": 0.75,
            }
        ],
    }
```

- [ ] **Step 2: Run FastAPI tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_fastapi_routes.py -q
```

Expected before implementation:

- `/evaluation/history` is captured by `/{run_id}` or returns 404/validation error
- `/evaluation/history/trends` does not exist

- [ ] **Step 3: Add API schemas**

In `api/schemas.py`, after `EvaluationRunResponse`, add:

```python
class EvaluationHistoryRun(BaseModel):
    """Historical evaluation run row."""

    run_id: str
    created_at: str
    source: str
    workspace_id: str | None = None
    status: str
    mode: str
    schema_version: int | None = None
    evaluator_version: str | None = None
    prompt_manifest_hash: str = ""
    question_count: int = 0
    result_path: str | None = None


class EvaluationHistoryListResponse(BaseModel):
    """Historical evaluation run listing."""

    runs: list[EvaluationHistoryRun]


class EvaluationTrendRow(BaseModel):
    """One historical metric point."""

    created_at: str
    run_id: str
    system_id: str
    system_label: str
    evaluator_version: str
    prompt_manifest_hash: str
    metric_name: str
    metric_value: float | None = None


class EvaluationTrendResponse(BaseModel):
    """Historical trend response."""

    metric: str
    system: str | None = None
    rows: list[EvaluationTrendRow]
```

- [ ] **Step 4: Add service methods**

In `api/services/evaluation.py`, import:

```python
from evaluation.history_service import EvaluationHistoryService
```

Modify `EvaluationService.__init__()`:

```python
    def __init__(
        self,
        settings: Settings | None = None,
        history_service: EvaluationHistoryService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._history = history_service or EvaluationHistoryService(settings=self.settings)
```

After writing the FastAPI payload in `run_evaluation()`, add:

```python
        history_status = self._history.record_payload(
            payload,
            source="api",
            result_path=str(result_path),
            run_id=run_id,
        )
        payload["history"] = history_status
        result_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
```

Keep `_public_run()` unchanged so public response fields remain stable.

Add methods:

```python
    def list_history_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent SQLite-backed evaluation run rows."""

        return self._history.list_runs(limit=limit)

    def query_history_trends(
        self,
        metric: str = "correctness_score",
        system: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return SQLite-backed evaluation trend rows."""

        return self._history.query_trends(metric=metric, system=system, limit=limit)
```

- [ ] **Step 5: Add routes before `/{run_id}`**

In `api/routes/evaluation.py`, update imports:

```python
from api.schemas import (
    EvaluationHistoryListResponse,
    EvaluationRunRequest,
    EvaluationRunResponse,
    EvaluationTrendResponse,
)
```

Add before `@router.get("/{run_id}"...)`:

```python
@router.get("/history", response_model=EvaluationHistoryListResponse)
def list_evaluation_history(
    limit: int = 20,
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
) -> EvaluationHistoryListResponse:
    """Return recent historical evaluation runs."""

    return EvaluationHistoryListResponse(
        runs=evaluation_service.list_history_runs(limit=limit)
    )


@router.get("/history/trends", response_model=EvaluationTrendResponse)
def get_evaluation_history_trends(
    metric: str = "correctness_score",
    system: str | None = None,
    limit: int = 20,
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
) -> EvaluationTrendResponse:
    """Return historical trend rows for one metric."""

    return EvaluationTrendResponse(
        metric=metric,
        system=system,
        rows=evaluation_service.query_history_trends(
            metric=metric,
            system=system,
            limit=limit,
        ),
    )
```

- [ ] **Step 6: Run FastAPI tests and verify pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_fastapi_routes.py -q
```

Expected after implementation:

- history routes pass
- existing evaluation run/get tests still pass

- [ ] **Step 7: Commit Task 5**

Run:

```bash
git add api/services/evaluation.py api/schemas.py api/routes/evaluation.py tests/test_fastapi_routes.py
git commit -m "feat: expose evaluation history api"
```

---

### Task 6: Add Dashboard History Models, Formatters, And Service Methods

**Files:**
- Modify: `evaluation/dashboard_models.py`
- Modify: `evaluation/dashboard_formatters.py`
- Modify: `evaluation/dashboard_service.py`
- Modify: `tests/test_dashboard_service.py`

- [ ] **Step 1: Write failing Dashboard history tests**

Append to `tests/test_dashboard_service.py`:

```python
from evaluation.dashboard_formatters import (
    history_runs_to_table,
    history_trends_to_table,
)
from evaluation.dashboard_models import (
    HISTORY_RUN_COLUMNS,
    HISTORY_TREND_COLUMNS,
)


class FakeHistoryService:
    def __init__(self, fail=False, runs=None, trends=None):
        self.fail = fail
        self.runs = runs if runs is not None else [
            {
                "run_id": "eval_1",
                "created_at": "2026-06-27T12:00:00.000000Z",
                "source": "api",
                "workspace_id": "workspace_1",
                "status": "completed",
                "mode": "comparison",
                "schema_version": 4,
                "evaluator_version": "p5b",
                "prompt_manifest_hash": "sha256:abcdef",
                "question_count": 5,
                "result_path": "data/evaluation_runs/eval_1.json",
            }
        ]
        self.trends = trends if trends is not None else [
            {
                "created_at": "2026-06-27T12:00:00.000000Z",
                "run_id": "eval_1",
                "system_id": "agentic",
                "system_label": "Agentic RAG",
                "evaluator_version": "p5b",
                "prompt_manifest_hash": "sha256:abcdef",
                "metric_name": "correctness_score",
                "metric_value": 0.75,
            }
        ]

    def list_runs(self, limit=20):
        if self.fail:
            raise RuntimeError("database unavailable")
        return self.runs[:limit]

    def query_trends(self, metric="correctness_score", system=None, limit=20):
        if self.fail:
            raise RuntimeError("database unavailable")
        return self.trends[:limit]

    def metric_names(self):
        return ("correctness_score", "average_latency")


def test_history_table_contracts_are_stable():
    assert HISTORY_RUN_COLUMNS == [
        "Run ID",
        "Created At",
        "Source",
        "Workspace",
        "Status",
        "Mode",
        "Evaluator",
        "Schema",
        "Prompt Hash",
        "Questions",
        "Result Path",
    ]
    assert HISTORY_TREND_COLUMNS == [
        "Created At",
        "Run ID",
        "System",
        "Evaluator",
        "Prompt Hash",
        "Metric",
        "Value",
    ]


def test_history_formatters_convert_rows_for_tables():
    run_rows = history_runs_to_table(FakeHistoryService().runs)
    trend_rows = history_trends_to_table(FakeHistoryService().trends)

    assert run_rows[0] == [
        "eval_1",
        "2026-06-27T12:00:00.000000Z",
        "api",
        "workspace_1",
        "completed",
        "comparison",
        "p5b",
        4,
        "sha256:abcde",
        5,
        "data/evaluation_runs/eval_1.json",
    ]
    assert trend_rows[0] == [
        "2026-06-27T12:00:00.000000Z",
        "eval_1",
        "Agentic RAG",
        "p5b",
        "sha256:abcde",
        "correctness_score",
        0.75,
    ]


def test_dashboard_service_loads_history_snapshot_and_trends():
    service = EvaluationDashboardService(history_service=FakeHistoryService())

    snapshot = service.load_history_snapshot(limit=10)
    trends = service.load_history_trends(metric="correctness_score", limit=10)

    assert snapshot["status"] == "completed"
    assert snapshot["run_rows"][0][0] == "eval_1"
    assert snapshot["trend_rows"] == []
    assert "Loaded 1 historical evaluation run(s)." in snapshot["message"]
    assert trends["status"] == "completed"
    assert trends["trend_rows"][0][2] == "Agentic RAG"


def test_dashboard_service_returns_unavailable_history_view_on_failure():
    service = EvaluationDashboardService(history_service=FakeHistoryService(fail=True))

    view = service.load_history_snapshot(limit=10)

    assert view["status"] == "unavailable"
    assert view["run_rows"] == []
    assert view["trend_rows"] == []
    assert "History unavailable: RuntimeError: database unavailable" == view["message"]
```

- [ ] **Step 2: Run Dashboard tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_dashboard_service.py -q
```

Expected before implementation:

- missing imports or constructor parameter failures

- [ ] **Step 3: Add Dashboard model contracts**

In `evaluation/dashboard_models.py`, add:

```python
HISTORY_RUN_COLUMNS = [
    "Run ID",
    "Created At",
    "Source",
    "Workspace",
    "Status",
    "Mode",
    "Evaluator",
    "Schema",
    "Prompt Hash",
    "Questions",
    "Result Path",
]
HISTORY_TREND_COLUMNS = [
    "Created At",
    "Run ID",
    "System",
    "Evaluator",
    "Prompt Hash",
    "Metric",
    "Value",
]


class HistoryDashboardView(TypedDict):
    status: DashboardStatus
    run_rows: list[list[Any]]
    trend_rows: list[list[Any]]
    metric_choices: list[str]
    message: str
```

- [ ] **Step 4: Add pure formatters**

In `evaluation/dashboard_formatters.py`, add:

```python
def history_runs_to_table(rows: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    """Convert historical run records to table rows."""

    return [
        [
            str(row.get("run_id") or ""),
            str(row.get("created_at") or ""),
            str(row.get("source") or ""),
            str(row.get("workspace_id") or ""),
            str(row.get("status") or ""),
            str(row.get("mode") or ""),
            str(row.get("evaluator_version") or "legacy"),
            row.get("schema_version") if row.get("schema_version") is not None else "legacy",
            _hash_prefix(row.get("prompt_manifest_hash")),
            row.get("question_count") or 0,
            str(row.get("result_path") or ""),
        ]
        for row in rows
    ]


def history_trends_to_table(rows: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    """Convert historical trend records to table rows."""

    return [
        [
            str(row.get("created_at") or ""),
            str(row.get("run_id") or ""),
            str(row.get("system_label") or row.get("system_id") or ""),
            str(row.get("evaluator_version") or "legacy"),
            _hash_prefix(row.get("prompt_manifest_hash")),
            str(row.get("metric_name") or ""),
            _metric_value(row.get("metric_value")),
        ]
        for row in rows
    ]


def _hash_prefix(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    if len(value) <= 12:
        return value
    return value[:12]
```

- [ ] **Step 5: Wire Dashboard service**

In `evaluation/dashboard_service.py`, import:

```python
from evaluation.history_service import EvaluationHistoryService
```

Update formatter imports:

```python
    history_runs_to_table,
    history_trends_to_table,
```

Update model imports:

```python
    HistoryDashboardView,
```

Modify `EvaluationDashboardService.__init__()` signature:

```python
        history_service: EvaluationHistoryService | None = None,
```

Inside `__init__`, add:

```python
        self._history_service = history_service or EvaluationHistoryService()
```

Add methods:

```python
    def load_history_snapshot(self, limit: int = 20) -> HistoryDashboardView:
        """Load recent SQLite-backed historical run rows."""

        try:
            runs = self._history_service.list_runs(limit=limit)
            message = (
                "No historical evaluation runs found. Run an evaluation or import "
                "an artifact to populate SQLite history."
                if not runs
                else f"Loaded {len(runs)} historical evaluation run(s)."
            )
            return {
                "status": "completed",
                "run_rows": history_runs_to_table(runs),
                "trend_rows": [],
                "metric_choices": list(self._history_service.metric_names()),
                "message": message,
            }
        except Exception as exc:  # noqa: BLE001 - UI-safe history failure.
            return {
                "status": "unavailable",
                "run_rows": [],
                "trend_rows": [],
                "metric_choices": [],
                "message": f"History unavailable: {type(exc).__name__}: {exc}",
            }

    def load_history_trends(
        self,
        metric: str = "correctness_score",
        system: str | None = None,
        limit: int = 20,
    ) -> HistoryDashboardView:
        """Load SQLite-backed trend rows for one metric."""

        try:
            trends = self._history_service.query_trends(
                metric=metric,
                system=None if system in {None, "", "all"} else system,
                limit=limit,
            )
            message = (
                f"Loaded {len(trends)} trend row(s) for {metric}."
                if trends
                else f"No trend rows found for {metric}."
            )
            return {
                "status": "completed",
                "run_rows": [],
                "trend_rows": history_trends_to_table(trends),
                "metric_choices": list(self._history_service.metric_names()),
                "message": message,
            }
        except Exception as exc:  # noqa: BLE001 - UI-safe history failure.
            return {
                "status": "unavailable",
                "run_rows": [],
                "trend_rows": [],
                "metric_choices": [],
                "message": f"History unavailable: {type(exc).__name__}: {exc}",
            }
```

- [ ] **Step 6: Run Dashboard tests and verify pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_dashboard_service.py -q
```

Expected after implementation:

- all selected Dashboard tests pass

- [ ] **Step 7: Commit Task 6**

Run:

```bash
git add evaluation/dashboard_models.py evaluation/dashboard_formatters.py evaluation/dashboard_service.py tests/test_dashboard_service.py
git commit -m "feat: add evaluation history dashboard service"
```

---

### Task 7: Add Minimal Gradio History Trends Tab

**Files:**
- Modify: `ui/gradio_app.py`
- Modify: `tests/test_gradio_app.py`

- [ ] **Step 1: Write failing Gradio helper tests**

Add to `tests/test_gradio_app.py`:

```python
def test_load_history_dashboard_returns_runs_and_metric_choices():
    class FakeService:
        def load_history_snapshot(self, limit=20):
            return {
                "status": "completed",
                "run_rows": [["eval_1"]],
                "trend_rows": [],
                "metric_choices": ["correctness_score", "average_latency"],
                "message": "Loaded 1 historical evaluation run(s).",
            }

    status, runs, metric_update, trends = gradio_app.load_history_dashboard(
        service=FakeService()
    )

    assert status == "Loaded 1 historical evaluation run(s)."
    assert runs == [["eval_1"]]
    assert metric_update["choices"] == ["correctness_score", "average_latency"]
    assert metric_update["value"] == "correctness_score"
    assert trends == []


def test_load_history_trends_returns_trend_rows():
    class FakeService:
        def load_history_trends(self, metric="correctness_score", system=None, limit=20):
            return {
                "status": "completed",
                "run_rows": [],
                "trend_rows": [["2026-06-27T12:00:00.000000Z"]],
                "metric_choices": ["correctness_score"],
                "message": "Loaded 1 trend row(s) for correctness_score.",
            }

    status, rows = gradio_app.load_history_trends(
        metric="correctness_score",
        service=FakeService(),
    )

    assert status == "Loaded 1 trend row(s) for correctness_score."
    assert rows == [["2026-06-27T12:00:00.000000Z"]]
```

Ensure the file imports the module as:

```python
import ui.gradio_app as gradio_app
```

- [ ] **Step 2: Run Gradio tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_gradio_app.py -q
```

Expected before implementation:

- `load_history_dashboard` and `load_history_trends` are missing

- [ ] **Step 3: Import history table constants**

In `ui/gradio_app.py`, extend imports from `evaluation.dashboard_models`:

```python
    HISTORY_RUN_COLUMNS,
    HISTORY_TREND_COLUMNS,
```

- [ ] **Step 4: Add Gradio helper functions**

Add near other dashboard helper functions:

```python
def load_history_dashboard(
    service: EvaluationDashboardService | None = None,
) -> tuple[str, list[list[Any]], dict[str, Any], list[list[Any]]]:
    """Load recent historical runs for the History Trends tab."""

    resolved_service = service or EvaluationDashboardService()
    view = resolved_service.load_history_snapshot()
    metric_choices = list(view.get("metric_choices", []))
    metric_value = metric_choices[0] if metric_choices else None
    return (
        str(view["message"]),
        list(view.get("run_rows", [])),
        gr.update(choices=metric_choices, value=metric_value),
        list(view.get("trend_rows", [])),
    )


def load_history_trends(
    metric: str | None,
    service: EvaluationDashboardService | None = None,
) -> tuple[str, list[list[Any]]]:
    """Load trend rows for the selected metric."""

    resolved_service = service or EvaluationDashboardService()
    selected_metric = metric or "correctness_score"
    view = resolved_service.load_history_trends(metric=selected_metric)
    return str(view["message"]), list(view.get("trend_rows", []))
```

- [ ] **Step 5: Add the `History Trends` sub-tab**

Inside `_build_evaluation_tab()`, after the `Ablation Snapshot` tab block, add:

```python
        with gr.Tab("History Trends"):
            gr.Markdown(
                "This view reads SQLite sidecar history and does not rerun models."
            )
            refresh_history = gr.Button("Refresh History", variant="primary")
            history_status = gr.Markdown(
                "Load SQLite-backed historical evaluation runs.",
                elem_classes="dashboard-status",
            )
            history_runs = gr.Dataframe(
                headers=HISTORY_RUN_COLUMNS,
                value=[],
                label="Recent evaluation runs",
                interactive=False,
                wrap=True,
            )
            history_metric = gr.Dropdown(
                choices=["correctness_score"],
                value="correctness_score",
                label="Trend metric",
            )
            history_trends = gr.Dataframe(
                headers=HISTORY_TREND_COLUMNS,
                value=[],
                label="Metric trend rows",
                interactive=False,
                wrap=True,
            )

            refresh_history.click(
                fn=lambda: load_history_dashboard(service=service),
                outputs=[
                    history_status,
                    history_runs,
                    history_metric,
                    history_trends,
                ],
            )
            history_metric.change(
                fn=lambda metric: load_history_trends(metric, service=service),
                inputs=history_metric,
                outputs=[history_status, history_trends],
                queue=False,
            )
```

- [ ] **Step 6: Run Gradio and Dashboard tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_gradio_app.py tests/test_dashboard_service.py -q
```

Expected after implementation:

- selected UI and service tests pass

- [ ] **Step 7: Commit Task 7**

Run:

```bash
git add ui/gradio_app.py tests/test_gradio_app.py
git commit -m "feat: add evaluation history trend tab"
```

---

### Task 8: Documentation, Compatibility Sweep, Review, And Version Marking

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/github_release_checklist.md`
- Modify: `docs/superpowers/plans/2026-06-27-p5b-sqlite-evaluation-history.md`

- [ ] **Step 1: Update documentation**

Update `README.md`:

- move P5b into Completed Work
- describe SQLite sidecar history under the evaluation section
- document `EVALUATION_HISTORY_ENABLED` and `EVALUATION_HISTORY_DB`
- state that JSON artifacts remain the full compatibility payload
- state that Dashboard trends are read-only summaries
- keep Background Evaluation and Trace Drill-down in Next Milestones

Add `CHANGELOG.md` entry:

```markdown
## v0.5.1-p5b - SQLite Historical Evaluation + Trend Dashboard

Date: 2026-06-27

### Added

- Added SQLite sidecar storage for historical evaluation runs.
- Added prompt-aware historical metric trends for CLI/API evaluation artifacts.
- Added FastAPI and Gradio read-only history views for recent runs and metric trends.

### Changed

- Advanced evaluation runtime metadata to schema version `4` and evaluator version `p5b`.
- Preserved JSON artifacts as the complete compatibility payload while indexing summaries into SQLite.

### Notes

- SQLite history is local runtime data and is disabled safely with `EVALUATION_HISTORY_ENABLED=false`.
- Legacy artifacts without schema or evaluator metadata import as `legacy`.
- Background Evaluation and Trace Drill-down remain future work.

### Verification

- Full test suite: update this bullet with the exact terminal output after the
  command runs.
- Focused history tests: update this bullet with the exact terminal output
  after the command runs.
- API/Dashboard compatibility tests: update this bullet with the exact
  terminal output after the command runs.
```

Update `docs/github_release_checklist.md` with a P5b section containing:

- branch name
- tag target `v0.5.1-p5b`
- exact verification commands
- note that `data/evaluation_history.sqlite3` is ignored runtime data
- note that no secrets, prompt templates, or rendered prompts are stored

- [ ] **Step 2: Run focused compatibility suites**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_evaluation_history_store.py \
  tests/test_evaluation_storage.py \
  tests/test_evaluate.py \
  tests/test_fastapi_routes.py \
  tests/test_dashboard_service.py \
  tests/test_gradio_app.py \
  tests/test_ablation.py \
  tests/test_evaluation_matrix.py \
  -q
```

Expected:

- all selected tests pass
- record exact count in `CHANGELOG.md` and this plan

- [ ] **Step 3: Run full verification**

Run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m compileall prompting agent rag api evaluation experiments baseline tools observability
git diff --check
```

Expected:

- full test suite passes
- ruff passes
- compileall succeeds
- whitespace check is clean

- [ ] **Step 4: Scan for forbidden persistence content**

Run:

```bash
rg -n 'OPENAI_API_KEY|EVALUATION_JUDGE_API_KEY|Bearer |rendered prompt|full prompt template' \
  evaluation api ui README.md CHANGELOG.md docs/github_release_checklist.md \
  .env.example
```

Expected:

- `.env.example` references environment variable names only
- documentation mentions that full prompt templates and rendered prompts are not stored
- no literal secrets are present

- [ ] **Step 5: Record observed verification**

In this plan, add an `## Observed Verification` section after the verification
commands run. Each bullet must contain concrete command output copied from the
terminal, such as the exact pytest pass count and elapsed time. Do not commit
projected counts or estimates.

- [ ] **Step 6: Commit docs and final plan status**

Run:

```bash
git add .env.example README.md CHANGELOG.md docs/github_release_checklist.md
git add -f docs/superpowers/plans/2026-06-27-p5b-sqlite-evaluation-history.md
git commit -m "docs: publish p5b sqlite evaluation history"
```

- [ ] **Step 7: Request independent code review**

Use `superpowers:requesting-code-review` against the full branch diff from
`82b3e4e` to `HEAD`.

Review focus:

- SQLite schema and idempotent initialization
- JSON compatibility preservation
- sidecar write failure isolation
- route ordering for `/evaluation/history`
- prompt manifest hashing without prompt text storage
- legacy artifact metadata behavior
- tests covering disabled and failed history writes

Address confirmed findings with focused commits, rerun affected tests, then run
the full verification suite again.

- [ ] **Step 8: Finish the branch**

After fresh verification passes, use `superpowers:finishing-a-development-branch`.

Offer the user:

1. Merge back to `main` locally
2. Push and create a Pull Request
3. Keep the branch as-is
4. Discard this work

Create tag `v0.5.1-p5b` only after:

- the user explicitly chooses integration
- integration into updated `main` succeeds
- merged `main` passes the full suite
- final integration record is committed

## Observed Verification

- Focused compatibility suite:
  `.venv/bin/python -m pytest tests/test_evaluation_history_store.py tests/test_evaluation_storage.py tests/test_evaluate.py tests/test_fastapi_routes.py tests/test_dashboard_service.py tests/test_gradio_app.py tests/test_ablation.py tests/test_evaluation_matrix.py -q`
  → `193 passed in 4.52s`.
- Full test suite:
  `.venv/bin/python -m pytest -q`
  → `671 passed in 4.34s`.
- Focused history tests:
  `.venv/bin/python -m pytest tests/test_evaluation_history_store.py tests/test_evaluation_storage.py tests/test_evaluate.py -q`
  → `73 passed in 2.00s`.
- API/Dashboard compatibility tests:
  `.venv/bin/python -m pytest tests/test_fastapi_routes.py tests/test_dashboard_service.py tests/test_gradio_app.py -q`
  → `83 passed in 4.50s`.
- Ruff:
  `.venv/bin/python -m ruff check .`
  → `All checks passed!`.
- Python compileall:
  `.venv/bin/python -m compileall prompting agent rag api evaluation experiments baseline tools observability`
  → `Listing 'prompting'...` through `Listing 'observability'...`, exit code `0`.
- Whitespace check:
  `git diff --check`
  → no output, exit code `0`.
- Forbidden persistence scan:
  `rg -n 'OPENAI_API_KEY|EVALUATION_JUDGE_API_KEY|Bearer |rendered prompt|full prompt template' evaluation api ui README.md CHANGELOG.md docs/github_release_checklist.md .env.example`
  → matches were limited to environment variable names/placeholders,
  documentation safety statements that prompt templates/rendered prompts are
  not stored, and sanitizer code references; no literal secrets were found.

## Coverage Checklist

- [x] P5b config defaults and overrides covered
- [x] runtime metadata reports schema `4` / evaluator `p5b`
- [x] SQLite schema initializes idempotently
- [x] single, comparison, matrix, ablation, and API wrapper payloads extract correctly
- [x] legacy artifacts display as legacy
- [x] prompt manifest hash is canonical and template-free
- [x] history disabled path avoids DB writes
- [x] history write failure does not remove JSON artifacts
- [x] CLI artifact writer records history after JSON writes
- [x] FastAPI exposes history runs and trends without route capture
- [x] Dashboard service exposes UI-safe run and trend views
- [x] Gradio adds a read-only History Trends tab
- [x] docs record Background Evaluation and Trace Drill-down as deferred
