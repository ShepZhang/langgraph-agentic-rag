"""SQLite sidecar history store for evaluation runs."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EVALUATION_HISTORY_DB_SCHEMA_VERSION = 1
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

_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")


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


def _safe_limit(limit: int) -> int:
    return max(1, min(int(limit), 200))


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
