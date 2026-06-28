"""SQLite sidecar history store for evaluation runs."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
_COMPLETED_ABLATION_STATUSES = {"completed", "completed_with_errors"}
_SYSTEM_ID_ALIASES = {
    "agentic_rag": "agentic",
    "naive_rag": "naive",
}
_SYSTEM_LABELS = {
    "agentic": "Agentic RAG",
    "naive": "Naive RAG",
    "agentic_reranker": "Agentic + Reranker",
}
_SAFE_RUNTIME_SECTION_KEYS = {
    "agent_features": (
        "query_transformation_enabled",
        "retrieval_grading_enabled",
        "conditional_retry_enabled",
        "citation_verification_enabled",
    ),
    "judge": ("enabled", "provider", "model", "temperature"),
    "llm": ("provider", "model", "temperature"),
    "retriever": (
        "top_k",
        "hybrid_retrieval_enabled",
        "dense_top_k",
        "bm25_top_k",
        "fusion_top_k",
    ),
    "reranker": ("enabled", "model", "top_n", "candidate_top_k"),
    "vectorstore": ("collection_name",),
}
_PROMPT_MANIFEST_KEYS = ("version", "fingerprint")
_PROMPT_TEXT_KEYS = {
    "prompt",
    "prompt_text",
    "system_prompt",
    "user_prompt",
    "rendered_prompt",
    "rendered_prompt_payload",
    "full_prompt_template",
    "prompt_template",
    "template",
}
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "authorization",
    "bearer",
)
_MISSING = object()


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


def extract_history_record(
    payload: Mapping[str, Any],
    *,
    run_id: str,
    created_at: str,
    source: str,
    result_path: str | None = None,
) -> HistoryRecord:
    """Normalize supported evaluation artifact shapes into a history record."""

    if not isinstance(payload, Mapping):
        raise ValueError("history payload must be a JSON object")

    report = _nested_report(payload)
    mode = _history_mode(payload, report)
    runtime_config = _sanitize_runtime_config(
        _runtime_config_for_history(payload, report, mode)
    )
    prompt_manifest = _prompt_manifest(runtime_config)
    system_summaries = _system_summaries(payload, report, mode)
    question_ids = _question_ids(payload, report, mode)
    summary = _sanitize_json_object(_mapping(report.get("summary")))

    return HistoryRecord(
        run_id=run_id,
        created_at=created_at,
        source=source,
        workspace_id=_optional_str(payload.get("workspace_id"))
        or _optional_str(report.get("workspace_id")),
        status=_optional_str(payload.get("status"))
        or _optional_str(report.get("status"))
        or "completed",
        mode=mode,
        schema_version=_schema_version(runtime_config),
        evaluator_version=_evaluator_version(runtime_config),
        result_path=result_path
        or _optional_str(payload.get("result_path"))
        or _optional_str(report.get("result_path")),
        question_count=_question_count(payload, report, mode, question_ids),
        question_ids=question_ids,
        runtime_config=runtime_config,
        prompt_manifest=prompt_manifest,
        prompt_manifest_hash=compute_prompt_manifest_hash(prompt_manifest),
        summary=summary,
        metrics=_metric_records(system_summaries),
        failure_counts=_failure_counts(system_summaries),
    )


def import_history_artifact(
    path: str | Path,
    *,
    store: HistoryStore,
    source: str = "import",
) -> dict[str, str | None]:
    """Import a JSON evaluation artifact into a history store."""

    artifact_path = Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return {
            "status": "failed",
            "run_id": None,
            "error": "JSON artifact must contain an object",
        }

    run_id = _stable_import_run_id(artifact_path, payload)
    record = extract_history_record(
        payload,
        run_id=run_id,
        created_at=_utc_now(),
        source=source,
        result_path=str(artifact_path),
    )
    stored_run_id = store.save_record(record)
    return {"status": "stored", "run_id": stored_run_id, "error": None}


def _nested_report(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    report = payload.get("report")
    if isinstance(report, Mapping):
        return report
    return payload


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _history_mode(payload: Mapping[str, Any], report: Mapping[str, Any]) -> str:
    if payload.get("kind") == "ablation_result" or report.get("kind") == "ablation_result":
        return "ablation"
    if isinstance(payload.get("runs"), list) or isinstance(report.get("runs"), list):
        return "ablation"

    summary = _mapping(report.get("summary"))
    mode = _optional_str(summary.get("mode"))
    if mode:
        return mode
    if isinstance(summary.get("variants"), Mapping):
        return "matrix"
    if isinstance(summary.get("naive"), Mapping) and isinstance(
        summary.get("agentic"), Mapping
    ):
        return "comparison"
    return "single"


def _runtime_config_for_history(
    payload: Mapping[str, Any],
    report: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    runtime_config = report.get("runtime_config")
    if isinstance(runtime_config, Mapping):
        return dict(runtime_config)

    runtime_config = payload.get("runtime_config")
    if isinstance(runtime_config, Mapping):
        return dict(runtime_config)

    if mode == "ablation":
        for run in _ablation_runs(payload, report):
            if run.get("status") not in _COMPLETED_ABLATION_STATUSES:
                continue
            runtime_config = run.get("runtime_config")
            if isinstance(runtime_config, Mapping):
                return dict(runtime_config)
    return {}


def _sanitize_runtime_config(runtime_config: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _metadata_runtime_config(runtime_config)
    sanitized: dict[str, Any] = {}

    schema_version = _schema_version(metadata)
    if schema_version is not None:
        sanitized["schema_version"] = schema_version

    evaluator_version = _evaluator_version(metadata)
    if evaluator_version is not None:
        sanitized["evaluator_version"] = evaluator_version

    for section, allowed_keys in _SAFE_RUNTIME_SECTION_KEYS.items():
        section_value = metadata.get(section)
        if not isinstance(section_value, Mapping):
            continue
        sanitized_section = _sanitize_allowed_mapping(section_value, allowed_keys)
        if sanitized_section:
            sanitized[section] = sanitized_section

    prompt_manifest = _sanitize_prompt_manifest(metadata.get("prompts"))
    if prompt_manifest:
        sanitized["prompts"] = prompt_manifest

    return sanitized


def _sanitize_allowed_mapping(
    value: Mapping[str, Any],
    allowed_keys: tuple[str, ...],
) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key in allowed_keys:
        if key not in value:
            continue
        safe_value = _safe_json_scalar(value[key])
        if safe_value is not _MISSING:
            sanitized[key] = safe_value
    return sanitized


def _sanitize_prompt_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}

    sanitized: dict[str, Any] = {}
    for prompt_id, metadata in value.items():
        if not isinstance(prompt_id, str) or not isinstance(metadata, Mapping):
            continue
        prompt_metadata = _sanitize_allowed_mapping(metadata, _PROMPT_MANIFEST_KEYS)
        if prompt_metadata:
            sanitized[prompt_id] = prompt_metadata
    return sanitized


def _sanitize_json_object(value: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = _sanitize_json_value(value)
    return sanitized if isinstance(sanitized, dict) else {}


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): sanitized_value
            for key, item in value.items()
            if not _is_unsafe_json_key(key)
            for sanitized_value in [_sanitize_json_value(item)]
            if sanitized_value is not _MISSING
        }
    if isinstance(value, list):
        return [
            sanitized_item
            for item in value
            for sanitized_item in [_sanitize_json_value(item)]
            if sanitized_item is not _MISSING
        ]
    return _safe_json_scalar(value)


def _safe_json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return _MISSING


def _is_unsafe_json_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    normalized = key.strip().lower().replace("-", "_")
    if normalized in _PROMPT_TEXT_KEYS:
        return True
    if "rendered" in normalized and "prompt" in normalized:
        return True
    if "template" in normalized:
        return True
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _metadata_runtime_config(runtime_config: Mapping[str, Any]) -> Mapping[str, Any]:
    if any(
        key in runtime_config
        for key in ("schema_version", "evaluator_version", "prompts")
    ):
        return runtime_config

    for system_id in ("agentic", "naive"):
        system_config = runtime_config.get(system_id)
        if isinstance(system_config, Mapping):
            return system_config

    for system_config in runtime_config.values():
        if isinstance(system_config, Mapping):
            return system_config
    return {}


def _schema_version(runtime_config: Mapping[str, Any]) -> int | None:
    value = _metadata_runtime_config(runtime_config).get("schema_version")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _evaluator_version(runtime_config: Mapping[str, Any]) -> str | None:
    return _optional_str(_metadata_runtime_config(runtime_config).get("evaluator_version"))


def _prompt_manifest(runtime_config: Mapping[str, Any]) -> dict[str, Any]:
    prompts = _metadata_runtime_config(runtime_config).get("prompts")
    if isinstance(prompts, Mapping):
        return dict(prompts)
    return {}


def _system_summaries(
    payload: Mapping[str, Any],
    report: Mapping[str, Any],
    mode: str,
) -> list[tuple[str, str, Mapping[str, Any]]]:
    summary = _mapping(report.get("summary"))
    if mode == "ablation":
        return _ablation_system_summaries(payload, report)
    if mode == "comparison":
        return [
            (system_id, _system_label(system_id), _mapping(summary.get(system_id)))
            for system_id in ("naive", "agentic")
            if isinstance(summary.get(system_id), Mapping)
        ]
    if mode == "matrix":
        variants = _mapping(summary.get("variants"))
        return [
            (str(system_id), _system_label(str(system_id)), _mapping(system_summary))
            for system_id, system_summary in variants.items()
            if isinstance(system_summary, Mapping)
        ]

    system_id = _system_id(report.get("system") or payload.get("system"))
    return [(system_id, _system_label(system_id), summary)]


def _ablation_system_summaries(
    payload: Mapping[str, Any],
    report: Mapping[str, Any],
) -> list[tuple[str, str, Mapping[str, Any]]]:
    summaries: list[tuple[str, str, Mapping[str, Any]]] = []
    for run in _ablation_runs(payload, report):
        if run.get("status") not in _COMPLETED_ABLATION_STATUSES:
            continue
        system_id = _optional_str(run.get("id"))
        if not system_id:
            continue
        method = _optional_str(run.get("method"))
        label = f"{system_id} {method}" if method else _system_label(system_id)
        summaries.append((system_id, label, _mapping(run.get("summary"))))
    return summaries


def _ablation_runs(
    payload: Mapping[str, Any],
    report: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    runs = report.get("runs")
    if not isinstance(runs, list):
        runs = payload.get("runs")
    if not isinstance(runs, list):
        return []
    return [run for run in runs if isinstance(run, Mapping)]


def _metric_records(
    system_summaries: list[tuple[str, str, Mapping[str, Any]]],
) -> list[MetricRecord]:
    records: list[MetricRecord] = []
    for system_id, system_label, summary in system_summaries:
        for metric_name in HISTORY_METRIC_NAMES:
            if metric_name not in summary:
                continue
            metric_value, metric_text = _metric_value(summary[metric_name])
            records.append(
                MetricRecord(
                    system_id=system_id,
                    system_label=system_label,
                    metric_name=metric_name,
                    metric_value=metric_value,
                    metric_text=metric_text,
                )
            )
    return records


def _metric_value(value: Any) -> tuple[float | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value), None
    return None, str(value)


def _failure_counts(
    system_summaries: list[tuple[str, str, Mapping[str, Any]]],
) -> dict[str, dict[str, int]]:
    failures: dict[str, dict[str, int]] = {}
    for system_id, _system_label, summary in system_summaries:
        raw_counts = _mapping(summary.get("failure_type_counts"))
        counts = {
            str(failure_type): int(count)
            for failure_type, count in raw_counts.items()
            if _is_int_like(count)
        }
        if counts:
            failures[system_id] = counts
    return failures


def _question_ids(
    payload: Mapping[str, Any],
    report: Mapping[str, Any],
    mode: str,
) -> list[str]:
    direct_ids = _direct_question_ids(report) or _direct_question_ids(payload)
    if direct_ids:
        return direct_ids

    if mode == "ablation":
        for run in _ablation_runs(payload, report):
            if run.get("status") not in _COMPLETED_ABLATION_STATUSES:
                continue
            run_ids = _question_ids_from_results(run.get("results"))
            if run_ids:
                return run_ids

    return _question_ids_from_results(report.get("results"))


def _direct_question_ids(payload: Mapping[str, Any]) -> list[str]:
    question_ids = payload.get("question_ids")
    if not isinstance(question_ids, list):
        return []
    return [str(question_id) for question_id in question_ids if question_id is not None]


def _question_ids_from_results(results: Any) -> list[str]:
    if not isinstance(results, list):
        return []

    question_ids: list[str] = []
    seen: set[str] = set()
    for result in results:
        for question_id in _result_question_ids(result):
            if question_id not in seen:
                question_ids.append(question_id)
                seen.add(question_id)
    return question_ids


def _result_question_ids(result: Any) -> list[str]:
    if not isinstance(result, Mapping):
        return []

    question_id = _optional_str(result.get("question_id"))
    if question_id:
        return [question_id]

    nested_ids: list[str] = []
    for key in ("naive", "agentic"):
        nested = result.get(key)
        if isinstance(nested, Mapping):
            nested_id = _optional_str(nested.get("question_id"))
            if nested_id:
                nested_ids.append(nested_id)

    systems = result.get("systems")
    if isinstance(systems, Mapping):
        for system_result in systems.values():
            if isinstance(system_result, Mapping):
                nested_id = _optional_str(system_result.get("question_id"))
                if nested_id:
                    nested_ids.append(nested_id)
    return nested_ids


def _question_count(
    payload: Mapping[str, Any],
    report: Mapping[str, Any],
    mode: str,
    question_ids: list[str],
) -> int:
    total_questions = _mapping(report.get("summary")).get("total_questions")
    if _is_int_like(total_questions):
        return int(total_questions)
    if question_ids:
        return len(question_ids)

    if mode == "ablation":
        for run in _ablation_runs(payload, report):
            if run.get("status") not in _COMPLETED_ABLATION_STATUSES:
                continue
            results = run.get("results")
            if isinstance(results, list):
                return len(results)

    results = report.get("results")
    if isinstance(results, list):
        return len(results)
    return 0


def _system_id(system: Any) -> str:
    raw_system_id = _optional_str(system) or "agentic"
    return _SYSTEM_ID_ALIASES.get(raw_system_id, raw_system_id)


def _system_label(system_id: str) -> str:
    return _SYSTEM_LABELS.get(system_id, system_id)


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _is_int_like(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _stable_import_run_id(path: Path, payload: Mapping[str, Any]) -> str:
    canonical_payload = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256()
    digest.update(str(path.resolve()).encode("utf-8"))
    digest.update(b"\0")
    digest.update(canonical_payload.encode("utf-8"))
    return f"hist_{digest.hexdigest()[:16]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00",
        "Z",
    )


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
