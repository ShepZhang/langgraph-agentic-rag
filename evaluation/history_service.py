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
            return {
                "status": "failed",
                "run_id": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

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
    return (
        f"eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_"
        f"{uuid4().hex[:8]}"
    )
