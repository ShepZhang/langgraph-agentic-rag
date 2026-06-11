"""Trace lookup service."""

from __future__ import annotations

from typing import Any

from config import Settings, get_settings
from observability.storage import JsonlTraceStore


class TraceService:
    """Read local JSONL traces."""

    def __init__(
        self,
        settings: Settings | None = None,
        store: JsonlTraceStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or JsonlTraceStore(self.settings.trace_log_dir)

    def get_trace(
        self,
        session_id: str,
        trace_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Return a trace by id, or latest trace for the session."""

        if trace_id:
            record = self.store.get(trace_id)
            if record and record.get("session_id") == session_id:
                return record
            return None
        return self.store.find_latest(session_id=session_id)
