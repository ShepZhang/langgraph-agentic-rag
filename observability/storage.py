"""Local trace storage backends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlTraceStore:
    """Append-only JSONL trace storage with lookup by trace id."""

    def __init__(self, log_dir: str | Path, filename: str = "traces.jsonl") -> None:
        self.log_dir = Path(log_dir)
        self.filename = filename

    @property
    def path(self) -> Path:
        """Return the JSONL file path."""

        return self.log_dir / self.filename

    def save(self, record: dict[str, Any]) -> None:
        """Append a trace record."""

        self.log_dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, default=str))
            file.write("\n")

    def get(self, trace_id: str) -> dict[str, Any] | None:
        """Return the newest trace record matching `trace_id`."""

        for record in self._iter_records_newest_first():
            if record.get("trace_id") == trace_id:
                return record
        return None

    def find_latest(
        self,
        session_id: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Return the newest trace matching the provided filters."""

        for record in self._iter_records_newest_first():
            if session_id is not None and record.get("session_id") != session_id:
                continue
            if workspace_id is not None and record.get("workspace_id") != workspace_id:
                continue
            return record
        return None

    def _iter_records_newest_first(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []

        records: list[dict[str, Any]] = []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
        return records
