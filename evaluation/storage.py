"""Evaluation result storage helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol


class ResultStore(Protocol):
    """Persistence boundary for evaluation result payloads."""

    def save(self, run_id: str, payload: dict[str, Any]) -> str:
        """Persist a result payload and return the final storage path."""
        ...

    def load(self, run_id: str) -> dict[str, Any] | None:
        """Load a result payload, returning None when it is missing."""
        ...


class JsonResultStore:
    """Atomic UTF-8 JSON result store rooted at a directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def save(self, run_id: str, payload: dict[str, Any]) -> str:
        if not isinstance(payload, dict):
            raise ValueError("result payload must be a JSON object")

        path = self._path_for(run_id)
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_path.replace(path)
        return str(path)

    def load(self, run_id: str) -> dict[str, Any] | None:
        path = self._path_for(run_id)
        if not path.exists():
            return None

        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("stored result payload must be a JSON object")
        return payload

    def _path_for(self, run_id: str) -> Path:
        return self.root / f"{run_id}.json"


def write_compatibility_artifacts(
    report: dict[str, Any],
    output_dir: str | Path,
    runtime_config: dict[str, Any],
) -> None:
    """Write legacy evaluation artifact filenames and payload layouts."""

    store = JsonResultStore(output_dir)
    summary = report.get("summary", {})
    if summary.get("mode") == "comparison":
        paired_results = report.get("results", [])
        store.save(
            "baseline_result",
            {
                "system": "naive_rag",
                "runtime_config": runtime_config,
                "summary": summary.get("naive", {}),
                "results": [paired.get("naive", {}) for paired in paired_results],
            },
        )
        store.save(
            "agentic_result",
            {
                "system": "agentic_rag",
                "runtime_config": runtime_config,
                "summary": summary.get("agentic", {}),
                "results": [paired.get("agentic", {}) for paired in paired_results],
            },
        )
        store.save("comparison_result", {"runtime_config": runtime_config, **report})
        return

    store.save(
        "agentic_result",
        {
            "system": "agentic_rag",
            "runtime_config": runtime_config,
            "summary": report.get("summary", {}),
            "results": report.get("results", []),
        },
    )
