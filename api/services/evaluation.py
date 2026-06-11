"""Synchronous evaluation service for the API layer."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from baseline.naive_rag import run_naive_rag
from config import Settings, get_settings
from evaluation.evaluate import evaluate_questions, load_eval_questions


class EvaluationService:
    """Run and persist lightweight evaluation reports."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def run_evaluation(
        self,
        workspace_id: str,
        question_ids: list[str] | None = None,
        include_baseline: bool = True,
    ) -> dict[str, Any]:
        """Run evaluation and write an artifact."""

        questions = load_eval_questions()
        if question_ids:
            selected = set(question_ids)
            questions = [question for question in questions if question["id"] in selected]

        report = evaluate_questions(
            questions,
            run_naive_fn=run_naive_rag if include_baseline else None,
        )
        run_id = f"eval_{uuid.uuid4().hex}"
        result_path = self._run_path(run_id)
        payload = {
            "run_id": run_id,
            "workspace_id": workspace_id,
            "status": "completed",
            "summary": report.get("summary", {}),
            "result_path": str(result_path),
            "report": report,
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return _public_run(payload)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Read a persisted evaluation run."""

        path = self._run_path(run_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _public_run(payload)

    def _run_path(self, run_id: str) -> Path:
        return self.settings.evaluation_run_dir / f"{run_id}.json"


def _public_run(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": payload["run_id"],
        "workspace_id": payload["workspace_id"],
        "status": payload["status"],
        "summary": payload.get("summary", {}),
        "result_path": payload["result_path"],
    }
