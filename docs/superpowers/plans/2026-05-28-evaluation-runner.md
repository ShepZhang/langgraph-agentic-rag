# Evaluation Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight evaluation runner that executes `eval_questions.json` through the Agentic RAG graph and reports answer, citation, source-hit, latency, and rewrite metrics.

**Architecture:** Keep evaluation independent from Gradio and backend services. `evaluation/evaluate.py` owns dataset loading, per-question evaluation, metric aggregation, report formatting, and a `python -m evaluation.evaluate` CLI entrypoint. Tests inject fake `run_agent` functions so no real LLM, Chroma, or embeddings are used.

**Tech Stack:** Python standard library (`argparse`, `json`, `time`, `pathlib`), existing `agent.graph.run_agent`, pytest.

---

## File Structure

- `evaluation/evaluate.py`: New CLI and reusable functions for loading questions, evaluating them, aggregating metrics, and formatting a report.
- `tests/test_evaluate.py`: Unit tests for loader validation, metric computation, failure handling, and report text.
- `README.md`: Roadmap status update after the runner is implemented.

## Task 1: Evaluation Core

**Files:**
- Create: `evaluation/evaluate.py`
- Create: `tests/test_evaluate.py`

- [ ] **Step 1: Write failing tests for loader and metrics**

Create `tests/test_evaluate.py` with tests for:

```python
"""Tests for lightweight evaluation runner."""

from __future__ import annotations

import json

import pytest

from evaluation.evaluate import evaluate_questions, load_eval_questions


def test_load_eval_questions_reads_json_file(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question": "What is Agentic RAG?",
                    "expected_keywords": ["agent", "retrieval"],
                    "expected_source": "notes.md",
                }
            ]
        ),
        encoding="utf-8",
    )

    questions = load_eval_questions(path)

    assert questions == [
        {
            "question": "What is Agentic RAG?",
            "expected_keywords": ["agent", "retrieval"],
            "expected_source": "notes.md",
        }
    ]


def test_load_eval_questions_rejects_missing_question(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps([{"expected_source": "notes.md"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="question"):
        load_eval_questions(path)


def test_evaluate_questions_computes_summary_metrics():
    questions = [
        {
            "question": "What is Agentic RAG?",
            "expected_keywords": ["agentic", "retrieval"],
            "expected_source": "notes.md",
        },
        {
            "question": "What is missing?",
            "expected_keywords": ["missing"],
            "expected_source": "missing.md",
        },
    ]
    timer_values = iter([0.0, 1.0, 1.0, 3.0])

    def fake_timer():
        return next(timer_values)

    def fake_run_agent(question):
        if question == "What is Agentic RAG?":
            return {
                "answer": "Agentic RAG uses retrieval.",
                "citations": [{"source": "notes.md"}],
                "retrieved_documents": [{"source": "notes.md", "content": "Agentic RAG uses retrieval."}],
                "rewrite_count": 1,
            }
        return {
            "answer": "",
            "citations": [],
            "retrieved_documents": [{"source": "other.md", "content": "Other content"}],
            "rewrite_count": 0,
        }

    report = evaluate_questions(questions, run_agent_fn=fake_run_agent, timer=fake_timer)

    assert report["summary"] == {
        "total_questions": 2,
        "answer_rate": 0.5,
        "citation_rate": 0.5,
        "source_hit_rate": 0.5,
        "average_latency": 1.5,
        "rewrite_triggered_count": 1,
        "keyword_hit_rate": 0.5,
        "error_count": 0,
    }
    assert report["results"][0]["source_hit"] is True
    assert report["results"][0]["keyword_hit"] is True
    assert report["results"][1]["answer_returned"] is False


def test_evaluate_questions_records_agent_errors():
    questions = [{"question": "Broken?", "expected_source": "notes.md"}]
    timer_values = iter([0.0, 0.25])

    def fake_timer():
        return next(timer_values)

    def failing_run_agent(question):
        raise RuntimeError("Missing LLM configuration")

    report = evaluate_questions(questions, run_agent_fn=failing_run_agent, timer=fake_timer)

    assert report["summary"]["answer_rate"] == 0.0
    assert report["summary"]["error_count"] == 1
    assert report["results"][0]["error"] == "Missing LLM configuration"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluate.py -v
```

Expected: tests fail because `evaluation.evaluate` is missing.

- [ ] **Step 3: Implement evaluation core**

Create `evaluation/evaluate.py` with:

```python
"""Lightweight evaluation runner for Agentic RAG."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agent.graph import run_agent


DEFAULT_EVAL_PATH = Path(__file__).with_name("eval_questions.json")


def load_eval_questions(path: str | Path = DEFAULT_EVAL_PATH) -> list[dict[str, Any]]:
    """Load and validate evaluation questions."""
    ...


def evaluate_questions(
    questions: list[dict[str, Any]],
    run_agent_fn=run_agent,
    timer=time.perf_counter,
) -> dict[str, Any]:
    """Evaluate questions and return per-question results plus summary metrics."""
    ...
```

Implementation behavior:

- `load_eval_questions` reads JSON, requires a list of objects, requires non-empty string `question`, normalizes `expected_keywords` to a list, and keeps `expected_source`.
- `evaluate_questions` calls `run_agent_fn(question)` for each item.
- Measures latency as `timer()` after call minus `timer()` before call.
- Per result includes `question`, `answer_returned`, `citation_returned`, `source_hit`, `keyword_hit`, `rewrite_triggered`, `latency`, `error`, `answer`, `citations`, and `retrieved_documents`.
- `source_hit` checks whether `expected_source` matches any `source` in `retrieved_documents`.
- `keyword_hit` checks whether all expected keywords appear in the lowercased answer. If no expected keywords are provided, `keyword_hit` is `False`.
- Summary metrics include:
  - `total_questions`
  - `answer_rate`
  - `citation_rate`
  - `source_hit_rate`
  - `average_latency`
  - `rewrite_triggered_count`
  - `keyword_hit_rate`
  - `error_count`
- Rates are rounded to 4 decimal places.
- Empty question lists return zero counts and zero rates.
- Agent exceptions are captured in result `error` and count toward `error_count`.

- [ ] **Step 4: Run core tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluate.py -v
```

Expected: all evaluation core tests pass.

- [ ] **Step 5: Commit core evaluator**

Run:

```bash
git add evaluation/evaluate.py tests/test_evaluate.py
git commit -m "feat: add evaluation metrics runner"
```

Expected: git creates a commit containing the evaluator and tests.

## Task 2: Evaluation CLI Report

**Files:**
- Modify: `evaluation/evaluate.py`
- Modify: `tests/test_evaluate.py`

- [ ] **Step 1: Add report formatting tests**

Append tests:

```python
from evaluation.evaluate import format_report, main


def test_format_report_includes_summary_and_question_rows():
    report = {
        "summary": {
            "total_questions": 1,
            "answer_rate": 1.0,
            "citation_rate": 1.0,
            "source_hit_rate": 1.0,
            "average_latency": 0.25,
            "rewrite_triggered_count": 1,
            "keyword_hit_rate": 1.0,
            "error_count": 0,
        },
        "results": [
            {
                "question": "What is Agentic RAG?",
                "answer_returned": True,
                "citation_returned": True,
                "source_hit": True,
                "keyword_hit": True,
                "rewrite_triggered": True,
                "latency": 0.25,
                "error": "",
            }
        ],
    }

    text = format_report(report)

    assert "Evaluation Report" in text
    assert "total_questions: 1" in text
    assert "source_hit_rate: 1.0" in text
    assert "What is Agentic RAG?" in text


def test_main_prints_report_with_injected_runner(tmp_path, capsys):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps([{"question": "What is Agentic RAG?", "expected_source": "notes.md"}]), encoding="utf-8")

    def fake_run_agent(question):
        return {
            "answer": "Agentic RAG uses retrieval.",
            "citations": [{"source": "notes.md"}],
            "retrieved_documents": [{"source": "notes.md", "content": "Agentic RAG uses retrieval."}],
            "rewrite_count": 0,
        }

    exit_code = main(["--questions", str(path)], run_agent_fn=fake_run_agent)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Evaluation Report" in captured.out
    assert "total_questions: 1" in captured.out
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluate.py -v
```

Expected: report tests fail because `format_report` and `main` are missing.

- [ ] **Step 3: Implement CLI functions**

Add:

```python
def format_report(report: dict[str, Any]) -> str:
    """Format an evaluation report for terminal output."""
    ...


def main(argv: list[str] | None = None, run_agent_fn=run_agent) -> int:
    """CLI entrypoint."""
    ...


if __name__ == "__main__":
    raise SystemExit(main())
```

Behavior:

- CLI supports `--questions` with default `evaluation/eval_questions.json`.
- CLI prints `format_report(evaluate_questions(...))`.
- `format_report` includes a summary section and per-question rows with booleans and latency.
- `main` returns `0` on success.

- [ ] **Step 4: Run CLI tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluate.py -v
```

Expected: all evaluation tests pass.

- [ ] **Step 5: Run CLI smoke with injected-free default**

Run:

```bash
.venv/bin/python -m evaluation.evaluate --questions evaluation/eval_questions.json
```

Expected: the command prints an evaluation report. If LLM/vector store configuration is missing, rows may contain captured errors, but the process should still exit successfully.

- [ ] **Step 6: Commit CLI report**

Run:

```bash
git add evaluation/evaluate.py tests/test_evaluate.py
git commit -m "feat: add evaluation cli report"
```

Expected: git creates a commit containing report formatting and CLI behavior.

## Task 3: README Status and Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run evaluation and full tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluate.py -v
.venv/bin/python -m pytest -v
.venv/bin/python -m compileall evaluation tests
```

Expected: all tests pass and compileall reports no syntax errors.

- [ ] **Step 2: Update README roadmap**

Modify the evaluation roadmap item from:

```markdown
- Implement evaluation runner.
```

to:

```markdown
- Evaluation runner implemented: answer rate, citation rate, source hit rate, latency, keyword hit rate, and rewrite-trigger metrics.
```

- [ ] **Step 3: Verify README text**

Run:

```bash
rg "Evaluation runner implemented" README.md
```

Expected output includes the updated roadmap bullet.

- [ ] **Step 4: Commit README update**

Run:

```bash
git add README.md
git commit -m "docs: update evaluation status"
```

Expected: git creates a commit containing the roadmap update.

## Task 4: Final Evaluation Verification

**Files:**
- Read: `evaluation/evaluate.py`, `tests/test_evaluate.py`, `README.md`

- [ ] **Step 1: Run full test suite**

Run:

```bash
.venv/bin/python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Run CLI smoke**

Run:

```bash
.venv/bin/python -m evaluation.evaluate --questions evaluation/eval_questions.json
```

Expected: command exits with code 0 and prints `Evaluation Report`.

- [ ] **Step 3: Confirm clean git status**

Run:

```bash
git status --short
```

Expected output is empty.

- [ ] **Step 4: Inspect recent commits**

Run:

```bash
git log --oneline -10
```

Expected recent commits include:

```text
docs: update evaluation status
feat: add evaluation cli report
feat: add evaluation metrics runner
```

## Self-Review

- Spec coverage: The plan implements loading questions, calling `run_agent`, latency, answer rate, citation rate, source hit rate, rewrite-trigger count, keyword hit rate, error handling, and CLI output.
- Placeholder scan: No unfinished placeholder markers remain in the plan.
- Type consistency: `evaluate_questions` returns a dictionary with `summary` and `results`; `format_report` and tests use the same keys.
