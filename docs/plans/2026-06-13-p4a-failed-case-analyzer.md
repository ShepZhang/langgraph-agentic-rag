# P4a Failed Case Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic failed-case attribution to evaluation artifacts and ablation reports so each failed question has a primary failure type, reason, and suggested next action.

**Architecture:** Create a small `evaluation.failure_analyzer` module with pure functions and no LLM calls. Integrate it at the normalized evaluation-result layer so single-system, comparison, and ablation artifacts all inherit the same `failure_analysis` payload and `failure_type_counts` summary. Extend the ablation Markdown report with count and representative-case tables while keeping dashboard and prompt-versioning work out of scope.

**Tech Stack:** Python 3.12, pytest, existing evaluation runner, existing ablation runner.

---

## File Map

### Create

- `evaluation/failure_analyzer.py`: deterministic failure taxonomy, source matching helpers, one-case analyzer, and summary counter.
- `tests/test_failure_analyzer.py`: focused rule-priority and source-matching tests.

### Modify

- `evaluation/evaluate.py`: attach `failure_analysis` to every result row and add `failure_type_counts` to summary output.
- `experiments/run_ablation.py`: add Failed Case Analysis tables to Markdown report.
- `tests/test_evaluate.py`: verify evaluation rows and summaries include failure diagnostics.
- `tests/test_ablation.py`: verify ablation Markdown includes failure type counts and representative failed cases.
- `README.md`: document deterministic failed-case analysis in Evaluation.
- `CHANGELOG.md`: add P4a entry.
- `docs/resume_bullets.md`: mention failed-case attribution in evaluation diagnostics.
- `api/main.py`: bump FastAPI version to `0.4.0-p4a`.

## Stable Interfaces

Create these public functions in `evaluation/failure_analyzer.py`:

```python
FailureType = Literal[
    "no_failure",
    "tool_failure",
    "fallback_failure",
    "query_rewrite_failure",
    "retrieval_failure",
    "reranking_failure",
    "citation_failure",
    "generation_failure",
]


def analyze_failure(
    question: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, str]:
    """Return primary failure attribution for one normalized evaluation row."""


def summarize_failure_types(
    results: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    """Count failure_analysis.failure_type values in evaluation results."""
```

Each `failure_analysis` record must include exactly these keys:

```python
{
    "question_id": "q001",
    "failure_type": "retrieval_failure",
    "reason": "Expected source was not retrieved in candidate documents.",
    "suggestion": "Try query expansion, hybrid retrieval, or increasing retrieval top_k.",
}
```

---

### Task 1: Deterministic Failure Analyzer

**Files:**
- Create: `evaluation/failure_analyzer.py`
- Create: `tests/test_failure_analyzer.py`

- [ ] **Step 1: Write failing analyzer tests**

Create `tests/test_failure_analyzer.py`:

```python
from __future__ import annotations

from evaluation.failure_analyzer import analyze_failure, summarize_failure_types


def _question(**overrides):
    question = {
        "id": "q001",
        "question_type": "single_doc",
        "question": "What is Agentic RAG?",
        "expected_sources": ["agentic_rag_notes.md"],
        "expected_keywords": ["retrieval"],
        "answerable": True,
        "should_answer": True,
        "requires_rewrite": False,
    }
    question.update(overrides)
    return question


def _result(**overrides):
    result = {
        "question_id": "q001",
        "question_type": "single_doc",
        "question": "What is Agentic RAG?",
        "answer_returned": True,
        "fallback_triggered": False,
        "fallback_correct": True,
        "correct": True,
        "context_relevant": True,
        "citation_hit": True,
        "citation_returned": True,
        "source_hit": True,
        "keyword_hit": True,
        "citation_verification_applicable": True,
        "unsupported_claim_count": 0,
        "retry_count": 0,
        "retrieved_doc_count": 1,
        "relevant_doc_count": 1,
        "error": None,
        "answer": "Agentic RAG uses retrieval [1].",
        "citations": [{"source": "agentic_rag_notes.md"}],
        "retrieved_documents": [{"source": "agentic_rag_notes.md"}],
        "relevant_documents": [{"source": "agentic_rag_notes.md"}],
    }
    result.update(overrides)
    return result


def test_analyze_failure_returns_no_failure_for_successful_case():
    analysis = analyze_failure(_question(), _result())

    assert analysis == {
        "question_id": "q001",
        "failure_type": "no_failure",
        "reason": "The case satisfied correctness, fallback, and evidence checks.",
        "suggestion": "No action required.",
    }


def test_analyze_failure_prioritizes_tool_failure():
    analysis = analyze_failure(
        _question(),
        _result(
            error="ToolExecutionError: retriever unavailable",
            correct=False,
            source_hit=False,
        ),
    )

    assert analysis["failure_type"] == "tool_failure"
    assert "retriever unavailable" in analysis["reason"]
    assert "trace" in analysis["suggestion"].lower()


def test_analyze_failure_detects_false_fallback_for_answerable_case():
    analysis = analyze_failure(
        _question(answerable=True, should_answer=True),
        _result(
            answer_returned=False,
            fallback_triggered=True,
            fallback_correct=False,
            correct=False,
        ),
    )

    assert analysis["failure_type"] == "fallback_failure"
    assert "answerable" in analysis["reason"]


def test_analyze_failure_detects_missed_fallback_for_unanswerable_case():
    analysis = analyze_failure(
        _question(
            answerable=False,
            should_answer=False,
            expected_sources=[],
            expected_keywords=[],
        ),
        _result(
            answer_returned=True,
            fallback_triggered=False,
            fallback_correct=False,
            correct=False,
            source_hit=False,
            citation_hit=False,
            citations=[],
            retrieved_documents=[],
            relevant_documents=[],
        ),
    )

    assert analysis["failure_type"] == "fallback_failure"
    assert "unanswerable" in analysis["reason"]


def test_analyze_failure_detects_query_rewrite_failure_before_retrieval_failure():
    analysis = analyze_failure(
        _question(requires_rewrite=True, question_type="follow_up"),
        _result(
            correct=False,
            context_relevant=False,
            citation_hit=False,
            source_hit=False,
            keyword_hit=False,
            retry_count=0,
            retrieved_documents=[],
            relevant_documents=[],
            citations=[],
        ),
    )

    assert analysis["failure_type"] == "query_rewrite_failure"
    assert "standalone" in analysis["suggestion"].lower()


def test_analyze_failure_detects_retrieval_failure_when_expected_source_absent():
    analysis = analyze_failure(
        _question(),
        _result(
            correct=False,
            context_relevant=False,
            citation_hit=False,
            source_hit=False,
            keyword_hit=False,
            retrieved_documents=[{"source": "unrelated.md"}],
            relevant_documents=[],
            citations=[],
        ),
    )

    assert analysis["failure_type"] == "retrieval_failure"
    assert "Expected source" in analysis["reason"]


def test_analyze_failure_detects_reranking_failure_when_retrieval_hit_is_filtered():
    analysis = analyze_failure(
        _question(),
        _result(
            correct=False,
            context_relevant=False,
            citation_hit=False,
            source_hit=True,
            keyword_hit=False,
            retrieved_documents=[{"source": "/tmp/agentic_rag_notes.md"}],
            relevant_documents=[{"source": "other.md"}],
            citations=[],
        ),
    )

    assert analysis["failure_type"] == "reranking_failure"
    assert "retrieved" in analysis["reason"]


def test_analyze_failure_detects_citation_failure_for_unsupported_claims():
    analysis = analyze_failure(
        _question(),
        _result(
            correct=False,
            citation_hit=True,
            source_hit=True,
            unsupported_claim_count=2,
        ),
    )

    assert analysis["failure_type"] == "citation_failure"
    assert "unsupported" in analysis["reason"].lower()


def test_analyze_failure_detects_generation_failure_after_evidence_hit():
    analysis = analyze_failure(
        _question(),
        _result(
            correct=False,
            keyword_hit=False,
            context_relevant=True,
            citation_hit=True,
            source_hit=True,
        ),
    )

    assert analysis["failure_type"] == "generation_failure"
    assert "answer" in analysis["suggestion"].lower()


def test_summarize_failure_types_counts_records_with_missing_analysis():
    counts = summarize_failure_types(
        [
            {"failure_analysis": {"failure_type": "no_failure"}},
            {"failure_analysis": {"failure_type": "retrieval_failure"}},
            {"failure_analysis": {"failure_type": "retrieval_failure"}},
            {},
        ]
    )

    assert counts == {
        "no_failure": 1,
        "retrieval_failure": 2,
        "tool_failure": 1,
    }
```

- [ ] **Step 2: Run analyzer tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_failure_analyzer.py -q
```

Expected: FAIL because `evaluation.failure_analyzer` does not exist.

- [ ] **Step 3: Implement the analyzer**

Create `evaluation/failure_analyzer.py`:

```python
"""Deterministic failed-case attribution for evaluation results."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping


FailureType = Literal[
    "no_failure",
    "tool_failure",
    "fallback_failure",
    "query_rewrite_failure",
    "retrieval_failure",
    "reranking_failure",
    "citation_failure",
    "generation_failure",
]

_QUERY_REWRITE_TYPES = {"ambiguous", "follow_up"}


def analyze_failure(
    question: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, str]:
    """Return primary failure attribution for one normalized evaluation row."""

    question_id = str(result.get("question_id") or question.get("id") or "")
    expected_sources = _string_list(question.get("expected_sources", []))
    retrieved_hit = _has_source(expected_sources, result.get("retrieved_documents", []))
    relevant_hit = _has_source(expected_sources, result.get("relevant_documents", []))
    citation_hit = bool(result.get("citation_hit")) or _has_source(
        expected_sources,
        result.get("citations", []),
    )
    any_evidence_hit = bool(result.get("source_hit")) or retrieved_hit or relevant_hit or citation_hit

    error = _clean_text(result.get("error"))
    if error:
        return _analysis(
            question_id,
            "tool_failure",
            f"Evaluation recorded an execution error: {error}",
            "Inspect tool-call traces, runner configuration, model settings, and retriever availability.",
        )

    if _is_successful_case(result, expected_sources, any_evidence_hit):
        return _analysis(
            question_id,
            "no_failure",
            "The case satisfied correctness, fallback, and evidence checks.",
            "No action required.",
        )

    should_answer = _answerable(question)
    fallback_triggered = bool(result.get("fallback_triggered"))
    answer_returned = bool(result.get("answer_returned"))
    if should_answer and fallback_triggered:
        return _analysis(
            question_id,
            "fallback_failure",
            "The question is answerable, but the system fell back instead of answering.",
            "Inspect retrieval grading thresholds, citation verification strictness, and fallback routing.",
        )
    if not should_answer and answer_returned:
        return _analysis(
            question_id,
            "fallback_failure",
            "The question is unanswerable, but the system returned an answer.",
            "Tighten fallback policy, retrieval grading, and unsupported-answer checks.",
        )

    if (
        expected_sources
        and not any_evidence_hit
        and _requires_query_rewrite(question)
        and int(result.get("retry_count") or 0) <= 0
    ):
        return _analysis(
            question_id,
            "query_rewrite_failure",
            "The case needed query rewriting, but no expected evidence was found and no retry/rewrite signal was recorded.",
            "Improve standalone question rewrite, multi-query expansion, or follow-up context handling.",
        )

    if expected_sources and not any_evidence_hit:
        return _analysis(
            question_id,
            "retrieval_failure",
            "Expected source was not found in retrieved, relevant, or cited documents.",
            "Try query expansion, hybrid retrieval, or increasing retrieval top_k.",
        )

    if expected_sources and retrieved_hit and not relevant_hit and not citation_hit:
        return _analysis(
            question_id,
            "reranking_failure",
            "Expected evidence was retrieved but did not survive relevance filtering or citation selection.",
            "Inspect reranker scores, retrieval grading labels, top_n settings, and evidence filtering.",
        )

    unsupported_count = _safe_int(result.get("unsupported_claim_count"))
    if unsupported_count > 0:
        return _analysis(
            question_id,
            "citation_failure",
            f"Claim verification found {unsupported_count} unsupported claim(s).",
            "Inspect citation selection, claim verifier feedback, and answer revision behavior.",
        )
    if expected_sources and answer_returned and not citation_hit:
        return _analysis(
            question_id,
            "citation_failure",
            "The system returned an answer, but citations did not hit the expected source.",
            "Improve citation selection and require cited chunks to support generated claims.",
        )

    if not bool(result.get("correct")):
        return _analysis(
            question_id,
            "generation_failure",
            "Retrieved evidence and fallback behavior were not the primary issue, but the answer failed correctness checks.",
            "Improve answer generation prompts, keyword coverage, and stricter grounding in selected evidence.",
        )

    return _analysis(
        question_id,
        "no_failure",
        "No failed evaluation signal was detected.",
        "No action required.",
    )


def summarize_failure_types(
    results: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    """Count failure_analysis.failure_type values in evaluation results."""

    counts: Counter[str] = Counter()
    for result in results:
        analysis = result.get("failure_analysis")
        if isinstance(analysis, Mapping):
            failure_type = analysis.get("failure_type")
            if isinstance(failure_type, str) and failure_type:
                counts[failure_type] += 1
                continue
        counts["tool_failure"] += 1
    return dict(counts)


def _analysis(
    question_id: str,
    failure_type: FailureType,
    reason: str,
    suggestion: str,
) -> dict[str, str]:
    return {
        "question_id": question_id,
        "failure_type": failure_type,
        "reason": reason,
        "suggestion": suggestion,
    }


def _is_successful_case(
    result: Mapping[str, Any],
    expected_sources: list[str],
    any_evidence_hit: bool,
) -> bool:
    if not bool(result.get("correct")) or not bool(result.get("fallback_correct")):
        return False
    if expected_sources and not any_evidence_hit:
        return False
    if result.get("citation_verification_applicable"):
        unsupported = result.get("unsupported_claim_count")
        if unsupported not in (None, 0):
            return False
    return True


def _answerable(question: Mapping[str, Any]) -> bool:
    value = question.get("answerable", question.get("should_answer", True))
    return bool(value)


def _requires_query_rewrite(question: Mapping[str, Any]) -> bool:
    return bool(question.get("requires_rewrite")) or str(
        question.get("question_type", "")
    ) in _QUERY_REWRITE_TYPES


def _has_source(expected_sources: list[str], documents: Any) -> bool:
    if not expected_sources or not isinstance(documents, list):
        return False
    expected = [_normalize_source(source) for source in expected_sources]
    for document in documents:
        if not isinstance(document, Mapping):
            continue
        candidates = [
            document.get("source"),
            document.get("source_path"),
            document.get("document_id"),
            document.get("file_hash"),
        ]
        for candidate in candidates:
            normalized = _normalize_source(candidate)
            if normalized and any(_source_matches(item, normalized) for item in expected):
                return True
    return False


def _source_matches(expected: str, observed: str) -> bool:
    if expected == observed:
        return True
    expected_name = Path(expected).name
    observed_name = Path(observed).name
    return (
        bool(expected_name and expected_name == observed_name)
        or expected in observed
        or observed in expected
    )


def _normalize_source(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().replace("\\", "/")


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _safe_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value
```

- [ ] **Step 4: Run analyzer tests and verify pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_failure_analyzer.py -q
```

Expected: all analyzer tests pass.

- [ ] **Step 5: Commit core analyzer**

```bash
git add evaluation/failure_analyzer.py tests/test_failure_analyzer.py
git commit -m "feat: add deterministic failure analyzer"
```

---

### Task 2: Evaluation Runner Integration

**Files:**
- Modify: `evaluation/evaluate.py`
- Modify: `tests/test_evaluate.py`

- [ ] **Step 1: Add failing evaluation integration tests**

Append to `tests/test_evaluate.py`:

```python
def test_evaluate_questions_attaches_failure_analysis_and_summary_counts():
    questions = [
        {
            "id": "q001",
            "question": "What is Agentic RAG?",
            "expected_sources": ["notes.md"],
            "expected_keywords": ["agentic"],
        },
        {
            "id": "q002",
            "question": "What is missing?",
            "expected_sources": ["missing.md"],
            "expected_keywords": ["missing"],
        },
    ]

    def runner(question):
        if question == "What is Agentic RAG?":
            return {
                "answer": "Agentic RAG uses retrieval [1].",
                "citations": [{"source": "notes.md"}],
                "retrieved_documents": [{"source": "notes.md"}],
                "relevant_documents": [{"source": "notes.md"}],
                "citation_verification_enabled": False,
            }
        return {
            "answer": "Wrong answer.",
            "citations": [],
            "retrieved_documents": [{"source": "other.md"}],
            "relevant_documents": [],
            "citation_verification_enabled": False,
        }

    report = evaluate_questions(questions, run_agent_fn=runner)

    assert report["results"][0]["failure_analysis"]["failure_type"] == "no_failure"
    assert report["results"][1]["failure_analysis"]["failure_type"] == "retrieval_failure"
    assert report["summary"]["failure_type_counts"] == {
        "no_failure": 1,
        "retrieval_failure": 1,
    }


def test_evaluate_comparison_preserves_failure_analysis_for_each_system():
    questions = [
        {
            "id": "q001",
            "question": "What is Agentic RAG?",
            "expected_sources": ["notes.md"],
            "expected_keywords": ["agentic"],
        }
    ]

    def naive(question):
        return {
            "answer": "Wrong answer.",
            "citations": [],
            "retrieved_documents": [{"source": "other.md"}],
            "relevant_documents": [],
            "citation_verification_enabled": False,
        }

    def agentic(question):
        return {
            "answer": "Agentic RAG uses retrieval [1].",
            "citations": [{"source": "notes.md"}],
            "retrieved_documents": [{"source": "notes.md"}],
            "relevant_documents": [{"source": "notes.md"}],
            "citation_verification_enabled": False,
        }

    report = evaluate_questions(
        questions,
        run_agent_fn=agentic,
        run_naive_fn=naive,
    )

    pair = report["results"][0]
    assert pair["naive"]["failure_analysis"]["failure_type"] == "retrieval_failure"
    assert pair["agentic"]["failure_analysis"]["failure_type"] == "no_failure"
    assert report["summary"]["naive"]["failure_type_counts"] == {
        "retrieval_failure": 1,
    }
    assert report["summary"]["agentic"]["failure_type_counts"] == {
        "no_failure": 1,
    }
```

- [ ] **Step 2: Run evaluation tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluate.py -q
```

Expected: FAIL because result rows and summaries do not include failure analysis.

- [ ] **Step 3: Wire analyzer into `evaluation/evaluate.py`**

At the top of `evaluation/evaluate.py`, add:

```python
from evaluation.failure_analyzer import analyze_failure, summarize_failure_types
```

In `_evaluate_single_system()`, after setting latency and error, attach analysis:

```python
    result["latency"] = latency
    result["error"] = error
    result["failure_analysis"] = analyze_failure(item, result)
    return result
```

In the zero-question branch of `_summarize()`, add:

```python
            "failure_type_counts": {},
```

In the normal `_summarize()` return payload, add:

```python
        "failure_type_counts": summarize_failure_types(results),
```

- [ ] **Step 4: Run evaluation tests and verify pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_failure_analyzer.py tests/test_evaluate.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit evaluation integration**

```bash
git add evaluation/evaluate.py tests/test_evaluate.py
git commit -m "feat: attach failure analysis to evaluation results"
```

---

### Task 3: Ablation Markdown Report Integration

**Files:**
- Modify: `experiments/run_ablation.py`
- Modify: `tests/test_ablation.py`

- [ ] **Step 1: Add failing ablation report test**

Append to `tests/test_ablation.py`:

```python
def test_format_ablation_report_includes_failed_case_analysis():
    payload = {
        "runs": [
            {
                "id": "v6_citation_verification",
                "method": "Claim-level Citation Verification",
                "status": "completed",
                "summary": {
                    "correctness_score": 0.5,
                    "context_relevance_score": 0.5,
                    "citation_hit_rate": 0.5,
                    "fallback_accuracy": 0.5,
                    "unsupported_claim_count": 1,
                    "supported_claim_ratio": 0.5,
                    "average_retry_count": 0,
                    "average_latency": 1.2,
                    "error_count": 0,
                    "failure_type_counts": {
                        "no_failure": 1,
                        "retrieval_failure": 1,
                    },
                },
                "results": [
                    {
                        "question_id": "q001",
                        "question_type": "single_doc",
                        "failure_analysis": {
                            "question_id": "q001",
                            "failure_type": "no_failure",
                            "reason": "The case satisfied correctness, fallback, and evidence checks.",
                            "suggestion": "No action required.",
                        },
                    },
                    {
                        "question_id": "q002",
                        "question_type": "multi_chunk",
                        "failure_analysis": {
                            "question_id": "q002",
                            "failure_type": "retrieval_failure",
                            "reason": "Expected source was not found in retrieved, relevant, or cited documents.",
                            "suggestion": "Try query expansion, hybrid retrieval, or increasing retrieval top_k.",
                        },
                    },
                ],
            }
        ]
    }

    report = format_ablation_report(payload)

    assert "## Failed Case Analysis" in report
    assert "| retrieval_failure | 1 |" in report
    assert "## Representative Failed Cases" in report
    assert "| q002 | multi_chunk | retrieval_failure |" in report
```

- [ ] **Step 2: Run ablation tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_ablation.py -q
```

Expected: FAIL because `format_ablation_report()` does not include failed-case tables.

- [ ] **Step 3: Add report helpers**

In `experiments/run_ablation.py`, add helper functions near the existing report
helpers:

```python
def _format_failed_case_analysis(runs: list[dict[str, Any]]) -> list[str]:
    counts = _aggregate_failure_counts(runs)
    cases = _representative_failed_cases(runs)
    lines = ["", "## Failed Case Analysis", ""]
    if counts:
        lines.extend(["| Failure Type | Count |", "|---|---:|"])
        for failure_type, count in sorted(counts.items()):
            if failure_type == "no_failure":
                continue
            lines.append(f"| {failure_type} | {count} |")
        if all(key == "no_failure" for key in counts):
            lines.append("| no_failure | 0 |")
    else:
        lines.append("No failure analysis records are available.")

    lines.extend(["", "## Representative Failed Cases", ""])
    if not cases:
        lines.append("No failed cases were recorded.")
        return lines

    lines.extend(
        [
            "| Variant | Question ID | Type | Failure | Reason | Suggestion |",
            "|---|---|---|---|---|---|",
        ]
    )
    for case in cases[:10]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table_cell(case["variant"]),
                    _escape_table_cell(case["question_id"]),
                    _escape_table_cell(case["question_type"]),
                    _escape_table_cell(case["failure_type"]),
                    _escape_table_cell(case["reason"]),
                    _escape_table_cell(case["suggestion"]),
                ]
            )
            + " |"
        )
    return lines


def _aggregate_failure_counts(runs: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for run in runs:
        if run.get("status") != "completed":
            continue
        summary_counts = run.get("summary", {}).get("failure_type_counts", {})
        if not isinstance(summary_counts, dict):
            continue
        for failure_type, count in summary_counts.items():
            if not isinstance(failure_type, str) or not isinstance(count, int):
                continue
            counts[failure_type] = counts.get(failure_type, 0) + count
    return counts


def _representative_failed_cases(runs: list[dict[str, Any]]) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    for run in runs:
        if run.get("status") != "completed":
            continue
        variant = _display_method(run)
        for result in run.get("results", []):
            if not isinstance(result, dict):
                continue
            analysis = result.get("failure_analysis", {})
            if not isinstance(analysis, dict):
                continue
            failure_type = str(analysis.get("failure_type", ""))
            if not failure_type or failure_type == "no_failure":
                continue
            cases.append(
                {
                    "variant": variant,
                    "question_id": str(result.get("question_id", analysis.get("question_id", ""))),
                    "question_type": str(result.get("question_type", "")),
                    "failure_type": failure_type,
                    "reason": str(analysis.get("reason", "")),
                    "suggestion": str(analysis.get("suggestion", "")),
                }
            )
    return cases


def _escape_table_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()
```

In `format_ablation_report()`, insert after the main metrics table and before
Observed Trade-offs:

```python
    lines.extend(_format_failed_case_analysis(payload.get("runs", [])))
```

- [ ] **Step 4: Run ablation tests and verify pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_ablation.py -q
```

Expected: all ablation tests pass.

- [ ] **Step 5: Commit ablation report integration**

```bash
git add experiments/run_ablation.py tests/test_ablation.py
git commit -m "feat: report failure analysis in ablations"
```

---

### Task 4: Documentation and Versioning

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/resume_bullets.md`
- Modify: `api/main.py`

- [ ] **Step 1: Update changelog**

Add above the P3d entry in `CHANGELOG.md`:

```markdown
## v0.4.0-p4a - Failed Case Analyzer

Date: 2026-06-13

### Added

- Added deterministic failed-case analysis for evaluation results with primary
  failure types, reasons, and suggested next actions.
- Added `failure_analysis` per question and `failure_type_counts` in evaluation
  summaries.
- Added failed-case count and representative-case sections to ablation reports.

### Notes

- P4a does not use an LLM judge and does not automatically repair failures.
- Failure attribution is rule-based and intended for debugging, regression
  triage, and portfolio explanation rather than benchmark-grade causal proof.
```

- [ ] **Step 2: Update README**

In the Evaluation section, add:

```markdown
### Failed Case Analysis

P4a adds deterministic failure attribution to evaluation artifacts. Each
question result includes a `failure_analysis` object with `failure_type`,
`reason`, and `suggestion`. The analyzer uses existing evaluation signals such
as source hits, citation hits, fallback correctness, unsupported claims,
retrieved documents, relevant documents, and runner errors.

Initial failure types are `retrieval_failure`, `reranking_failure`,
`query_rewrite_failure`, `generation_failure`, `citation_failure`,
`fallback_failure`, `tool_failure`, and `no_failure`. The analyzer is
rule-based and does not automatically repair failures.
```

Also add `failure_type_counts` to the metric field list and mention failed-case
analysis in Resume Highlights and Completed Work.

- [ ] **Step 3: Update resume bullets**

In `docs/resume_bullets.md`, revise the evaluation bullet so it mentions:

```text
失败样本归因（retrieval / reranking / query rewrite / generation / citation / fallback / tool failure）
```

- [ ] **Step 4: Update API version**

In `api/main.py`, change:

```python
version="0.3.3-p3d"
```

to:

```python
version="0.4.0-p4a"
```

- [ ] **Step 5: Run documentation checks**

Run:

```bash
rg -n "production-ready|autonomous agent|automatic repair|auto-fix" README.md CHANGELOG.md docs/resume_bullets.md
git diff --check
.venv/bin/python -c "from api.main import app; print(app.title, app.version, len(app.routes))"
```

Expected:

- no inappropriate hype wording except historical changelog text if already present
- no whitespace errors
- API version output includes `0.4.0-p4a`

- [ ] **Step 6: Commit docs**

```bash
git add README.md CHANGELOG.md docs/resume_bullets.md api/main.py
git commit -m "docs: publish p4a failure analysis release"
```

---

### Task 5: Final Verification

**Files:**
- Verify repository state only.

- [ ] **Step 1: Run static checks**

```bash
git diff --check
.venv/bin/python -m compileall evaluation experiments api
```

Expected: no whitespace errors and compileall succeeds.

- [ ] **Step 2: Run focused P4a tests**

```bash
.venv/bin/python -m pytest \
  tests/test_failure_analyzer.py \
  tests/test_evaluate.py \
  tests/test_ablation.py \
  tests/test_fastapi_routes.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 3: Run complete suite**

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 4: Verify API import and version**

```bash
.venv/bin/python -c "from api.main import app; print(app.title, app.version, len(app.routes))"
```

Expected:

```text
Reliability-oriented Agentic RAG API 0.4.0-p4a 12
```

- [ ] **Step 5: Inspect final scope**

```bash
git status --short
git log --oneline -8
```

Expected: clean worktree with P4a commits above `40b9713 docs: design p4a failed case analyzer`.

Do not merge to `main` or tag `v0.4.0-p4a` until the user reviews the
implementation and explicitly requests version integration.
