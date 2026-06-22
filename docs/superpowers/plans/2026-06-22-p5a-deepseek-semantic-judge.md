# P5a DeepSeek Semantic Judge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional, independently configured DeepSeek semantic Judge that scores evaluation results for semantic correctness and groundedness while preserving deterministic metrics and disabled-by-default behavior.

**Architecture:** Keep system execution in `evaluation.runners`, invoke the optional Judge in `evaluation.comparison` after deterministic scoring, and aggregate additive Judge fields in `evaluation.metrics`. Build the DeepSeek client from isolated `EVALUATION_JUDGE_*` settings, render one versioned prompt, strictly parse one two-dimension JSON response per successful system result, and expose the result through the existing evaluation facade and consumers.

**Tech Stack:** Python 3.12, frozen dataclasses, LangChain `ChatOpenAI`, versioned prompt registry, typed evaluation schemas, JSON, pytest, FastAPI/Gradio compatibility paths, and existing evaluation/ablation tooling.

---

## Design Reference

Implement the approved specification:

`docs/superpowers/specs/2026-06-22-p5a-deepseek-semantic-judge-design.md`

Target release: `v0.5.0-p5a`.

The implementation starts from commit `f13c3c9` on
`codex/p5a-deepseek-semantic-judge`. Before Task 1:

```bash
git status --short --branch
.venv/bin/python -m pytest -q
```

Expected:

- branch is `codex/p5a-deepseek-semantic-judge`
- only the pre-existing root `.superpowers/` directory is untracked
- baseline suite reports `489 passed`

Do not add, delete, or modify the root `.superpowers/` directory.

## File Map

### New Files

- `evaluation/judge_config.py`: isolated Judge environment loading, validation,
  ChatOpenAI construction, and safe runtime metadata.
- `evaluation/judge_evidence.py`: deterministic bounded evidence and citation
  formatting.
- `evaluation/judge_parsing.py`: strict semantic-Judge JSON validation and score
  normalization.
- `tests/test_evaluation_judge_config.py`
- `tests/test_evaluation_judge_evidence.py`
- `tests/test_evaluation_judge_parsing.py`

### Modified Files

- `prompting/catalog.py`: active `evaluation.semantic_judge@v1` definition.
- `evaluation/schemas.py`: additive Judge result, per-question field, and
  aggregate summary fields.
- `evaluation/judges.py`: `DeepSeekJudge`, configured factory, and sanitized
  failures.
- `evaluation/runners.py`: keep execution and system latency only.
- `evaluation/comparison.py`: Judge orchestration and post-Judge failure
  analysis.
- `evaluation/evaluate.py`: public Judge injection and configured default.
- `evaluation/metrics.py`: Judge aggregation.
- `evaluation/reporting.py`: Judge rows in terminal reports.
- `evaluation/runtime_config.py`: schema `3`, evaluator `p5a`, and safe Judge
  metadata.
- `evaluation/matrix.py`: reuse one configured Judge per matrix run and show
  Judge metrics.
- `experiments/run_ablation.py`: show Judge metrics and cost implications in
  generated reports.
- `.env.example`: disabled-by-default Judge variables.
- `tests/test_prompt_catalog.py`
- `tests/test_evaluation_schemas.py`
- `tests/test_evaluation_judges.py`
- `tests/test_evaluation_runners.py`
- `tests/test_evaluation_comparison.py`
- `tests/test_evaluate.py`
- `tests/test_evaluation_metrics.py`
- `tests/test_evaluation_reporting.py`
- `tests/test_evaluation_matrix.py`
- `tests/test_ablation.py`
- `tests/test_evaluation_storage.py`
- `tests/test_dashboard_service.py`
- `tests/test_fastapi_routes.py`
- `README.md`
- `CHANGELOG.md`
- `docs/github_release_checklist.md`
- `docs/superpowers/specs/2026-06-22-p5a-deepseek-semantic-judge-design.md`:
  approved status.
- `docs/superpowers/plans/2026-06-22-p5a-deepseek-semantic-judge.md`

No Dashboard component, column, chart, toggle, or filter is added in P5a.

## Pinned P5a Prompt Contract

| Prompt ID | Active | Variables | `v1` fingerprint |
|---|---|---|---|
| `evaluation.semantic_judge` | yes | `citations`, `evidence`, `fallback_triggered`, `gold_answer`, `question`, `should_answer`, `system_answer` | `sha256:58c0f2bcecbd34afbf4f30054281daf62a25fff311aefe3c4377b759f1095462` |

The prompt is active even when the Judge is disabled. The active project prompt
manifest therefore contains 9 prompts.

## Stable Result Contract

A completed result serializes this additive structure:

```json
{
  "judge": {
    "status": "completed",
    "scores": {
      "semantic_correctness": 0.75,
      "groundedness": 1.0
    },
    "reason": "Semantic Judge completed.",
    "error": null,
    "raw_scores": {
      "semantic_correctness": 3,
      "groundedness": 4
    },
    "reasons": {
      "semantic_correctness": "The answer matches the reference meaning.",
      "groundedness": "The factual claims are supported by the evidence."
    },
    "model": "deepseek-chat",
    "prompt_id": "evaluation.semantic_judge",
    "prompt_version": "v1",
    "prompt_fingerprint": "sha256:58c0f2bcecbd34afbf4f30054281daf62a25fff311aefe3c4377b759f1095462"
  }
}
```

The existing positional field order of `JudgeResult(status, scores, reason,
error)` remains valid. New fields are appended with defaults.

---

### Task 1: Add Independent Judge Configuration

**Files:**
- Create: `evaluation/judge_config.py`
- Create: `tests/test_evaluation_judge_config.py`

- [ ] **Step 1: Write failing configuration tests**

Create `tests/test_evaluation_judge_config.py`:

```python
from __future__ import annotations

from typing import Any

import pytest

from evaluation.judge_config import (
    EvaluationJudgeSettings,
    build_judge_runtime_metadata,
    create_evaluation_judge_model,
    load_evaluation_judge_settings,
)


def test_judge_is_disabled_by_default_and_does_not_reuse_system_llm_env() -> None:
    settings = load_evaluation_judge_settings(
        {
            "OPENAI_API_KEY": "system-secret",
            "OPENAI_BASE_URL": "https://system.example/v1",
            "OPENAI_MODEL": "system-model",
        }
    )

    assert settings == EvaluationJudgeSettings(
        enabled=False,
        api_key="",
        base_url="",
        model="",
        temperature=0.0,
    )


def test_disabled_judge_ignores_invalid_unused_temperature() -> None:
    settings = load_evaluation_judge_settings(
        {
            "EVALUATION_JUDGE_ENABLED": "false",
            "EVALUATION_JUDGE_TEMPERATURE": "not-a-number",
        }
    )

    assert settings.enabled is False
    assert settings.temperature == 0.0


@pytest.mark.parametrize(
    ("missing_name", "environment"),
    [
        (
            "EVALUATION_JUDGE_API_KEY",
            {
                "EVALUATION_JUDGE_ENABLED": "true",
                "EVALUATION_JUDGE_BASE_URL": "https://judge.example/v1",
                "EVALUATION_JUDGE_MODEL": "deepseek-chat",
            },
        ),
        (
            "EVALUATION_JUDGE_BASE_URL",
            {
                "EVALUATION_JUDGE_ENABLED": "true",
                "EVALUATION_JUDGE_API_KEY": "judge-secret",
                "EVALUATION_JUDGE_MODEL": "deepseek-chat",
            },
        ),
        (
            "EVALUATION_JUDGE_MODEL",
            {
                "EVALUATION_JUDGE_ENABLED": "true",
                "EVALUATION_JUDGE_API_KEY": "judge-secret",
                "EVALUATION_JUDGE_BASE_URL": "https://judge.example/v1",
            },
        ),
    ],
)
def test_enabled_judge_requires_independent_fields(
    missing_name: str,
    environment: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match=missing_name):
        load_evaluation_judge_settings(environment)


@pytest.mark.parametrize("value", ["-0.1", "2.1", "not-a-number"])
def test_enabled_judge_rejects_invalid_temperature(value: str) -> None:
    environment = {
        "EVALUATION_JUDGE_ENABLED": "true",
        "EVALUATION_JUDGE_API_KEY": "judge-secret",
        "EVALUATION_JUDGE_BASE_URL": "https://judge.example/v1",
        "EVALUATION_JUDGE_MODEL": "deepseek-chat",
        "EVALUATION_JUDGE_TEMPERATURE": value,
    }

    with pytest.raises(ValueError, match="EVALUATION_JUDGE_TEMPERATURE"):
        load_evaluation_judge_settings(environment)


def test_disabled_judge_does_not_construct_client() -> None:
    calls: list[dict[str, Any]] = []

    def client_factory(**kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    with pytest.raises(RuntimeError, match="disabled"):
        create_evaluation_judge_model(
            EvaluationJudgeSettings(
                enabled=False,
                api_key="",
                base_url="",
                model="",
                temperature=0.0,
            ),
            client_factory=client_factory,
        )

    assert calls == []


def test_enabled_judge_constructs_independent_openai_compatible_client() -> None:
    calls: list[dict[str, Any]] = []
    sentinel = object()

    def client_factory(**kwargs: Any) -> object:
        calls.append(kwargs)
        return sentinel

    settings = EvaluationJudgeSettings(
        enabled=True,
        api_key="judge-secret",
        base_url="https://judge.example/v1",
        model="deepseek-chat",
        temperature=0.0,
    )

    model = create_evaluation_judge_model(
        settings,
        client_factory=client_factory,
    )

    assert model is sentinel
    assert calls == [
        {
            "model": "deepseek-chat",
            "api_key": "judge-secret",
            "base_url": "https://judge.example/v1",
            "temperature": 0.0,
        }
    ]


def test_runtime_metadata_excludes_secret_and_base_url() -> None:
    metadata = build_judge_runtime_metadata(
        EvaluationJudgeSettings(
            enabled=True,
            api_key="judge-secret",
            base_url="https://judge.example/v1",
            model="deepseek-chat",
            temperature=0.0,
        )
    )

    assert metadata == {
        "enabled": True,
        "provider": "openai_compatible",
        "model": "deepseek-chat",
        "temperature": 0.0,
    }
    assert "judge-secret" not in repr(metadata)
    assert "judge.example" not in repr(metadata)
```

- [ ] **Step 2: Run tests to verify the module is missing**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_judge_config.py -q
```

Expected: collection fails because `evaluation.judge_config` does not exist.

- [ ] **Step 3: Implement isolated settings and client construction**

Create `evaluation/judge_config.py`:

```python
"""Independent configuration for the optional evaluation Judge."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI


@dataclass(frozen=True)
class EvaluationJudgeSettings:
    enabled: bool = False
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        required = {
            "EVALUATION_JUDGE_API_KEY": self.api_key,
            "EVALUATION_JUDGE_BASE_URL": self.base_url,
            "EVALUATION_JUDGE_MODEL": self.model,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise ValueError(
                f"{', '.join(missing)} must not be empty when "
                "EVALUATION_JUDGE_ENABLED is true"
            )
        if not 0 <= self.temperature <= 2:
            raise ValueError(
                "EVALUATION_JUDGE_TEMPERATURE must be between 0 and 2"
            )


def load_evaluation_judge_settings(
    environ: Mapping[str, str] | None = None,
) -> EvaluationJudgeSettings:
    source = os.environ if environ is None else environ
    enabled = _parse_bool(
        source.get("EVALUATION_JUDGE_ENABLED"),
        default=False,
    )
    temperature = (
        _parse_float(
            source.get("EVALUATION_JUDGE_TEMPERATURE"),
            default=0.0,
        )
        if enabled
        else 0.0
    )
    return EvaluationJudgeSettings(
        enabled=enabled,
        api_key=source.get("EVALUATION_JUDGE_API_KEY", "").strip(),
        base_url=source.get("EVALUATION_JUDGE_BASE_URL", "").strip(),
        model=source.get("EVALUATION_JUDGE_MODEL", "").strip(),
        temperature=temperature,
    )


def create_evaluation_judge_model(
    settings: EvaluationJudgeSettings,
    *,
    client_factory: Callable[..., Any] | None = None,
) -> Any:
    if not settings.enabled:
        raise RuntimeError("Evaluation Judge is disabled")
    factory = client_factory or ChatOpenAI
    return factory(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=settings.temperature,
    )


def build_judge_runtime_metadata(
    settings: EvaluationJudgeSettings,
) -> dict[str, object]:
    return {
        "enabled": settings.enabled,
        "provider": "openai_compatible",
        "model": settings.model or None,
        "temperature": settings.temperature,
    }


def _parse_bool(raw_value: str | None, *, default: bool) -> bool:
    if raw_value is None or not raw_value.strip():
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(
        "EVALUATION_JUDGE_ENABLED must be a boolean, "
        f"got {raw_value!r}"
    )


def _parse_float(raw_value: str | None, *, default: float) -> float:
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return float(raw_value.strip())
    except ValueError as exc:
        raise ValueError(
            "EVALUATION_JUDGE_TEMPERATURE must be a number, "
            f"got {raw_value!r}"
        ) from exc
```

- [ ] **Step 4: Run the focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_judge_config.py -q
```

Expected: all configuration tests pass.

- [ ] **Step 5: Commit**

```bash
git add evaluation/judge_config.py tests/test_evaluation_judge_config.py
git commit -m "feat: add independent evaluation judge config"
```

---

### Task 2: Register The Versioned Semantic Judge Prompt

**Files:**
- Modify: `prompting/catalog.py`
- Modify: `tests/test_prompt_catalog.py`

- [ ] **Step 1: Add the failing prompt identity test**

Extend `EXPECTED_PROMPTS` in `tests/test_prompt_catalog.py`:

```python
    "evaluation.semantic_judge": (
        (
            "citations",
            "evidence",
            "fallback_triggered",
            "gold_answer",
            "question",
            "should_answer",
            "system_answer",
        ),
        "sha256:58c0f2bcecbd34afbf4f30054281daf62a25fff311aefe3c4377b759f1095462",
    ),
```

Add `"evaluation.semantic_judge"` to `ACTIVE_PROMPT_IDS`, then add:

```python
def test_semantic_judge_prompt_renders_strict_json_contract() -> None:
    rendered = render_prompt(
        "evaluation.semantic_judge",
        question="What is RAG?",
        gold_answer="Retrieval-augmented generation.",
        should_answer="true",
        fallback_triggered="false",
        system_answer="RAG uses retrieved evidence.",
        citations="[]",
        evidence="[]",
    )

    assert "semantic_correctness" in rendered
    assert "groundedness" in rendered
    assert "integer from 0 through 4" in rendered
    assert '"applicable": true' in rendered
    assert "Question:\nWhat is RAG?" in rendered
    assert rendered.endswith("JSON:")
```

- [ ] **Step 2: Run the prompt catalog tests to verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_prompt_catalog.py -q
```

Expected: failure because `evaluation.semantic_judge@v1` is not registered.

- [ ] **Step 3: Add the exact immutable prompt**

Add this constant to `prompting/catalog.py`:

```python
SEMANTIC_JUDGE_V1 = """You are an evaluation judge for a private document QA system.

Evaluate the system result on two independent dimensions:
1. semantic_correctness: whether the result has the same essential meaning as the gold answer and follows expected answerability.
2. groundedness: whether every substantive factual statement in a non-fallback answer is supported by the supplied evidence.

Scoring rubric for each applicable dimension:
- 0: completely incorrect or unsupported
- 1: mostly incorrect or mostly unsupported
- 2: partially correct or partially supported
- 3: mostly correct or mostly supported
- 4: fully correct or fully supported

Rules:
- Judge semantic meaning, not exact wording.
- Use the gold answer and should_answer for semantic correctness.
- An appropriate refusal can be semantically correct when should_answer is false.
- An inappropriate refusal is semantically incorrect when should_answer is true.
- Judge groundedness only from the supplied evidence and citation metadata.
- Do not use outside knowledge to award groundedness.
- If fallback_triggered is true, groundedness.applicable must be false and groundedness.score must be null.
- If fallback_triggered is false, groundedness.applicable must be true and groundedness.score must be an integer from 0 through 4.
- Every reason must be a non-empty concise string and must not reproduce long evidence passages.
- Return exactly the JSON object below, with no Markdown fences or surrounding prose:
{{"semantic_correctness": {{"score": 0, "reason": "concise reason"}}, "groundedness": {{"applicable": true, "score": 0, "reason": "concise reason"}}}}

Question:
{question}

Gold answer:
{gold_answer}

Should answer:
{should_answer}

Fallback triggered:
{fallback_triggered}

System answer:
{system_answer}

Citations:
{citations}

Evidence:
{evidence}

JSON:"""
```

Add the definition to `_PROJECT_PROMPT_DEFINITIONS`:

```python
    PromptDefinition(
        prompt_id="evaluation.semantic_judge",
        version="v1",
        template=SEMANTIC_JUDGE_V1,
        description=(
            "Score evaluation answers for semantic correctness and groundedness."
        ),
    ),
```

Add the active version:

```python
        "evaluation.semantic_judge": "v1",
```

- [ ] **Step 4: Run prompt tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_prompt_registry.py \
  tests/test_prompt_catalog.py \
  -q
```

Expected: all prompt tests pass and the active manifest contains 9 entries.

- [ ] **Step 5: Commit**

```bash
git add prompting/catalog.py tests/test_prompt_catalog.py
git commit -m "feat: register semantic judge prompt"
```

---

### Task 3: Add Bounded Judge Evidence Formatting

**Files:**
- Create: `evaluation/judge_evidence.py`
- Create: `tests/test_evaluation_judge_evidence.py`

- [ ] **Step 1: Write failing evidence tests**

Create `tests/test_evaluation_judge_evidence.py`:

```python
from __future__ import annotations

import json
from copy import deepcopy

from evaluation.judge_evidence import (
    MAX_JUDGE_EVIDENCE_CHARS,
    format_judge_citations,
    format_judge_evidence,
    select_judge_evidence,
)
from evaluation.schemas import EvaluationResult


def _result() -> EvaluationResult:
    return EvaluationResult.empty("q001", "single_doc", "What is RAG?")


def test_relevant_documents_take_priority_without_mutation() -> None:
    result = _result()
    result.retrieved_documents = [
        {"source": "retrieved.md", "content": "retrieved"}
    ]
    result.relevant_documents = [
        {
            "source": "/private/docs/relevant.md",
            "page": 2,
            "chunk_id": "chunk-2",
            "content": "  Relevant\n evidence.  ",
            "source_path": "/private/docs/relevant.md",
        }
    ]
    original = deepcopy(result.relevant_documents)

    selected = select_judge_evidence(result)
    payload = json.loads(format_judge_evidence(result))

    assert selected == result.relevant_documents
    assert selected is not result.relevant_documents
    assert payload == [
        {
            "source": "relevant.md",
            "page": 2,
            "chunk_id": "chunk-2",
            "content": "Relevant evidence.",
        }
    ]
    assert result.relevant_documents == original
    assert "source_path" not in payload[0]


def test_retrieved_documents_are_used_when_relevant_documents_are_empty() -> None:
    result = _result()
    result.retrieved_documents = [
        {"source": "retrieved.md", "content": "retrieved evidence"}
    ]

    payload = json.loads(format_judge_evidence(result))

    assert payload[0]["source"] == "retrieved.md"
    assert payload[0]["content"] == "retrieved evidence"


def test_evidence_is_limited_to_eight_chunks_and_1200_characters() -> None:
    result = _result()
    result.relevant_documents = [
        {
            "source": f"doc-{index}.md",
            "chunk_id": f"chunk-{index}",
            "content": "x" * (MAX_JUDGE_EVIDENCE_CHARS + 50),
        }
        for index in range(10)
    ]

    payload = json.loads(format_judge_evidence(result))

    assert len(payload) == 8
    assert [item["chunk_id"] for item in payload] == [
        f"chunk-{index}" for index in range(8)
    ]
    assert all(
        len(item["content"]) == MAX_JUDGE_EVIDENCE_CHARS
        for item in payload
    )


def test_citations_are_metadata_only_and_bounded() -> None:
    result = _result()
    result.citations = [
        {
            "source": r"C:\private\notes.md",
            "page": 3,
            "chunk_id": "chunk-3",
            "snippet": "  cited\n text  ",
        }
    ]

    payload = json.loads(format_judge_citations(result))

    assert payload == [
        {
            "source": "notes.md",
            "page": 3,
            "chunk_id": "chunk-3",
            "snippet": "cited text",
        }
    ]
    assert "private" not in format_judge_citations(result)
```

- [ ] **Step 2: Run tests to verify the module is missing**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_judge_evidence.py -q
```

Expected: collection fails because `evaluation.judge_evidence` does not exist.

- [ ] **Step 3: Implement pure bounded formatting**

Create `evaluation/judge_evidence.py`:

```python
"""Bounded prompt inputs for semantic evaluation judging."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from evaluation.schemas import EvaluationResult


MAX_JUDGE_EVIDENCE_CHUNKS = 8
MAX_JUDGE_EVIDENCE_CHARS = 1200


def select_judge_evidence(
    result: EvaluationResult,
) -> list[dict[str, Any]]:
    source = (
        result.relevant_documents
        if result.relevant_documents
        else result.retrieved_documents
    )
    return deepcopy(list(source[:MAX_JUDGE_EVIDENCE_CHUNKS]))


def format_judge_evidence(result: EvaluationResult) -> str:
    records = [
        _format_evidence_record(record)
        for record in select_judge_evidence(result)
    ]
    return json.dumps(
        records,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def format_judge_citations(result: EvaluationResult) -> str:
    records = [
        _format_citation_record(record)
        for record in result.citations[:MAX_JUDGE_EVIDENCE_CHUNKS]
    ]
    return json.dumps(
        records,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _format_evidence_record(record: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "content": _bounded_text(record.get("content")),
    }
    _copy_metadata(payload, record)
    return _ordered_record(payload, content_key="content")


def _format_citation_record(record: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "snippet": _bounded_text(record.get("snippet")),
    }
    _copy_metadata(payload, record)
    return _ordered_record(payload, content_key="snippet")


def _copy_metadata(
    payload: dict[str, Any],
    record: dict[str, Any],
) -> None:
    source = _safe_source(record.get("source"))
    if source is not None:
        payload["source"] = source
    page = record.get("page")
    if isinstance(page, int) and not isinstance(page, bool):
        payload["page"] = page
    chunk_id = record.get("chunk_id")
    if isinstance(chunk_id, str) and chunk_id.strip():
        payload["chunk_id"] = chunk_id.strip()


def _ordered_record(
    payload: dict[str, Any],
    *,
    content_key: str,
) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in ("source", "page", "chunk_id", content_key)
        if key in payload
    }


def _safe_source(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("\\", "/")
    return normalized.rsplit("/", 1)[-1]


def _bounded_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = " ".join(value.split())
    return normalized[:MAX_JUDGE_EVIDENCE_CHARS]
```

- [ ] **Step 4: Run focused evidence tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_judge_evidence.py -q
```

Expected: all evidence tests pass.

- [ ] **Step 5: Commit**

```bash
git add evaluation/judge_evidence.py tests/test_evaluation_judge_evidence.py
git commit -m "feat: bound semantic judge evidence"
```

---

### Task 4: Add Strict Semantic Judge Parsing

**Files:**
- Create: `evaluation/judge_parsing.py`
- Create: `tests/test_evaluation_judge_parsing.py`

- [ ] **Step 1: Write failing parser tests**

Create `tests/test_evaluation_judge_parsing.py`:

```python
from __future__ import annotations

import json

import pytest

from evaluation.judge_parsing import parse_semantic_judge_response


def _payload(
    *,
    semantic_score: object = 3,
    semantic_reason: object = "Meaning matches.",
    applicable: object = True,
    groundedness_score: object = 4,
    groundedness_reason: object = "Evidence supports the answer.",
) -> str:
    return json.dumps(
        {
            "semantic_correctness": {
                "score": semantic_score,
                "reason": semantic_reason,
            },
            "groundedness": {
                "applicable": applicable,
                "score": groundedness_score,
                "reason": groundedness_reason,
            },
        }
    )


def test_parser_accepts_normal_answer_and_normalizes_scores() -> None:
    parsed = parse_semantic_judge_response(
        _payload(),
        fallback_triggered=False,
    )

    assert parsed.raw_scores == {
        "semantic_correctness": 3,
        "groundedness": 4,
    }
    assert parsed.scores == {
        "semantic_correctness": 0.75,
        "groundedness": 1.0,
    }
    assert parsed.reasons["semantic_correctness"] == "Meaning matches."


def test_parser_accepts_fallback_with_unavailable_groundedness() -> None:
    parsed = parse_semantic_judge_response(
        _payload(
            semantic_score=4,
            applicable=False,
            groundedness_score=None,
            groundedness_reason="No substantive answer was provided.",
        ),
        fallback_triggered=True,
    )

    assert parsed.raw_scores["groundedness"] is None
    assert parsed.scores["groundedness"] is None


@pytest.mark.parametrize(
    "semantic_score",
    [True, 2.5, "3", -1, 5],
)
def test_parser_rejects_invalid_semantic_score_types_and_ranges(
    semantic_score: object,
) -> None:
    with pytest.raises(ValueError, match="semantic_correctness.score"):
        parse_semantic_judge_response(
            _payload(semantic_score=semantic_score),
            fallback_triggered=False,
        )


@pytest.mark.parametrize(
    ("raw_text", "error_pattern"),
    [
        ("", "blank"),
        ("```json\n{}\n```", "valid JSON"),
        ('prefix {"semantic_correctness": {}}', "valid JSON"),
        (
            json.dumps(
                {
                    "semantic_correctness": {
                        "score": 3,
                        "reason": "ok",
                    },
                    "groundedness": {
                        "applicable": True,
                        "score": 4,
                        "reason": "ok",
                    },
                    "extra": {},
                }
            ),
            "top-level keys",
        ),
    ],
)
def test_parser_rejects_non_contract_responses(
    raw_text: str,
    error_pattern: str,
) -> None:
    with pytest.raises(ValueError, match=error_pattern):
        parse_semantic_judge_response(
            raw_text,
            fallback_triggered=False,
        )


def test_parser_rejects_applicability_mismatch() -> None:
    with pytest.raises(ValueError, match="groundedness.applicable"):
        parse_semantic_judge_response(
            _payload(
                applicable=False,
                groundedness_score=None,
            ),
            fallback_triggered=False,
        )


def test_parser_rejects_fallback_with_non_null_groundedness_score() -> None:
    with pytest.raises(ValueError, match="groundedness.score"):
        parse_semantic_judge_response(
            _payload(
                applicable=False,
                groundedness_score=0,
            ),
            fallback_triggered=True,
        )


@pytest.mark.parametrize(
    "groundedness_score",
    [False, 1.5, "4", -1, 5],
)
def test_parser_rejects_invalid_groundedness_scores(
    groundedness_score: object,
) -> None:
    with pytest.raises(ValueError, match="groundedness.score"):
        parse_semantic_judge_response(
            _payload(groundedness_score=groundedness_score),
            fallback_triggered=False,
        )


def test_parser_rejects_missing_or_extra_nested_keys() -> None:
    missing = json.loads(_payload())
    missing["groundedness"].pop("reason")
    extra = json.loads(_payload())
    extra["semantic_correctness"]["confidence"] = 0.9

    for payload in (missing, extra):
        with pytest.raises(ValueError, match="keys"):
            parse_semantic_judge_response(
                json.dumps(payload),
                fallback_triggered=False,
            )


@pytest.mark.parametrize(
    ("semantic_reason", "groundedness_reason"),
    [
        ("", "grounded"),
        ("semantic", "   "),
        (1, "grounded"),
    ],
)
def test_parser_rejects_blank_or_non_string_reasons(
    semantic_reason: object,
    groundedness_reason: object,
) -> None:
    with pytest.raises(ValueError, match="reason"):
        parse_semantic_judge_response(
            _payload(
                semantic_reason=semantic_reason,
                groundedness_reason=groundedness_reason,
            ),
            fallback_triggered=False,
        )
```

- [ ] **Step 2: Run tests to verify the module is missing**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_judge_parsing.py -q
```

Expected: collection fails because `evaluation.judge_parsing` does not exist.

- [ ] **Step 3: Implement exact-key parsing and normalization**

Create `evaluation/judge_parsing.py`:

```python
"""Strict parser for the versioned semantic Judge response."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedJudgeResult:
    raw_scores: dict[str, int | None]
    scores: dict[str, float | None]
    reasons: dict[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_scores", deepcopy(self.raw_scores))
        object.__setattr__(self, "scores", deepcopy(self.scores))
        object.__setattr__(self, "reasons", deepcopy(self.reasons))


def parse_semantic_judge_response(
    raw_text: str,
    *,
    fallback_triggered: bool,
) -> ParsedJudgeResult:
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("Judge response must not be blank")
    try:
        payload = json.loads(raw_text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("Judge response must be valid JSON only") from exc
    if not isinstance(payload, dict):
        raise ValueError("Judge response must be a JSON object")
    _require_exact_keys(
        payload,
        {"semantic_correctness", "groundedness"},
        "top-level keys",
    )

    semantic = _require_object(
        payload["semantic_correctness"],
        "semantic_correctness",
    )
    groundedness = _require_object(
        payload["groundedness"],
        "groundedness",
    )
    _require_exact_keys(
        semantic,
        {"score", "reason"},
        "semantic_correctness keys",
    )
    _require_exact_keys(
        groundedness,
        {"applicable", "score", "reason"},
        "groundedness keys",
    )

    semantic_score = _require_score(
        semantic["score"],
        "semantic_correctness.score",
    )
    semantic_reason = _require_reason(
        semantic["reason"],
        "semantic_correctness.reason",
    )

    applicable = groundedness["applicable"]
    if type(applicable) is not bool:
        raise ValueError("groundedness.applicable must be a boolean")
    expected_applicable = not fallback_triggered
    if applicable is not expected_applicable:
        raise ValueError(
            "groundedness.applicable contradicts fallback_triggered"
        )
    if applicable:
        groundedness_score: int | None = _require_score(
            groundedness["score"],
            "groundedness.score",
        )
    else:
        if groundedness["score"] is not None:
            raise ValueError(
                "groundedness.score must be null when not applicable"
            )
        groundedness_score = None
    groundedness_reason = _require_reason(
        groundedness["reason"],
        "groundedness.reason",
    )

    return ParsedJudgeResult(
        raw_scores={
            "semantic_correctness": semantic_score,
            "groundedness": groundedness_score,
        },
        scores={
            "semantic_correctness": semantic_score / 4,
            "groundedness": (
                None
                if groundedness_score is None
                else groundedness_score / 4
            ),
        },
        reasons={
            "semantic_correctness": semantic_reason,
            "groundedness": groundedness_reason,
        },
    )


def _require_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return value


def _require_exact_keys(
    payload: dict[str, Any],
    expected: set[str],
    field_name: str,
) -> None:
    if set(payload) != expected:
        raise ValueError(
            f"Judge response {field_name} must be exactly {sorted(expected)}"
        )


def _require_score(value: Any, field_name: str) -> int:
    if type(value) is not int or not 0 <= value <= 4:
        raise ValueError(f"{field_name} must be an integer from 0 through 4")
    return value


def _require_reason(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()
```

- [ ] **Step 4: Run focused parser tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_judge_parsing.py -q
```

Expected: all parser tests pass.

- [ ] **Step 5: Commit**

```bash
git add evaluation/judge_parsing.py tests/test_evaluation_judge_parsing.py
git commit -m "feat: parse semantic judge responses strictly"
```

---

### Task 5: Expand Evaluation Schemas Compatibly

**Files:**
- Modify: `evaluation/schemas.py`
- Modify: `tests/test_evaluation_schemas.py`

- [ ] **Step 1: Add failing schema compatibility tests**

Append to `tests/test_evaluation_schemas.py`:

```python
def test_judge_result_preserves_old_factories_and_adds_metadata() -> None:
    old_completed = JudgeResult.completed(
        {"semantic_correctness": 0.8},
        reason="Legacy reason.",
    )
    old_failed = JudgeResult.failed("RuntimeError: unavailable")
    direct = JudgeResult(
        status="completed",
        scores={"semantic_correctness": 0.8},
    )
    enriched = JudgeResult.completed(
        {
            "semantic_correctness": 0.75,
            "groundedness": 1.0,
        },
        reason="Semantic Judge completed.",
        raw_scores={
            "semantic_correctness": 3,
            "groundedness": 4,
        },
        reasons={
            "semantic_correctness": "Meaning matches.",
            "groundedness": "Evidence supports the answer.",
        },
        model="deepseek-chat",
        prompt_id="evaluation.semantic_judge",
        prompt_version="v1",
        prompt_fingerprint="sha256:test",
    )

    assert old_completed.scores == {"semantic_correctness": 0.8}
    assert old_failed.error == "RuntimeError: unavailable"
    assert direct.status == "completed"
    assert enriched.raw_scores["groundedness"] == 4
    assert enriched.model == "deepseek-chat"
    assert enriched.prompt_id == "evaluation.semantic_judge"


def test_result_defaults_judge_to_disabled_and_accepts_nested_dict() -> None:
    default_result = EvaluationResult.empty(
        "q001",
        "single_doc",
        "What is RAG?",
    )
    restored = EvaluationResult.from_compat_dict(
        {
            "question_id": "q001",
            "question_type": "single_doc",
            "question": "What is RAG?",
            "judge": {
                "status": "completed",
                "scores": {"semantic_correctness": 0.75},
                "raw_scores": {"semantic_correctness": 3},
                "reasons": {"semantic_correctness": "Meaning matches."},
                "model": "deepseek-chat",
            },
        }
    )

    assert default_result.judge.status == "disabled"
    assert restored.judge.status == "completed"
    assert restored.judge.raw_scores == {"semantic_correctness": 3}
    assert restored.to_dict()["judge"]["model"] == "deepseek-chat"


def test_summary_defaults_unavailable_judge_metrics_to_none() -> None:
    payload = EvaluationSummary.empty().to_dict()

    assert payload["judge_completed_count"] == 0
    assert payload["judge_failed_count"] == 0
    assert payload["judge_completion_rate"] is None
    assert payload["average_semantic_correctness"] is None
    assert payload["average_groundedness"] is None
    assert payload["groundedness_applicable_count"] == 0


def test_judge_result_defensively_copies_all_nested_maps() -> None:
    raw_scores = {"semantic_correctness": 3}
    scores = {"semantic_correctness": 0.75}
    reasons = {"semantic_correctness": "Meaning matches."}
    result = JudgeResult(
        status="completed",
        scores=scores,
        raw_scores=raw_scores,
        reasons=reasons,
    )

    raw_scores["semantic_correctness"] = 0
    scores["semantic_correctness"] = 0.0
    reasons["semantic_correctness"] = "mutated"

    assert result.raw_scores == {"semantic_correctness": 3}
    assert result.scores == {"semantic_correctness": 0.75}
    assert result.reasons == {
        "semantic_correctness": "Meaning matches."
    }
```

Update the exact key set in
`test_empty_result_covers_current_single_question_result_shape()` to include
`"judge"`.

- [ ] **Step 2: Run schema tests to verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_schemas.py -q
```

Expected: failures for missing Judge metadata, `EvaluationResult.judge`, and
summary fields.

- [ ] **Step 3: Move and expand `JudgeResult` without breaking old call forms**

Move `JudgeResult` from the bottom of `evaluation/schemas.py` to immediately
after `EvaluationQuestion` and before `EvaluationResult`, then replace it with:

```python
@dataclass(frozen=True)
class JudgeResult:
    status: JudgeStatus
    scores: dict[str, float | None] = field(default_factory=dict)
    reason: str = ""
    error: str | None = None
    raw_scores: dict[str, int | None] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    model: str | None = None
    prompt_id: str | None = None
    prompt_version: str | None = None
    prompt_fingerprint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", deepcopy(self.scores))
        object.__setattr__(self, "raw_scores", deepcopy(self.raw_scores))
        object.__setattr__(self, "reasons", deepcopy(self.reasons))

    @classmethod
    def disabled(cls) -> JudgeResult:
        return cls(status="disabled")

    @classmethod
    def completed(
        cls,
        scores: dict[str, float | None],
        reason: str = "",
        *,
        raw_scores: dict[str, int | None] | None = None,
        reasons: dict[str, str] | None = None,
        model: str | None = None,
        prompt_id: str | None = None,
        prompt_version: str | None = None,
        prompt_fingerprint: str | None = None,
    ) -> JudgeResult:
        return cls(
            status="completed",
            scores=dict(scores),
            reason=reason,
            raw_scores=dict(raw_scores or {}),
            reasons=dict(reasons or {}),
            model=model,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            prompt_fingerprint=prompt_fingerprint,
        )

    @classmethod
    def failed(
        cls,
        error: str,
        *,
        model: str | None = None,
        prompt_id: str | None = None,
        prompt_version: str | None = None,
        prompt_fingerprint: str | None = None,
    ) -> JudgeResult:
        return cls(
            status="failed",
            error=error,
            model=model,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            prompt_fingerprint=prompt_fingerprint,
        )

    @classmethod
    def from_compat_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> JudgeResult:
        allowed_fields = {item.name for item in fields(cls)}
        values = {
            key: deepcopy(value)
            for key, value in payload.items()
            if key in allowed_fields
        }
        values.setdefault("status", "disabled")
        return cls(**values)
```

- [ ] **Step 4: Add Judge fields to result and summary records**

Add to `EvaluationResult`:

```python
    judge: JudgeResult = field(default_factory=JudgeResult.disabled)
```

Add this conversion to `EvaluationResult.__post_init__()` before copying the
other nested fields:

```python
        if isinstance(self.judge, Mapping):
            self.judge = JudgeResult.from_compat_dict(self.judge)
        elif not isinstance(self.judge, JudgeResult):
            raise ValueError("judge must be a JudgeResult or mapping")
```

Add to `EvaluationSummary`:

```python
    judge_completed_count: int = 0
    judge_failed_count: int = 0
    judge_completion_rate: float | None = None
    average_semantic_correctness: float | None = None
    average_groundedness: float | None = None
    groundedness_applicable_count: int = 0
```

- [ ] **Step 5: Run schema and Judge contract tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_evaluation_schemas.py \
  tests/test_evaluation_judges.py \
  -q
```

Expected: all existing and new compatibility tests pass.

- [ ] **Step 6: Commit**

```bash
git add evaluation/schemas.py tests/test_evaluation_schemas.py
git commit -m "feat: add semantic judge result schemas"
```

---

### Task 6: Implement `DeepSeekJudge`

**Files:**
- Modify: `evaluation/judges.py`
- Modify: `tests/test_evaluation_judges.py`

- [ ] **Step 1: Add failing fake-LLM and factory tests**

Extend `tests/test_evaluation_judges.py` with:

```python
import json
from typing import Any

from evaluation.judge_config import EvaluationJudgeSettings
from evaluation.judges import (
    DeepSeekJudge,
    build_configured_judge,
)


class FakeLLM:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> Any:
        self.prompts.append(prompt)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _completed_response(
    *,
    applicable: bool = True,
    groundedness_score: int | None = 4,
) -> str:
    return json.dumps(
        {
            "semantic_correctness": {
                "score": 3,
                "reason": "Meaning matches.",
            },
            "groundedness": {
                "applicable": applicable,
                "score": groundedness_score,
                "reason": "Groundedness assessed.",
            },
        }
    )


def test_deepseek_judge_calls_llm_once_and_records_prompt_metadata() -> None:
    llm = FakeLLM(_completed_response())
    result = _result()
    result.answer = "RAG uses retrieved evidence."
    result.citations = [{"source": "notes.md", "chunk_id": "chunk-1"}]
    result.relevant_documents = [
        {
            "source": "notes.md",
            "chunk_id": "chunk-1",
            "content": "RAG uses retrieved evidence.",
        }
    ]
    judge = DeepSeekJudge(
        llm,
        model="deepseek-chat",
        api_key="judge-secret",
    )

    judged = judge.evaluate(_question(), result)

    assert len(llm.prompts) == 1
    assert "Question:\nWhat is RAG?" in llm.prompts[0]
    assert "Gold answer:\nRetrieval augmented generation." in llm.prompts[0]
    assert '"chunk_id":"chunk-1"' in llm.prompts[0]
    assert judged.status == "completed"
    assert judged.raw_scores == {
        "semantic_correctness": 3,
        "groundedness": 4,
    }
    assert judged.scores == {
        "semantic_correctness": 0.75,
        "groundedness": 1.0,
    }
    assert judged.model == "deepseek-chat"
    assert judged.prompt_id == "evaluation.semantic_judge"
    assert judged.prompt_version == "v1"
    assert judged.prompt_fingerprint == (
        "sha256:58c0f2bcecbd34afbf4f30054281daf62a25fff311aefe3c4377b759f1095462"
    )


def test_deepseek_judge_accepts_message_content_and_fallback_null_score() -> None:
    class Message:
        content = _completed_response(
            applicable=False,
            groundedness_score=None,
        )

    llm = FakeLLM(Message())
    result = _result()
    result.answer = "I cannot answer from the current documents."
    result.fallback_triggered = True

    judged = DeepSeekJudge(
        llm,
        model="deepseek-chat",
        api_key="judge-secret",
    ).evaluate(_question(), result)

    assert judged.status == "completed"
    assert judged.raw_scores["groundedness"] is None
    assert judged.scores["groundedness"] is None


def test_deepseek_judge_isolates_errors_and_redacts_credentials() -> None:
    llm = FakeLLM(
        RuntimeError(
            "Bearer token-123 failed for api_key=judge-secret\n"
            "with Authorization: secret-value"
        )
    )
    judged = DeepSeekJudge(
        llm,
        model="deepseek-chat",
        api_key="judge-secret",
    ).evaluate(_question(), _result())

    assert judged.status == "failed"
    assert judged.model == "deepseek-chat"
    assert judged.prompt_id == "evaluation.semantic_judge"
    assert "judge-secret" not in str(judged.error)
    assert "token-123" not in str(judged.error)
    assert "secret-value" not in str(judged.error)
    assert len(str(judged.error)) <= 500


def test_deepseek_judge_marks_malformed_response_failed() -> None:
    judged = DeepSeekJudge(
        FakeLLM('{"semantic_correctness": {}}'),
        model="deepseek-chat",
        api_key="judge-secret",
    ).evaluate(_question(), _result())

    assert judged.status == "failed"
    assert judged.scores == {}
    assert judged.raw_scores == {}
    assert judged.reasons == {}


def test_configured_judge_does_not_construct_model_when_disabled() -> None:
    calls: list[EvaluationJudgeSettings] = []

    def model_factory(settings: EvaluationJudgeSettings) -> object:
        calls.append(settings)
        return object()

    judge = build_configured_judge(
        settings=EvaluationJudgeSettings(),
        model_factory=model_factory,
    )

    assert isinstance(judge, DisabledJudge)
    assert calls == []


def test_configured_judge_constructs_deepseek_judge_when_enabled() -> None:
    llm = FakeLLM(_completed_response())
    settings = EvaluationJudgeSettings(
        enabled=True,
        api_key="judge-secret",
        base_url="https://judge.example/v1",
        model="deepseek-chat",
        temperature=0.0,
    )

    judge = build_configured_judge(
        settings=settings,
        model_factory=lambda received: llm,
    )

    assert isinstance(judge, DeepSeekJudge)
    assert judge.evaluate(_question(), _result()).status == "completed"
```

- [ ] **Step 2: Run Judge tests to verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_judges.py -q
```

Expected: failures because `DeepSeekJudge` and `build_configured_judge` do not
exist.

- [ ] **Step 3: Implement the Judge and sanitized error boundary**

Replace `evaluation/judges.py` with:

```python
"""Optional semantic evaluation Judge implementations."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Protocol

from evaluation.judge_config import (
    EvaluationJudgeSettings,
    create_evaluation_judge_model,
    load_evaluation_judge_settings,
)
from evaluation.judge_evidence import (
    format_judge_citations,
    format_judge_evidence,
)
from evaluation.judge_parsing import parse_semantic_judge_response
from evaluation.schemas import EvaluationQuestion, EvaluationResult, JudgeResult
from prompting import get_prompt_definition, render_prompt
from tools.base import coerce_llm_text


SEMANTIC_JUDGE_PROMPT_ID = "evaluation.semantic_judge"
SEMANTIC_JUDGE_PROMPT_VERSION = "v1"
MAX_JUDGE_ERROR_CHARS = 500


class Judge(Protocol):
    def evaluate(
        self,
        question: EvaluationQuestion,
        result: EvaluationResult,
    ) -> JudgeResult:
        ...


class JudgeLLM(Protocol):
    def invoke(self, prompt: str) -> Any:
        ...


class DisabledJudge:
    def evaluate(
        self,
        question: EvaluationQuestion,
        result: EvaluationResult,
    ) -> JudgeResult:
        return JudgeResult.disabled()


class DeepSeekJudge:
    def __init__(
        self,
        llm: JudgeLLM,
        *,
        model: str,
        api_key: str,
    ) -> None:
        self._llm = llm
        self._model = model
        self._api_key = api_key
        self._prompt = get_prompt_definition(
            SEMANTIC_JUDGE_PROMPT_ID,
            version=SEMANTIC_JUDGE_PROMPT_VERSION,
        )

    def evaluate(
        self,
        question: EvaluationQuestion,
        result: EvaluationResult,
    ) -> JudgeResult:
        try:
            prompt = render_prompt(
                self._prompt.prompt_id,
                version=self._prompt.version,
                question=question.question,
                gold_answer=question.gold_answer,
                should_answer=str(question.answerable).lower(),
                fallback_triggered=str(result.fallback_triggered).lower(),
                system_answer=result.answer,
                citations=format_judge_citations(result),
                evidence=format_judge_evidence(result),
            )
            raw_text = coerce_llm_text(self._llm.invoke(prompt))
            parsed = parse_semantic_judge_response(
                raw_text,
                fallback_triggered=result.fallback_triggered,
            )
            return JudgeResult.completed(
                parsed.scores,
                reason="Semantic Judge completed.",
                raw_scores=parsed.raw_scores,
                reasons=parsed.reasons,
                model=self._model,
                prompt_id=self._prompt.prompt_id,
                prompt_version=self._prompt.version,
                prompt_fingerprint=self._prompt.fingerprint,
            )
        except Exception as exc:
            return JudgeResult.failed(
                sanitize_judge_error(exc, api_key=self._api_key),
                model=self._model,
                prompt_id=self._prompt.prompt_id,
                prompt_version=self._prompt.version,
                prompt_fingerprint=self._prompt.fingerprint,
            )


def build_configured_judge(
    settings: EvaluationJudgeSettings | None = None,
    *,
    model_factory: Callable[[EvaluationJudgeSettings], Any] | None = None,
) -> Judge:
    resolved = settings or load_evaluation_judge_settings()
    if not resolved.enabled:
        return DisabledJudge()
    factory = model_factory or create_evaluation_judge_model
    return DeepSeekJudge(
        factory(resolved),
        model=resolved.model,
        api_key=resolved.api_key,
    )


def invoke_judge(
    judge: Judge,
    question: EvaluationQuestion,
    result: EvaluationResult,
) -> JudgeResult:
    try:
        return judge.evaluate(question, result)
    except Exception as exc:
        return JudgeResult.failed(sanitize_judge_error(exc))


def sanitize_judge_error(
    exc: Exception,
    *,
    api_key: str = "",
) -> str:
    message = str(exc)
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    message = re.sub(
        r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+",
        "Bearer [REDACTED]",
        message,
    )
    message = re.sub(
        r"(?i)\b(api[_ -]?key|authorization)\s*[:=]\s*\S+",
        r"\1=[REDACTED]",
        message,
    )
    message = " ".join(message.split())
    formatted = type(exc).__name__
    if message:
        formatted = f"{formatted}: {message}"
    return formatted[:MAX_JUDGE_ERROR_CHARS]


__all__ = [
    "DeepSeekJudge",
    "DisabledJudge",
    "Judge",
    "build_configured_judge",
    "invoke_judge",
    "sanitize_judge_error",
]
```

- [ ] **Step 4: Run Judge, parser, evidence, and prompt tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_evaluation_judges.py \
  tests/test_evaluation_judge_parsing.py \
  tests/test_evaluation_judge_evidence.py \
  tests/test_prompt_catalog.py \
  -q
```

Expected: all focused tests pass, with no network access.

- [ ] **Step 5: Commit**

```bash
git add evaluation/judges.py tests/test_evaluation_judges.py
git commit -m "feat: implement deepseek semantic judge"
```

---

### Task 7: Integrate Judge Invocation Into Evaluation Orchestration

**Files:**
- Modify: `evaluation/runners.py`
- Modify: `evaluation/comparison.py`
- Modify: `evaluation/evaluate.py`
- Modify: `evaluation/matrix.py`
- Modify: `tests/test_evaluation_runners.py`
- Modify: `tests/test_evaluation_comparison.py`
- Modify: `tests/test_evaluate.py`
- Modify: `tests/test_evaluation_matrix.py`

- [ ] **Step 1: Add failing orchestration tests**

In `tests/test_evaluation_comparison.py`, add a recording Judge:

```python
from evaluation.schemas import EvaluationQuestion, JudgeResult


class RecordingJudge:
    def __init__(self, result: JudgeResult | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.result = result or JudgeResult.completed(
            {
                "semantic_correctness": 0.75,
                "groundedness": 1.0,
            }
        )

    def evaluate(
        self,
        question: EvaluationQuestion,
        result: EvaluationResult,
    ) -> JudgeResult:
        self.calls.append((question.id, result.answer))
        return self.result
```

Add these tests:

```python
def test_single_system_invokes_injected_judge_before_failure_analysis() -> None:
    questions = normalize_questions([{"id": "q001", "question": "Q?"}])
    judge = RecordingJudge()
    runner = StaticRunner(
        {"Q?": {"answer": "A"}},
        [],
        "agentic",
    )
    timer_values = iter([0.0, 0.1])

    report = evaluate_single_system(
        questions,
        runner,
        timer=lambda: next(timer_values),
        judge=judge,
    )

    assert judge.calls == [("q001", "A")]
    assert report.results[0].judge.status == "completed"
    assert report.results[0].failure_analysis["failure_type"] == "no_failure"


def test_failed_judge_preserves_deterministic_result_fields() -> None:
    questions = normalize_questions(
        [
            {
                "id": "q001",
                "question": "Q?",
                "expected_keywords": ["answer"],
                "expected_sources": ["notes.md"],
            }
        ]
    )
    judge = RecordingJudge(
        JudgeResult.failed("RuntimeError: judge unavailable")
    )
    runner = StaticRunner(
        {
            "Q?": {
                "answer": "answer",
                "citations": [{"source": "notes.md"}],
            }
        },
        [],
        "agentic",
    )
    timer_values = iter([0.0, 0.1])

    report = evaluate_single_system(
        questions,
        runner,
        timer=lambda: next(timer_values),
        judge=judge,
    )

    result = report.results[0]
    assert result.judge.status == "failed"
    assert result.answer_returned is True
    assert result.keyword_hit is True
    assert result.source_hit is True
    assert result.failure_analysis["failure_type"] == "no_failure"


def test_comparison_invokes_judge_once_per_system_result() -> None:
    questions = normalize_questions([{"id": "q001", "question": "Q?"}])
    judge = RecordingJudge()
    timer_values = iter([0.0, 0.1, 0.1, 0.2])

    report = evaluate_comparison(
        questions,
        naive_runner=StaticRunner({"Q?": {"answer": "naive"}}, [], "naive"),
        agentic_runner=StaticRunner(
            {"Q?": {"answer": "agentic"}},
            [],
            "agentic",
        ),
        timer=lambda: next(timer_values),
        judge=judge,
    )

    assert judge.calls == [
        ("q001", "naive"),
        ("q001", "agentic"),
    ]
    assert report.results[0].naive.judge.status == "completed"
    assert report.results[0].agentic.judge.status == "completed"


def test_system_error_skips_judge_and_records_local_failure() -> None:
    questions = normalize_questions([{"id": "q001", "question": "Q?"}])
    judge = RecordingJudge()

    class BrokenRunner:
        def run(
            self,
            question: str,
            chat_history: list[ChatMessage],
        ) -> dict[str, object]:
            raise RuntimeError("offline")

    timer_values = iter([0.0, 0.1])
    report = evaluate_single_system(
        questions,
        BrokenRunner(),
        timer=lambda: next(timer_values),
        judge=judge,
    )

    assert judge.calls == []
    assert report.results[0].judge.status == "failed"
    assert report.results[0].judge.error == (
        "SystemResultUnavailable: system execution failed; Judge was not invoked"
    )
    assert report.results[0].error == "RuntimeError: offline"
    assert report.results[0].failure_analysis["failure_type"] == "tool_failure"
```

In `tests/test_evaluate.py`, add:

```python
class FacadeRecordingJudge:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def evaluate(self, question, result):
        self.calls.append(question.id)
        return JudgeResult.completed(
            {"semantic_correctness": 0.75, "groundedness": 1.0}
        )


def test_evaluate_questions_uses_injected_judge_without_building_default(
    monkeypatch,
) -> None:
    def fail_builder():
        raise AssertionError("configured Judge builder must not run")

    monkeypatch.setattr(evaluator, "build_configured_judge", fail_builder)
    judge = FacadeRecordingJudge()

    report = evaluator.evaluate_questions(
        [{"id": "q001", "question": "Q?"}],
        run_agent_fn=lambda question: {"answer": "A"},
        timer=iter([0.0, 0.1]).__next__,
        judge=judge,
    )

    assert report["results"][0]["judge"]["status"] == "completed"


def test_evaluate_questions_builds_configured_judge_when_omitted(
    monkeypatch,
) -> None:
    judge = FacadeRecordingJudge()
    calls: list[str] = []

    def builder():
        calls.append("build")
        return judge

    monkeypatch.setattr(evaluator, "build_configured_judge", builder)
    report = evaluator.evaluate_questions(
        [{"id": "q001", "question": "Q?"}],
        run_agent_fn=lambda question: {"answer": "A"},
        timer=iter([0.0, 0.1]).__next__,
    )

    assert calls == ["build"]
    assert report["results"][0]["judge"]["status"] == "completed"
```

Use the module import already present in the file or add:

```python
import evaluation.evaluate as evaluator
from evaluation.schemas import JudgeResult
```

- [ ] **Step 2: Update runner tests for the new layer boundary**

Rename the runner tests to:

```python
def test_evaluate_question_records_runner_errors_and_latency() -> None:
def test_evaluate_question_scores_successful_runner_output_without_postprocessing() -> None:
```

Change their final assertions so `evaluate_question()` returns:

```python
    assert result.failure_analysis == {}
```

Failure analysis is verified in comparison/orchestration tests after Judge
attachment. Also add to the existing typed single-system test:

```python
    assert report.results[0].judge.status == "disabled"
```

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_evaluation_runners.py \
  tests/test_evaluation_comparison.py \
  tests/test_evaluate.py \
  -q
```

Expected: failures for missing `judge` parameters and because failure analysis
still runs inside the runner.

- [ ] **Step 3: Keep `evaluate_question()` limited to system execution**

In `evaluation/runners.py`, remove `attach_failure_analysis` from the imports
and replace the final return:

```python
    result.latency = latency
    result.error = error
    return result
```

This preserves `latency` as system latency and does not measure Judge time.

- [ ] **Step 4: Add orchestration finalization**

In `evaluation/comparison.py`, import:

```python
from evaluation.judges import DisabledJudge, Judge, invoke_judge
from evaluation.metrics import attach_failure_analysis, summarize_results
from evaluation.schemas import EvaluationResult, JudgeResult
```

Add:

```python
SYSTEM_RESULT_UNAVAILABLE_ERROR = (
    "SystemResultUnavailable: system execution failed; Judge was not invoked"
)


def _finalize_result(
    question: EvaluationQuestion,
    result: EvaluationResult,
    judge: Judge,
) -> EvaluationResult:
    if result.error:
        result.judge = JudgeResult.failed(SYSTEM_RESULT_UNAVAILABLE_ERROR)
    else:
        result.judge = invoke_judge(judge, question, result)
    return attach_failure_analysis(question, result)
```

Add `judge: Judge | None = None` to both typed orchestration functions. Resolve
it once per call:

```python
    resolved_judge = judge if judge is not None else DisabledJudge()
```

In `evaluate_single_system()`, finalize each runner result:

```python
    results = [
        _finalize_result(
            question,
            evaluate_question(question, runner, timer),
            resolved_judge,
        )
        for question in questions
    ]
```

In `evaluate_comparison()`, finalize naive and Agentic independently:

```python
        naive_result = _finalize_result(
            question,
            evaluate_question(question, naive_runner, timer),
            resolved_judge,
        )
        agentic_result = _finalize_result(
            question,
            evaluate_question(question, agentic_runner, timer),
            resolved_judge,
        )
```

- [ ] **Step 5: Add Judge injection to the compatibility facade**

In `evaluation/evaluate.py`, import:

```python
from evaluation.judges import Judge, build_configured_judge
```

Add `judge: Judge | None = None` as the final parameter of
`evaluate_questions()` and `evaluate_single_system()`. Resolve it once:

```python
    resolved_judge = judge if judge is not None else build_configured_judge()
```

Pass `judge=resolved_judge` to typed single-system or comparison orchestration.
The complete single-item facade call becomes:

```python
    report = evaluate_typed_single_system(
        [normalize_question(item, 0)],
        CallableRunnerAdapter(runner),
        timer,
        judge=(
            judge
            if judge is not None
            else build_configured_judge()
        ),
    )
```

- [ ] **Step 6: Reuse one Judge for all matrix variants**

In `evaluation/matrix.py`, import `Judge` and `build_configured_judge`. Add
`judge: Judge | None = None` to `evaluate_matrix()` and resolve once:

```python
    resolved_judge = judge if judge is not None else build_configured_judge()
```

Pass it to each facade call:

```python
            result = evaluate_single_system(
                item,
                runner,
                timer,
                judge=resolved_judge,
            )
```

In `tests/test_evaluation_matrix.py`, import:

```python
from evaluation.judges import Judge
from evaluation.schemas import JudgeResult
```

Update the expected public wrapper type hints:

```python
    assert single_hints == {
        "item": dict[str, Any],
        "runner": Callable[[str], dict[str, Any]],
        "timer": Callable[[], float],
        "judge": Judge | None,
        "return": dict[str, Any],
    }
```

Add:

```python
def test_evaluate_matrix_reuses_one_judge_across_variants() -> None:
    matrix = _matrix_module()

    class MatrixJudge:
        def __init__(self) -> None:
            self.answers: list[str] = []

        def evaluate(self, question, result):
            self.answers.append(result.answer)
            return JudgeResult.completed(
                {
                    "semantic_correctness": 1.0,
                    "groundedness": 1.0,
                }
            )

    judge = MatrixJudge()
    report = matrix.evaluate_matrix(
        [{"id": "q001", "question": "Q?"}],
        {
            "naive": lambda question: _system_result("naive"),
            "agentic": lambda question: _system_result("agentic"),
            "agentic_reranker": lambda question: _system_result("reranked"),
        },
        timer=StepTimer(),
        judge=judge,
    )

    assert judge.answers == ["naive", "agentic", "reranked"]
    assert all(
        result["judge"]["status"] == "completed"
        for result in report["results"][0]["systems"].values()
    )
```

- [ ] **Step 7: Run orchestration and consumer tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_evaluation_runners.py \
  tests/test_evaluation_comparison.py \
  tests/test_evaluate.py \
  tests/test_evaluation_matrix.py \
  -q
```

Expected: all tests pass. Existing public calls without a Judge remain valid.

- [ ] **Step 8: Commit**

```bash
git add \
  evaluation/runners.py \
  evaluation/comparison.py \
  evaluation/evaluate.py \
  evaluation/matrix.py \
  tests/test_evaluation_runners.py \
  tests/test_evaluation_comparison.py \
  tests/test_evaluate.py \
  tests/test_evaluation_matrix.py
git commit -m "feat: integrate semantic judge orchestration"
```

---

### Task 8: Aggregate, Report, And Record Judge Metadata

**Files:**
- Modify: `evaluation/metrics.py`
- Modify: `evaluation/comparison.py`
- Modify: `evaluation/reporting.py`
- Modify: `evaluation/runtime_config.py`
- Modify: `evaluation/matrix.py`
- Modify: `experiments/run_ablation.py`
- Modify: `tests/test_evaluation_metrics.py`
- Modify: `tests/test_evaluation_comparison.py`
- Modify: `tests/test_evaluation_reporting.py`
- Modify: `tests/test_evaluation_matrix.py`
- Modify: `tests/test_ablation.py`
- Modify: `tests/test_evaluation_storage.py`
- Modify: `tests/test_dashboard_service.py`
- Modify: `tests/test_fastapi_routes.py`

- [ ] **Step 1: Add failing aggregation tests**

Append to `tests/test_evaluation_metrics.py`:

```python
from evaluation.schemas import JudgeResult


def test_summary_aggregates_completed_failed_and_applicable_scores() -> None:
    questions = [
        normalize_question({"id": "q001", "question": "One?"}, 0),
        normalize_question({"id": "q002", "question": "Two?"}, 1),
        normalize_question({"id": "q003", "question": "Three?"}, 2),
        normalize_question({"id": "q004", "question": "Four?"}, 3),
    ]
    results = [
        EvaluationResult.empty(question.id, question.question_type, question.question)
        for question in questions
    ]
    results[0].judge = JudgeResult.completed(
        {
            "semantic_correctness": 1.0,
            "groundedness": 0.75,
        }
    )
    results[1].judge = JudgeResult.completed(
        {
            "semantic_correctness": 0.5,
            "groundedness": None,
        }
    )
    results[2].judge = JudgeResult.failed("RuntimeError: unavailable")

    summary = summarize_results(results, questions)

    assert summary.judge_completed_count == 2
    assert summary.judge_failed_count == 1
    assert summary.judge_completion_rate == 0.6667
    assert summary.average_semantic_correctness == 0.75
    assert summary.average_groundedness == 0.75
    assert summary.groundedness_applicable_count == 1


def test_disabled_only_summary_keeps_judge_metrics_unavailable() -> None:
    question = normalize_question({"id": "q001", "question": "Q?"}, 0)
    result = EvaluationResult.empty(
        question.id,
        question.question_type,
        question.question,
    )

    summary = summarize_results([result], [question])

    assert summary.judge_completed_count == 0
    assert summary.judge_failed_count == 0
    assert summary.judge_completion_rate is None
    assert summary.average_semantic_correctness is None
    assert summary.average_groundedness is None
    assert summary.groundedness_applicable_count == 0
```

Extend the comparison key assertion in
`tests/test_evaluation_comparison.py` with:

```python
        "naive_average_semantic_correctness",
        "agentic_average_semantic_correctness",
        "naive_average_groundedness",
        "agentic_average_groundedness",
        "naive_judge_completion_rate",
        "agentic_judge_completion_rate",
```

- [ ] **Step 2: Add failing report and metadata tests**

In `tests/test_evaluation_reporting.py`, extend the single summary fixture:

```python
            "judge_completion_rate": 1.0,
            "average_semantic_correctness": 0.75,
            "average_groundedness": 0.5,
            "judge_failed_count": 0,
```

Add:

```python
    assert "judge_completion_rate: 1.0" in rendered
    assert "average_semantic_correctness: 0.75" in rendered
    assert "average_groundedness: 0.5" in rendered
```

Extend the comparison fixture:

```python
            "naive": {
                "source_hit_rate": 0.5,
                "judge_completion_rate": 1.0,
                "average_semantic_correctness": 0.75,
                "average_groundedness": None,
            },
            "agentic": {
                "source_hit_rate": 1.0,
                "judge_completion_rate": 1.0,
                "average_semantic_correctness": 1.0,
                "average_groundedness": 0.5,
            },
```

Add:

```python
    assert "| Judge Completion Rate | 1.0 | 1.0 |" in rendered
    assert "| Semantic Correctness | 0.75 | 1.0 |" in rendered
    assert "| Groundedness | N/A | 0.5 |" in rendered
```

Update `tests/test_ablation.py` runtime expectations:

```python
    assert snapshot["schema_version"] == 3
    assert snapshot["evaluator_version"] == "p5a"
    assert snapshot["judge"] == {
        "enabled": False,
        "provider": "openai_compatible",
        "model": None,
        "temperature": 0.0,
    }
    assert "evaluation.semantic_judge" in snapshot["prompts"]
    assert len(snapshot["prompts"]) == 9
```

Update `tests/test_evaluation_storage.py` schema/evaluator assertions to `3`
and `p5a`.

Add a runtime redaction test:

```python
def test_runtime_snapshot_records_safe_enabled_judge_metadata() -> None:
    from evaluation.judge_config import EvaluationJudgeSettings

    snapshot = build_runtime_config_snapshot(
        judge_settings=EvaluationJudgeSettings(
            enabled=True,
            api_key="judge-secret",
            base_url="https://judge.example/v1",
            model="deepseek-chat",
            temperature=0.0,
        )
    )

    assert snapshot["judge"] == {
        "enabled": True,
        "provider": "openai_compatible",
        "model": "deepseek-chat",
        "temperature": 0.0,
    }
    serialized = json.dumps(snapshot)
    assert "judge-secret" not in serialized
    assert "judge.example" not in serialized
```

- [ ] **Step 3: Run focused tests to verify missing aggregation**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_evaluation_metrics.py \
  tests/test_evaluation_comparison.py \
  tests/test_evaluation_reporting.py \
  tests/test_ablation.py \
  tests/test_evaluation_storage.py \
  -q
```

Expected: failures for missing summary fields, comparison keys, report rows,
runtime schema `3`, and Judge metadata.

- [ ] **Step 4: Aggregate optional Judge values**

In `evaluation/metrics.py`, calculate before constructing
`EvaluationSummary`:

```python
    judge_completed_count = sum(
        1 for result in results if result.judge.status == "completed"
    )
    judge_failed_count = sum(
        1 for result in results if result.judge.status == "failed"
    )
    judge_attempted_count = judge_completed_count + judge_failed_count
    semantic_scores = _judge_scores(results, "semantic_correctness")
    groundedness_scores = _judge_scores(results, "groundedness")
```

Pass these fields to `EvaluationSummary`:

```python
        judge_completed_count=judge_completed_count,
        judge_failed_count=judge_failed_count,
        judge_completion_rate=(
            None
            if judge_attempted_count == 0
            else _rate(judge_completed_count, judge_attempted_count)
        ),
        average_semantic_correctness=_optional_average(semantic_scores),
        average_groundedness=_optional_average(groundedness_scores),
        groundedness_applicable_count=len(groundedness_scores),
```

Add:

```python
def _judge_scores(
    results: list[EvaluationResult],
    dimension: str,
) -> list[float]:
    values: list[float] = []
    for result in results:
        if result.judge.status != "completed":
            continue
        value = result.judge.scores.get(dimension)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
        ):
            continue
        values.append(float(value))
    return values


def _optional_average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)
```

- [ ] **Step 5: Add flattened comparison fields**

Extend `build_comparison_summary()` in `evaluation/comparison.py`:

```python
        "naive_average_semantic_correctness": (
            naive.average_semantic_correctness
        ),
        "agentic_average_semantic_correctness": (
            agentic.average_semantic_correctness
        ),
        "naive_average_groundedness": naive.average_groundedness,
        "agentic_average_groundedness": agentic.average_groundedness,
        "naive_judge_completion_rate": naive.judge_completion_rate,
        "agentic_judge_completion_rate": agentic.judge_completion_rate,
```

- [ ] **Step 6: Render Judge metrics in terminal reports**

The single-system formatter already iterates every summary field, so retain
that behavior. Add these rows to `_format_comparison_report()` in
`evaluation/reporting.py` after fallback correctness:

```python
        (
            f"| Judge Completion Rate | "
            f"{_format_optional(naive.get('judge_completion_rate'))} | "
            f"{_format_optional(agentic.get('judge_completion_rate'))} |"
        ),
        (
            f"| Semantic Correctness | "
            f"{_format_optional(naive.get('average_semantic_correctness'))} | "
            f"{_format_optional(agentic.get('average_semantic_correctness'))} |"
        ),
        (
            f"| Groundedness | "
            f"{_format_optional(naive.get('average_groundedness'))} | "
            f"{_format_optional(agentic.get('average_groundedness'))} |"
        ),
```

Add:

```python
def _format_optional(value: Any) -> Any:
    return "N/A" if value is None else value
```

- [ ] **Step 7: Advance runtime metadata safely**

In `evaluation/runtime_config.py`, import:

```python
from evaluation.judge_config import (
    EvaluationJudgeSettings,
    build_judge_runtime_metadata,
    load_evaluation_judge_settings,
)
```

Set:

```python
EVALUATION_SCHEMA_VERSION = 3
EVALUATOR_VERSION = "p5a"
```

Add `judge_settings: EvaluationJudgeSettings | None = None` to both public
functions and pass it through. In `build_runtime_metadata()` resolve:

```python
    resolved_judge = (
        judge_settings
        if judge_settings is not None
        else load_evaluation_judge_settings()
    )
```

Add to the config dictionary:

```python
            "judge": build_judge_runtime_metadata(resolved_judge),
```

- [ ] **Step 8: Show Judge metrics in matrix and ablation reports**

Add to `MATRIX_METRICS` in `evaluation/matrix.py`:

```python
    ("Semantic Correctness", "average_semantic_correctness"),
    ("Groundedness", "average_groundedness"),
    ("Judge Completion Rate", "judge_completion_rate"),
```

In `experiments/run_ablation.py`, expand the Markdown table with columns
`Semantic Correctness`, `Groundedness`, and `Judge Completion`, then format:

```python
                f"{_format_metric(summary.get('average_semantic_correctness'))} | "
                f"{_format_metric(summary.get('average_groundedness'))} | "
                f"{_format_metric(summary.get('judge_completion_rate'))} | "
```

Add these limitation lines:

```python
            "- Semantic Judge scores are model-based signals and can reflect model bias.",
            "- Enabling the Judge adds one model call per successful system result; comparison and ablation runs multiply latency and cost.",
```

Update the fixed-order expectation in `tests/test_evaluation_matrix.py` by
inserting:

```python
        "| Semantic Correctness | N/A | N/A | N/A |",
        "| Groundedness | N/A | N/A | N/A |",
        "| Judge Completion Rate | N/A | N/A | N/A |",
```

In
`test_format_ablation_report_uses_observed_metrics_and_explicit_limitations()`,
add to the first summary:

```python
                    "average_semantic_correctness": 0.5,
                    "average_groundedness": None,
                    "judge_completion_rate": 1.0,
```

Add to the second summary:

```python
                    "average_semantic_correctness": 0.75,
                    "average_groundedness": 0.8,
                    "judge_completion_rate": 0.9,
```

Add these assertions:

```python
    assert "Semantic Correctness" in report
    assert "Groundedness" in report
    assert "Judge Completion" in report
    assert "model-based signals" in report
    assert "one model call per successful system result" in report
```

- [ ] **Step 9: Verify API and Dashboard inherit additive raw fields without UI changes**

In the parameterized
`test_run_quick_evaluation_dispatches_runners_and_builds_summary_rows()` test
in `tests/test_dashboard_service.py`, add:

```python
    raw_report = view["raw_report"]
    if system_mode == "comparison":
        assert raw_report["summary"]["naive"]["judge_completion_rate"] is None
        assert raw_report["summary"]["agentic"]["judge_completion_rate"] is None
        assert all(
            result["naive"]["judge"]["status"] == "disabled"
            and result["agentic"]["judge"]["status"] == "disabled"
            for result in raw_report["results"]
        )
    else:
        assert raw_report["summary"]["judge_completion_rate"] is None
        assert all(
            result["judge"]["status"] == "disabled"
            for result in raw_report["results"]
        )
```

In `tests/test_fastapi_routes.py`, change both fake evaluation summaries to:

```python
            "summary": {
                "total_questions": 1,
                "judge_completion_rate": 1.0,
            },
```

Then change the route assertion to:

```python
    assert run_response.json()["summary"] == {
        "total_questions": 1,
        "judge_completion_rate": 1.0,
    }
```

After the existing `get_response = client.get(...)` line, add:

```python
    assert get_response.json()["summary"] == {
        "total_questions": 1,
        "judge_completion_rate": 1.0,
    }
```

Do not modify `evaluation/dashboard_models.py`,
`evaluation/dashboard_formatters.py`, or `ui/gradio_app.py`.

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_evaluation_metrics.py \
  tests/test_evaluation_comparison.py \
  tests/test_evaluation_reporting.py \
  tests/test_evaluation_matrix.py \
  tests/test_ablation.py \
  tests/test_evaluation_storage.py \
  tests/test_dashboard_service.py \
  tests/test_fastapi_routes.py \
  -q
```

Expected: all aggregation, report, metadata, and direct-consumer tests pass.

- [ ] **Step 10: Commit**

```bash
git add \
  evaluation/metrics.py \
  evaluation/comparison.py \
  evaluation/reporting.py \
  evaluation/runtime_config.py \
  evaluation/matrix.py \
  experiments/run_ablation.py \
  tests/test_evaluation_metrics.py \
  tests/test_evaluation_comparison.py \
  tests/test_evaluation_reporting.py \
  tests/test_evaluation_matrix.py \
  tests/test_ablation.py \
  tests/test_evaluation_storage.py \
  tests/test_dashboard_service.py \
  tests/test_fastapi_routes.py
git commit -m "feat: report semantic judge metrics"
```

---

### Task 9: Document, Verify, Review, And Prepare P5a Integration

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/github_release_checklist.md`
- Modify: `docs/superpowers/plans/2026-06-22-p5a-deepseek-semantic-judge.md`

- [ ] **Step 1: Add the disabled-by-default environment contract**

Add to `.env.example` after the primary chat LLM settings:

```text
# Optional independent semantic evaluation Judge.
# Enabling this adds one Judge model call per successful system result.
EVALUATION_JUDGE_ENABLED=false
EVALUATION_JUDGE_API_KEY=
EVALUATION_JUDGE_BASE_URL=
EVALUATION_JUDGE_MODEL=
EVALUATION_JUDGE_TEMPERATURE=0
```

- [ ] **Step 2: Document behavior, cost, and limitations**

Add a `DeepSeek Semantic Judge` section to `README.md` after the versioned
prompt section. It must state:

- deterministic metrics remain unchanged and remain independent signals
- the Judge has no fallback to the evaluated system model settings
- disabled mode creates no Judge client and makes no Judge calls
- semantic correctness and groundedness use raw `0–4` and normalized `0–1`
  values
- fallback results receive semantic correctness but groundedness is `null`
- evidence uses relevant documents first, then retrieved documents, with 8
  chunks and 1,200 characters per chunk
- one successful result creates one Judge call; comparison creates two calls
  per question
- Judge failures are isolated and do not remove deterministic metrics
- LLM-as-a-Judge can be biased and is not a human ground truth
- P5a adds no Dashboard UI; raw reports carry the fields

Update the metric list with:

```text
- `judge_completion_rate`
- `average_semantic_correctness`
- `average_groundedness`
- `judge_failed_count`
```

Move P5a into `Completed Work` and leave the next route:

1. P5b SQLite historical evaluation and trend dashboard
2. Background Evaluation
3. Trace Drill-down

- [ ] **Step 3: Update changelog and release checklist**

Prepend to `CHANGELOG.md`:

```markdown
## v0.5.0-p5a - DeepSeek Semantic Judge

Date: 2026-06-22

### Added

- Added an optional independently configured OpenAI-compatible DeepSeek Judge
  for semantic correctness and groundedness.
- Added strict `0-4` Judge scoring, normalized `0-1` metrics, bounded evidence,
  versioned prompt metadata, and isolated per-result failures.
- Added Judge summaries to JSON artifacts, terminal reports, matrix output, and
  ablation output.

### Changed

- Advanced evaluation artifact metadata to schema version `3` and evaluator
  version `p5a`.
- Moved failure analysis after optional Judge invocation while preserving
  system-only latency semantics.

### Notes

- The Judge is disabled by default and never reuses the evaluated system's LLM
  configuration.
- Enabling it adds one model call per successful system result and can increase
  latency and cost substantially for comparison and ablation runs.
- Judge scores are model-based signals, not human ground truth.

### Verification

- Full test suite: record the observed passing count from Step 7.
- Focused Judge and evaluation tests: record the observed passing count from
  Step 7.
- CLI compatibility smoke tests: record the observed passing count from Step 7.
- Ablation, matrix, Dashboard, and FastAPI compatibility tests: record the
  observed passing count from Step 7.
```

Update `docs/github_release_checklist.md` to:

- version `v0.5.0-p5a`
- schema `3`, evaluator `p5a`
- 11 registered prompts and 9 active prompts
- include Judge-focused verification commands
- include Judge call-cost and bias scope notes
- keep tag creation after integration and explicit user approval

- [ ] **Step 4: Run static safety checks**

Run:

```bash
.venv/bin/python -m compileall \
  prompting agent rag api evaluation experiments baseline tools observability
rg -n \
  'EVALUATION_JUDGE_API_KEY|api_key|base_url|rendered_prompt|raw_response' \
  evaluation/runtime_config.py evaluation/schemas.py
rg -n \
  'evaluation\\.semantic_judge|EVALUATION_JUDGE_' \
  README.md .env.example prompting evaluation tests
git diff --check
```

Expected:

- compile succeeds
- runtime metadata contains no Judge API key, base URL, rendered prompt, or raw
  response field
- the prompt and environment contract are documented and tested
- `git diff --check` reports no whitespace errors

- [ ] **Step 5: Run focused Judge verification**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_evaluation_judge_config.py \
  tests/test_evaluation_judge_evidence.py \
  tests/test_evaluation_judge_parsing.py \
  tests/test_evaluation_judges.py \
  tests/test_evaluation_schemas.py \
  tests/test_evaluation_runners.py \
  tests/test_evaluation_metrics.py \
  tests/test_evaluation_comparison.py \
  tests/test_evaluation_reporting.py \
  tests/test_prompt_registry.py \
  tests/test_prompt_catalog.py \
  -q
```

Expected: all focused Judge, schema, orchestration, metric, report, and prompt
tests pass with no live network call.

- [ ] **Step 6: Run direct-consumer and CLI verification**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_evaluate.py::test_main_prints_report_with_injected_runner \
  tests/test_evaluate.py::test_main_writes_comparison_artifacts \
  tests/test_evaluate.py::test_main_writes_single_system_agentic_artifact_schema \
  -q
.venv/bin/python -m pytest \
  tests/test_ablation.py \
  tests/test_evaluation_matrix.py \
  tests/test_dashboard_service.py \
  tests/test_fastapi_routes.py \
  -q
```

Expected: both commands pass. Record exact observed counts in the changelog and
release checklist.

- [ ] **Step 7: Run the full project suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass. Record the exact observed count; do not retain the
P4d `489 passed` count after new tests are added.

- [ ] **Step 8: Inspect scope and secret safety**

Run:

```bash
git status --short
git diff --stat f13c3c9...HEAD
git log --oneline --decorate f13c3c9..HEAD
rg -n \
  'judge-secret|system-secret|Bearer token-123|secret-value' \
  README.md CHANGELOG.md docs evaluation prompting .env.example \
  -g '!docs/superpowers/plans/2026-06-22-p5a-deepseek-semantic-judge.md'
```

Expected:

- only P5a implementation, tests, and owned documentation changed
- root `.superpowers/` remains untracked and unstaged
- no test credential appears in production code or public documentation
- commits remain separated by configuration, prompt, evidence, parser, schema,
  Judge implementation, orchestration, metrics/metadata, and docs

- [ ] **Step 9: Mark the plan complete and commit release documentation**

Mark completed checkboxes and record observed verification output in this plan.
Then run:

```bash
git add \
  .env.example \
  README.md \
  CHANGELOG.md \
  docs/github_release_checklist.md
git add -f \
  docs/superpowers/plans/2026-06-22-p5a-deepseek-semantic-judge.md
git commit -m "docs: publish p5a semantic judge"
```

- [ ] **Step 10: Request independent code review**

Invoke `superpowers:requesting-code-review` against
`codex/p5a-deepseek-semantic-judge`. Address confirmed specification,
compatibility, secret-safety, or test findings with focused commits. Rerun
affected focused tests and the full suite after fixes.

- [ ] **Step 11: Finish the development branch**

After fresh verification, invoke `superpowers:finishing-a-development-branch`.
Offer merge, pull request, keep, or cleanup choices. Create tag
`v0.5.0-p5a` only after:

- the user explicitly chooses integration
- integration into updated `main` succeeds
- merged `main` passes the full suite
- the final integration record is committed

## Final Verification Matrix

| Concern | Verification |
|---|---|
| Independent settings and no system-model fallback | `tests/test_evaluation_judge_config.py` |
| Prompt ID, variables, fingerprint, active manifest | `tests/test_prompt_catalog.py` |
| Relevant/retrieved evidence selection and bounds | `tests/test_evaluation_judge_evidence.py` |
| Strict JSON, score, reason, and fallback parsing | `tests/test_evaluation_judge_parsing.py` |
| DeepSeek call count, metadata, coercion, redaction | `tests/test_evaluation_judges.py` |
| Old and new schema compatibility | `tests/test_evaluation_schemas.py` |
| Runner latency and layer boundary | `tests/test_evaluation_runners.py` |
| Single/comparison Judge orchestration | `tests/test_evaluation_comparison.py` |
| Public facade and CLI compatibility | `tests/test_evaluate.py` |
| Completion rate and score averages | `tests/test_evaluation_metrics.py` |
| Terminal report labels | `tests/test_evaluation_reporting.py` |
| Schema `3`, evaluator `p5a`, safe metadata | `tests/test_ablation.py`, `tests/test_evaluation_storage.py` |
| Matrix Judge reuse and metrics | `tests/test_evaluation_matrix.py` |
| Dashboard raw-report inheritance without UI expansion | `tests/test_dashboard_service.py` |
| FastAPI summary inheritance | `tests/test_fastapi_routes.py` |
| Full regression | `.venv/bin/python -m pytest -q` |
