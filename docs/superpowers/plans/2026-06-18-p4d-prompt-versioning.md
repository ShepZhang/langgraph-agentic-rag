# P4d Prompt Versioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add immutable, code-native prompt versions and fingerprints, route all current LLM prompts through the registry without text changes, and record the active prompt manifest in evaluation artifacts and Agent traces.

**Architecture:** Add a dependency-free `prompting` package containing the prompt domain model, registry, catalog, and facade. Keep `agent.prompts` as a compatibility facade for existing constants and formatting helpers; runtime callers render by stable prompt ID. Evaluation runtime metadata and trace records receive a safe copied manifest containing only prompt IDs, versions, and SHA-256 fingerprints.

**Tech Stack:** Python 3.12, frozen dataclasses, `string.Formatter`, `hashlib`, pytest, existing LangGraph Agent, evaluation runtime metadata, and JSONL traces.

---

## Design Reference

Implement the approved specification:

`docs/superpowers/specs/2026-06-18-p4d-prompt-versioning-design.md`

Target release: `v0.4.3-p4d`.

The implementation starts from commit `8877f01` on
`codex/p4d-prompt-versioning`. Before Task 1:

```bash
git status --short --branch
.venv/bin/python -m pytest -q
```

Expected:

- branch is `codex/p4d-prompt-versioning`
- only the pre-existing root `.superpowers/` directory is untracked
- baseline suite reports `469 passed`

Do not add, delete, or modify `.superpowers/`.

## File Map

### New Files

- `prompting/__init__.py`: stable project prompt facade.
- `prompting/registry.py`: immutable prompt definitions, strict rendering,
  fingerprints, active lookup, and safe manifests.
- `prompting/catalog.py`: exact `v1` templates and active-version mapping.
- `tests/test_prompt_registry.py`: generic registry contract tests.
- `tests/test_prompt_catalog.py`: project prompt identity and regression tests.

### Modified Files

- `agent/prompts.py`: compatibility constants backed by the registry; retain
  formatting helpers.
- `agent/query_transform.py`: render the initial query-transform prompt through
  the registry.
- `agent/nodes.py`: render retry, grading, answer, claim extraction, and answer
  revision prompts through the registry.
- `baseline/naive_rag.py`: render the shared answer-generation prompt.
- `tools/citation_verifier_tool.py`: render the citation-verification prompt.
- `tools/document_summary_tool.py`: render the document-summary prompt.
- `evaluation/runtime_config.py`: schema `2`, evaluator `p4d`, active prompt
  manifest.
- `observability/trace.py`: defensively store a prompt manifest per recorder.
- `agent/graph.py`: snapshot the active manifest when trace recording starts.
- `tests/test_query_transform.py`
- `tests/test_agent_nodes.py`
- `tests/test_baselines.py`
- `tests/test_citation_verifier_tool.py`
- `tests/test_document_summary_tool.py`
- `tests/test_ablation.py`
- `tests/test_evaluation_storage.py`
- `tests/test_evaluate.py`
- `tests/test_trace_logging.py`
- `README.md`
- `CHANGELOG.md`
- `docs/github_release_checklist.md`
- `docs/superpowers/plans/2026-06-14-p4c-modular-evaluation-framework.md`
- `docs/superpowers/plans/2026-06-18-p4d-prompt-versioning.md`

## Pinned P4d Prompt Contract

| Prompt ID | Active | Variables | `v1` fingerprint |
|---|---|---|---|
| `agent.query_transform` | yes | `chat_history`, `question` | `sha256:24a29bac995a196aa1315c7515832e2cf5c14d4f0d8eacb53f5ec584a057e849` |
| `agent.retry_query_rewrite` | yes | `current_query`, `documents`, `grading_reason`, `partial_relevance_context`, `previous_queries`, `question` | `sha256:d80156327d23940a7e97c64409a08a1cce5b5e6ceb165dcc445ae42b0041363b` |
| `agent.retrieval_grading` | yes | `current_query`, `documents`, `question` | `sha256:78c000ae92d4d549b8ef63800bd473df34adcbca88d52347648c531df086e004` |
| `agent.answer_generation` | yes | `current_query`, `documents`, `question` | `sha256:ef456ee86f56b4b61d2908d077d2d89f7b24995a18bd5ceb0240d86b3f370922` |
| `agent.claim_extraction` | yes | `answer`, `documents`, `question` | `sha256:d249a0daf94d425fa2476efc4f2adabd55a42402bcf83f3bf1e869cb8e167eb2` |
| `agent.citation_verification` | yes | `answer`, `claims`, `documents`, `question` | `sha256:8c833a798f37b875561e16279e655bf393764cdebd42b2198f7f369ba1044eac` |
| `agent.answer_revision` | yes | `answer`, `documents`, `question`, `unsupported_claims` | `sha256:253006a5843c9a514299aec08520f2ea4f2ec57735f161dd6485d7a2bd81329a` |
| `tool.document_summary` | yes | `content`, `max_points`, `title` | `sha256:2c37f768a5fa0d1f01b5d7445c2fd04fd756f4f1c87549706b629216866ecde9` |
| `agent.query_rewrite` | no | `chat_history`, `question` | `sha256:3ccba60fa7582f1b319e3b07422ad569d212af7f49a7eb2a1c98872c5f32f4c7` |
| `agent.claim_verification` | no | `answer`, `documents`, `question` | `sha256:cc44c443a01e6f672dcacb2e9003f22a809742b54ee5b762d222ee17f4f00c17` |

---

### Task 1: Add The Generic Prompt Registry

**Files:**
- Create: `prompting/registry.py`
- Create: `tests/test_prompt_registry.py`

- [x] **Step 1: Write the failing registry contract tests**

Create `tests/test_prompt_registry.py`:

```python
from __future__ import annotations

import pytest

from prompting.registry import PromptDefinition, PromptRegistry


def definition(
    prompt_id: str = "agent.example",
    version: str = "v1",
    template: str = "Hello {name}",
) -> PromptDefinition:
    return PromptDefinition(
        prompt_id=prompt_id,
        version=version,
        template=template,
        description="Example prompt.",
    )


def test_registry_gets_active_definition_and_renders_strict_variables():
    registry = PromptRegistry(
        [definition()],
        active_versions={"agent.example": "v1"},
    )

    resolved = registry.get("agent.example")

    assert resolved.version == "v1"
    assert resolved.variables == ("name",)
    assert registry.render(
        "agent.example",
        variables={"name": "RAG"},
    ) == "Hello RAG"


def test_registry_can_get_an_explicit_historical_version():
    registry = PromptRegistry(
        [
            definition(version="v1", template="Hello {name}"),
            definition(version="v2", template="Hi {name}"),
        ],
        active_versions={"agent.example": "v2"},
    )

    assert registry.get("agent.example").template == "Hi {name}"
    assert registry.get("agent.example", version="v1").template == "Hello {name}"


def test_registry_rejects_duplicate_prompt_versions():
    with pytest.raises(ValueError, match="duplicate prompt definition"):
        PromptRegistry(
            [definition(), definition()],
            active_versions={"agent.example": "v1"},
        )


def test_prompt_definition_rejects_invalid_ids_versions_and_empty_templates():
    invalid_cases = [
        {"prompt_id": "Agent.Example", "version": "v1", "template": "Hello"},
        {"prompt_id": "agent", "version": "v1", "template": "Hello"},
        {"prompt_id": "agent.example", "version": "1", "template": "Hello"},
        {"prompt_id": "agent.example", "version": "v0", "template": "Hello"},
        {"prompt_id": "agent.example", "version": "v1", "template": ""},
    ]

    for values in invalid_cases:
        with pytest.raises(ValueError):
            PromptDefinition(
                description="Invalid prompt.",
                **values,
            )


def test_prompt_definition_rejects_malformed_or_advanced_format_fields():
    invalid_templates = [
        "Hello {name",
        "Hello {user.name}",
        "Hello {items[0]}",
        "Hello {name!r}",
        "Hello {name:>10}",
    ]

    for template in invalid_templates:
        with pytest.raises(ValueError, match="template"):
            definition(template=template)


def test_registry_rejects_invalid_active_version_mappings():
    with pytest.raises(ValueError, match="active prompt"):
        PromptRegistry(
            [definition()],
            active_versions={"agent.missing": "v1"},
        )

    with pytest.raises(ValueError, match="active prompt"):
        PromptRegistry(
            [definition()],
            active_versions={"agent.example": "v2"},
        )


def test_registry_rejects_missing_and_unexpected_render_variables_without_values():
    registry = PromptRegistry(
        [definition(template="Question: {question}\nContext: {context}")],
        active_versions={"agent.example": "v1"},
    )

    with pytest.raises(
        ValueError,
        match=r"missing=\['context'\].*unexpected=\[\]",
    ):
        registry.render(
            "agent.example",
            variables={"question": "What is RAG?"},
        )

    with pytest.raises(
        ValueError,
        match=r"missing=\[\].*unexpected=\['secret'\]",
    ) as exc_info:
        registry.render(
            "agent.example",
            variables={
                "question": "What is RAG?",
                "context": "Private text",
                "secret": "must-not-appear-in-error",
            },
        )
    assert "must-not-appear-in-error" not in str(exc_info.value)


def test_definition_fingerprint_and_literal_json_rendering_are_deterministic():
    prompt = definition(
        template='Return {{"answer": "ok"}} for {name}.',
    )
    registry = PromptRegistry(
        [prompt],
        active_versions={"agent.example": "v1"},
    )

    assert prompt.fingerprint == (
        "sha256:"
        "028f419a8c14cbe0efcb58e5b0f67cbfdd54cdcd2ee4c4ab3a065c5f103eb72e"
    )
    assert registry.render(
        "agent.example",
        variables={"name": "RAG"},
    ) == 'Return {"answer": "ok"} for RAG.'


def test_active_manifest_is_sorted_excludes_inactive_versions_and_is_copied():
    registry = PromptRegistry(
        [
            definition("tool.summary", "v1", "Summarize {content}"),
            definition("agent.example", "v1", "Hello {name}"),
            definition("agent.legacy", "v1", "Legacy {value}"),
        ],
        active_versions={
            "tool.summary": "v1",
            "agent.example": "v1",
        },
    )

    manifest = registry.active_manifest()

    assert list(manifest) == ["agent.example", "tool.summary"]
    assert manifest["agent.example"] == {
        "version": "v1",
        "fingerprint": (
            "sha256:"
            "833583c574131c1ec81313e982643b9f5fba312df50f7c97db0572ad1ce5a929"
        ),
    }
    manifest["agent.example"]["version"] = "mutated"
    assert registry.active_manifest()["agent.example"]["version"] == "v1"
```

- [x] **Step 2: Run the tests and verify the package is missing**

Run:

```bash
.venv/bin/python -m pytest tests/test_prompt_registry.py -q
```

Expected: collection fails with
`ModuleNotFoundError: No module named 'prompting'`.

- [x] **Step 3: Implement the immutable registry**

Create `prompting/registry.py`:

```python
"""Immutable prompt definitions, strict rendering, and safe manifests."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from string import Formatter
from typing import Any


_PROMPT_ID_PATTERN = re.compile(
    r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+"
)
_PROMPT_VERSION_PATTERN = re.compile(r"v[1-9][0-9]*")


@dataclass(frozen=True)
class PromptDefinition:
    """One immutable prompt template version."""

    prompt_id: str
    version: str
    template: str
    description: str
    variables: tuple[str, ...] = field(init=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if _PROMPT_ID_PATTERN.fullmatch(self.prompt_id) is None:
            raise ValueError(f"invalid prompt_id: {self.prompt_id!r}")
        if _PROMPT_VERSION_PATTERN.fullmatch(self.version) is None:
            raise ValueError(f"invalid prompt version: {self.version!r}")
        if not self.template:
            raise ValueError("prompt template must not be empty")

        variables = _parse_template_variables(self.template)
        digest = hashlib.sha256(self.template.encode("utf-8")).hexdigest()
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "fingerprint", f"sha256:{digest}")


class PromptRegistry:
    """Resolve, render, and describe project prompt versions."""

    def __init__(
        self,
        definitions: Iterable[PromptDefinition],
        *,
        active_versions: Mapping[str, str],
    ) -> None:
        registered: dict[tuple[str, str], PromptDefinition] = {}
        for definition in definitions:
            key = (definition.prompt_id, definition.version)
            if key in registered:
                raise ValueError(
                    "duplicate prompt definition: "
                    f"{definition.prompt_id}@{definition.version}"
                )
            registered[key] = definition

        resolved_active_versions = dict(active_versions)
        for prompt_id, version in resolved_active_versions.items():
            if (prompt_id, version) not in registered:
                raise ValueError(
                    f"active prompt {prompt_id}@{version} is not registered"
                )

        self._definitions = registered
        self._active_versions = resolved_active_versions

    def get(
        self,
        prompt_id: str,
        *,
        version: str | None = None,
    ) -> PromptDefinition:
        """Return an exact prompt version or the configured active version."""

        resolved_version = version
        if resolved_version is None:
            resolved_version = self._active_versions.get(prompt_id)
            if resolved_version is None:
                raise KeyError(f"prompt {prompt_id!r} has no active version")

        definition = self._definitions.get((prompt_id, resolved_version))
        if definition is None:
            raise KeyError(
                f"prompt {prompt_id!r} version {resolved_version!r} is not registered"
            )
        return definition

    def render(
        self,
        prompt_id: str,
        *,
        variables: Mapping[str, Any],
        version: str | None = None,
    ) -> str:
        """Render a prompt only when variables exactly match its contract."""

        definition = self.get(prompt_id, version=version)
        supplied = set(variables)
        expected = set(definition.variables)
        missing = sorted(expected - supplied)
        unexpected = sorted(supplied - expected)
        if missing or unexpected:
            raise ValueError(
                f"prompt {definition.prompt_id}@{definition.version} "
                f"variables mismatch: missing={missing}, unexpected={unexpected}"
            )
        return definition.template.format(**dict(variables))

    def active_manifest(self) -> dict[str, dict[str, str]]:
        """Return prompt IDs, versions, and fingerprints without prompt text."""

        manifest = {
            prompt_id: {
                "version": definition.version,
                "fingerprint": definition.fingerprint,
            }
            for prompt_id in sorted(self._active_versions)
            for definition in [self.get(prompt_id)]
        }
        return deepcopy(manifest)


def _parse_template_variables(template: str) -> tuple[str, ...]:
    variables: set[str] = set()
    try:
        parsed = Formatter().parse(template)
        for _literal, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if (
                not field_name
                or "." in field_name
                or "[" in field_name
                or conversion is not None
                or format_spec
            ):
                raise ValueError(
                    "prompt template fields must use simple names without "
                    "conversion or format specifications"
                )
            variables.add(field_name)
    except ValueError as exc:
        raise ValueError(f"invalid prompt template: {exc}") from exc
    return tuple(sorted(variables))
```

- [x] **Step 4: Run the focused registry tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_prompt_registry.py -q
```

Expected: `9 passed`.

- [x] **Step 5: Commit the registry foundation**

```bash
git add prompting/registry.py tests/test_prompt_registry.py
git commit -m "feat: add versioned prompt registry"
```

---

### Task 2: Add The Project Prompt Catalog And Compatibility Facade

**Files:**
- Create: `prompting/catalog.py`
- Create: `prompting/__init__.py`
- Create: `tests/test_prompt_catalog.py`
- Modify: `agent/prompts.py`

- [x] **Step 1: Write the failing project catalog regression tests**

Create `tests/test_prompt_catalog.py`:

```python
from __future__ import annotations

from agent.prompts import (
    ANSWER_GENERATION_PROMPT,
    ANSWER_REVISION_PROMPT,
    CLAIM_EXTRACTION_PROMPT,
    CLAIM_VERIFICATION_PROMPT,
    CITATION_VERIFICATION_PROMPT,
    QUERY_REWRITE_PROMPT,
    RETRIEVAL_GRADING_PROMPT,
    RETRY_QUERY_REWRITE_PROMPT,
)
from prompting import (
    get_active_prompt_manifest,
    get_prompt_definition,
    get_prompt_template,
    render_prompt,
)


EXPECTED_PROMPTS = {
    "agent.query_transform": {
        "active": True,
        "variables": ("chat_history", "question"),
        "fingerprint": "sha256:24a29bac995a196aa1315c7515832e2cf5c14d4f0d8eacb53f5ec584a057e849",
    },
    "agent.retry_query_rewrite": {
        "active": True,
        "variables": (
            "current_query",
            "documents",
            "grading_reason",
            "partial_relevance_context",
            "previous_queries",
            "question",
        ),
        "fingerprint": "sha256:d80156327d23940a7e97c64409a08a1cce5b5e6ceb165dcc445ae42b0041363b",
    },
    "agent.retrieval_grading": {
        "active": True,
        "variables": ("current_query", "documents", "question"),
        "fingerprint": "sha256:78c000ae92d4d549b8ef63800bd473df34adcbca88d52347648c531df086e004",
    },
    "agent.answer_generation": {
        "active": True,
        "variables": ("current_query", "documents", "question"),
        "fingerprint": "sha256:ef456ee86f56b4b61d2908d077d2d89f7b24995a18bd5ceb0240d86b3f370922",
    },
    "agent.claim_extraction": {
        "active": True,
        "variables": ("answer", "documents", "question"),
        "fingerprint": "sha256:d249a0daf94d425fa2476efc4f2adabd55a42402bcf83f3bf1e869cb8e167eb2",
    },
    "agent.citation_verification": {
        "active": True,
        "variables": ("answer", "claims", "documents", "question"),
        "fingerprint": "sha256:8c833a798f37b875561e16279e655bf393764cdebd42b2198f7f369ba1044eac",
    },
    "agent.answer_revision": {
        "active": True,
        "variables": (
            "answer",
            "documents",
            "question",
            "unsupported_claims",
        ),
        "fingerprint": "sha256:253006a5843c9a514299aec08520f2ea4f2ec57735f161dd6485d7a2bd81329a",
    },
    "tool.document_summary": {
        "active": True,
        "variables": ("content", "max_points", "title"),
        "fingerprint": "sha256:2c37f768a5fa0d1f01b5d7445c2fd04fd756f4f1c87549706b629216866ecde9",
    },
    "agent.query_rewrite": {
        "active": False,
        "variables": ("chat_history", "question"),
        "fingerprint": "sha256:3ccba60fa7582f1b319e3b07422ad569d212af7f49a7eb2a1c98872c5f32f4c7",
    },
    "agent.claim_verification": {
        "active": False,
        "variables": ("answer", "documents", "question"),
        "fingerprint": "sha256:cc44c443a01e6f672dcacb2e9003f22a809742b54ee5b762d222ee17f4f00c17",
    },
}


def test_catalog_pins_prompt_ids_versions_variables_and_fingerprints():
    manifest = get_active_prompt_manifest()

    assert set(manifest) == {
        prompt_id
        for prompt_id, contract in EXPECTED_PROMPTS.items()
        if contract["active"]
    }
    for prompt_id, contract in EXPECTED_PROMPTS.items():
        definition = get_prompt_definition(prompt_id, version="v1")
        assert definition.version == "v1"
        assert definition.variables == contract["variables"]
        assert definition.fingerprint == contract["fingerprint"]
        if contract["active"]:
            assert manifest[prompt_id] == {
                "version": "v1",
                "fingerprint": contract["fingerprint"],
            }


def test_agent_prompt_constants_are_exact_registry_compatibility_exports():
    expected_constants = {
        "agent.query_rewrite": QUERY_REWRITE_PROMPT,
        "agent.retry_query_rewrite": RETRY_QUERY_REWRITE_PROMPT,
        "agent.retrieval_grading": RETRIEVAL_GRADING_PROMPT,
        "agent.answer_generation": ANSWER_GENERATION_PROMPT,
        "agent.claim_extraction": CLAIM_EXTRACTION_PROMPT,
        "agent.citation_verification": CITATION_VERIFICATION_PROMPT,
        "agent.answer_revision": ANSWER_REVISION_PROMPT,
        "agent.claim_verification": CLAIM_VERIFICATION_PROMPT,
    }

    for prompt_id, constant in expected_constants.items():
        assert constant == get_prompt_template(prompt_id, version="v1")


def test_query_transform_v1_renders_the_existing_text_contract():
    rendered = render_prompt(
        "agent.query_transform",
        chat_history="user: Discuss Agentic RAG.",
        question="How does it compare?",
    )

    assert rendered.startswith(
        "You transform user questions for private knowledge-base retrieval."
    )
    assert (
        '{"strategy": "rewrite", "rewritten_query": '
        '"standalone retrieval query"'
    ) in rendered
    assert "Chat history:\nuser: Discuss Agentic RAG." in rendered
    assert "Original question:\nHow does it compare?" in rendered


def test_document_summary_v1_renders_the_existing_text_contract():
    rendered = render_prompt(
        "tool.document_summary",
        max_points=2,
        title="Spec Notes",
        content="Grounded answers require retrieved evidence.",
    )

    assert rendered == (
        "Summarize the document using only the supplied text. "
        "Return at most 2 concise bullet points. "
        "Do not add unsupported facts.\n\n"
        "Title: Spec Notes\n\n"
        "Document:\nGrounded answers require retrieved evidence."
    )
```

- [x] **Step 2: Run the tests and verify the catalog facade is missing**

Run:

```bash
.venv/bin/python -m pytest tests/test_prompt_catalog.py -q
```

Expected: collection fails because `prompting` does not yet export the catalog
facade functions.

- [x] **Step 3: Create the exact project catalog**

Create `prompting/catalog.py`. Copy every template exactly, including blank
lines, doubled JSON braces, punctuation, and final labels:

```python
"""Project prompt catalog and active-version mapping."""

from __future__ import annotations

from prompting.registry import PromptDefinition, PromptRegistry


QUERY_REWRITE_V1 = """You are rewriting a user question for private knowledge-base retrieval.

Use the chat history only to resolve references or missing context.
Return one standalone retrieval question.
If the original question is already clear, return it unchanged.

Chat history:
{chat_history}

Original question:
{question}

Standalone retrieval question:"""

QUERY_TRANSFORM_V1 = """You transform user questions for private knowledge-base retrieval.

Use chat history only to resolve references or missing context.
Choose exactly one strategy:
- rewrite: for simple factual questions that only need a standalone retrieval query.
- multi_query: for ambiguous or context-dependent questions that benefit from equivalent or complementary retrieval queries.
- decomposition: for complex comparison or multi-hop questions that benefit from sub-questions.

Return JSON only in this shape:
{{"strategy": "rewrite", "rewritten_query": "standalone retrieval query", "expanded_queries": [], "sub_questions": [], "reason": "short reason"}}

Rules:
- rewritten_query must be a standalone question suitable for retrieval.
- Do not use decomposition for simple factual questions.
- expanded_queries should be empty unless strategy is multi_query.
- sub_questions should be empty unless strategy is decomposition.
- Return JSON only. No markdown fences.

Chat history:
{chat_history}

Original question:
{question}

JSON:"""

RETRY_QUERY_REWRITE_V1 = """You are improving a failed private knowledge-base retrieval query.

The previous retrieval did not find enough relevant evidence. Rewrite the query to improve retrieval.
Avoid repeating the same query. Keep it concise, specific, and search-oriented.

Original question:
{question}

Previous retrieval query:
{current_query}

Previous queries:
{previous_queries}

Previous grading reason:
{grading_reason}

Partial relevance recovery:
{partial_relevance_context}

Previously retrieved chunks:
{documents}

Improved retrieval query:"""

RETRIEVAL_GRADING_V1 = """You are grading whether retrieved chunks can answer a user's original question.

Do not mark chunks relevant just because they share keywords.
Mark them relevant only if they contain enough factual information to answer the original user question.
Return JSON only in this shape:
{{"grades": [{{"document_index": 1, "relevance": "relevant", "confidence": 0.91, "reason": "short reason"}}], "reason": "short overall reason"}}

Rules:
- The retrieval query is only used to explain how the chunks were searched.
- You must grade the retrieved chunks against the original user question.
- Do not grade the chunks as relevant only because they match the retrieval query.
- document_index must use 1-based indexes matching the retrieved chunk numbers.
- relevance must be exactly one of: relevant, partially_relevant, irrelevant.
- Use relevant only when the chunk directly contains enough evidence to answer the original question.
- Use partially_relevant when the chunk is related but does not contain enough evidence to answer.
- Use irrelevant when the chunk does not help answer the original question.
- confidence must be a number between 0 and 1.
- Return JSON only. No markdown fences.

Original user question:
{question}

Retrieval query:
{current_query}

Retrieved chunks:
{documents}

JSON:"""

ANSWER_GENERATION_V1 = """You answer questions using only the retrieved chunks.

Rules:
- You must answer the original user question.
- The retrieval query is provided only to explain how the documents were searched.
- Do not answer the retrieval query as if it were the user's question.
- Use only facts from the retrieved chunks.
- Do not invent facts that are not present in the retrieved chunks.
- For key facts, include citation markers like [1] and [2] that correspond to chunk numbers.
- If the retrieved chunks do not contain the answer, say you cannot answer from the current documents.
- Distinguish workflow cases: weak retrieval evidence can trigger retry rewriting before answer generation; unsupported claims in a generated answer are handled by citation safety fallback, not by retry rewriting.
- Keep the answer concise and useful.
- Return JSON only in this shape:
  {{"answer": "Final answer text with citation markers like [1].", "used_citation_indices": [1]}}
- used_citation_indices must contain only the 1-based chunk numbers actually used as evidence.
- The citation markers in answer must exactly match used_citation_indices.

Original user question:
{question}

Retrieval query:
{current_query}

Retrieved chunks:
{documents}

JSON:"""

CLAIM_EXTRACTION_V1 = """You extract atomic factual claims from a draft answer.

Return JSON only in this shape:
{{"claims": [{{"claim_id": "c001", "claim": "short factual claim", "cited_chunk_ids": ["chunk_id"]}}], "reason": "short reason"}}

Rules:
- Extract only factual claims that need citation support.
- Do not create claims for connective text, hedging, or citation markers by themselves.
- claim_id values must be stable IDs like c001, c002, c003.
- cited_chunk_ids must come only from the selected citation chunks below.
- If the answer contains no factual claims, return an empty claims list.
- Return JSON only. No markdown fences.

Original user question:
{question}

Draft answer:
{answer}

Selected citation chunks:
{documents}

JSON:"""

CITATION_VERIFICATION_V1 = """You verify each extracted claim against its cited chunks.

Return JSON only in this shape:
{{"results": [{{"claim_id": "c001", "claim": "short factual claim", "cited_chunk_ids": ["chunk_id"], "verification_label": "supported", "confidence": 0.91, "reason": "short reason"}}], "reason": "short overall reason"}}

Rules:
- verification_label must be exactly one of: supported, partially_supported, unsupported.
- Use supported only when the cited chunks directly support the claim.
- Use partially_supported when the cited chunks support part of the claim but the claim is too broad or too strong.
- Use unsupported when the cited chunks do not support the claim.
- Do not give credit for vague keyword overlap.
- confidence must be a number between 0 and 1.
- Return JSON only. No markdown fences.

Original user question:
{question}

Draft answer:
{answer}

Extracted claims:
{claims}

Selected citation chunks:
{documents}

JSON:"""

ANSWER_REVISION_V1 = """You revise an answer after claim-level citation verification found unsupported content.

Return JSON only in this shape:
{{"answer": "Revised answer with citation markers like [1].", "used_citation_indices": [1]}}

Rules:
- Remove unsupported claims.
- Narrow partially supported claims to exactly what the cited chunks support.
- Preserve valid citation markers like [1] that refer to selected citation chunks.
- Do not introduce new facts or new citations.
- If no supported answer remains, say you cannot answer from the current documents and return an empty used_citation_indices list.
- used_citation_indices must contain only the 1-based chunk numbers actually used as evidence.
- The citation markers in answer must exactly match used_citation_indices.
- Return JSON only. No markdown fences.

Original user question:
{question}

Current draft answer:
{answer}

Unsupported or partially supported claims:
{unsupported_claims}

Selected citation chunks:
{documents}

JSON:"""

CLAIM_VERIFICATION_V1 = """You are a claim-level citation verifier for private document QA.

Verify whether the answer is fully supported by the selected citation chunks.
Return JSON only in this shape:
{{"verified": true, "claims": [{{"claim": "short factual claim", "supported": true, "citation_indices": [1]}}], "reason": "short reason"}}

Rules:
- Split the answer into factual claims.
- Every factual claim must be supported by at least one selected citation chunk.
- citation_indices must use 1-based indexes matching the selected citation chunk numbers below.
- Mark verified false if any important factual claim is unsupported.
- Do not give credit for vague keyword overlap. Check whether the citation actually supports the claim.
- Return JSON only. No markdown fences.

Original user question:
{question}

Answer to verify:
{answer}

Selected citation chunks:
{documents}

JSON:"""

DOCUMENT_SUMMARY_V1 = """Summarize the document using only the supplied text. Return at most {max_points} concise bullet points. Do not add unsupported facts.

Title: {title}

Document:
{content}"""


PROJECT_PROMPT_REGISTRY = PromptRegistry(
    [
        PromptDefinition(
            prompt_id="agent.query_rewrite",
            version="v1",
            template=QUERY_REWRITE_V1,
            description="Legacy standalone query rewrite prompt.",
        ),
        PromptDefinition(
            prompt_id="agent.query_transform",
            version="v1",
            template=QUERY_TRANSFORM_V1,
            description="Structured query transformation prompt.",
        ),
        PromptDefinition(
            prompt_id="agent.retry_query_rewrite",
            version="v1",
            template=RETRY_QUERY_REWRITE_V1,
            description="Failed-retrieval query refinement prompt.",
        ),
        PromptDefinition(
            prompt_id="agent.retrieval_grading",
            version="v1",
            template=RETRIEVAL_GRADING_V1,
            description="Chunk-level retrieval grading prompt.",
        ),
        PromptDefinition(
            prompt_id="agent.answer_generation",
            version="v1",
            template=ANSWER_GENERATION_V1,
            description="Grounded answer generation prompt.",
        ),
        PromptDefinition(
            prompt_id="agent.claim_extraction",
            version="v1",
            template=CLAIM_EXTRACTION_V1,
            description="Atomic claim extraction prompt.",
        ),
        PromptDefinition(
            prompt_id="agent.citation_verification",
            version="v1",
            template=CITATION_VERIFICATION_V1,
            description="Claim citation verification prompt.",
        ),
        PromptDefinition(
            prompt_id="agent.answer_revision",
            version="v1",
            template=ANSWER_REVISION_V1,
            description="Unsupported-claim answer revision prompt.",
        ),
        PromptDefinition(
            prompt_id="agent.claim_verification",
            version="v1",
            template=CLAIM_VERIFICATION_V1,
            description="Legacy combined claim verification prompt.",
        ),
        PromptDefinition(
            prompt_id="tool.document_summary",
            version="v1",
            template=DOCUMENT_SUMMARY_V1,
            description="Grounded document summary prompt.",
        ),
    ],
    active_versions={
        "agent.query_transform": "v1",
        "agent.retry_query_rewrite": "v1",
        "agent.retrieval_grading": "v1",
        "agent.answer_generation": "v1",
        "agent.claim_extraction": "v1",
        "agent.citation_verification": "v1",
        "agent.answer_revision": "v1",
        "tool.document_summary": "v1",
    },
)
```

- [x] **Step 4: Add the stable prompting facade**

Create `prompting/__init__.py`:

```python
"""Versioned project prompt facade."""

from __future__ import annotations

from typing import Any

from prompting.catalog import PROJECT_PROMPT_REGISTRY
from prompting.registry import PromptDefinition


def get_prompt_definition(
    prompt_id: str,
    *,
    version: str | None = None,
) -> PromptDefinition:
    """Return an exact or active project prompt definition."""

    return PROJECT_PROMPT_REGISTRY.get(prompt_id, version=version)


def get_prompt_template(
    prompt_id: str,
    *,
    version: str | None = None,
) -> str:
    """Return an exact or active prompt template without rendering."""

    return get_prompt_definition(prompt_id, version=version).template


def render_prompt(
    prompt_id: str,
    *,
    version: str | None = None,
    **variables: Any,
) -> str:
    """Render a project prompt through strict registry validation."""

    return PROJECT_PROMPT_REGISTRY.render(
        prompt_id,
        version=version,
        variables=variables,
    )


def get_active_prompt_manifest() -> dict[str, dict[str, str]]:
    """Return safe active prompt metadata for artifacts and traces."""

    return PROJECT_PROMPT_REGISTRY.active_manifest()


__all__ = [
    "PromptDefinition",
    "get_active_prompt_manifest",
    "get_prompt_definition",
    "get_prompt_template",
    "render_prompt",
]
```

- [x] **Step 5: Convert `agent.prompts` into a compatibility facade**

Replace the prompt bodies at the top of `agent/prompts.py` with exact registry
lookups while preserving `format_chat_history()` and `format_documents()`
unchanged:

```python
"""Prompt compatibility exports and formatting helpers for Agentic RAG."""

from __future__ import annotations

from agent.state import ChatMessage, RetrievedDocument
from prompting import get_prompt_template


QUERY_REWRITE_PROMPT = get_prompt_template(
    "agent.query_rewrite",
    version="v1",
)
RETRY_QUERY_REWRITE_PROMPT = get_prompt_template(
    "agent.retry_query_rewrite",
    version="v1",
)
RETRIEVAL_GRADING_PROMPT = get_prompt_template(
    "agent.retrieval_grading",
    version="v1",
)
ANSWER_GENERATION_PROMPT = get_prompt_template(
    "agent.answer_generation",
    version="v1",
)
CLAIM_EXTRACTION_PROMPT = get_prompt_template(
    "agent.claim_extraction",
    version="v1",
)
CITATION_VERIFICATION_PROMPT = get_prompt_template(
    "agent.citation_verification",
    version="v1",
)
ANSWER_REVISION_PROMPT = get_prompt_template(
    "agent.answer_revision",
    version="v1",
)
CLAIM_VERIFICATION_PROMPT = get_prompt_template(
    "agent.claim_verification",
    version="v1",
)
```

Keep the existing helper implementations below these exports:

```python
def format_chat_history(chat_history: list[ChatMessage]) -> str:
    """Format chat history for prompts."""

    if not chat_history:
        return "No prior chat history."

    lines = []
    for message in chat_history:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def format_documents(documents: list[RetrievedDocument]) -> str:
    """Format retrieved documents for prompts."""

    if not documents:
        return "No retrieved chunks."

    blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        source = document.get("source")
        page = document.get("page")
        chunk_id = document.get("chunk_id")
        score = document.get("score")
        rerank_score = document.get("rerank_score")
        content = document.get("content", "")
        rerank_part = (
            f" rerank_score={rerank_score}" if rerank_score is not None else ""
        )
        blocks.append(
            f"[{index}] source={source} page={page} chunk_id={chunk_id} "
            f"score={score}{rerank_part}\n{content}"
        )
    return "\n\n".join(blocks)
```

- [x] **Step 6: Run catalog and compatibility tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_prompt_registry.py \
  tests/test_prompt_catalog.py \
  tests/test_agent_state_prompts.py -q
```

Expected: `18 passed`.

- [x] **Step 7: Verify no prompt text changed**

Run:

```bash
.venv/bin/python - <<'PY'
from prompting import get_prompt_definition

for prompt_id in [
    "agent.query_rewrite",
    "agent.query_transform",
    "agent.retry_query_rewrite",
    "agent.retrieval_grading",
    "agent.answer_generation",
    "agent.claim_extraction",
    "agent.citation_verification",
    "agent.answer_revision",
    "agent.claim_verification",
    "tool.document_summary",
]:
    prompt = get_prompt_definition(prompt_id, version="v1")
    print(prompt_id, prompt.version, prompt.fingerprint)
PY
```

Expected: output matches the ten fingerprints in the pinned contract table.

- [x] **Step 8: Commit the catalog and compatibility layer**

```bash
git add \
  prompting/__init__.py \
  prompting/catalog.py \
  agent/prompts.py \
  tests/test_prompt_catalog.py
git commit -m "feat: catalog versioned project prompts"
```

---

### Task 3: Route Agent Prompt Calls Through The Registry

**Files:**
- Modify: `agent/query_transform.py`
- Modify: `agent/nodes.py`
- Modify: `tests/test_query_transform.py`
- Modify: `tests/test_agent_nodes.py`

- [x] **Step 1: Add failing tests that observe registry prompt IDs**

Append to `tests/test_query_transform.py`:

```python
def test_build_query_transform_prompt_uses_registered_prompt(monkeypatch):
    import agent.query_transform as query_transform_module

    captured = {}

    def fake_render(prompt_id, **variables):
        captured["prompt_id"] = prompt_id
        captured["variables"] = variables
        return "registered query transform"

    monkeypatch.setattr(query_transform_module, "render_prompt", fake_render)

    result = query_transform_module.build_query_transform_prompt(
        question="How does it compare?",
        chat_history=[{"role": "user", "content": "Discuss Agentic RAG."}],
    )

    assert result == "registered query transform"
    assert captured == {
        "prompt_id": "agent.query_transform",
        "variables": {
            "chat_history": "user: Discuss Agentic RAG.",
            "question": "How does it compare?",
        },
    }
```

Append to `tests/test_agent_nodes.py`:

```python
def test_generate_answer_uses_registered_answer_prompt(monkeypatch):
    import agent.nodes as nodes_module

    captured = {}

    def fake_render(prompt_id, **variables):
        captured["prompt_id"] = prompt_id
        captured["variables"] = variables
        return "registered answer prompt"

    monkeypatch.setattr(nodes_module, "render_prompt", fake_render)
    llm = FakeLLM(
        [
            (
                '{"answer": "Agentic RAG grades evidence [1].", '
                '"used_citation_indices": [1]}'
            )
        ]
    )
    nodes = nodes_module.AgentNodes(
        llm,
        features=AgentFeatureFlags(citation_verification_enabled=False),
        retriever_fn=lambda query: [],
    )
    document = {
        "content": "Agentic RAG grades evidence.",
        "source": "notes.md",
        "chunk_id": "notes.md:pNA:c1",
    }

    result = nodes.generate_answer_node(
        {
            "question": "What does Agentic RAG do?",
            "current_query": "Agentic RAG evidence grading",
            "relevant_documents": [document],
        }
    )

    assert captured == {
        "prompt_id": "agent.answer_generation",
        "variables": {
            "question": "What does Agentic RAG do?",
            "current_query": "Agentic RAG evidence grading",
            "documents": format_documents([document]),
        },
    }
    assert result["draft_answer"] == "Agentic RAG grades evidence [1]."
```

Ensure the existing imports in `tests/test_agent_nodes.py` include:

```python
from agent.features import AgentFeatureFlags
from agent.prompts import format_documents
```

- [x] **Step 2: Run both new tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_query_transform.py::test_build_query_transform_prompt_uses_registered_prompt \
  tests/test_agent_nodes.py::test_generate_answer_uses_registered_answer_prompt -q
```

Expected: both tests fail because the modules do not expose `render_prompt`.

- [x] **Step 3: Route initial query transformation through the registry**

In `agent/query_transform.py`, add:

```python
from prompting import render_prompt
```

Replace `build_query_transform_prompt()` with:

```python
def build_query_transform_prompt(
    question: str,
    chat_history: list[ChatMessage],
) -> str:
    """Build the initial query transformation prompt."""

    return render_prompt(
        "agent.query_transform",
        chat_history=format_chat_history(chat_history),
        question=question,
    )
```

- [x] **Step 4: Route all Agent node prompts through the registry**

In `agent/nodes.py`, change prompt imports to:

```python
from agent.prompts import format_documents
from prompting import render_prompt
```

Replace the retry render:

```python
prompt = render_prompt(
    "agent.retry_query_rewrite",
    question=state["question"],
    current_query=state.get("current_query") or state["question"],
    previous_queries=_format_previous_queries(
        state.get("previous_queries", [])
    ),
    grading_reason=state.get("grading_reason") or "No grading reason.",
    partial_relevance_context=_format_partial_relevance_context(state),
    documents=format_documents(state.get("documents", [])),
)
```

Replace the retrieval-grading render:

```python
prompt = render_prompt(
    "agent.retrieval_grading",
    question=state["question"],
    current_query=state.get("current_query") or state["question"],
    documents=format_documents(documents),
)
```

Replace the answer-generation render:

```python
prompt = render_prompt(
    "agent.answer_generation",
    question=state["question"],
    current_query=state.get("current_query") or state["question"],
    documents=format_documents(documents),
)
```

Replace the claim-extraction render:

```python
prompt = render_prompt(
    "agent.claim_extraction",
    question=state["question"],
    answer=draft_answer,
    documents=format_documents(cited_documents),
)
```

Replace the answer-revision render:

```python
prompt = render_prompt(
    "agent.answer_revision",
    question=state["question"],
    answer=state.get("draft_answer", ""),
    unsupported_claims=json.dumps(
        state.get("unsupported_claims", []),
        ensure_ascii=False,
    ),
    documents=format_documents(cited_documents),
)
```

- [x] **Step 5: Run Agent prompt and node tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_query_transform.py \
  tests/test_agent_nodes.py \
  tests/test_agent_graph.py \
  tests/test_agent_state_prompts.py -q
```

Expected: all tests pass, including the two new registry-observation tests.

- [x] **Step 6: Verify Agent runtime code no longer formats prompt constants**

Run:

```bash
rg -n '_PROMPT\.format|ANSWER_GENERATION_PROMPT|ANSWER_REVISION_PROMPT|CLAIM_EXTRACTION_PROMPT|RETRY_QUERY_REWRITE_PROMPT|RETRIEVAL_GRADING_PROMPT' \
  agent/nodes.py agent/query_transform.py
```

Expected: no matches.

- [x] **Step 7: Commit the Agent migration**

```bash
git add \
  agent/query_transform.py \
  agent/nodes.py \
  tests/test_query_transform.py \
  tests/test_agent_nodes.py
git commit -m "refactor: render agent prompts through registry"
```

---

### Task 4: Route Baseline And LLM-backed Tools Through The Registry

**Files:**
- Modify: `baseline/naive_rag.py`
- Modify: `tools/citation_verifier_tool.py`
- Modify: `tools/document_summary_tool.py`
- Modify: `tests/test_baselines.py`
- Modify: `tests/test_citation_verifier_tool.py`
- Modify: `tests/test_document_summary_tool.py`

- [x] **Step 1: Add failing baseline and tool registry-observation tests**

Append to `tests/test_baselines.py`:

```python
def test_run_naive_rag_uses_registered_answer_prompt(monkeypatch):
    import baseline.naive_rag as naive_module

    captured = {}

    def fake_render(prompt_id, **variables):
        captured["prompt_id"] = prompt_id
        captured["variables"] = variables
        return "registered baseline answer prompt"

    monkeypatch.setattr(naive_module, "render_prompt", fake_render)
    llm = FakeLLM(
        [
            (
                '{"answer": "Naive RAG retrieves once [1].", '
                '"used_citation_indices": [1]}'
            )
        ]
    )
    documents = [
        {
            "content": "Naive RAG retrieves once.",
            "source": "notes.md",
            "chunk_id": "notes.md:pNA:c1",
        }
    ]

    result = naive_module.run_naive_rag(
        "What is naive RAG?",
        retriever_fn=lambda query: documents,
        llm=llm,
    )

    assert captured["prompt_id"] == "agent.answer_generation"
    assert captured["variables"]["question"] == "What is naive RAG?"
    assert captured["variables"]["current_query"] == "What is naive RAG?"
    assert result["answer"] == "Naive RAG retrieves once [1]."
```

Append to `tests/test_citation_verifier_tool.py`:

```python
def test_citation_verifier_uses_registered_prompt(monkeypatch):
    import tools.citation_verifier_tool as verifier_module

    captured = {}

    def fake_render(prompt_id, **variables):
        captured["prompt_id"] = prompt_id
        captured["variables"] = variables
        return "registered citation verifier prompt"

    monkeypatch.setattr(verifier_module, "render_prompt", fake_render)
    llm = FakeLLM(
        [
            (
                '{"results": [{"claim_id": "c001", '
                '"claim": "The answer cites one fact.", '
                '"cited_chunk_ids": ["chunk-1"], '
                '"verification_label": "supported", "confidence": 0.9, '
                '"reason": "Direct support."}], "reason": "checked"}'
            )
        ]
    )
    registry = ToolRegistry()
    registry.register(CitationVerifierTool(ToolContext(llm=llm)))

    result = registry.invoke("verify_citations", _base_arguments())

    assert result.success is True
    assert captured["prompt_id"] == "agent.citation_verification"
    assert captured["variables"]["question"] == "What does the document say?"
    assert "chunk-1" in captured["variables"]["claims"]
```

Append to `tests/test_document_summary_tool.py`:

```python
def test_document_summary_uses_registered_prompt(monkeypatch):
    import tools.document_summary_tool as summary_module

    captured = {}

    def fake_render(prompt_id, **variables):
        captured["prompt_id"] = prompt_id
        captured["variables"] = variables
        return "registered document summary prompt"

    monkeypatch.setattr(summary_module, "render_prompt", fake_render)
    llm = FakeLLM(["summary"])
    registry = ToolRegistry()
    registry.register(DocumentSummaryTool(ToolContext(llm=llm)))

    result = registry.invoke(
        "summarize_document",
        {
            "title": "Spec Notes",
            "content": "Grounded answers use evidence.",
            "max_points": 2,
        },
    )

    assert result.success is True
    assert captured == {
        "prompt_id": "tool.document_summary",
        "variables": {
            "max_points": 2,
            "title": "Spec Notes",
            "content": "Grounded answers use evidence.",
        },
    }
```

- [x] **Step 2: Run the new tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_baselines.py::test_run_naive_rag_uses_registered_answer_prompt \
  tests/test_citation_verifier_tool.py::test_citation_verifier_uses_registered_prompt \
  tests/test_document_summary_tool.py::test_document_summary_uses_registered_prompt \
  -q
```

Expected: three failures because the modules do not expose `render_prompt`.

- [x] **Step 3: Migrate the naive baseline**

In `baseline/naive_rag.py`, change imports to:

```python
from agent.prompts import format_documents
from prompting import render_prompt
```

Replace prompt construction with:

```python
prompt = render_prompt(
    "agent.answer_generation",
    question=question,
    current_query=question,
    documents=format_documents(documents),
)
```

- [x] **Step 4: Migrate the citation verifier tool**

In `tools/citation_verifier_tool.py`, change imports to:

```python
from agent.prompts import format_documents
from prompting import render_prompt
```

Replace prompt construction with:

```python
prompt = render_prompt(
    "agent.citation_verification",
    question=arguments.question,
    answer=arguments.answer,
    claims=json.dumps(arguments.claims, ensure_ascii=False),
    documents=format_documents(arguments.documents),
)
```

- [x] **Step 5: Migrate the document summary tool**

Add this import in `tools/document_summary_tool.py`:

```python
from prompting import render_prompt
```

Replace the inline prompt construction with:

```python
prompt = render_prompt(
    "tool.document_summary",
    max_points=arguments.max_points,
    title=title,
    content=arguments.content,
)
```

- [x] **Step 6: Run baseline and tool tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_baseline.py \
  tests/test_baselines.py \
  tests/test_citation_verifier_tool.py \
  tests/test_document_summary_tool.py \
  tests/test_tool_registry.py -q
```

Expected: all tests pass.

- [x] **Step 7: Verify every runtime LLM prompt is registry-backed**

Run:

```bash
rg -n 'prompt = \\(|_PROMPT\\.format|return f\"\"\"You transform' \
  agent baseline tools -g '*.py'
```

Expected: no runtime prompt construction matches. Compatibility constants in
`agent/prompts.py` remain allowed because they are registry lookups.

- [x] **Step 8: Commit the baseline and tool migration**

```bash
git add \
  baseline/naive_rag.py \
  tools/citation_verifier_tool.py \
  tools/document_summary_tool.py \
  tests/test_baselines.py \
  tests/test_citation_verifier_tool.py \
  tests/test_document_summary_tool.py
git commit -m "refactor: render tool prompts through registry"
```

---

### Task 5: Record Prompt Versions In Evaluation Runtime Metadata

**Files:**
- Modify: `evaluation/runtime_config.py`
- Modify: `tests/test_ablation.py`
- Modify: `tests/test_evaluation_storage.py`
- Modify: `tests/test_evaluate.py`

- [x] **Step 1: Write the failing runtime metadata safety test**

Append to `tests/test_ablation.py`:

```python
def test_runtime_config_snapshot_includes_safe_active_prompt_manifest():
    snapshot = build_runtime_config_snapshot()

    assert snapshot["schema_version"] == 2
    assert snapshot["evaluator_version"] == "p4d"
    assert set(snapshot["prompts"]) == {
        "agent.query_transform",
        "agent.retry_query_rewrite",
        "agent.retrieval_grading",
        "agent.answer_generation",
        "agent.claim_extraction",
        "agent.citation_verification",
        "agent.answer_revision",
        "tool.document_summary",
    }
    assert all(
        set(metadata) == {"version", "fingerprint"}
        for metadata in snapshot["prompts"].values()
    )
    assert "template" not in json.dumps(snapshot["prompts"])
```

Add `import json` at the top of `tests/test_ablation.py`.

- [x] **Step 2: Update existing expected runtime versions**

In `tests/test_evaluation_storage.py`, change:

```python
assert comparison_payload["runtime_config"]["schema_version"] == 2
assert comparison_payload["runtime_config"]["evaluator_version"] == "p4d"
assert comparison_payload["runtime_config"]["prompts"] == runtime_config["prompts"]
```

In both artifact tests in `tests/test_evaluate.py`, change all actual runtime
metadata assertions to:

```python
assert baseline_payload["runtime_config"]["schema_version"] == 2
assert agentic_payload["runtime_config"]["evaluator_version"] == "p4d"
assert comparison_payload["runtime_config"]["schema_version"] == 2
assert comparison_payload["runtime_config"]["evaluator_version"] == "p4d"
assert comparison_payload["runtime_config"]["prompts"]
```

and:

```python
assert payload["runtime_config"]["schema_version"] == 2
assert payload["runtime_config"]["evaluator_version"] == "p4d"
assert "prompts" in payload["runtime_config"]
```

Do not change the generic `RuntimeMetadata` unit tests in
`tests/test_evaluation_schemas.py`; their `p4c` values are arbitrary constructor
inputs, not current runtime constants.

- [x] **Step 3: Run the focused tests and verify the new manifest test fails**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_ablation.py::test_runtime_config_snapshot_includes_safe_active_prompt_manifest \
  tests/test_evaluation_storage.py \
  tests/test_evaluate.py::test_main_writes_comparison_artifacts \
  tests/test_evaluate.py::test_main_writes_single_system_agentic_artifact_schema \
  -q
```

Expected: failures report schema `1`, evaluator `p4c`, or missing `prompts`.

- [x] **Step 4: Add prompt metadata to the runtime snapshot**

In `evaluation/runtime_config.py`, add:

```python
from prompting import get_active_prompt_manifest
```

Change constants to:

```python
EVALUATION_SCHEMA_VERSION = 2
EVALUATOR_VERSION = "p4d"
```

Add the manifest to the `RuntimeMetadata` config:

```python
config={
    "agent_features": resolved_features.to_dict(),
    "llm": {
        "provider": resolved.llm_provider,
        "model": resolved.effective_llm_model,
        "temperature": resolved.temperature,
    },
    "prompts": get_active_prompt_manifest(),
    "retriever": {
        "top_k": resolved.top_k,
        "hybrid_retrieval_enabled": resolved.hybrid_retrieval_enabled,
        "dense_top_k": resolved.dense_top_k,
        "bm25_top_k": resolved.bm25_top_k,
        "fusion_top_k": resolved.fusion_top_k,
    },
    "reranker": {
        "enabled": resolved.reranker_enabled,
        "model": resolved.reranker_model,
        "top_n": resolved.reranker_top_n,
        "candidate_top_k": resolved.reranker_candidate_top_k,
    },
    "vectorstore": {
        "collection_name": resolved.chroma_collection_name,
    },
},
```

- [x] **Step 5: Run evaluation metadata and artifact tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_ablation.py \
  tests/test_evaluation_storage.py \
  tests/test_evaluate.py \
  tests/test_dashboard_service.py \
  tests/test_evaluation_matrix.py -q
```

Expected: all tests pass with additive prompt metadata and unchanged artifact
filenames.

- [x] **Step 6: Commit evaluation metadata**

```bash
git add \
  evaluation/runtime_config.py \
  tests/test_ablation.py \
  tests/test_evaluation_storage.py \
  tests/test_evaluate.py
git commit -m "feat: record prompt versions in evaluation metadata"
```

---

### Task 6: Record A Safe Prompt Manifest In Agent Traces

**Files:**
- Modify: `observability/trace.py`
- Modify: `agent/graph.py`
- Modify: `tests/test_trace_logging.py`

- [x] **Step 1: Write the failing defensive-copy trace test**

Append to `tests/test_trace_logging.py`:

```python
def test_trace_recorder_copies_safe_prompt_manifest():
    from observability.trace import TraceRecorder

    manifest = {
        "agent.answer_generation": {
            "version": "v1",
            "fingerprint": "sha256:abc",
        }
    }
    recorder = TraceRecorder(
        original_question="question",
        prompts=manifest,
    )
    manifest["agent.answer_generation"]["version"] = "mutated-input"

    record = recorder.build_record({}, latency_ms=1)
    record["prompts"]["agent.answer_generation"]["version"] = "mutated-output"

    assert recorder.build_record({}, latency_ms=1)["prompts"] == {
        "agent.answer_generation": {
            "version": "v1",
            "fingerprint": "sha256:abc",
        }
    }
```

In `test_run_agent_writes_node_events_and_route_decisions_to_trace`, add:

```python
from prompting import get_active_prompt_manifest
```

and these assertions:

```python
assert trace["prompts"] == get_active_prompt_manifest()
assert all(
    set(metadata) == {"version", "fingerprint"}
    for metadata in trace["prompts"].values()
)
```

- [x] **Step 2: Run the focused trace tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_trace_logging.py::test_trace_recorder_copies_safe_prompt_manifest \
  tests/test_trace_logging.py::test_run_agent_writes_node_events_and_route_decisions_to_trace \
  -q
```

Expected: the first test fails because `TraceRecorder` has no `prompts`
argument; the second fails because traces have no `prompts` field.

- [x] **Step 3: Store a copied manifest in `TraceRecorder`**

In `observability/trace.py`, add:

```python
from copy import deepcopy
```

Change the constructor to:

```python
def __init__(
    self,
    original_question: str,
    session_id: str | None = None,
    workspace_id: str | None = None,
    trace_id: str | None = None,
    prompts: dict[str, dict[str, str]] | None = None,
) -> None:
    self.trace_id = trace_id or f"trace_{uuid.uuid4().hex}"
    self.session_id = session_id
    self.workspace_id = workspace_id
    self.original_question = original_question
    self.prompts = deepcopy(prompts or {})
    self.started_at = time.perf_counter()
    self.events: list[dict[str, Any]] = []
    self.route_decisions: list[dict[str, Any]] = []
    self.tool_calls: list[dict[str, Any]] = []
```

Add this field to the top-level dictionary in `build_record()`:

```python
"prompts": deepcopy(self.prompts),
```

- [x] **Step 4: Snapshot the active manifest when an Agent trace starts**

In `agent/graph.py`, add:

```python
from prompting import get_active_prompt_manifest
```

Change trace-recorder construction to:

```python
trace_recorder = (
    TraceRecorder(
        original_question=question,
        session_id=session_id,
        workspace_id=workspace_id,
        prompts=get_active_prompt_manifest(),
    )
    if trace_enabled
    else None
)
```

Do not add prompt metadata to `AgentState`, the public `run_agent()` result, API
schemas, or Gradio payloads.

- [x] **Step 5: Run all trace and API compatibility tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_trace_logging.py \
  tests/test_fastapi_routes.py \
  tests/test_agent_graph.py -q
```

Expected: all tests pass and trace lookup remains compatible with the additive
top-level `prompts` field.

- [x] **Step 6: Commit trace metadata**

```bash
git add \
  observability/trace.py \
  agent/graph.py \
  tests/test_trace_logging.py
git commit -m "feat: record prompt versions in agent traces"
```

---

### Task 7: Document P4d, Verify The Project, And Prepare Review

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/github_release_checklist.md`
- Modify: `docs/superpowers/plans/2026-06-14-p4c-modular-evaluation-framework.md`
- Modify: `docs/superpowers/plans/2026-06-18-p4d-prompt-versioning.md`

- [x] **Step 1: Document the prompt-versioning architecture**

Add this section after `### Modular Evaluation Framework` in `README.md`:

```markdown
### Versioned Prompt Registry

P4d moves Agent, baseline, and LLM-backed tool prompts into a code-native
registry with 10 registered `v1` templates: 8 active runtime prompts and 2
inactive compatibility-only templates. Each template has a stable prompt ID,
immutable version, strict input variables, and a SHA-256 fingerprint. Existing
`agent.prompts` constants remain available as compatibility exports, while
runtime call sites render active versions through the registry.

The new evaluation `runtime_config.prompts` and trace `prompts` fields record
only the 8 active prompt IDs, versions, and fingerprints; those fields do not
store templates or rendered prompt payloads. Existing trace records still
contain the original question, compact document snippets, answers, citations,
and diagnostics as documented observability data. P4d detects template drift
but does not yet run LLM-based behavioral prompt regression or support dynamic
version selection.
```

In the Project Structure tree add:

```text
├── prompting/
│   ├── __init__.py
│   ├── catalog.py
│   └── registry.py
```

Update the runtime-config paragraph to include prompt versions and
fingerprints. Scope trace safety wording to the new `prompts` manifest field:
it contains only IDs, versions, and fingerprints, without templates or rendered
prompt payloads. Explicitly retain the documented fact that trace records also
contain the original question, compact document snippets, answers, citations,
and diagnostics. Add this Completed Work item:

```markdown
- P4d prompt versioning implemented: 10 registered `v1` templates—8 active runtime prompts and 2 inactive compatibility-only templates—have stable IDs, strict rendering contracts, SHA-256 fingerprints, compatibility exports, and safe evaluation/trace manifests.
```

Replace the beginning of Next Milestones with the confirmed route:

```markdown
### Next Milestones

- P5a: implement a configurable DeepSeek semantic correctness and groundedness judge through the existing judge protocol, with its judge prompt registered and versioned.
- P5b: add SQLite-backed historical evaluation runs, prompt-aware run comparison, and a trend dashboard.
- Add background evaluation status, progress, cancellation, and checkpoint recovery.
- Link failed evaluation cases to `trace_id` and a node-level trace drill-down view.
```

Keep the remaining retrieval, workspace, model-tuning, and human-label roadmap
items after these four entries. Remove the now-completed prompt-version tracking
item and avoid claiming behavioral prompt regression is complete.

- [x] **Step 2: Add the P4d changelog entry**

Insert at the top of `CHANGELOG.md`:

```markdown
## v0.4.3-p4d - Prompt Versioning

Date: 2026-06-18

### Added

- Added a code-native prompt registry with stable prompt IDs, immutable versions,
  strict rendering contracts, and deterministic SHA-256 fingerprints.
- Registered 10 exact `v1` templates: 8 active runtime prompts plus 2 inactive
  compatibility-only templates.
- Added safe active prompt manifests to evaluation runtime metadata and local
  Agent traces.

### Changed

- Routed all current runtime LLM prompt construction through the registry while
  preserving prompt text, invocation order, parser contracts, and public
  `agent.prompts` constants.
- Advanced evaluation artifact metadata to schema version `2` and evaluator
  version `p4d`.

### Notes

- P4d detects template drift and records reproducibility metadata. It does not
  add dynamic prompt selection, online prompt editing, or LLM-based behavioral
  prompt regression.
- P5a DeepSeek semantic judging and P5b SQLite historical trends remain the next
  evaluation milestones.

### Verification

- Full test suite: `489 passed`.
- CLI compatibility smoke tests: `3 passed`.
- Prompt registry and catalog tests: `13 passed`.
- Ablation, matrix, dashboard, and FastAPI compatibility tests: `83 passed`.
```

- [x] **Step 3: Align release and prior-plan documentation**

In `docs/github_release_checklist.md`:

- change the version label and tag commands from `v0.4.2-p4c` to
  `v0.4.3-p4d`
- add `prompting` to the compile command
- change the expected full suite to `489 passed`
- retain CLI smoke expectation `3 passed`
- change the ablation, matrix, dashboard, and FastAPI compatibility expectation
  from `82 passed` to `83 passed`
- add prompt versioning to the project narrative and honest scope notes

In
`docs/superpowers/plans/2026-06-14-p4c-modular-evaluation-framework.md`,
change the already-integrated Task 11 Step 8 checkbox from:

```markdown
- [ ] **Step 8: Finish the development branch**
```

to:

```markdown
- [x] **Step 8: Finish the development branch**
```

Add an execution note stating that P4c was merged to `main` and tagged
`v0.4.2-p4c` before P4d began.

- [x] **Step 4: Run compile and import verification**

Run:

```bash
.venv/bin/python -m compileall \
  prompting agent rag api evaluation experiments baseline tools observability
.venv/bin/python - <<'PY'
from evaluation.evaluate import format_report, load_eval_questions
from prompting import get_active_prompt_manifest

print(len(load_eval_questions()))
print(len(get_active_prompt_manifest()))
PY
```

Expected:

```text
36
8
```

- [x] **Step 5: Run prompt-focused verification**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_prompt_registry.py \
  tests/test_prompt_catalog.py -q
```

Expected: `13 passed`.

- [x] **Step 6: Run CLI compatibility smoke tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_evaluate.py::test_main_prints_report_with_injected_runner \
  tests/test_evaluate.py::test_main_writes_comparison_artifacts \
  tests/test_evaluate.py::test_main_writes_single_system_agentic_artifact_schema \
  -q
```

Expected: `3 passed`.

- [x] **Step 7: Run direct consumer regression tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_ablation.py \
  tests/test_evaluation_matrix.py \
  tests/test_dashboard_service.py \
  tests/test_fastapi_routes.py \
  tests/test_gradio_app.py \
  tests/test_trace_logging.py -q
```

Expected: all tests pass.

- [x] **Step 8: Run the full project test suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: `489 passed`.

- [x] **Step 9: Verify prompt-source and metadata safety invariants**

Run:

```bash
rg -n 'return f\"\"\"You transform|prompt = \\(|_PROMPT\\.format' \
  agent baseline tools -g '*.py'
rg -n 'template|rendered_prompt|OPENAI_API_KEY' \
  evaluation/runtime_config.py observability/trace.py
git diff --check
```

Expected:

- no runtime inline prompt construction or `_PROMPT.format` matches
- no prompt template, rendered prompt, or API-key field is added to evaluation
  runtime metadata or trace construction
- `git diff --check` reports no whitespace errors

- [x] **Step 10: Inspect final scope**

Run:

```bash
git status --short
git diff --stat 59f58b5...32f825e
git log --oneline --decorate 59f58b5..32f825e
git diff --stat 59f58b5...HEAD
git log --oneline --decorate 59f58b5..HEAD
```

Expected:

- the pinned `59f58b5...32f825e` comparison reports the pre-documentation P4d
  implementation scope
- the `59f58b5...HEAD` comparison reports the current scope including
  documentation follow-ups
- only P4d prompt, metadata, trace, tests, and documentation files are changed
- `.venv` remains untracked and unstaged
- commits remain separated by registry, catalog, Agent migration, tool
  migration, evaluation metadata, trace metadata, and documentation

Execution notes:

- Compile verification completed successfully for `prompting`, `agent`, `rag`,
  `api`, `evaluation`, `experiments`, `baseline`, `tools`, and
  `observability`.
- The import smoke test printed `36` evaluation questions and `8` active
  prompt-manifest entries.
- Prompt registry and catalog verification passed with `13 passed in 0.02s`.
- CLI compatibility verification passed with `3 passed in 2.09s`.
- The exact four-file ablation, matrix, dashboard, and FastAPI compatibility
  command passed with `83 passed in 3.80s`.
- The broader direct-consumer command, adding Gradio and trace tests, passed
  with `110 passed in 4.63s`.
- The full suite passed with `489 passed in 3.63s`.
- The runtime-inline prompt search returned no matches.
- The exact metadata safety search for `template`, `rendered_prompt`, and
  `OPENAI_API_KEY` returned no matches. A broader semantic inspection confirmed
  that the new evaluation `runtime_config.prompts` and trace `prompts` fields
  contain only IDs, versions, and fingerprints, not templates or rendered
  prompt payloads. This safety statement is scoped to those fields: the trace
  record still stores its pre-existing original question, compact document
  snippets, answers, citations, and diagnostics.
- `git diff --check` reported no whitespace errors.
- Pre-commit `git status --short` showed only the five owned documentation
  files modified plus untracked `.venv`.
- The pinned pre-documentation command
  `git diff --stat 59f58b5...32f825e` showed the P4d implementation scope:
  `25 files changed, 3950 insertions(+), 268 deletions(-)`.
- Before the follow-up clarification commit, the current working-tree comparison
  `git diff --stat 59f58b5` showed the complete P4d scope including
  documentation follow-ups: `29 files changed, 4127 insertions(+), 294
  deletions(-)`.
- No expected verification count differed from the observed release targets.
  The plan did not previously pin a broader-consumer count; the observed count
  is `110`.
- Commit series before the documentation commit:
  - `8877f01 docs: design p4d prompt versioning`
  - `2e2a9c8 docs: plan p4d prompt versioning`
  - `43cd06c feat: add versioned prompt registry`
  - `41387d7 feat: catalog versioned project prompts`
  - `bee872d refactor: make prompt catalog versions explicit`
  - `89bfbf4 refactor: render agent prompts through registry`
  - `29e2f83 refactor: render tool prompts through registry`
  - `169ea9e feat: record prompt versions in evaluation metadata`
  - `3442e6d feat: record prompt versions in agent traces`
  - `32f825e refactor: avoid redundant trace manifest copy`
- Step 11 publishes these documentation updates with
  `docs: publish p4d prompt versioning`.
- Follow-up review clarification updates only owned documentation and is
  committed as `docs: clarify p4d release guidance`; Steps 12-13 remain
  controller-owned and unchecked.
- Documentation commit series:
  - `c7b0132 docs: publish p4d prompt versioning`
  - `docs: clarify p4d release guidance`
- Follow-up verification:
  - Markdown consistency checked four modified Markdown files: balanced code
    fences and no trailing whitespace.
  - Prompt registry and catalog tests: `13 passed in 0.02s`.
  - CLI compatibility smoke tests: `3 passed in 1.60s`.
  - Exact four-file compatibility command: `83 passed in 3.15s`.
  - Full suite: `489 passed in 4.49s`.

- [x] **Step 11: Mark the implementation plan complete and commit docs**

Mark every completed checkbox in this plan, record observed verification output,
then run:

```bash
git add \
  README.md \
  CHANGELOG.md \
  docs/github_release_checklist.md
git add -f \
  docs/superpowers/plans/2026-06-14-p4c-modular-evaluation-framework.md \
  docs/superpowers/plans/2026-06-18-p4d-prompt-versioning.md
git commit -m "docs: publish p4d prompt versioning"
```

- [x] **Step 12: Request code review before integration**

Invoke `superpowers:requesting-code-review` against the implementation branch.
Address confirmed correctness, compatibility, metadata-safety, or test findings
with focused commits and rerun the affected tests plus the full suite.

Review notes:

- Every implementation task passed an independent specification-compliance
  review followed by a code-quality review.
- Task 2 made prompt-version constant names explicit and added exact coverage
  for inactive compatibility definitions.
- Task 6 removed a redundant trace-manifest copy while preserving constructor
  and output isolation.
- Task 7 clarified manifest privacy scope, corrected the release flow, and
  standardized the `10 registered / 8 active` wording.
- The final whole-branch review reported no critical, important, or minor
  findings and approved P4d for integration.

- [ ] **Step 13: Finish the development branch**

After fresh verification, invoke
`superpowers:finishing-a-development-branch`. Offer merge, pull request, keep,
or cleanup choices. Create tag `v0.4.3-p4d` only after the user explicitly
chooses integration, integration succeeds, and `main` passes the final suite.

## Final Verification Matrix

| Concern | Verification |
|---|---|
| Registry validation | `tests/test_prompt_registry.py` |
| Prompt IDs, versions, variables, fingerprints | `tests/test_prompt_catalog.py` |
| Compatibility constants | `tests/test_prompt_catalog.py`, `tests/test_agent_state_prompts.py` |
| Agent runtime rendering | `tests/test_query_transform.py`, `tests/test_agent_nodes.py`, `tests/test_agent_graph.py` |
| Baseline shared prompt | `tests/test_baselines.py`, `tests/test_baseline.py` |
| Tool prompt rendering | `tests/test_citation_verifier_tool.py`, `tests/test_document_summary_tool.py` |
| Evaluation metadata | `tests/test_ablation.py`, `tests/test_evaluation_storage.py`, `tests/test_evaluate.py` |
| Dashboard and matrix compatibility | `tests/test_dashboard_service.py`, `tests/test_evaluation_matrix.py`, `tests/test_gradio_app.py` |
| Trace metadata and copying | `tests/test_trace_logging.py` |
| FastAPI compatibility | `tests/test_fastapi_routes.py` |
| Full regression | `.venv/bin/python -m pytest -q` |
