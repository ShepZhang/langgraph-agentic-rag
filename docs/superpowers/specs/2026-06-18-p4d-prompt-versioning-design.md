# P4d Prompt Versioning Design

Date: 2026-06-18

Status: Approved architecture; awaiting written-spec review

Target version: `v0.4.3-p4d`

## Goal

Introduce deterministic prompt identity, versioning, rendering, and runtime
metadata without changing the current prompt text or Agent behavior.

P4d creates a small code-native prompt registry that becomes the source of
truth for Agent, baseline, and LLM-backed tool prompts. Each prompt has a stable
ID, an immutable version, and a SHA-256 fingerprint. Evaluation artifacts and
persisted traces record the active prompt manifest so later P5a semantic-judge
results and P5b historical trends can be attributed to the prompt set that
produced them.

The migration must preserve all existing public imports, rendered prompt text,
LLM call order, parser contracts, evaluation behavior, API behavior, and trace
redaction guarantees.

## Current Baseline

The design targets `main` commit `59f58b5`, which contains tag
`v0.4.2-p4c` at commit `e326fbc`. The baseline test suite passes with
`469 passed`.

Prompt definitions are currently split across three locations:

- `agent/prompts.py` contains eight string constants.
- `agent/query_transform.py` builds the initial query-transform prompt inline.
- `tools/document_summary_tool.py` builds the document-summary prompt inline.

Eight prompts are used by runtime LLM calls. Two constants remain importable
but are not called by current runtime code:

- `QUERY_REWRITE_PROMPT`
- `CLAIM_VERIFICATION_PROMPT`

The current system has no prompt ID, prompt version, fingerprint, active-version
mapping, prompt manifest, or evaluation/trace prompt metadata. Existing tests
assert selected guardrail text, but they do not prevent a prompt body from
changing without a corresponding version change.

## Scope

P4d includes:

- a code-native, immutable prompt definition model
- a registry keyed by stable prompt ID and explicit version
- one statically configured active version for each runtime prompt
- strict rendering with missing-variable and unexpected-variable validation
- deterministic SHA-256 fingerprints over exact UTF-8 prompt templates
- migration of every current runtime LLM prompt to registry rendering
- compatibility exports for existing constants from `agent.prompts`
- registration of the two unused compatibility prompts as non-active versions
- an active prompt manifest in evaluation runtime metadata
- the same active prompt manifest in persisted Agent traces
- deterministic regression tests for prompt versions, fingerprints, variables,
  rendering, and compatibility imports
- README, CHANGELOG, roadmap, and release documentation updates

P4d does not include:

- editing or tuning current prompt wording
- a DeepSeek semantic correctness or groundedness judge
- LLM-based prompt-quality regression runs
- SQLite historical result storage or trend views
- environment-variable, API, or UI prompt-version selection
- loading prompts from YAML, JSON, a database, or a remote service
- runtime prompt editing or hot reload
- storing complete rendered prompts in evaluation artifacts or traces
- recording per-call prompt inputs, user questions, retrieved chunks, or
  document content as prompt metadata
- background evaluation, cancellation, checkpoint recovery, or trace
  drill-down UI

Deterministic template regression belongs to P4d. Behavioral prompt regression,
where multiple prompt versions are run and scored against evaluation questions,
is deferred until the judge and historical evaluation foundations exist.

## Confirmed Decisions

### Code-Native Registry

Prompt definitions remain Python source.

This matches the current architecture, keeps prompt changes reviewable in the
same commits as their consumers and parsers, avoids a new runtime I/O failure
mode, and allows ordinary unit tests to enforce prompt invariants.

External prompt files and dynamic providers are intentionally deferred. They
would add configuration, deployment, caching, validation, and fallback
semantics that are not required for P4d.

### Version Format

Prompt versions use monotonic strings:

```text
v1
v2
v3
```

Versions must match `v[1-9][0-9]*`. A released `(prompt_id, version)` pair is
immutable. Changing any character, whitespace, placeholder, output schema, or
instruction requires registering a new version and changing the active mapping.

P4d does not infer semantic-version compatibility between prompts. Prompt
versions identify exact templates rather than advertise API compatibility.

### Stable Prompt IDs

Prompt IDs are lowercase dotted identifiers. The initial catalog uses:

| Prompt ID | Current source | Status |
|---|---|---|
| `agent.query_transform` | inline in `agent/query_transform.py` | active `v1` |
| `agent.retry_query_rewrite` | `RETRY_QUERY_REWRITE_PROMPT` | active `v1` |
| `agent.retrieval_grading` | `RETRIEVAL_GRADING_PROMPT` | active `v1` |
| `agent.answer_generation` | `ANSWER_GENERATION_PROMPT` | active `v1` |
| `agent.claim_extraction` | `CLAIM_EXTRACTION_PROMPT` | active `v1` |
| `agent.citation_verification` | `CITATION_VERIFICATION_PROMPT` | active `v1` |
| `agent.answer_revision` | `ANSWER_REVISION_PROMPT` | active `v1` |
| `tool.document_summary` | inline in `tools/document_summary_tool.py` | active `v1` |
| `agent.query_rewrite` | `QUERY_REWRITE_PROMPT` | compatibility-only `v1` |
| `agent.claim_verification` | `CLAIM_VERIFICATION_PROMPT` | compatibility-only `v1` |

Compatibility-only definitions are registered and addressable by explicit
version, but they are not present in the active runtime manifest because no
current LLM call uses them.

### Active Version Selection

The catalog contains one static active-version mapping.

Runtime callers normally request a prompt by ID and receive its active version.
Tests and future comparison code may request an explicit version. An unknown
prompt, unknown version, or missing active mapping raises an explicit error;
the registry never silently falls back to another prompt version.

P4d does not add active-version settings to `Settings` or environment variables.
Changing an active prompt is a reviewed source-code change with tests and a
release marker.

### Fingerprint Definition

The fingerprint is:

```text
sha256:<lowercase 64-character hexadecimal digest>
```

The digest is computed from the exact prompt template encoded as UTF-8.
Whitespace and escaped literal braces are significant. Description text,
active status, prompt ID, and version are not included in the digest.

This definition allows the same exact template to have the same fingerprint
across machines while the `(prompt_id, version)` pair supplies semantic
identity.

### Strict Rendering

Template variables are derived through Python format-string parsing. Rendering
requires the supplied variable names to exactly equal the template variable
set.

- Missing variables fail before an LLM call.
- Unexpected variables fail before an LLM call.
- Malformed templates fail when the catalog is constructed.
- Literal JSON braces remain escaped in source templates and render exactly as
  they do today.

Strict rendering catches prompt/caller drift early and prevents unused caller
data from being mistaken for part of a prompt contract.

## Architecture

```text
agent nodes / baseline / LLM-backed tools
                   |
                   v
          prompting.render_prompt
                   |
                   v
       PromptRegistry + active mapping
                   |
            +------+------+
            |             |
            v             v
     rendered text   active manifest
            |             |
            v             +------------------+
         LLM call                            |
                                             v
                         evaluation runtime metadata / trace record
```

Prompt text and prompt metadata have separate flows. Runtime callers receive
only rendered text. Evaluation and observability consumers receive only the
safe active manifest.

## Package Design

### `prompting/registry.py`

This module owns the generic prompt domain model and registry behavior. It has
no dependency on Agent state, evaluation, tools, LLM clients, or environment
configuration.

`PromptDefinition` is a frozen dataclass with:

- `prompt_id`
- `version`
- `template`
- `description`

It exposes derived template variables and the deterministic fingerprint.

`PromptRegistry` owns:

- definition registration
- duplicate `(prompt_id, version)` rejection
- active-version validation
- explicit-version lookup
- active-version lookup
- strict rendering
- active-manifest construction

The registry copies externally supplied mappings and returns new manifest
dictionaries so callers cannot mutate catalog state.

### `prompting/catalog.py`

This module is the single source of truth for project prompt text.

It registers the ten current prompt definitions without changing their text.
It declares active versions for the eight runtime prompts and omits active
mappings for the two compatibility-only prompts.

The module exports one constructed project registry. It performs no file I/O
and no LLM calls.

### `prompting/__init__.py`

The package facade exposes the small public surface required by consumers:

- render a prompt using the active or an explicit version
- retrieve an exact template for compatibility exports
- retrieve the active prompt manifest
- retrieve a prompt definition for tests or future orchestration

Internal catalog construction details remain private.

### `agent/prompts.py`

Formatting helpers remain in this module:

- `format_chat_history`
- `format_documents`

Existing prompt constant names remain importable. Their values are fetched from
the registry by exact ID and `v1`, preserving callers and tests that import and
format these constants directly.

The prompt bodies no longer live in this module, preventing duplicate sources
of truth.

### Runtime Call Sites

All current LLM call paths use registry rendering through the eight active
prompt definitions:

- initial query transformation
- retry query rewrite
- retrieval grading
- Agent answer generation
- naive baseline answer generation
- claim extraction
- citation verification tool
- answer revision
- document summary tool

Agent and naive answer generation intentionally share
`agent.answer_generation`. The registry does not create a duplicate baseline
prompt ID when the underlying template and contract are the same.

Parser modules remain unchanged unless import movement is required. Prompt
versioning does not alter structured response parsing or fallback behavior.

## Manifest And Metadata

The active manifest is a mapping ordered by prompt ID:

```json
{
  "agent.answer_generation": {
    "version": "v1",
    "fingerprint": "sha256:<digest>"
  }
}
```

Only active prompts appear. The manifest contains no prompt template, rendered
prompt, model response, user data, retrieved evidence, secrets, or local paths.

### Evaluation Artifacts

`evaluation/runtime_config.py` adds the manifest under:

```text
runtime_config.prompts
```

P4d updates:

- `EVALUATION_SCHEMA_VERSION` from `1` to `2`
- `EVALUATOR_VERSION` from `p4c` to `p4d`

The field is additive, and existing artifact filenames and top-level report
shapes remain unchanged. Older dashboard and analysis paths must continue to
ignore fields they do not use.

All compatibility artifacts written by `write_compatibility_artifacts()` receive
the same manifest through the existing runtime-config snapshot.

### Persisted Traces

`run_agent()` snapshots the active manifest when a trace-enabled run starts and
passes it to `TraceRecorder`. The persisted trace stores the copied manifest
under top-level `prompts`.

The manifest describes the active project prompt set for the run. It does not
claim that every listed prompt was invoked. Per-call prompt events and
prompt-to-node drill-down are deferred to the later trace drill-down milestone.

No `AgentState` field, public `run_agent()` response field, FastAPI schema, or
Gradio response shape is added for P4d.

## Compatibility Requirements

P4d preserves:

- all current constant imports from `agent.prompts`
- exact rendered text for all ten migrated templates
- current LLM invocation count and ordering
- current Agent graph topology and routing
- current baseline behavior
- current structured JSON output instructions
- current parser and fallback behavior
- current evaluation facade and artifact filenames
- current Dashboard, FastAPI, ablation, and matrix consumers
- current trace content, with only the additive safe manifest

Prompt registry failures are programming/configuration errors and must not be
converted into normal Agent fallback answers. They should fail before the LLM
call with an error that identifies the prompt ID, requested version, and
variable mismatch without including user content.

## TDD And Verification Strategy

Implementation follows red-green-refactor in small commits.

### Registry Unit Tests

Tests cover:

- valid registration and active lookup
- duplicate definition rejection
- invalid ID and version rejection
- missing active version rejection
- explicit historical-version lookup
- missing and unexpected render variables
- literal JSON brace rendering
- deterministic SHA-256 fingerprints
- defensive copying of manifests
- active manifest exclusion of compatibility-only prompts

### Prompt Regression Tests

Tests pin:

- the complete set of prompt IDs
- each active version
- each `v1` fingerprint
- each prompt variable set
- exact equality between compatibility constants and registered templates
- exact rendered output for representative inputs

A prompt-text edit with an unchanged version must fail the pinned fingerprint
test. The correct change procedure is to add a new version, add its fingerprint
expectation, and update the active mapping.

### Integration Tests

Existing fake-LLM tests continue verifying prompt content and call order.
Focused updates verify that:

- Agent nodes render through the registry without behavior changes
- naive RAG uses the shared answer-generation prompt
- citation verification and document summary use registered prompts
- evaluation runtime metadata contains schema `2`, evaluator `p4d`, and the
  active manifest
- trace records contain a defensive copy of the same manifest
- no prompt template or rendered prompt appears in runtime metadata

### Full Regression

Before completion:

- focused prompt, Agent, baseline, tool, evaluation-storage, runtime-config, and
  trace tests pass
- the full project test suite passes
- Python compilation and import smoke tests pass
- `git diff --check` passes
- CLI compatibility tests pass without live model calls

No live DeepSeek or Ollama call is required to validate P4d because prompt text
and behavior are intentionally unchanged.

## Documentation And Release

The final documentation phase will:

- add P4d to README Completed Work
- reorder Next Milestones to match the confirmed route:
  P5a DeepSeek Semantic Judge, then P5b SQLite Historical Evaluation and Trend
  Dashboard, then Background Evaluation and Trace Drill-down
- explain the prompt registry, version rule, fingerprint, and metadata fields
- add a `v0.4.3-p4d` CHANGELOG entry
- correct the stale unchecked P4c branch-finish marker based on the already
  merged `main` history and existing `v0.4.2-p4c` tag
- preserve honest limitations around behavioral prompt regression and dynamic
  version selection

The `v0.4.3-p4d` tag is created only after implementation, review, full
verification, integration, and explicit user approval.

## Implementation Sequence

The later implementation plan will use this order:

1. add the prompt definition, registry, catalog, and focused unit tests
2. expose compatibility constants and prove exact template parity
3. migrate Agent, baseline, and tool LLM call sites one group at a time
4. add prompt manifests to evaluation runtime metadata
5. add the safe manifest snapshot to persisted traces
6. run compatibility and full regression verification
7. update README, CHANGELOG, roadmap, and release documentation
8. request code review before integration and version tagging

Each behavior-bearing step starts with a failing test and ends with a focused
commit.

## Success Criteria

P4d is complete when:

- all ten current prompt templates have stable IDs and `v1` definitions
- all eight runtime prompt paths render through the registry
- the two unused compatibility constants remain importable but non-active
- no prompt text changes relative to `v0.4.2-p4c`
- unchanged prompt versions cannot silently change content
- evaluation artifacts record schema `2`, evaluator `p4d`, and active prompt
  versions/fingerprints
- persisted traces record the same safe active manifest
- no artifact or trace stores full prompt templates or rendered prompts
- public APIs, artifact filenames, Agent routes, and parser contracts remain
  compatible
- focused tests and the full project suite pass
- documentation accurately marks P4d complete and preserves the confirmed
  P5a/P5b/background-evaluation route

## Deferred Roadmap

After P4d:

1. P5a implements a configurable DeepSeek semantic correctness and groundedness
   judge through the existing evaluation judge protocol. Its judge prompt must
   be registered and versioned by this P4d foundation.
2. P5b adds SQLite historical evaluation storage and a trend dashboard so runs
   can be compared by model, configuration, evaluator version, and prompt
   manifest.
3. Background Evaluation adds durable run state, progress, cancellation, and
   checkpoint recovery.
4. Trace Drill-down links evaluation failures to trace IDs, nodes, tool calls,
   and eventually the exact prompt ID/version invoked at each LLM call.
5. Behavioral prompt regression can compare selected prompt versions using the
   semantic judge and historical storage rather than relying only on template
   fingerprints.
