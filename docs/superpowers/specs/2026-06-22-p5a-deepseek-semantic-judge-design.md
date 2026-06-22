# P5a DeepSeek Semantic Judge Design

Date: 2026-06-22

Status: Approved

Target version: `v0.5.0-p5a`

## Goal

Add an optional, independently configured DeepSeek LLM-as-a-Judge that scores
evaluation answers for semantic correctness and groundedness without replacing
or mutating the existing deterministic evaluation metrics.

P5a extends the P4c Judge protocol and the P4d prompt-versioning foundation.
The Judge runs after deterministic system scoring, produces additive typed
results, and fails independently per evaluated system result. Existing
evaluation behavior remains the default when the Judge is disabled.

The feature must work through the shared evaluation facade so CLI, FastAPI,
Gradio Dashboard service, matrix, and ablation consumers inherit the same
configuration and result semantics without separate integrations.

## Current Baseline

The design targets `main` commit `e93fca4`, tagged `v0.4.3-p4d`. Local and
remote `main` were synchronized before P5a design began.

The verified P4d baseline is:

- full test suite: `489 passed`
- evaluation schema version: `2`
- evaluator version: `p4d`
- registered prompts: 10
- active runtime prompts: 8

Existing foundations:

- `evaluation.judges.Judge` defines
  `evaluate(question, result) -> JudgeResult`.
- `DisabledJudge` performs no network call.
- `invoke_judge()` converts Judge exceptions into failed results.
- `JudgeResult` supports disabled, completed, and failed states.
- `evaluation.comparison` owns single-system and comparison orchestration.
- `evaluation.metrics` owns deterministic scoring and aggregation.
- `evaluation.evaluate` is the public compatibility facade.
- `prompting` supplies immutable prompt IDs, versions, strict rendering, and
  fingerprints.

Current limitations:

- no live Judge implementation
- no Judge configuration
- no Judge invocation in evaluation orchestration
- no semantic correctness or groundedness fields in result artifacts
- no Judge aggregate metrics or report rendering

## Scope

P5a includes:

- independent `EVALUATION_JUDGE_*` configuration
- disabled-by-default behavior
- an OpenAI-compatible DeepSeek Judge client independent of the evaluated
  system's chat model
- one versioned `evaluation.semantic_judge@v1` prompt
- one Judge call for each successful system result
- simultaneous semantic correctness and groundedness scoring in that call
- strict JSON parsing and `0–4` integer score validation
- normalized `0–1` scores in result artifacts
- groundedness marked unavailable for fallback results
- bounded evidence formatting
- additive per-result Judge status, scores, reasons, model, and prompt metadata
- additive Judge aggregate metrics
- failure isolation for model errors and malformed responses
- sanitized Judge runtime metadata
- shared integration through CLI, FastAPI, Dashboard service, matrix, and
  ablation evaluation paths
- terminal report support
- focused offline tests using fake LLMs and fake Judges
- README, CHANGELOG, roadmap, and release documentation updates

P5a does not include:

- replacing deterministic correctness, source, citation, fallback, or claim
  metrics
- using the evaluated system's LLM configuration as an implicit Judge fallback
- a Dashboard chart, filter, toggle, or other new UI component
- SQLite historical evaluation storage
- historical trend comparison
- background evaluation jobs, cancellation, or recovery
- repeated trials, confidence intervals, or inter-Judge agreement
- model ensembles or multiple Judge providers
- human calibration labels
- automatic prompt optimization
- live DeepSeek calls in the default test suite
- persisting raw Judge prompts or raw model responses

## Confirmed Decisions

### Independent Judge Configuration

The Judge has its own environment variables:

```text
EVALUATION_JUDGE_ENABLED=false
EVALUATION_JUDGE_API_KEY=
EVALUATION_JUDGE_BASE_URL=
EVALUATION_JUDGE_MODEL=
EVALUATION_JUDGE_TEMPERATURE=0
```

There is no fallback to `OPENAI_*`, `OLLAMA_*`, or the evaluated system's
effective model settings.

This prevents accidental self-evaluation, makes Judge changes explicit, and
allows multiple system models to be evaluated against one stable Judge.

When disabled:

- no Judge model is constructed
- no Judge network dependency is initialized
- no Judge call occurs
- existing evaluation output remains valid with additive disabled Judge fields

When enabled, API key, base URL, and model must all be non-empty. Temperature
must be between `0` and `2`.

### Scoring Scale

The Judge returns integer scores from `0` through `4`:

| Score | Meaning |
|---|---|
| `0` | completely incorrect or unsupported |
| `1` | mostly incorrect or mostly unsupported |
| `2` | partially correct or partially supported |
| `3` | mostly correct or mostly supported |
| `4` | fully correct or fully supported |

Artifacts store both:

- raw integer scores
- normalized scores calculated as `raw_score / 4`

The normalized range is `0.0–1.0`. The parser rejects booleans, floats,
numeric strings, negative values, and values above `4`.

### Fallback Semantics

Every result with `fallback_triggered=True` receives:

- a semantic correctness score that judges whether refusing to answer was
  appropriate
- groundedness marked not applicable
- groundedness raw and normalized scores set to `null`

This applies to both correct and incorrect fallbacks. Semantic correctness
distinguishes them.

A non-fallback answer must receive an applicable groundedness score. A Judge
response that contradicts these rules is invalid.

### One Call Per System Result

One Judge request returns both dimensions and both reasons.

For comparison mode, naive and Agentic results are separate system results and
each receives one Judge call. A 36-question comparison can therefore produce
up to 72 Judge calls when enabled.

This keeps correctness and groundedness based on identical context while
avoiding two network calls per result.

### Evidence Selection And Budget

Groundedness evidence is selected in this order:

1. use `relevant_documents` when non-empty
2. otherwise use `retrieved_documents`

The prompt also receives the result's citations as separate selected-evidence
metadata.

Limits:

- at most 8 evidence chunks
- at most 1,200 characters of content per chunk
- preserve `source`, `page`, and `chunk_id`
- preserve input order
- do not mutate the stored `EvaluationResult` evidence

The formatter omits local source paths, file contents beyond the bounded
snippet, credentials, and unrelated result diagnostics.

### Failure Isolation

Judge failure never terminates an evaluation run.

Failures include:

- client invocation exceptions
- blank responses
- invalid JSON
- missing or extra top-level score objects
- invalid applicability
- invalid score types or ranges
- missing or blank reasons

On failure:

- `judge.status` is `failed`
- raw and normalized scores are unavailable
- Judge reasons are empty
- configured model and prompt identity remain available when known
- a sanitized error is recorded
- deterministic result fields are preserved
- evaluation continues with the next system result

If system execution already failed, the Judge is not called. The Judge result
is marked failed with a local system-result-unavailable error so aggregate
completion accurately reflects the missing semantic evaluation.

Errors must not expose API keys. Error messages are limited to the exception
class and a sanitized, bounded message.

### No Dashboard UI Expansion

Dashboard-triggered evaluations use the configured Judge through the shared
facade. Their raw reports contain Judge fields.

P5a does not add visible Judge columns, charts, controls, or filters to Gradio.
Historical visualization belongs to P5b.

## Architecture

```text
CLI / FastAPI / Dashboard / Matrix / Ablation
                       |
                       v
             evaluation.evaluate facade
                       |
             build_configured_judge()
                       |
          +------------+-------------+
          |                          |
          v                          v
   DisabledJudge              DeepSeekJudge
                                      |
                                      v
Runner -> deterministic scoring -> one Judge call -> failure analysis
                                      |
                                      v
                    typed result -> summary -> report/storage
```

The runner remains responsible only for system execution and deterministic
result construction. The orchestration layer owns optional Judge invocation.
The metrics layer owns aggregation of both deterministic and additive Judge
fields.

## Module Design

### `evaluation/judge_config.py`

This module defines:

```python
@dataclass(frozen=True)
class EvaluationJudgeSettings:
    enabled: bool
    api_key: str
    base_url: str
    model: str
    temperature: float
```

Responsibilities:

- load `EVALUATION_JUDGE_*` environment variables
- parse the enabled boolean and temperature
- validate fields only when enabled
- create the independent OpenAI-compatible chat model
- build sanitized Judge runtime metadata

Sanitized metadata:

```json
{
  "enabled": true,
  "provider": "openai_compatible",
  "model": "deepseek-chat",
  "temperature": 0.0
}
```

It excludes API key and base URL to match the project's current runtime
redaction policy.

The client factory uses `ChatOpenAI` with the Judge's independent model, API
key, base URL, and temperature.

### `evaluation/judge_evidence.py`

This pure module owns bounded evidence and citation formatting.

It exposes focused functions that:

- select relevant or retrieved documents
- copy at most 8 records
- normalize source, page, and chunk ID
- truncate normalized content to 1,200 characters
- format citations without source paths or full document content
- return deterministic JSON strings for prompt variables

No LLM or file I/O occurs in this module.

### `evaluation/judge_parsing.py`

This module defines the normalized semantic Judge payload and strict parser.

Expected raw JSON:

```json
{
  "semantic_correctness": {
    "score": 3,
    "reason": "The answer captures the reference meaning."
  },
  "groundedness": {
    "applicable": true,
    "score": 4,
    "reason": "Every factual statement is supported by the supplied evidence."
  }
}
```

For fallback:

```json
{
  "semantic_correctness": {
    "score": 4,
    "reason": "The documents do not answer the question, so refusal is correct."
  },
  "groundedness": {
    "applicable": false,
    "score": null,
    "reason": "No substantive factual answer was provided."
  }
}
```

Parsing rules:

- use `json.loads(raw_text.strip())`
- reject Markdown fences and surrounding prose
- require exactly the two top-level keys
- require exactly `score` and `reason` for semantic correctness
- require exactly `applicable`, `score`, and `reason` for groundedness
- require non-empty string reasons
- reject booleans as scores
- require integer `0–4` scores when applicable
- require `score=null` when groundedness is not applicable
- validate groundedness applicability against `result.fallback_triggered`
- do not clamp, coerce, infer, retry parsing, or repair malformed output

### `evaluation/judges.py`

The existing module remains the Judge boundary and gains:

- `DeepSeekJudge`
- `build_configured_judge()`
- sanitized error formatting

`DeepSeekJudge` dependencies:

- an injected LLM-compatible object
- model name
- prompt definition metadata from `prompting`

Its `evaluate()` flow:

1. select and bound evidence
2. format citations and evidence
3. render `evaluation.semantic_judge`
4. call the LLM exactly once
5. coerce the response to text using the project's established message/text
   compatibility pattern
6. strictly parse the JSON
7. normalize valid integer scores
8. return a completed `JudgeResult`

`DeepSeekJudge.evaluate()` catches its own client and parsing failures so it can
redact the configured Judge API key and return failed results that still include
known model and prompt identity. `invoke_judge()` remains the outer safety net
for arbitrary injected Judge implementations and formats failures without
assuming provider-specific credentials.

`build_configured_judge()`:

- loads settings when none are injected
- returns `DisabledJudge` when disabled
- validates configuration before constructing the model
- returns `DeepSeekJudge` when enabled

Tests can inject either a fake Judge into evaluation orchestration or a fake LLM
into `DeepSeekJudge`.

## Versioned Judge Prompt

P5a adds one active prompt definition:

```text
evaluation.semantic_judge@v1
```

Prompt variables:

- `question`
- `gold_answer`
- `should_answer`
- `fallback_triggered`
- `system_answer`
- `citations`
- `evidence`

The prompt instructs the Judge to:

- evaluate semantic meaning rather than exact wording
- use the gold answer and expected answerability for correctness
- treat an appropriate refusal as semantically correct
- evaluate groundedness only against supplied evidence
- not use outside knowledge for groundedness
- use the `0–4` rubric
- return strict JSON only
- set groundedness not applicable for fallback results
- provide concise reasons without reproducing long evidence passages

P4d version rules apply:

- `v1` is immutable after release
- prompt text changes require a new version
- the fingerprint is pinned in tests
- evaluation runtime metadata automatically records the active Judge prompt

Adding the active Judge prompt increases the active prompt manifest from 8 to
9 definitions, even when the Judge is disabled. The manifest describes the
active project prompt set, not prompts invoked in a specific run.

## Data Model

### `JudgeResult`

The current record expands compatibly:

```python
JudgeStatus = Literal["disabled", "completed", "failed"]

@dataclass(frozen=True)
class JudgeResult:
    status: JudgeStatus
    raw_scores: dict[str, int | None]
    scores: dict[str, float | None]
    reasons: dict[str, str]
    reason: str
    model: str | None
    prompt_id: str | None
    prompt_version: str | None
    prompt_fingerprint: str | None
    error: str | None
```

The existing aggregate `reason` field remains for compatibility. For completed
semantic Judge results it can contain a short general status such as
`"Semantic Judge completed."`; dimension-specific explanations live in
`reasons`.

Factories:

- `JudgeResult.disabled()`
- `JudgeResult.completed(...)`
- `JudgeResult.failed(error, ...)`

The existing call forms remain valid:

```python
JudgeResult.completed({"semantic_correctness": 0.8}, reason="...")
JudgeResult.failed("RuntimeError: unavailable")
JudgeResult(status="completed", scores={"semantic_correctness": 0.8})
```

New raw-score, dimension-reason, model, and prompt fields use defaults so P4c
callers and tests remain compatible.

All nested dictionaries are defensively copied.

### `EvaluationResult`

Add:

```python
judge: JudgeResult
```

The default is `JudgeResult.disabled()`.

External serialization:

```json
{
  "judge": {
    "status": "completed",
    "raw_scores": {
      "semantic_correctness": 3,
      "groundedness": 4
    },
    "scores": {
      "semantic_correctness": 0.75,
      "groundedness": 1.0
    },
    "reasons": {
      "semantic_correctness": "...",
      "groundedness": "..."
    },
    "reason": "Semantic Judge completed.",
    "model": "deepseek-chat",
    "prompt_id": "evaluation.semantic_judge",
    "prompt_version": "v1",
    "prompt_fingerprint": "sha256:...",
    "error": null
  }
}
```

`EvaluationResult.from_compat_dict()` accepts either an existing `JudgeResult`
or a Judge dictionary and defaults missing Judge data to disabled. This
preserves compatibility with pre-P5a payloads.

### `EvaluationSummary`

Add:

- `judge_completed_count: int`
- `judge_failed_count: int`
- `judge_completion_rate: float | None`
- `average_semantic_correctness: float | None`
- `average_groundedness: float | None`
- `groundedness_applicable_count: int`

Aggregation:

- attempted count = completed + failed
- completion rate = completed / attempted
- disabled results are excluded from attempted count
- if attempted count is zero, completion rate is `null`
- semantic average uses completed results with a semantic score
- groundedness average uses completed results with an applicable groundedness
  score
- unavailable averages serialize as `null`
- groundedness applicable count is the number of scores included in the
  groundedness average

### Comparison Summary

Naive and Agentic nested summaries receive the new fields automatically.

The flattened comparison object adds:

- `naive_average_semantic_correctness`
- `agentic_average_semantic_correctness`
- `naive_average_groundedness`
- `agentic_average_groundedness`
- `naive_judge_completion_rate`
- `agentic_judge_completion_rate`

These fields are additive.

## Evaluation Data Flow

### Runner Layer

`evaluation.runners.evaluate_question()` continues to:

- call the system runner
- create deterministic result fields
- capture system exceptions as result data
- measure system latency

Failure analysis moves out of this function so orchestration can preserve the
confirmed order:

```text
deterministic score -> optional Judge -> failure analysis
```

Judge latency is not added to the current system `latency` metric. P5a records
Judge model latency separately only if a dedicated field is included in
`JudgeResult`; otherwise Judge latency remains unreported rather than silently
changing the meaning of existing latency metrics.

P5a will not add a Judge latency field unless implementation planning confirms
it is required by an existing consumer. YAGNI favors preserving current latency
semantics.

### Orchestration Layer

`evaluate_single_system()` and `evaluate_comparison()` accept a `Judge`.

For each system result:

1. run deterministic evaluation
2. if the system result has an execution error, attach a failed Judge result
   without a model call
3. otherwise call `invoke_judge()`
4. attach Judge output
5. attach deterministic failure analysis

The default typed orchestration Judge is `DisabledJudge`.

Comparison mode performs this independently for naive and Agentic results.

### Compatibility Facade

Public facade functions add optional Judge injection:

```python
evaluate_questions(..., judge: Judge | None = None)
evaluate_single_system(..., judge: Judge | None = None)
```

When `judge` is:

- supplied: use the injected Judge
- omitted: call `build_configured_judge()`

Default environment configuration returns `DisabledJudge`, preserving all
existing callers.

CLI, FastAPI, Dashboard service, matrix, and ablation paths already call the
facade and therefore inherit the same configured Judge behavior.

## Runtime Metadata

P5a updates:

- evaluation schema version from `2` to `3`
- evaluator version from `p4d` to `p5a`

`runtime_config` adds:

```json
{
  "judge": {
    "enabled": true,
    "provider": "openai_compatible",
    "model": "deepseek-chat",
    "temperature": 0.0
  }
}
```

The existing prompt manifest includes
`evaluation.semantic_judge@v1` and its fingerprint.

When a custom Judge is injected through the public facade, the artifact records
sanitized runtime metadata for that resolved Judge instead of rebuilding
contradictory Judge metadata from the environment.

No runtime metadata contains:

- Judge API key
- Judge base URL
- rendered Judge prompt
- raw Judge response
- full evidence

## Reporting And Consumer Behavior

### Terminal Report

Single-system reports add:

- Judge completion rate
- average semantic correctness
- average groundedness
- Judge failure count

Comparison reports add naive and Agentic Judge averages and completion rates.

Question-level failed Judge status and sanitized errors remain available in JSON
but do not require verbose terminal output for every successful question.

### JSON Artifacts

All existing artifact names remain unchanged.

Per-result and summary Judge fields are additive. Older payloads without Judge
fields remain readable through defaults and compatibility constructors.

### FastAPI

The existing synchronous evaluation endpoint uses the configured Judge through
the facade.

Its public run summary naturally includes additive Judge aggregate fields. No
new endpoint or request field is added in P5a.

### Dashboard Service

Quick evaluation uses the configured Judge through the facade. `raw_report`
contains Judge results and summaries.

Existing visible metric tables remain unchanged in P5a.

### Matrix And Ablation

When the environment explicitly enables the Judge, matrix and ablation runs
also score each system result.

Documentation must warn that comparison and ablation runs can multiply Judge
calls, latency, and cost. Disabled remains the default.

## Error Handling

Configuration errors:

- disabled configuration never errors because credentials are absent
- enabled configuration with missing values raises before the evaluation starts
- invalid Judge temperature raises a clear configuration error

Per-result errors:

- network, model, and parsing failures become failed `JudgeResult` records
- evaluation continues
- deterministic metrics remain available

Error sanitization:

- redact the configured API key if it appears in an exception message
- redact common bearer-token and API-key patterns
- normalize whitespace
- cap stored error messages at 500 characters
- preserve the exception class

Malformed Judge reasons or scores fail the whole Judge result. P5a does not
partially accept one dimension while rejecting the other.

## TDD And Verification Strategy

Implementation follows red-green-refactor with focused commits.

### Configuration Tests

Cover:

- Judge disabled by default
- independent environment variables
- no fallback to system LLM settings
- enabled configuration requires API key, base URL, and model
- temperature parsing and bounds
- disabled mode constructs no client
- sanitized metadata excludes API key and base URL

### Prompt Tests

Cover:

- exact prompt ID and active `v1`
- pinned fingerprint
- exact variable set
- strict rendering
- active manifest count increases from 8 to 9
- no prompt body changes without a version change

### Parser Tests

Cover:

- valid normal-answer payload
- valid fallback payload
- score normalization
- strict top-level and nested keys
- boolean, float, string, and out-of-range score rejection
- fallback/applicability mismatch
- non-fallback missing groundedness rejection
- blank reason rejection
- Markdown fence and surrounding-text rejection

### Evidence Tests

Cover:

- relevant-document priority
- retrieved-document fallback
- 8-document limit
- 1,200-character truncation
- stable order
- source/page/chunk metadata
- input records are not mutated

### DeepSeek Judge Tests

Cover:

- exactly one fake LLM call
- exact prompt variables
- completed raw and normalized scores
- fallback groundedness `null`
- model and prompt metadata
- response-message coercion
- network exception isolation
- malformed response isolation
- API key redaction from failures

### Orchestration Tests

Cover:

- disabled Judge performs no call
- single-system Judge injection
- comparison invokes once for naive and once for Agentic
- system execution error skips the Judge model call
- Judge failure preserves deterministic metrics
- failure analysis remains attached
- pre-P5a public call signatures remain valid

### Aggregation And Reporting Tests

Cover:

- completion and failure counts
- completion rate denominator
- disabled-only summaries return `null` averages and rate
- semantic average
- groundedness applicable count and average
- fallback exclusion from groundedness average
- flattened comparison fields
- terminal report labels
- JSON compatibility for old and new result dictionaries

### Consumer Regression

Run:

- prompt registry and catalog tests
- Judge configuration, parser, evidence, and implementation tests
- evaluation schemas, runners, metrics, comparison, reporting, and storage
- CLI compatibility smoke tests
- FastAPI route tests
- Dashboard service tests
- matrix tests
- ablation tests
- full project suite

A live DeepSeek smoke test is optional, explicit, and excluded from the default
test suite.

## Documentation And Release

README will document:

- what LLM-as-a-Judge means
- why the Judge is independently configured
- environment variables
- score rubric
- fallback groundedness semantics
- call-count and cost implications
- deterministic metrics remain authoritative independent signals
- Judge limitations and bias

CHANGELOG target:

```text
v0.5.0-p5a - DeepSeek Semantic Judge
```

Roadmap moves P5a into Completed Work and keeps:

1. P5b SQLite historical evaluation and trend dashboard
2. Background Evaluation
3. Trace Drill-down

The release tag is created only after implementation, independent review,
fresh full verification, integration into `main`, and explicit user approval.

## Implementation Sequence

The later implementation plan will:

1. add independent Judge configuration and client construction
2. add the versioned semantic Judge prompt
3. add strict evidence formatting and response parsing
4. implement `DeepSeekJudge`
5. expand Judge, result, and summary schemas compatibly
6. integrate Judge invocation in single and comparison orchestration
7. aggregate and render Judge metrics
8. connect configured Judge behavior through the public facade
9. update runtime metadata and direct consumers
10. run full compatibility verification
11. update documentation and release records
12. request independent review before integration

Each behavior-bearing task starts with a failing test and ends with a focused
commit.

## Success Criteria

P5a is complete when:

- Judge configuration is independent and disabled by default
- disabled evaluation performs no Judge model initialization or call
- every enabled, successfully executed system result receives exactly one Judge
  attempt
- system execution errors receive no Judge model call and record failed Judge
  status locally
- comparison mode independently judges naive and Agentic results
- semantic correctness and groundedness use strict `0–4` integer scores
- normalized scores use `raw / 4`
- semantic correctness is unavailable when no nonblank gold answer exists
- fallback groundedness is unavailable and excluded from its average
- evidence selection and size limits are deterministic
- evidence source URLs cannot expose user information, queries, or fragments
- invalid Judge output fails without partial acceptance
- Judge failures never terminate evaluation
- deterministic metrics are unchanged by Judge output
- per-result Judge model and prompt metadata are reproducible
- evaluation runtime metadata is schema `3`, evaluator `p5a`, and secret-free
- CLI reports Judge aggregates
- API and Dashboard service inherit configured Judge behavior
- no Dashboard UI is added
- old evaluation payloads remain readable
- focused and full tests pass
- documentation accurately describes cost, limitations, and the P5b route

## Deferred Roadmap

After P5a:

1. P5b stores evaluation runs in SQLite and adds prompt-aware historical trend
   comparison.
2. Background Evaluation adds durable job state, progress, cancellation, and
   checkpoint recovery.
3. Trace Drill-down links failed cases to traces and eventually records the
   exact prompt ID/version invoked per LLM call.
4. Behavioral prompt regression compares prompt versions using semantic Judge
   scores stored by P5b.
5. Human-reviewed labels and repeated trials can calibrate Judge reliability
   and estimate variance.
