# P4c Modular Evaluation Framework Design

Date: 2026-06-14

Status: Approved interaction and architecture design; pending written-spec review

Target version: `v0.4.2-p4c`

## Goal

Refactor the current Approach A evaluator into a modular, typed Approach B
evaluation framework without changing the established evaluation behavior or
breaking its consumers.

The current `evaluation/evaluate.py` combines dataset validation, runner
adaptation, per-question scoring, metric aggregation, comparison logic, report
formatting, and artifact persistence in one module. It is directly consumed by
the CLI, FastAPI evaluation service, Gradio Evaluation Dashboard, ablation
runner, and compatibility tests.

P4c separates those responsibilities behind stable interfaces while keeping
the current public API and JSON contracts intact. This provides a reliable
foundation for later LLM judges, alternative storage backends, prompt
regression tests, historical run comparison, and background evaluation jobs.

## Scope

P4c includes:

- typed internal records for evaluation questions, per-question results,
  summaries, comparison reports, and runtime metadata
- dedicated modules for dataset loading, runner execution, deterministic
  metrics, comparison, reporting, judges, and result storage
- a compatibility facade in `evaluation/evaluate.py`
- a metric registry that keeps metric calculation outside runner control flow
- a runner protocol and adapters for one-argument and history-aware functions
- a judge protocol with a disabled default implementation
- a result-store protocol with an atomic JSON implementation
- additive evaluator and schema version metadata
- preserved runtime snapshots for model, temperature, feature flags,
  retriever, reranker, and vectorstore configuration
- focused tests for each new module and compatibility regression tests
- README and roadmap updates for the modular framework

P4c does not include:

- changing the current correctness or relevance algorithms
- a live DeepSeek or other LLM-as-judge implementation
- SQLite or remote result storage
- background evaluation jobs, progress polling, cancellation, or recovery
- repeated trials, confidence intervals, or statistical significance
- historical trend views
- prompt regression execution
- changes to the Gradio Evaluation Dashboard user interface

## Confirmed Decisions

### Compatibility Strategy

P4c uses a compatibility-first migration.

The following functions remain importable from `evaluation.evaluate`:

```python
load_eval_questions(...)
evaluate_questions(...)
format_report(...)
write_evaluation_artifacts(...)
main(...)
```

Their existing arguments, default behavior, return dictionaries, CLI
parameters, and artifact file names remain compatible. FastAPI, Gradio, and
ablation code can continue calling this facade during P4c.

The existing report fields are not removed or renamed. New metadata is
additive and must not prevent older dashboard or analysis code from reading a
report.

### Internal Record Strategy

Internal domain records use Python dataclasses.

Dataclasses provide explicit fields and type annotations without introducing
Pydantic serialization behavior into the evaluator's internal calculations.
Conversion to JSON-ready dictionaries occurs at the compatibility and storage
boundaries.

Pydantic remains appropriate for FastAPI request and response schemas, but is
not required for the internal P4c evaluation domain.

### Extension Strategy

P4c implements protocols and one conservative default for judges and result
storage:

- `Judge`: protocol plus a disabled implementation that performs no model call
- `ResultStore`: protocol plus an atomic JSON file implementation

DeepSeek judge execution and SQLite storage remain explicit roadmap items.
This keeps P4c focused on stable architecture rather than adding new scoring
behavior and persistence semantics at the same time.

## Architecture

```text
CLI / FastAPI / Gradio / Ablation
                 |
                 v
       evaluation.evaluate
        compatibility facade
                 |
                 v
     dataset -> runners -> metrics
                     |         |
                     v         v
                   judges   comparison
                       \       /
                        v     v
                    typed report
                       |     |
                       v     v
                  reporting storage
```

The facade accepts and returns the current dictionary contracts. The internal
pipeline operates on typed records.

## Module Design

### `evaluation/schemas.py`

This module defines the evaluation domain records and JSON conversion helpers.
It contains no file I/O and invokes no system runner.

Primary records:

- `EvaluationQuestion`
- `EvaluationResult`
- `EvaluationSummary`
- `PairedEvaluationResult`
- `ComparisonSummary`
- `EvaluationReport`
- `RuntimeMetadata`
- `JudgeResult`

Fields that can legitimately be unavailable, such as citation verification
metrics when verification is disabled, remain optional. They must serialize to
`null`, not an invented zero.

The records expose explicit conversion helpers for the established external
dictionary shape. Conversion helpers centralize compatibility aliases such as
`answerable` and `should_answer`.

### `evaluation/dataset.py`

This module owns:

- loading the evaluation JSON dataset
- validating that the root value is a list
- normalizing IDs and question types
- normalizing legacy `expected_source` into `expected_sources`
- validating `answerable`, `should_answer`, and `expected_behavior`
- validating and normalizing chat history
- returning `EvaluationQuestion` records

Validation messages retain the current semantic wording so existing CLI and
test expectations remain useful.

The public compatibility function returns dictionaries, while the internal
dataset API returns typed records.

### `evaluation/runners.py`

This module defines the callable runner contract and executes one question
against one system.

Responsibilities:

- adapt plain functions to a common history-aware runner interface
- preserve support for existing one-argument injected test runners
- invoke history-aware runners with `question` and `chat_history`
- measure latency through an injected timer
- validate that runner payloads are mappings with a string answer
- convert system exceptions into per-question error results

Runner failures remain data within an evaluation run. A failure in one
question does not stop the remaining questions.

### `evaluation/metrics.py`

This module owns deterministic per-question scoring and aggregate metrics.

It migrates the current behavior for:

- answer and fallback detection
- correctness
- context relevance
- source hit
- citation hit
- keyword hit
- fallback correctness
- claim support and unsupported-claim counts
- citation verification pass rate
- retry, retrieval, latency, token, and cost aggregation
- failure type aggregation

A small metric registry describes metric names and calculation functions. The
registry makes metric discovery and testing explicit, but does not introduce a
general plugin framework or dynamic imports.

After deterministic scoring, the metrics layer calls the existing independent
`failure_analyzer.py` classifier and attaches its diagnostic result. The
analyzer remains a separate module and does not enter runner control flow.

P4c must preserve the numerical output of the current deterministic metrics for
the same inputs. Metric quality improvements are a separate milestone.

### `evaluation/comparison.py`

This module owns Naive RAG versus Agentic RAG pairing and comparison summary
construction.

It evaluates both systems against the same normalized question order and
produces the current paired result shape:

```python
{
    "question": "...",
    "requires_rewrite": False,
    "naive": {...},
    "agentic": {...},
}
```

The current flattened comparison summary keys remain available for existing
reports and dashboards.

### `evaluation/judges.py`

This module defines the future semantic judge boundary:

```python
class Judge(Protocol):
    def evaluate(
        self,
        question: EvaluationQuestion,
        result: EvaluationResult,
    ) -> JudgeResult:
        ...
```

P4c provides a disabled judge that returns an unavailable result and performs
no network call. Judge output is additive and cannot overwrite deterministic
metrics.

Future LLM judges can be introduced through this protocol with explicit
configuration, prompt versions, model metadata, and failure handling.

### `evaluation/reporting.py`

This module contains pure renderers for:

- single-system terminal reports
- comparison terminal or Markdown reports

It performs no runner calls and no file I/O. Existing report labels and
question-level diagnostic fields remain compatible.

### `evaluation/storage.py`

This module defines:

```python
class ResultStore(Protocol):
    def save(self, run_id: str, payload: Mapping[str, Any]) -> str:
        ...

    def load(self, run_id: str) -> dict[str, Any] | None:
        ...
```

`JsonResultStore` writes through a temporary file and atomically replaces the
target file. It supports reading saved payloads and provides a stable boundary
for future SQLite or object-storage implementations.

The compatibility artifact writer continues producing:

- `baseline_result.json`
- `agentic_result.json`
- `comparison_result.json`

### `evaluation/evaluate.py`

This module becomes a compatibility facade and CLI entrypoint.

It owns:

- legacy public function forwarding
- dependency assembly
- CLI argument parsing
- top-level handling for dataset loading errors

It does not contain dataset normalization, metric calculations, comparison
logic, report rendering, or JSON serialization details.

The target is a focused module of approximately 150 lines. The line count is a
design guide, not a release condition; clear responsibility boundaries are the
actual requirement.

## Data Flow

### Single-System Evaluation

```text
raw question dictionaries
        |
        v
dataset.normalize_questions
        |
        v
EvaluationQuestion records
        |
        v
runners.evaluate_question
        |
        v
validated raw system output
        |
        v
metrics.score_result
        |
        v
EvaluationResult records
        |
        v
metrics.aggregate_results
        |
        v
EvaluationReport
        |
        +--> reporting renderer
        |
        `--> compatibility dict / ResultStore
```

### Comparison Evaluation

The same normalized question record is passed to both runners. Results are
scored independently, paired in dataset order, summarized independently, and
then passed to the comparison summary builder.

This preserves a fair Naive-versus-Agentic comparison and avoids each system
loading or normalizing a different dataset representation.

## Runtime Metadata

Runtime snapshots continue excluding secrets, credential-bearing URLs, and
local persistence paths.

P4c adds:

```json
{
  "schema_version": 1,
  "evaluator_version": "p4c"
}
```

These fields are additive. `schema_version: 1` identifies the preserved
external report contract rather than claiming a new incompatible V2 schema.

The snapshot continues recording:

- enabled Agent features
- LLM provider, model, and temperature
- dense, BM25, hybrid, and fusion retrieval configuration
- reranker enablement, model, candidate count, and top-N
- vectorstore collection name

Prompt version metadata remains a roadmap item until prompts are moved into the
planned versioned prompt registry.

## Error Handling

P4c distinguishes case-level failures from run-level failures.

Case-level failures include:

- runner exceptions
- malformed runner payloads
- invalid required answer fields
- invalid list-shaped citation, claim, or retrieval fields

These failures produce an `EvaluationResult` with an `error` field and
participate in the summary error count.

Malformed optional token-usage and estimated-cost values continue to be
excluded from aggregation rather than converting the whole question into an
execution failure.

Run-level failures include:

- unreadable or invalid dataset files
- an invalid dataset root
- incompatible question schema
- invalid evaluator construction or storage configuration

These failures stop the run before a misleading summary is generated.

Judge failures are recorded as judge diagnostics. They do not erase
deterministic scores or convert a successful system execution into a runner
failure.

Storage writes use temporary files and atomic replacement so interrupted
writes do not leave apparently complete artifacts.

## Compatibility Contract

P4c is accepted only if all of the following remain true:

- current imports from `evaluation.evaluate` work
- current public function signatures remain callable
- current CLI flags remain valid
- current single-system and comparison result keys remain present
- current numerical deterministic metrics remain unchanged for fixture inputs
- current artifact file names and top-level layouts remain readable
- P4b dashboard formatting can consume current and newly generated artifacts
- FastAPI evaluation routes continue returning their existing response schema
- ablation variants continue using the same evaluator entrypoint

The implementation may migrate consumers to narrower internal imports after
the facade is stable, but such migration is not required for P4c completion.

## Testing Strategy

P4c uses test-driven, incremental extraction.

### Focused Tests

- `tests/test_evaluation_schemas.py`
  - typed-record construction
  - optional metric serialization
  - compatibility dictionary conversion

- `tests/test_evaluation_dataset.py`
  - rich and legacy schema normalization
  - chat history validation
  - answerability and expected-behavior validation
  - malformed file and root handling

- `tests/test_evaluation_runners.py`
  - one-argument runner adaptation
  - history-aware runner invocation
  - latency measurement
  - exception and malformed-payload conversion

- `tests/test_evaluation_metrics.py`
  - per-question deterministic scoring
  - aggregate rates and averages
  - zero denominators
  - citation verification availability
  - malformed token and cost filtering
  - metric registry behavior

- `tests/test_evaluation_comparison.py`
  - paired question order
  - independent summaries
  - flattened comparison fields

- `tests/test_evaluation_reporting.py`
  - single-system report rendering
  - comparison report rendering
  - unavailable metrics

- `tests/test_evaluation_storage.py`
  - atomic JSON save
  - load and not-found behavior
  - compatibility artifact names and layouts

- `tests/test_evaluation_judges.py`
  - disabled judge behavior
  - judge failures remaining separate from deterministic metrics

### Compatibility Tests

The existing `tests/test_evaluate.py` remains a facade contract suite. It may
be reduced only after equivalent focused tests exist; it must not be deleted
as part of an unverified mechanical split.

Regression verification includes:

- evaluation tests
- dashboard service and formatter tests
- FastAPI evaluation route tests
- ablation tests
- the complete project test suite

No live model call is required for the core P4c tests.

## Documentation And Release

README updates will:

- describe the modular Approach B evaluation architecture
- distinguish deterministic metrics from optional future judges
- explain compatibility with the Evaluation Dashboard and ablation artifacts
- preserve the future roadmap for DeepSeek judge, SQLite storage, historical
  runs, prompt versioning, and background execution

The intended release marker is `v0.4.2-p4c` after implementation, full
verification, integration, and explicit user approval.

## Implementation Sequence

The implementation plan will extract behavior in this order:

1. add typed schemas and compatibility serialization
2. extract dataset normalization
3. extract deterministic metrics
4. extract runner execution
5. extract comparison logic
6. extract reporting
7. add judge and result-store protocols with conservative defaults
8. reduce `evaluation/evaluate.py` to the compatibility facade
9. verify Dashboard, FastAPI, ablation, and full-suite compatibility
10. update documentation and release metadata

Each extraction must leave the relevant tests passing before the next
responsibility moves.

## Success Criteria

P4c is complete when:

- the evaluator is split into focused modules with explicit typed boundaries
- `evaluation/evaluate.py` is a compatibility facade rather than the
  implementation center
- deterministic metric results match the pre-P4c evaluator for the same inputs
- existing CLI, API, Dashboard, and ablation consumers continue working
- JSON artifacts remain reproducible and compatible
- optional Judge and ResultStore interfaces are testable and replaceable
- runtime metadata includes additive evaluator and schema version fields
- focused module tests and the full project test suite pass
- README documents the architecture and honest limitations

## Deferred Roadmap

After P4c:

- implement a configurable DeepSeek semantic correctness and groundedness judge
- implement SQLite-backed run storage and historical run queries
- introduce a higher-level `EvaluationEngine` that composes runners, metrics,
  judges, and stores
- move prompts into a versioned registry and record prompt versions in runtime
  metadata
- add prompt regression and historical-run comparison
- add background runs, progress, cancellation, and checkpoint recovery
- share run IDs between Gradio and FastAPI
- link failed cases to trace IDs and node-level execution views
- add repeated trials, confidence intervals, and human-reviewed reference labels
