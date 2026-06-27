# P5b SQLite Historical Evaluation + Trend Dashboard Design

Date: 2026-06-27

Branch: `codex/p5b-sqlite-eval-history`

Target version: `v0.5.1-p5b`

## Status

Approved design direction:

- use SQLite as a sidecar historical index
- keep JSON evaluation artifacts as the complete compatibility payloads
- add prompt-aware trend queries and a minimal Dashboard history view
- defer background execution and trace drill-down

## Context

P5a completed the optional independent DeepSeek semantic Judge and advanced
evaluation artifacts to runtime schema `3` with evaluator version `p5a`.
Evaluation reports now include deterministic metrics, optional semantic Judge
scores, safe runtime metadata, and prompt manifests.

Current persistence is still file-oriented:

- `evaluation/storage.py` provides `ResultStore` and `JsonResultStore` for
  atomic JSON payloads.
- `evaluation.evaluate.write_evaluation_artifacts()` writes compatibility
  artifact names such as `baseline_result.json`, `agentic_result.json`, and
  `comparison_result.json`.
- `api.services.evaluation.EvaluationService` writes FastAPI-triggered run
  wrappers to `EVALUATION_RUN_DIR`.
- the Gradio Evaluation Dashboard can run synchronous quick comparisons and
  load one saved ablation snapshot.
- existing `experiments/results/*.json` artifacts are older payloads and may
  not contain `runtime_config.schema_version` or `runtime_config.evaluator_version`.

P5b should make evaluation results comparable over time without replacing the
stable JSON contracts that CLI, FastAPI, Dashboard, and experiments already
consume.

## Goals

1. Store evaluation run history in SQLite so multiple runs can be listed,
   filtered, and compared without scanning JSON files every time.
2. Preserve JSON artifacts as the full source of detail for compatibility and
   later trace drill-down.
3. Record prompt-aware metadata from P4d/P5a runtime manifests so trends can be
   grouped by evaluator version, model configuration, and prompt manifest.
4. Add a minimal trend view to the Evaluation Dashboard using existing Gradio
   table patterns.
5. Add FastAPI service and route support for listing historical runs and trend
   rows without changing the existing `/evaluation/run` and
   `/evaluation/{run_id}` behavior.
6. Keep history write failures isolated from the primary evaluation result so a
   SQLite problem does not invalidate a completed JSON artifact.
7. Use the Python standard library `sqlite3`; do not add a database dependency.

## Non-Goals

P5b does not implement:

- background evaluation jobs
- progress, cancellation, retries, or checkpoint recovery
- trace drill-down from failed cases
- full per-question relational storage
- online prompt editing or prompt selection
- replacement of existing JSON artifacts
- statistical significance or repeated-trial analysis
- secret storage, full prompt template storage, or rendered prompt storage

## Selected Approach

Use a SQLite sidecar index.

JSON remains the durable compatibility artifact. SQLite stores normalized,
queryable summaries:

- one row per historical run
- one row per system or variant metric set within a run
- one row per failure type count within a system or variant
- runtime config and prompt manifest as sanitized JSON text

This keeps P5b small enough to implement safely while still unlocking useful
trend views. Later milestones can add per-question and trace-aware tables
without changing the P5b run and metric tables.

## Alternatives Considered

### 1. SQLite sidecar index

This is the selected design.

Pros:

- preserves all existing artifact contracts
- low migration risk
- supports trend queries without rescanning JSON files
- leaves detailed drill-down to JSON and future trace work

Cons:

- per-question trend and trace drill-down remain deferred
- SQLite rows must be derived from several payload shapes

### 2. SQLite canonical store

The system could move every run, question result, Judge score, and failure row
into relational tables and export JSON only for compatibility.

Pros:

- strong foundation for trace drill-down
- richer SQL queries for per-question analysis

Cons:

- larger migration
- higher compatibility risk
- overreaches the P5b milestone

### 3. Read-time JSON scanner

The Dashboard could scan `experiments/results` and `data/evaluation_runs` on
every page load.

Pros:

- smallest implementation
- no database lifecycle

Cons:

- not real historical storage
- slow and unstable as artifacts accumulate
- does not satisfy the SQLite-backed roadmap item

## Configuration

Add two settings to `config.Settings` and `.env.example`:

```dotenv
EVALUATION_HISTORY_ENABLED=true
EVALUATION_HISTORY_DB=./data/evaluation_history.sqlite3
```

Behavior:

- when enabled, evaluation writers attempt to append a history record after
  writing the JSON artifact
- when disabled, JSON artifact behavior is unchanged
- the default DB path is under `data/`, already ignored by git
- `*.sqlite3` and `*.db` are already ignored by git
- local absolute paths are not added to runtime config snapshots

## Runtime Metadata Versioning

P5b advances evaluation runtime metadata:

- `EVALUATION_SCHEMA_VERSION = 4`
- `EVALUATOR_VERSION = "p5b"`

Schema version `4` means the report can be indexed into the P5b history store.
It does not mean old payloads are unreadable. Historical import and trend
queries must accept legacy payloads:

- missing schema version becomes `NULL` in SQLite and displays as `legacy`
- missing evaluator version becomes `legacy`
- missing prompt manifests produce an empty manifest fingerprint

## Proposed Modules

### `evaluation/history_store.py`

Owns SQLite persistence and extraction from existing report shapes.

Primary responsibilities:

- open SQLite connections from a configured DB path
- initialize schema idempotently
- insert or replace historical run metadata
- insert normalized system or variant metrics
- insert normalized failure counts
- list recent runs
- query trend rows
- import existing JSON artifacts

This module depends only on:

- Python standard library modules
- sanitized payload dictionaries passed by callers

It must not import Gradio, FastAPI, Agent runners, or model clients.

### `evaluation/history_service.py`

Provides a small application-level facade over the store.

Primary responsibilities:

- load settings
- respect `EVALUATION_HISTORY_ENABLED`
- build safe write status records
- expose dashboard/API-friendly list and trend methods
- isolate history write failures

This layer allows CLI, FastAPI, and Dashboard code to call the same behavior
without directly handling SQLite exceptions.

### `evaluation/dashboard_formatters.py`

Add pure formatting helpers for history rows:

- recent run table rows
- trend metric table rows
- trend summary labels

The formatters should accept already-normalized records and return lists for
Gradio tables. They should not open SQLite.

### `evaluation/dashboard_models.py`

Add table contracts:

- `HISTORY_RUN_COLUMNS`
- `HISTORY_TREND_COLUMNS`
- `HistoryRunRow`
- `HistoryTrendRow`
- `HistoryDashboardView`

### `evaluation/dashboard_service.py`

Add read-only history methods:

- `load_history_snapshot(limit: int = 20) -> HistoryDashboardView`
- `load_history_trends(metric: str, limit: int = 20) -> HistoryDashboardView`

History failures should produce a Dashboard view with status `unavailable` and
a UI-safe message.

### `ui/gradio_app.py`

Add a minimal Evaluation sub-tab named `History Trends`.

The tab contains:

- refresh button
- status markdown
- recent runs table
- metric selector
- trend table

It should reuse the existing Evaluation tab service object and Gradio table
patterns. It should not run evaluations or write artifacts.

### `api/schemas.py`

Add response models:

- `EvaluationHistoryRun`
- `EvaluationHistoryListResponse`
- `EvaluationTrendRow`
- `EvaluationTrendResponse`

Existing `EvaluationRunResponse` remains compatible. Additive fields are
allowed only if tests show existing callers remain valid.

### `api/routes/evaluation.py`

Add read routes:

```text
GET /evaluation/history
GET /evaluation/history/trends
```

Query parameters:

- `limit`: default `20`, bounded to a safe maximum
- `metric`: default `correctness_score`
- `system`: optional system or variant id filter

The existing route `GET /evaluation/{run_id}` must remain available. The
literal `/history` routes must be declared before `/{run_id}` to avoid route
capture.

## SQLite Schema

Use `PRAGMA foreign_keys = ON` per connection.

### `evaluation_history_meta`

```sql
CREATE TABLE IF NOT EXISTS evaluation_history_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

Required keys:

- `schema_version = "1"`

This is the SQLite database schema version, separate from evaluation runtime
schema version.

### `evaluation_runs`

```sql
CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    workspace_id TEXT,
    status TEXT NOT NULL,
    mode TEXT NOT NULL,
    schema_version INTEGER,
    evaluator_version TEXT,
    result_path TEXT,
    question_count INTEGER NOT NULL DEFAULT 0,
    question_ids_json TEXT NOT NULL DEFAULT '[]',
    runtime_config_json TEXT NOT NULL DEFAULT '{}',
    prompt_manifest_json TEXT NOT NULL DEFAULT '{}',
    prompt_manifest_hash TEXT NOT NULL DEFAULT '',
    summary_json TEXT NOT NULL DEFAULT '{}'
);
```

`source` values:

- `cli`
- `api`
- `ablation`
- `matrix`
- `import`

`mode` values:

- `single`
- `comparison`
- `matrix`
- `ablation`

### `evaluation_system_metrics`

```sql
CREATE TABLE IF NOT EXISTS evaluation_system_metrics (
    run_id TEXT NOT NULL,
    system_id TEXT NOT NULL,
    system_label TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL,
    metric_text TEXT,
    PRIMARY KEY (run_id, system_id, metric_name),
    FOREIGN KEY (run_id) REFERENCES evaluation_runs(run_id) ON DELETE CASCADE
);
```

Store numeric metrics in `metric_value`. Store non-numeric display values such
as `N/A` in `metric_text`. Most trend queries use `metric_value`.

Initial metric allowlist:

- `correctness_score`
- `context_relevance_score`
- `citation_hit_rate`
- `fallback_accuracy`
- `unsupported_claim_count`
- `average_latency`
- `average_retry_count`
- `error_count`
- `average_semantic_correctness`
- `average_groundedness`
- `judge_completion_rate`

The allowlist mirrors current dashboard and Judge-visible metrics. Other
summary fields remain available in `summary_json`.

### `evaluation_failure_counts`

```sql
CREATE TABLE IF NOT EXISTS evaluation_failure_counts (
    run_id TEXT NOT NULL,
    system_id TEXT NOT NULL,
    failure_type TEXT NOT NULL,
    count INTEGER NOT NULL,
    PRIMARY KEY (run_id, system_id, failure_type),
    FOREIGN KEY (run_id) REFERENCES evaluation_runs(run_id) ON DELETE CASCADE
);
```

This table is derived from `summary.failure_type_counts` when available. Legacy
artifacts without failure counts simply insert no failure-count rows.

### Indexes

```sql
CREATE INDEX IF NOT EXISTS idx_evaluation_runs_created_at
ON evaluation_runs(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_evaluation_runs_evaluator_version
ON evaluation_runs(evaluator_version);

CREATE INDEX IF NOT EXISTS idx_evaluation_runs_prompt_manifest_hash
ON evaluation_runs(prompt_manifest_hash);

CREATE INDEX IF NOT EXISTS idx_evaluation_metrics_name_system
ON evaluation_system_metrics(metric_name, system_id);
```

## Prompt Manifest Hashing

Runtime config already contains a safe prompt manifest shaped like:

```json
{
  "agent.answer_generation": {
    "version": "v1",
    "fingerprint": "sha256:..."
  },
  "evaluation.semantic_judge": {
    "version": "v1",
    "fingerprint": "sha256:..."
  }
}
```

P5b stores:

- `prompt_manifest_json`: the manifest JSON exactly as sanitized in runtime
  config
- `prompt_manifest_hash`: SHA-256 of canonical JSON with sorted keys

This hash groups runs by the effective prompt set without storing templates or
rendered prompt text.

## Payload Extraction Rules

The history writer accepts a top-level payload and extracts run and metric rows
without requiring callers to know the payload shape.

### Single-system report

Shape:

```json
{
  "system": "agentic_rag",
  "runtime_config": {},
  "summary": {},
  "results": []
}
```

Rules:

- mode: `single`
- system id: `agentic` for `agentic_rag`, `naive` for `naive_rag`, otherwise
  sanitized `system`
- metrics from `summary`
- question count from `summary.total_questions` or `len(results)`

### Comparison report

Shape:

```json
{
  "runtime_config": {},
  "summary": {
    "mode": "comparison",
    "naive": {},
    "agentic": {}
  },
  "results": []
}
```

Rules:

- mode: `comparison`
- insert `naive` metrics from `summary.naive`
- insert `agentic` metrics from `summary.agentic`
- store comparison summary in `summary_json`

### Matrix report

Shape:

```json
{
  "summary": {
    "mode": "matrix",
    "variants": {
      "naive": {},
      "agentic": {},
      "agentic_reranker": {}
    }
  },
  "results": []
}
```

Rules:

- mode: `matrix`
- one system metric group per `summary.variants` key
- runtime config may be missing in legacy matrix artifacts

### Ablation payload

Shape:

```json
{
  "kind": "ablation_result",
  "phase": "P0b",
  "runs": [
    {
      "id": "v0_naive",
      "method": "Naive RAG",
      "status": "completed",
      "runtime_config": {},
      "summary": {},
      "results": []
    }
  ]
}
```

Rules:

- mode: `ablation`
- one `evaluation_runs` row for the artifact
- one system metric group per completed or completed-with-errors variant
- system id: variant id
- system label: `"{id} {method}"`
- parent runtime metadata comes from the first run with runtime config, if
  available
- variant-specific runtime config remains in `summary_json`; a normalized
  variant runtime table is deferred until a later milestone requires it

### FastAPI wrapper

Shape:

```json
{
  "run_id": "eval_...",
  "workspace_id": "workspace_1",
  "status": "completed",
  "summary": {},
  "result_path": "...",
  "report": {}
}
```

Rules:

- use wrapper `run_id`
- use wrapper `workspace_id`
- use wrapper `status`
- extract metrics from nested `report`
- store wrapper `result_path`

## Run ID Policy

Callers may provide a run id. If a payload lacks one, the history service
generates a stable import id:

```text
hist_<sha256 of canonical payload and result path, first 16 hex chars>
```

For live CLI writes, the implementation should generate a timestamp-based id:

```text
eval_YYYYMMDDTHHMMSSffffffZ_<8 random hex chars>
```

Run ids are validated with the existing safe file-stem pattern:

```text
[A-Za-z0-9_.-]+
```

Invalid ids are rejected by the store and returned as a failed history write
status by the service.

## Write Path

### CLI compatibility artifacts

`write_evaluation_artifacts(report, output_dir)` keeps writing JSON exactly as
before. After JSON writes succeed, it calls the history service once for the
full report.

The history record uses:

- source: `cli`
- result path: the comparison artifact path for comparison reports, otherwise
  the single-system artifact path
- workspace id: `None`

If history is disabled or unavailable, JSON output remains valid.

### FastAPI evaluation runs

`EvaluationService.run_evaluation()` keeps writing its current JSON wrapper.
After the wrapper write succeeds, it records the wrapper in SQLite.

The public response remains:

- `run_id`
- `workspace_id`
- `status`
- `summary`
- `result_path`

The P5b API response model does not add `history_status`. The persisted JSON
wrapper can store a private `history` status object for diagnostics, but the
public response stays compatible.

### Ablation and matrix artifacts

P5b adds import helpers so historical dashboards can index existing artifacts:

```python
import_history_artifact(path: str | Path, source: str = "import") -> HistoryWriteStatus
```

The P5b implementation provides this importer helper for ablation and matrix
artifacts. It does not add new ablation or matrix runner flags, and it does not
rewrite ablation or matrix runner internals.

## Failure Handling

History write is sidecar persistence. It must not corrupt or block the primary
JSON artifact once the evaluation itself completed.

Return statuses:

- `stored`: SQLite write succeeded
- `disabled`: history is disabled by config
- `failed`: SQLite write failed or payload could not be indexed

Failure details:

- include a sanitized error type and message in the status object
- do not include secrets, API keys, full local stack traces, or prompt text
- Dashboard and API list endpoints return `unavailable` or HTTP `503` when the
  history DB cannot be opened for read
- write failures do not remove JSON artifacts

## Query Behavior

### Recent runs

Return most recent runs ordered by `created_at DESC`.

Each row contains:

- run id
- created at
- source
- workspace id
- status
- mode
- evaluator version or `legacy`
- schema version or `legacy`
- prompt manifest hash prefix
- question count
- result path

### Trends

Trend query inputs:

- metric name
- optional system id
- limit

Returned rows are ordered by `created_at ASC` so the table can be read as a
timeline.

Each row contains:

- created at
- run id
- system label
- evaluator version
- prompt manifest hash prefix
- metric name
- metric value

If no rows match, return an empty list and a completed status message.

## Dashboard UX

Add an Evaluation sub-tab named `History Trends`.

Controls:

- `Refresh History`
- metric dropdown with the initial metric allowlist
- optional system dropdown populated from available history rows

Displays:

- status message
- recent runs table
- trend rows table

Initial status when DB is empty:

```text
No historical evaluation runs found. Run an evaluation or import an artifact to populate SQLite history.
```

This tab is deliberately table-first. It avoids charting dependencies and keeps
P5b focused on storage and trend data contracts.

## FastAPI UX

Add:

```text
GET /evaluation/history?limit=20
GET /evaluation/history/trends?metric=correctness_score&system=agentic&limit=20
```

Response examples:

```json
{
  "runs": [
    {
      "run_id": "eval_20260627T120000000000Z_ab12cd34",
      "created_at": "2026-06-27T12:00:00.000000Z",
      "source": "api",
      "workspace_id": "workspace_1",
      "status": "completed",
      "mode": "comparison",
      "schema_version": 4,
      "evaluator_version": "p5b",
      "prompt_manifest_hash": "sha256:...",
      "question_count": 5,
      "result_path": "data/evaluation_runs/eval_...json"
    }
  ]
}
```

```json
{
  "metric": "correctness_score",
  "system": "agentic",
  "rows": [
    {
      "created_at": "2026-06-27T12:00:00.000000Z",
      "run_id": "eval_20260627T120000000000Z_ab12cd34",
      "system_id": "agentic",
      "system_label": "Agentic RAG",
      "evaluator_version": "p5b",
      "prompt_manifest_hash": "sha256:...",
      "metric_name": "correctness_score",
      "metric_value": 0.83
    }
  ]
}
```

## Compatibility Requirements

Existing behavior must remain valid:

- `JsonResultStore` atomic save/load behavior
- compatibility artifact filenames
- `evaluation.evaluate` public facade
- CLI output formatting
- FastAPI `/evaluation/run`
- FastAPI `/evaluation/{run_id}`
- Gradio Quick Compare
- Gradio Ablation Snapshot
- P5a Judge disabled-by-default semantics
- schema 1-3 JSON payload readability

P5b adds history but does not require callers to use it.

## Security and Privacy

SQLite stores only sanitized evaluation metadata and summaries.

It must not store:

- API keys
- authorization headers
- full prompt templates
- rendered prompts
- raw model request bodies
- unsanitized base URLs with credentials
- full exception stack traces

Runtime config persistence remains bounded by the existing sanitized runtime
snapshot rules. Prompt manifests store only prompt id, version, and fingerprint.

## Testing Strategy

### Store tests

Create `tests/test_evaluation_history_store.py` covering:

- schema initialization is idempotent
- disabled history returns `disabled`
- single-system report records one run and one system metric set
- comparison report records naive and agentic metric sets
- matrix report records all variants
- ablation payload records variant metric sets
- FastAPI wrapper extracts nested report and workspace id
- legacy artifacts without schema/evaluator versions display as legacy
- prompt manifest hash is deterministic for reordered manifest keys
- invalid run ids are rejected safely
- failed SQLite writes return `failed` without deleting JSON artifacts
- trend query orders rows chronologically

### Evaluation writer tests

Extend `tests/test_evaluation_storage.py` and `tests/test_evaluate.py`:

- JSON artifacts remain unchanged in shape
- history writer is called only after JSON write succeeds
- history disabled keeps existing behavior
- runtime config advances to schema `4` and evaluator `p5b`

### FastAPI tests

Extend `tests/test_fastapi_routes.py`:

- `/evaluation/history` returns recent runs
- `/evaluation/history/trends` returns trend rows
- `/evaluation/history` is not captured by `/{run_id}`
- existing run and get routes still pass

### Dashboard tests

Extend `tests/test_dashboard_service.py` and `tests/test_gradio_app.py`:

- history snapshot returns table-ready recent runs and trends
- unavailable DB returns UI-safe unavailable view
- empty history returns completed view with empty tables and clear message
- existing quick comparison and ablation snapshot behavior remains unchanged

### Full verification

Run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m compileall prompting agent rag api evaluation experiments baseline tools observability
git diff --check
```

## Documentation Updates

P5b implementation should update:

- `.env.example`
- `README.md`
- `CHANGELOG.md`
- `docs/github_release_checklist.md`
- `docs/superpowers/plans/2026-06-27-p5b-sqlite-evaluation-history.md`

Documentation must state:

- SQLite history is a sidecar index, not a replacement for JSON artifacts
- the default DB path is local ignored runtime data
- old JSON artifacts can be imported and appear as legacy when metadata is
  missing
- Dashboard trends are synchronous read-only summaries
- Background Evaluation and Trace Drill-down remain future work

## Acceptance Criteria

P5b is complete when:

- runtime metadata reports schema `4` and evaluator `p5b`
- JSON artifact compatibility tests still pass
- SQLite schema initializes automatically
- new evaluation writes record a history run when history is enabled
- history can be disabled without changing JSON outputs
- legacy JSON artifacts can be imported and displayed as legacy
- Dashboard exposes recent runs and trend rows
- FastAPI exposes recent runs and trend rows
- prompt manifest hashes group runs without storing prompt text
- all focused and full verification commands pass
- release documentation records observed test counts

## Deferred Roadmap

After P5b:

1. Background Evaluation adds durable job state, progress, cancellation, and
   checkpoint recovery.
2. Trace Drill-down links failed cases to traces, nodes, tool calls, and exact
   prompt ids or versions used at runtime.
3. Per-question relational history is deferred until deeper regression analysis
   requires it.
4. Repeated trials and human-reviewed labels can calibrate Judge trends.
