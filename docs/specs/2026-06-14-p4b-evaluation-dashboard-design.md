# P4b Evaluation Dashboard Design

Date: 2026-06-14

Status: Approved interaction and architecture design; pending written-spec review

Target version: `v0.4.1-p4b`

## Goal

Add an interactive Evaluation Dashboard to the existing Gradio application so
users can run a small, real evaluation and inspect evaluation or ablation
results without reading JSON artifacts manually.

This milestone turns the existing evaluation framework into a visible product
capability while preserving the current evaluation algorithms as the single
source of truth. It does not introduce a second metric implementation inside
the UI.

The dashboard must answer three practical questions:

- Does Agentic RAG outperform Naive RAG on the selected questions?
- Which reliability metrics improved or regressed?
- What kinds of failures occurred, and what evidence supports that diagnosis?

## Scope

P4b includes:

- a top-level `Evaluation` tab in the Gradio application
- real-time quick evaluation for Naive RAG, Agentic RAG, or both systems
- a five-question default smoke set with manual selection of any of the
  36 evaluation questions
- an ablation snapshot view for the existing V0-V6 experiment artifact
- metric summaries, failure type counts, filterable failed cases, and
  selected-case details
- a shared dashboard service and pure formatting layer outside the UI
- compatibility handling for older ablation artifacts created before P4a
- tests for service behavior, formatting, Gradio event helpers, and app
  construction
- documentation and release metadata for `v0.4.1-p4b`

P4b does not include:

- running the full V0-V6 ablation matrix from Gradio
- background jobs, progress polling, cancellation, or checkpoint recovery
- linking failed cases to trace IDs or a node-by-node trace viewer
- LLM-as-judge or manual human scoring
- historical run management or trend charts
- multi-user evaluation run isolation

## Confirmed Product Decisions

### Evaluation Modes

The quick evaluation control supports:

- `Naive RAG`
- `Agentic RAG`
- `Compare Both`

`Compare Both` evaluates the same selected questions with both systems and
uses the existing paired comparison report shape.

### Question Selection

The default smoke set is:

```text
q001, q016, q027, q030, q033
```

These cases provide a short, mixed reliability check rather than a statistically
complete benchmark. The user can manually select questions or choose all
36 records. The UI must warn that a full run can take substantially longer and
incur more model usage.

### Failed-Case Interaction

The dashboard shows:

- failure type counts
- a failed-case table
- filters for system and failure type
- a selected-case detail panel containing the reason and suggestion

The first release does not link a failed case to an execution trace.

### Ablation Behavior

The ablation view reads an existing V0-V6 result artifact. It does not run
ablation experiments during the Gradio request. This keeps the first dashboard
responsive, predictable, and compatible with the current synchronous
application.

The code will define a small future-facing runner protocol so a later
background implementation can be added without changing the dashboard view
model.

## User Interface

The Gradio application will have two top-level tabs:

1. `Document QA`
2. `Evaluation`

The current upload and question-answering workflow moves into `Document QA`
without changing its behavior.

The `Evaluation` tab contains two subviews:

1. `Quick Compare`
2. `Ablation Snapshot`

### Quick Compare Layout

The quick view contains:

- a system mode selector
- a multi-select question control
- `Select smoke set` and `Select all` actions
- a primary `Run Evaluation` action
- a run status message
- a compact metric table
- failure type counts
- failed-case filters and result table
- a selected-case detail panel

The dashboard reports these metrics when available:

- correctness
- context relevance
- citation accuracy
- fallback accuracy
- unsupported claim count
- average latency
- average retry count

Missing metrics render as `N/A`; they are not converted to zero.

### Ablation Snapshot Layout

The snapshot view contains:

- artifact status and source path
- V0-V6 metric rows
- runtime configuration details for a selected variant
- failure type counts for a selected variant
- filterable failed cases
- selected-case details

The screen clearly labels the data as a saved snapshot. It must not imply that
opening the tab reran the experiments.

## Architecture

```text
Gradio components and event handlers
              |
              v
EvaluationDashboardService
       |                    |
       v                    v
evaluation.evaluate     AblationResultProvider
       |                    |
       v                    v
Naive / Agentic runners  experiments/results/*.json
              \            /
               v          v
          dashboard_formatters
                  |
                  v
          JSON-ready view model
```

The dashboard calls local Python services directly. It does not make HTTP
requests to the FastAPI service running in the same process.

This boundary avoids duplicating orchestration and keeps the service reusable
by Gradio, tests, and a future API endpoint.

## Module Design

### `evaluation/dashboard_models.py`

This module defines JSON-ready types and constants for the dashboard contract.
It does not import Gradio.

Core types include:

```python
DashboardSystemMode = Literal["naive", "agentic", "comparison"]
DashboardStatus = Literal["completed", "failed", "unavailable"]
DiagnosticsSource = Literal["stored", "derived", "unavailable"]
```

The normalized dashboard view model is:

```python
{
    "status": "completed",
    "run_id": "quick-20260614-120000-ab12",
    "summary_rows": [],
    "failure_count_rows": [],
    "failure_cases": [],
    "raw_report": {},
    "message": "",
}
```

Each failure case contains enough stable data for table rendering and detail
lookup:

```python
{
    "case_key": "agentic:q023",
    "system": "agentic",
    "question_id": "q023",
    "question_type": "citation_sensitive",
    "question": "Which source supports the retry policy?",
    "failure_type": "citation_failure",
    "reason": "The answer cited a source that does not support the retry policy.",
    "suggestion": "Review citation selection and claim verification output.",
    "diagnostics_source": "stored",
}
```

### `evaluation/dashboard_formatters.py`

This module contains pure transformations from evaluation or ablation reports
to display rows. It performs no file I/O and invokes no model.

Responsibilities:

- convert single-system and comparison summaries into metric rows
- flatten single-system and paired results into a common failure-case shape
- count failure types by system or variant
- filter failure cases without rerunning evaluation
- build selected-case details
- preserve `None` as unavailable instead of coercing it to zero

### `evaluation/dashboard_service.py`

`EvaluationDashboardService` coordinates existing evaluation capabilities and
returns normalized view models.

Public operations:

```python
list_questions() -> list[QuestionOption]
run_quick_evaluation(
    question_ids: list[str],
    system_mode: DashboardSystemMode,
) -> DashboardView
load_ablation_snapshot() -> DashboardView
filter_failure_cases(
    dashboard_view: DashboardView,
    system: str | None,
    failure_type: str | None,
) -> list[FailureCaseRow]
get_failure_detail(
    dashboard_view: DashboardView,
    case_key: str | None,
) -> FailureCaseDetail
```

The service accepts injected question loaders, runners, artifact providers,
clock functions, and ID factories. Tests can therefore exercise behavior
without model calls, a vector store, or wall-clock assumptions.

Question IDs are validated against `load_eval_questions()`. Selected records
retain dataset order so repeated runs are comparable.

### Ablation Provider Boundary

The first implementation uses a read-only provider:

```python
class AblationResultProvider(Protocol):
    def load(self) -> dict[str, Any]:
        pass
```

`JsonAblationResultProvider` reads the configured
`experiments/results/ablation_result.json` path and validates the minimum
payload structure.

The future execution boundary is reserved separately:

```python
class AblationRunner(Protocol):
    def run(
        self,
        question_ids: list[str],
        variant_ids: list[str],
    ) -> AblationRunHandle:
        pass
```

P4b defines the protocol types only. It does not implement a synchronous or
background V0-V6 runner in the UI.

## Quick Evaluation Flow

1. Gradio passes selected question IDs and system mode to the dashboard
   service.
2. The service validates the selection and loads matching records in dataset
   order.
3. The service maps the mode to existing runners:
   - `naive` calls `evaluate_questions()` with the Naive runner as the single
     system runner.
   - `agentic` calls `evaluate_questions()` with the Agentic runner.
   - `comparison` calls `evaluate_questions()` with both existing runners.
4. Existing evaluation code computes per-case metrics and P4a failure
   analysis.
5. Dashboard formatters normalize the report into metric, failure count, and
   failed-case rows.
6. Gradio stores the normalized view and raw report in `gr.State`.
7. Later filter or selection events transform the stored view only; they do
   not call either RAG system again.

The dashboard does not alter the underlying evaluation output or write the
quick run to the canonical ablation artifact.

## Ablation Snapshot Compatibility

Some existing ablation artifacts were produced before P4a and therefore do not
contain `failure_analysis` or `failure_type_counts`.

The snapshot provider supports these artifacts through an in-memory
compatibility adapter:

1. Load the saved ablation artifact without modifying it.
2. Map result rows to current evaluation question metadata by `question_id`.
3. If a result already contains `failure_analysis`, use it and mark
   `diagnostics_source="stored"`.
4. If it is absent and the required question metadata is available, call the
   deterministic P4a `analyze_failure()` function and mark
   `diagnostics_source="derived"`.
5. If metadata is missing or the row is malformed, keep the metric row but
   mark failure diagnostics unavailable for that case.

Derived diagnostics are a compatibility view over stored metrics. They are not
a benchmark rerun, do not invoke an LLM, and do not rewrite the JSON artifact.
The UI must label derived diagnostics so users can distinguish them from data
saved by a newer evaluation run.

## Gradio State and Event Design

The UI uses two independent states:

```python
quick_evaluation_state: dict
ablation_snapshot_state: dict
```

This prevents one view from overwriting the other and allows filters to operate
locally.

Event handlers remain thin, testable functions:

- run a quick evaluation and return component updates
- load or refresh the ablation snapshot
- select the smoke or full question set
- filter stored failure rows
- render a selected failure detail

`ui/gradio_app.py` should be split into focused UI builders such as
`_build_document_qa_tab()` and `_build_evaluation_tab()` rather than placing
all components and handlers in one monolithic function.

The synchronous run button is disabled while the event executes and restored
after completion. The status text distinguishes running, completed, failed,
and unavailable states.

## Error Handling

### Invalid or Empty Selection

An empty question selection returns a clear validation message and does not
invoke evaluation. Unknown IDs are rejected before any runner call.

### Per-Case Runner Failure

The existing evaluation runner records an exception as result data. The run
continues, and P4a classifies the case as `tool_failure`. The dashboard renders
the completed report with its error count and failed case.

### Whole-Run Failure

If loading inputs or coordinating the evaluation fails before a report exists,
the service returns a failed view with a concise message. The UI displays the
error but retains the last successful `gr.State` data so the previous result
does not disappear.

### Missing or Invalid Ablation Artifact

The ablation view returns `status="unavailable"` with an actionable message.
The quick evaluation view remains usable.

### Partial Legacy Data

Malformed variants or result rows are skipped individually when possible.
Valid variants remain visible. The status message reports that the snapshot is
partial instead of failing the entire dashboard.

## Testing Strategy

### `tests/test_dashboard_service.py`

Cover:

- question listing and deterministic dataset ordering
- five-question smoke selection
- naive, agentic, and comparison runner mapping
- empty and unknown question validation
- normalized metric rows
- failure count and failed-case filtering
- selected-case detail lookup
- per-case runner errors remaining visible as `tool_failure`
- whole-run failure returning a failed view
- missing ablation artifact returning an unavailable view
- modern ablation diagnostics marked `stored`
- legacy ablation diagnostics derived in memory and marked `derived`
- artifact files remaining unchanged after compatibility enrichment
- partial malformed snapshot handling

### `tests/test_gradio_app.py`

Extend current UI tests to cover:

- `create_app()` still returns `gr.Blocks`
- Document QA helper behavior remains unchanged
- evaluation event helpers use the injected service
- smoke and full question selection helpers
- successful run component values
- failed run preserving the previous successful state
- filtering does not call evaluation again
- selected failure details render safely

### Regression Tests

Continue running:

- evaluation runner tests
- failure analyzer tests
- ablation tests
- FastAPI tests
- full repository test suite

### Visual Verification

After implementation:

- start the Gradio development server
- inspect `Document QA`, `Quick Compare`, and `Ablation Snapshot`
- verify desktop and narrow viewport layouts in the in-app browser
- confirm tables, selectors, buttons, and long error text do not overlap
- run a deterministic test-backed dashboard flow before using any external
  model-backed smoke run

## Documentation and Release

Update:

- `README.md` with Evaluation Dashboard usage and screenshots or a concise
  interface description
- `CHANGELOG.md` with P4b scope and limitations
- the project roadmap with the background live-ablation upgrade
- relevant run commands for Gradio and existing evaluation artifacts

The documentation must call this a reliability-oriented evaluation interface,
not an automated benchmark platform or production monitoring system.

## Future Upgrade Route

The post-P4b roadmap must explicitly retain:

1. `BackgroundAblationRunner` implementing the reserved `AblationRunner`
   protocol.
2. Background job status, progress, cancellation, and checkpoint recovery for
   live V0-V6 runs.
3. Shared evaluation run IDs across Gradio and FastAPI.
4. Failed-case `trace_id` linkage and a node-level trace viewer.
5. Prompt versioning and prompt regression comparisons.
6. Historical evaluation runs and metric trend comparison.

This route upgrades the snapshot view to live experiments without replacing
the dashboard service or its normalized view model.

## Acceptance Criteria

P4b is complete when:

- the Gradio app exposes working `Document QA` and `Evaluation` tabs
- quick evaluation runs Naive, Agentic, or paired comparison on selected
  records through the existing evaluation runner
- the five-question smoke set is selected by default
- all 36 evaluation questions can be selected
- the dashboard renders the agreed metric and failure views
- failed cases can be filtered and inspected without rerunning evaluation
- the V0-V6 snapshot loads from the existing artifact
- legacy P4a-missing diagnostics are derived transparently in memory
- missing snapshot data does not break quick evaluation
- the existing Document QA workflow remains functional
- focused dashboard tests and the full repository test suite pass
- desktop and narrow viewport browser checks pass
- README, changelog, roadmap, and version metadata describe the release
  accurately
