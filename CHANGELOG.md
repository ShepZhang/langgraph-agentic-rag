# Changelog

## v0.5.1-p5b - SQLite Historical Evaluation + Trend Dashboard

Date: 2026-06-27

### Added

- Added SQLite sidecar storage for historical evaluation runs.
- Added prompt-aware historical metric trends for CLI/API evaluation artifacts.
- Added FastAPI and Gradio read-only history views for recent runs and metric trends.

### Changed

- Advanced evaluation runtime metadata to schema version `4` and evaluator version `p5b`.
- Preserved JSON artifacts as the complete compatibility payload while indexing summaries into SQLite.

### Notes

- SQLite history is local runtime data and is disabled safely with `EVALUATION_HISTORY_ENABLED=false`.
- Legacy artifacts without schema or evaluator metadata import as `legacy`.
- Background Evaluation and Trace Drill-down remain future work.

### Verification

- Full test suite: `.venv/bin/python -m pytest -q` → `672 passed in 4.71s`.
- Focused history tests: `.venv/bin/python -m pytest tests/test_evaluation_history_store.py tests/test_evaluation_storage.py tests/test_evaluate.py -q` → `74 passed in 1.44s`.
- API/Dashboard compatibility tests: `.venv/bin/python -m pytest tests/test_fastapi_routes.py tests/test_dashboard_service.py tests/test_gradio_app.py -q` → `83 passed in 3.64s`.

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
- Semantic correctness is unavailable when `gold_answer` is blank, normalized
  aggregates reject scores outside `0-1`, and injected Judge metadata is
  preserved in artifacts.

### Notes

- The Judge is disabled by default and never reuses the evaluated system's LLM
  configuration.
- Enabling it adds one model call per successful system result and can increase
  latency and cost substantially for comparison and ablation runs.
- Judge scores are model-based signals, not human ground truth.
- Evidence source URLs are reduced to credential-free basenames before Judge
  invocation.

### Verification

- Full test suite: `641 passed`.
- Focused Judge and evaluation tests: `188 passed`.
- CLI compatibility smoke tests: `3 passed`.
- Ablation, matrix, Dashboard, and FastAPI compatibility tests: `92 passed`.

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

## v0.4.2-p4c - Modular Evaluation Framework

Date: 2026-06-16

### Changed

- Refactored `evaluation.evaluate` into a compatibility facade over focused
  evaluator modules for typed schemas, dataset normalization, deterministic
  metrics, runner execution, comparison orchestration, report rendering,
  optional judge contracts, atomic JSON storage, and sanitized runtime metadata.
- Preserved existing CLI, FastAPI, Gradio Evaluation Dashboard, ablation, and
  JSON artifact contracts while moving implementation details behind typed
  internal records and adapters.
- Kept deterministic offline evaluation behavior as the default path.
- Added extension boundaries for future semantic judges, historical result
  storage, and a higher-level evaluation engine without claiming those roadmap
  features as complete.

### Fixed

- Preserved stable generated question IDs for raw multi-question evaluation
  matrix inputs across variants.
- Preserved public `summarize_results()` compatibility with result dictionaries
  that contain additive external diagnostic fields.
- Restored the legacy `EvaluationRunner` callable alias shape from
  `evaluation.evaluate`.

### Verification

- Full test suite: `469 passed`.
- CLI compatibility smoke tests: `3 passed`.
- Ablation, matrix, dashboard, and FastAPI compatibility tests: `82 passed`.

## v0.4.1-p4b - Evaluation Dashboard

Date: 2026-06-14

### Added

- Added a Gradio Evaluation tab with a five-question smoke set, manual selection across all 36 questions, and Naive RAG, Agentic RAG, or Compare Both modes.
- Added reliability metrics, failure counts, filterable failed cases, and selected-case details.
- Added a read-only V0-V6 ablation snapshot with saved runtime configuration.
- Added deterministic in-memory diagnostics for pre-P4a artifacts, with transparent `stored` or `derived` labels and no writes back to source artifacts.

### Notes

- Quick evaluation is synchronous, so a 36-question run may take time and incur model cost.
- The ablation view reads existing artifacts; it does not run V0-V6 from the browser.
- Background progress, cancellation, checkpoint recovery, trace drill-down, and historical run views remain future work.

## v0.4.0-p4a - Failed Case Analyzer

Date: 2026-06-13

### Added

- Added deterministic failed-case analysis for evaluation results with primary failure types, reasons, and suggested next actions.
- Added `failure_analysis` per question and `failure_type_counts` in evaluation summaries.
- Added failed-case count and representative-case sections to ablation reports.

### Notes

- P4a does not use an LLM judge and does not automatically repair failures.
- Failure attribution is rule-based and intended for debugging, regression triage, and portfolio explanation rather than benchmark-grade causal proof.

## v0.3.3-p3d - Typed Internal Tool Registry

Date: 2026-06-12

### Added

- Added a typed internal Tool Registry with Pydantic argument validation,
  runtime dependency injection, normalized results, and compact diagnostics.
- Registered retriever, claim citation verifier, document summary, and safe
  calculator tools.
- Routed LangGraph retrieval and citation verification through the registry.
- Added tool-call trace events with success, latency, metadata, and sanitized
  errors.
- Preserved the existing LangChain retriever tool API through a compatibility
  adapter.

### Notes

- P3d does not add autonomous tool selection, planning, a ReAct loop, or a
  generic public tool execution endpoint.
- Document summary and calculator tools are registered for extension and
  independent use but are not part of the primary QA workflow.

## v0.3.2-p3c - Workspace-aware Retrieval Isolation

Date: 2026-06-11

### Added

- Added workspace-aware dense retrieval through Chroma metadata filters.
- Added workspace-filtered BM25 corpus loading for hybrid retrieval.
- Added `workspace_id` propagation through hybrid retrieval, project-level
  retriever normalization, Agent retriever tools, and `run_agent()`.
- Retriever outputs now preserve `workspace_id` and `document_id` metadata when
  present.
- Added workspace isolation tests covering dense, sparse, hybrid, retriever, and
  Agent default retrieval paths.

### Notes

- P3c uses one shared Chroma collection with metadata filters. Per-workspace
  collections and tenant authorization remain future hardening work.

## v0.3.1-p3b - FastAPI Service Layer

Date: 2026-06-11

### Added

- Added modular `api` package with FastAPI app factory, schemas,
  dependencies, route modules, and service modules.
- Added `POST /chat` and `GET /chat/{session_id}/trace`.
- Added API-managed document upload, indexing, listing, and deletion routes.
- Added synchronous evaluation run and retrieval routes.
- Added explicit FastAPI runtime dependencies to `requirements.txt`.
- Added FastAPI route tests using dependency overrides.

### Notes

- P3b is an integration-oriented backend layer, not a production API. Auth,
  async jobs, durable task state, and tenant authorization remain future work.

## v0.3.0-p3a - Local Trace Logging

Date: 2026-06-11

### Added

- Added `observability` package with trace record construction and JSONL trace
  storage.
- Added configurable trace logging through `TRACE_LOGGING_ENABLED` and
  `TRACE_LOG_DIR`.
- Instrumented LangGraph nodes and conditional edges through graph wrappers so
  trace logging stays outside Agent node business logic.
- `run_agent()` now returns `trace_id`, `trace_path`, and `latency_ms` when trace
  logging is enabled.
- Added trace tests covering JSONL persistence and Agent workflow trace output.

### Notes

- P3a stores local JSONL traces only. API trace lookup and a visual trace
  dashboard remain future milestones.

## v0.2.1-p0c - Documentation Consistency Pass

Date: 2026-06-11

### Changed

- Updated README positioning so P0b ablation is described as completed real
  V0-V6 evaluation rather than proxy or pending scaffolding.
- Split the roadmap into completed work and next milestones.
- Kept the Approach B evaluator upgrade and interactive Evaluation Dashboard as
  explicit future tasks.
- Updated resume bullets to describe the executable cumulative ablation
  framework.

## v0.2.0-p0b - Reliability-oriented Agentic RAG Upgrade

Date: 2026-06-11

This version upgrades the project from an Agentic RAG MVP into a reliability-oriented document QA system with real baseline comparison and cumulative ablation artifacts.

### Added

- Naive RAG baseline package and CLI for retrieve-once comparison.
- 36-question structured evaluation dataset covering single-doc, multi-chunk, ambiguous, unanswerable, distractor, comparison, follow-up, citation-sensitive, cross-file, and false-premise cases.
- Hybrid retrieval pipeline with dense retrieval, BM25 sparse retrieval, and Reciprocal Rank Fusion.
- Optional cross-encoder reranker with configurable candidate and output limits.
- Structured query transformation with rewrite, multi-query expansion, and decomposition metadata.
- Multi-query retrieval execution with deduplicated evidence merging.
- Structured retrieval grading with relevance labels, confidence scores, and grading reasons.
- Partial-relevance recovery and retry rewrites guided by weak evidence.
- Claim-level citation verification workflow with claim extraction, per-claim support labels, answer revision, and fallback.
- Agent feature flags for real V0-V6 cumulative ablation experiments.
- Recoverable ablation runner with per-variant checkpoints, canonical baseline/agentic artifacts, and derived comparison results.
- P0b DeepSeek evaluation artifacts under `experiments/results/`.
- Resume-ready project bullets under `docs/resume_bullets.md`.

### P0b Result Snapshot

- Dataset: 36 questions.
- All V0-V6 variants completed with `error_count = 0`.
- V0 Naive RAG correctness: `0.6111`, average latency: `4.4576s`.
- V5 Hybrid + Reranker correctness: `0.6389`, average latency: `28.6960s`.
- V6 Claim-level Citation Verification unsupported final claims: `0`, supported-claim ratio: `1.0000`, average latency: `41.2485s`.

### Notes

- The results are single-run heuristic metrics, not benchmark-grade claims.
- V6 improves claim-support diagnostics but does not outperform V0 on every metric.
- Token usage and cost are unavailable because the active DeepSeek integration did not expose reliable usage metadata.
- The project is reliability-oriented and production-oriented as an architecture exercise, not a complete production deployment.
