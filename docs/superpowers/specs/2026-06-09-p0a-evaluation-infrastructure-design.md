# P0a Evaluation Infrastructure Design

## Context

The project is currently a LangGraph-based Agentic RAG document QA system with query rewriting, retrieval, document grading, conditional retry, fallback, citation-aware answer generation, lightweight claim verification, Gradio UI, an evaluation runner, and tests.

The next upgrade should make the project stronger as a resume project without turning the first pass into a broad rewrite. The agreed scope is P0a: build the evaluation infrastructure first, then reuse it after P1/P2 algorithm upgrades to produce the final results in P0b.

## Decision

Use approach A for P0a: keep the current `evaluation/evaluate.py` flow as the main runner and add the missing baseline, dataset, metrics, JSON outputs, ablation framework, and report template incrementally.

The implementation should follow approach B's interface direction where practical, but it should not perform a full evaluation-package refactor in P0a.

## Goals

- Add an independent naive RAG baseline package that can be run from CLI.
- Expand the evaluation dataset to 30-50 structured questions.
- Keep naive RAG and Agentic RAG evaluated against the same documents and questions.
- Add richer metrics for reliability-oriented RAG evaluation.
- Save reproducible JSON artifacts for baseline, agentic, and ablation runs.
- Add an ablation-study framework that can be completed after P1/P2 algorithm work.
- Update documentation so readers understand P0a is the measurement foundation, not the final algorithm result.
- Preserve existing behavior and keep the current test suite passing.

## Non-Goals

- Do not implement the full P1/P2 algorithm upgrades in P0a.
- Do not claim final Agentic RAG performance improvements before the algorithm pass and final P0b evaluation.
- Do not fully split `evaluation/evaluate.py` into a larger framework yet.
- Do not rewrite the LangGraph agent workflow during P0a.
- Do not add FastAPI, trace logging, workspace isolation, or Gradio dashboard work in this phase.

## Proposed File Changes

New baseline package:

```text
baseline/
+-- __init__.py
+-- naive_rag.py
`-- run_baseline.py
```

Evaluation changes:

```text
evaluation/
+-- eval_questions.json
+-- evaluate.py
`-- baselines.py
```

`evaluation/baselines.py` may remain as a backward-compatible wrapper around `baseline.naive_rag`.

New experiments package:

```text
experiments/
+-- run_ablation.py
+-- configs/
|   +-- v0_naive.yaml
|   +-- v1_query_rewrite.yaml
|   +-- v2_retrieval_grading.yaml
|   +-- v3_retry_fallback.yaml
|   +-- v4_reranker.yaml
|   `-- v5_citation_verification.yaml
+-- results/
`-- report.md
```

Optional sample document expansion:

```text
sample_docs/
+-- agentic_rag_notes.md
+-- retrieval_pipeline_notes.md
+-- citation_verification_notes.md
+-- evaluation_notes.md
`-- distractor_company_policy.md
```

Documentation:

```text
docs/resume_bullets.md
README.md
```

## Dataset Schema

P0a should support a richer schema while remaining compatible with the current fields.

```json
{
  "id": "q001",
  "question": "How does retrieval grading improve RAG reliability?",
  "question_type": "single_doc",
  "gold_answer": "Retrieval grading checks whether retrieved chunks contain enough evidence before answer generation.",
  "expected_sources": ["agentic_rag_notes.md"],
  "expected_keywords": ["retrieval grading", "evidence", "answer generation"],
  "answerable": true,
  "expected_behavior": "answer_with_citation",
  "chat_history": []
}
```

Required question-type coverage:

- `single_doc`
- `multi_chunk`
- `ambiguous`
- `unanswerable`
- `distractor`
- `comparison`
- `follow_up`
- `citation_sensitive`
- `cross_file`
- `false_premise`

Legacy fields should be normalized:

- `should_answer` maps to `answerable`.
- `expected_source` maps to `expected_sources`.
- `requires_rewrite` remains accepted as optional metadata.

## Baseline Runner

The naive baseline should implement:

```text
question -> retrieve top-k documents -> generate answer
```

It should return the same public payload shape expected by the evaluation runner:

- `answer`
- `citations`
- `retrieved_documents`
- `relevant_documents`
- `claims`
- `claim_verification`
- `is_verified`
- `retry_count`
- `fallback_reason`
- optional usage and latency metadata

The CLI should support:

```bash
.venv/bin/python -m baseline.run_baseline \
  --questions evaluation/eval_questions.json \
  --output experiments/results/baseline_result.json
```

## Evaluation Runner

The existing runner should be extended instead of replaced. It should support:

- single-system evaluation
- naive vs agentic comparison
- writing `baseline_result.json`
- writing `agentic_result.json`
- writing a comparison summary
- preserving terminal-readable reports

Expected CLI:

```bash
.venv/bin/python -m evaluation.evaluate \
  --questions evaluation/eval_questions.json \
  --output-dir experiments/results
```

The output directory should be created when missing. Existing outputs may be overwritten by an explicit evaluation run.

## Metrics

P0a should compute these metrics with deterministic heuristics where possible:

- `answer_rate`
- `correctness_score`
- `context_relevance_score`
- `source_hit_rate`
- `citation_hit_rate`
- `fallback_accuracy`
- `unsupported_claim_count`
- `supported_claim_ratio`
- `citation_verification_pass_rate`
- `average_retry_count`
- `average_latency`
- `average_token_usage`
- `estimated_cost`
- `error_count`

Metric notes:

- `correctness_score` can initially use expected-keyword and gold-answer heuristics, not an LLM judge.
- `context_relevance_score` can initially use expected-source hits and accepted relevant-document counts.
- `citation_hit_rate` should prefer citations and fall back to retrieved documents only for diagnostics, not for citation accuracy.
- Token usage and cost should be nullable or zero when the active LLM client does not expose usage data.
- Unsupported claims should be counted from claim-verification payloads when present.

## Ablation Framework

P0a should create an ablation framework without pretending every algorithmic toggle is fully implemented yet.

Initial versions:

```text
V0 Naive RAG
V1 Agentic RAG + Query Rewrite
V2 + Retrieval Grading
V3 + Conditional Retry / Fallback
V4 + Reranker
V5 + Claim-level Citation Verification
```

`experiments/run_ablation.py` should:

- load YAML-like config files
- run each configured method when supported
- record unsupported or pending variants clearly
- write `experiments/results/ablation_result.json`
- update or generate an ablation summary suitable for `experiments/report.md`

If PyYAML is not available, configs may be simple key-value YAML that can be parsed with a small local parser, or the dependency can be added if needed.

## Report Template

`experiments/report.md` should include:

- project evaluation goal
- dataset description
- baseline vs agentic comparison section
- ablation table
- metric definitions
- limitations
- P0b final evaluation plan

The ablation table should use this shape:

```text
Method | Correctness | Context Relevance | Citation Accuracy | Fallback Accuracy | Unsupported Claims | Avg Latency
--- | --- | --- | --- | --- | --- | ---
Naive RAG | pending | pending | pending | pending | pending | pending
+ Query Rewrite | pending | pending | pending | pending | pending | pending
+ Retrieval Grading | pending | pending | pending | pending | pending | pending
+ Retry / Fallback | pending | pending | pending | pending | pending | pending
+ Reranker | pending | pending | pending | pending | pending | pending
+ Citation Verification | pending | pending | pending | pending | pending | pending
```

P0a may fill rows from the current implementation, but the report must clearly say final numbers will be regenerated after algorithm upgrades.

## Tests

Add or update focused tests for:

- baseline CLI and payload compatibility
- dataset schema normalization
- richer metric calculation
- output JSON writing
- naive vs agentic comparison output
- ablation config loading
- ablation result writing
- report formatting

Existing tests must continue passing.

Target command:

```bash
.venv/bin/python -m pytest -q
```

## Acceptance Criteria

P0a is complete when these commands work:

```bash
.venv/bin/python -m baseline.run_baseline \
  --questions evaluation/eval_questions.json \
  --output experiments/results/baseline_result.json

.venv/bin/python -m evaluation.evaluate \
  --questions evaluation/eval_questions.json \
  --output-dir experiments/results

.venv/bin/python -m experiments.run_ablation \
  --questions evaluation/eval_questions.json \
  --output-dir experiments/results

.venv/bin/python -m pytest -q
```

And these artifacts exist:

- `experiments/results/baseline_result.json`
- `experiments/results/agentic_result.json`
- `experiments/results/ablation_result.json`
- `experiments/report.md`
- `docs/resume_bullets.md`

## P0b Follow-Up

After P1/P2 algorithm upgrades, run P0b:

- rebuild or refresh the sample index
- run naive vs agentic evaluation
- run ablation study
- regenerate result JSON files
- update `experiments/report.md` with real observed metrics
- update README with honest, reproducible results and trade-off analysis

## Roadmap: Interactive Evaluation Dashboard

After P0b results are reproducible from CLI and JSON artifacts, add an interactive evaluation dashboard as a later product-facing milestone.

Candidate scope:

- Add an Evaluation tab to the existing Gradio app.
- Allow selecting `baseline`, `agentic`, or an ablation config.
- Run evaluation from the UI without requiring terminal commands.
- Display metric cards for correctness, context relevance, citation hit rate, fallback accuracy, unsupported claim count, average retry count, and average latency.
- Show a baseline-vs-agentic comparison table.
- Show failed cases with failure type, reason, expected sources, retrieved sources, fallback reason, and citation status.
- Allow opening a per-question detail view that includes answer, citations, retrieved chunks, relevant chunks, retry count, and error text.
- Link dashboard rows to trace IDs after trace logging is implemented.

Dashboard non-goals for P0a:

- Do not add interactive charts before the evaluation artifacts are stable.
- Do not make dashboard results the source of truth; JSON artifacts remain the reproducible evaluation record.
- Do not block P1/P2 algorithm work on UI polish.

## Roadmap: Upgrade To Approach B

After P0b, refactor the evaluation code into a fuller framework:

```text
evaluation/
+-- schemas.py
+-- dataset.py
+-- runners.py
+-- metrics.py
+-- comparison.py
+-- report.py
+-- result_io.py
`-- evaluate.py
```

Tasks:

- Replace dict-heavy payloads with typed dataclasses or Pydantic models.
- Move dataset loading and schema normalization into `evaluation/dataset.py`.
- Move metric functions into `evaluation/metrics.py`.
- Move report rendering into `evaluation/report.py`.
- Move JSON artifact writing into `evaluation/result_io.py`.
- Add a metric registry so new metrics can be added without editing runner control flow.
- Add run metadata snapshots for prompt version, model, temperature, retriever config, and reranker config.
- Add historical-run comparison for regression testing.

This roadmap keeps P0a small while preserving a clean path to a stronger evaluation subsystem.

## Risks And Mitigations

- Risk: Early ablation results are misleading before algorithm upgrades.
  Mitigation: label P0a reports as infrastructure-ready and regenerate final results in P0b.
- Risk: Dataset expansion over one source document is weak.
  Mitigation: add a small multi-document sample pack with distractor and comparison cases.
- Risk: More metrics make results look more rigorous than they are.
  Mitigation: document heuristic metrics clearly and avoid claiming benchmark-grade evaluation.
- Risk: Refactoring too early slows algorithm work.
  Mitigation: use approach A now and schedule approach B after P0b.
