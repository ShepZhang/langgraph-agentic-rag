# P0b Real Ablation Evaluation Design

## Goal

Regenerate the evaluation artifacts after the P1 and P2 algorithm upgrades, and
replace the P0a proxy ablation table with reproducible experiments in which each
row represents a genuinely different system configuration.

P0b remains on the lightweight evaluation architecture selected for P0a. It does
not perform the deferred package-wide evaluation framework rewrite or build the
interactive dashboard.

## Current Problems

The existing evaluation infrastructure is runnable, but it is not yet suitable
for final P0b claims:

- `run_ablation.py` can execute only the naive runner or the current complete
  Agentic RAG runner. V1-V3 therefore repeat the same workflow.
- Query transformation, retrieval grading, retry, and citation verification are
  fixed into the graph and cannot be independently ablated.
- Follow-up records contain `chat_history`, but the evaluator invokes runners
  with only the question.
- Unsupported-claim metrics inspect extracted claims rather than
  `claim_verification_results`.
- Existing artifacts contain failed placeholder runs and must not be presented
  as experiment results.

## Chosen Approach

Use independent feature flags and build one LangGraph workflow from those flags.
This keeps production defaults unchanged while allowing evaluation variants to
remove a capability at the graph boundary.

Rejected alternatives:

- Separate graph implementations for every variant would duplicate workflow
  definitions and allow variants to drift.
- Monkeypatching nodes or replacing them with evaluation-only no-ops would make
  the experiment fragile and difficult to explain.

## Feature Configuration

Introduce a typed `AgentFeatureFlags` value with these fields:

```python
query_transformation_enabled: bool = True
retrieval_grading_enabled: bool = True
conditional_retry_enabled: bool = True
citation_verification_enabled: bool = True
```

`build_graph()` and `run_agent()` accept an optional feature configuration.
When omitted, every flag is enabled, preserving the current application,
Gradio, tests, and public API behavior.

Hybrid retrieval and reranking continue to use the existing typed `Settings`
fields:

```python
hybrid_retrieval_enabled
reranker_enabled
```

The ablation runner creates immutable settings variants with
`dataclasses.replace()` instead of mutating process-wide environment variables.
Every result records both the feature flags and sanitized runtime settings.

## Workflow Composition

The graph builder changes edges according to the feature configuration. Disabled
features are absent from the execution path rather than simulated with prompts.

### Query Transformation Disabled

The graph starts at retrieval and uses the original question as the retrieval
query. No rewrite LLM call is made.

### Retrieval Grading Disabled

Retrieved documents flow directly to answer generation and are treated as the
available answer context. No grading LLM call is made.

### Retrieval Grading Enabled, Retry Disabled

Relevant evidence flows to answer generation. If grading finds no adequate
evidence, the graph immediately returns fallback. This retains safe behavior
without performing a retry.

### Conditional Retry Enabled

Insufficient or weak evidence can route back to query transformation until the
retry budget is exhausted. Exhaustion routes to fallback.

### Citation Verification Disabled

The generated cited answer is finalized after generation validation. Claim
extraction, claim verification, and answer revision are not executed.

### Citation Verification Enabled

The current P2 workflow remains active:

```text
generate_answer
-> extract_claims
-> verify_citations
-> finalize_answer
```

Unsupported claims route through answer revision and are verified again before
finalization or fallback.

## Ablation Matrix

All variants use the same 36-question dataset, indexed documents, LLM model,
temperature, embedding model, chunking configuration, and top-k values except
where the row explicitly adds retrieval stages.

| Version | Capabilities |
|---|---|
| V0 | Naive dense retrieve-generate RAG |
| V1 | V0-style retrieval plus Agentic query transformation |
| V2 | V1 plus structured retrieval grading |
| V3 | V2 plus conditional retry and fallback |
| V4 | V3 plus BM25 + dense retrieval + RRF fusion |
| V5 | V4 plus reranking |
| V6 | V5 plus claim-level citation verification and revision |

The variants are cumulative. V0 uses the baseline runner. V1-V6 use graph and
settings factories built from their config files. V0 and V6 also become the
canonical baseline and complete Agentic RAG comparison artifacts.

An ablation row can be marked `completed` only when its resolved flags differ
from the previous row as specified. Duplicate effective configurations are a
validation error.

## Evaluation Invocation

Evaluation runners receive both:

```python
question: str
chat_history: list[ChatMessage]
```

The Agentic runner passes history to `run_agent()`. The naive runner accepts the
same argument for interface consistency but intentionally ignores it. This
allows the follow-up cases to measure standalone-question transformation rather
than accidentally testing both systems without conversation context.

## Metric Corrections

Keep the P0a deterministic metrics, with these corrections:

- Claim labels come from `claim_verification_results`.
- `unsupported_claim_count` counts only `unsupported`.
- `supported_claim_ratio` is supported verification results divided by all
  verification results.
- `citation_verification_pass_rate` uses the explicit verification pass field.
- Citation-disabled variants report verification metrics as unavailable rather
  than as successful.
- Follow-up results record whether chat history was supplied.
- Runtime errors remain per-question data and contribute to `error_count`.

Correctness remains a documented heuristic based on expected keywords and gold
answer overlap. P0b does not add an LLM-as-judge because that would introduce a
second model-dependent measurement system and additional cost into this
milestone.

Token usage and estimated cost are recorded only if the active model integration
provides reliable metadata. Missing usage remains unavailable; the runner must
not invent estimates.

## Execution And Artifacts

The execution sequence is:

1. Rebuild the sample index from all files in `sample_docs/`.
2. Validate all configs and verify that the effective variants are distinct.
3. Run a smoke subset covering answerable, unanswerable, follow-up,
   citation-sensitive, and comparison cases.
4. Run the complete 36-question V0-V6 matrix sequentially.
5. Generate aggregate comparison and Markdown reports.
6. Run the local test suite after artifact generation.

Each variant is written atomically after completion so an API interruption does
not discard earlier results. The final output contains:

```text
experiments/results/
  variants/
    v0_naive.json
    v1_query_transform.json
    v2_retrieval_grading.json
    v3_retry_fallback.json
    v4_hybrid_retrieval.json
    v5_reranker.json
    v6_citation_verification.json
  baseline_result.json
  agentic_result.json
  comparison_result.json
  ablation_result.json
  ablation_report.md
```

`baseline_result.json` is derived from V0. `agentic_result.json` is derived from
V6. `comparison_result.json` pairs V0 and V6 cases without rerunning either
system.

Every artifact includes:

- dataset path and question count
- variant ID and effective feature flags
- sanitized model, retriever, reranker, and vector-store settings
- per-question results and errors
- summary metrics
- run completion status

Run status has three explicit values:

- `completed`: every question produced a result without runtime errors
- `completed_with_errors`: all questions were attempted, but at least one
  recorded an error
- `incomplete`: execution stopped before every question was attempted

Only `completed` variants are included as valid rows in the final trade-off
analysis. Other statuses remain in the artifacts for diagnosis.

## Reporting

Update `experiments/report.md` from the generated artifacts. The report contains:

- dataset composition
- V0 versus V6 comparison
- cumulative ablation table
- failed-case counts
- observed trade-offs
- metric limitations
- exact reproduction commands

Trade-off analysis must be evidence-based. Expected patterns such as higher
reranker latency or lower answer rate after stricter fallback are discussed only
when the generated numbers support them.

## Error Handling

- Invalid or duplicate variant configurations fail before API calls.
- A per-question exception is recorded without terminating the current variant.
- A fully attempted variant with any errors is marked `completed_with_errors`
  and its error count is reported.
- Index construction or model configuration failures stop the experiment before
  producing misleading aggregate artifacts.
- Secrets, base URLs containing credentials, and local persistence paths are
  excluded from runtime snapshots and reports.

## Testing

Add or update focused tests for:

- feature-flag defaults preserving the full graph
- graph paths for each disabled feature
- retry-disabled immediate fallback
- citation-disabled direct finalization
- distinct effective V0-V6 configurations
- history-aware evaluator invocation
- naive history-ignore behavior
- verifier-result metric aggregation
- unavailable verification metrics for citation-disabled variants
- per-variant artifact generation
- V0/V6 comparison derivation without duplicate execution
- incomplete/error run reporting
- secret-free runtime snapshots

The full existing test suite must continue to pass before live evaluation.

## Deferred Work

The following remain later roadmap items:

- Approach B: a modular, strongly typed evaluation package with pluggable
  metrics, runners, storage, and judges
- interactive Gradio evaluation dashboard
- trace-backed failed-case analyzer
- prompt version registry and prompt regression suite
- repeated trials and confidence intervals
- human- or judge-model semantic correctness scoring

P0b focuses on trustworthy single-run comparison and real module ablation using
the current lightweight evaluation architecture.
