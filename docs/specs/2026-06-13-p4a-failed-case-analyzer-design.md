# P4a Failed Case Analyzer Design

Date: 2026-06-13

## Goal

Add a deterministic failed-case analyzer to the evaluation pipeline so the
project can explain why an evaluation case failed, not only report aggregate
metrics. The analyzer should turn existing per-question evaluation fields into
a compact failure type, reason, and suggested next action.

This milestone covers P4a plus report integration. It does not implement the
interactive Gradio Evaluation Dashboard or prompt versioning; those remain
later P4 milestones.

## Problem

The evaluation framework already records useful signals:

- answer correctness
- context relevance
- source and citation hits
- fallback correctness
- unsupported claim counts
- retrieved and relevant documents
- errors from failed runs

However, a failed result still requires manual inspection. For a resume-quality
reliability-oriented RAG system, the evaluation artifact should directly answer:

- Did retrieval miss the evidence?
- Was the evidence retrieved but filtered out by reranking or grading?
- Did the model answer with weak or wrong citations?
- Did fallback trigger incorrectly?
- Did a tool or runner fail?

## Approach

Implement a deterministic analyzer under `evaluation/failure_analyzer.py`.
It will inspect one normalized evaluation result plus the corresponding
evaluation question metadata and return:

```json
{
  "question_id": "q023",
  "failure_type": "retrieval_failure",
  "reason": "Expected source was not found in retrieved, relevant, or cited documents.",
  "suggestion": "Try query expansion, hybrid retrieval, or increasing retrieval top_k."
}
```

The analyzer will not call an LLM judge. It will use rule-based checks derived
from existing fields, making the output fast, reproducible, and safe to include
in local evaluation artifacts.

## Failure Types

The initial failure taxonomy is:

- `no_failure`
- `tool_failure`
- `fallback_failure`
- `query_rewrite_failure`
- `retrieval_failure`
- `reranking_failure`
- `citation_failure`
- `generation_failure`

The analyzer returns exactly one primary `failure_type` per case. This keeps
reports scannable and avoids double-counting a single failed question across
multiple categories.

## Rule Priority

Rules run in a fixed order:

1. `tool_failure`
   - Triggered when `result["error"]` is non-empty.
   - Reason points to the recorded evaluation error.
   - Suggestion recommends checking tool traces, runner configuration, and
     model/retriever setup.

2. `no_failure`
   - Triggered when `correct` is true and `fallback_correct` is true.
   - If expected sources are present, source or citation hit must also be true.
   - If citation verification is applicable, unsupported claim count must be
     zero or absent.

3. `fallback_failure`
   - Triggered when fallback behavior is wrong for answerability:
     an answerable case falls back, or an unanswerable case returns an answer.
   - Reason distinguishes false fallback from missed fallback.

4. `query_rewrite_failure`
   - Triggered for `requires_rewrite=true` or question types `ambiguous` and
     `follow_up` when expected sources are not found and retry/rewrite signals
     are absent.
   - Suggestion recommends improving standalone question rewrite or
     multi-query expansion.

5. `retrieval_failure`
   - Triggered when expected sources exist but are absent from retrieved,
     relevant, and cited documents.
   - Suggestion recommends query expansion, hybrid retrieval, or increasing
     candidate top-k.

6. `reranking_failure`
   - Triggered when retrieved documents include an expected source, but
     relevant documents and citations do not.
   - This captures cases where recall succeeded but filtering, reranking, or
     retrieval grading lost the needed evidence.

7. `citation_failure`
   - Triggered when an answer is returned but citation hit is false for a case
     with expected sources, or when `unsupported_claim_count > 0`.
   - Suggestion recommends checking citation selection, claim verification, and
     answer revision behavior.

8. `generation_failure`
   - Triggered when evidence and fallback behavior look acceptable but
     `correct` or `keyword_hit` is false.
   - Suggestion recommends improving answer prompts or stricter grounding.

If no rule matches, the analyzer returns `generation_failure` for conservative
diagnostics when `correct` is false, otherwise `no_failure`.

## Integration Points

### Evaluation Runner

`evaluation.evaluate._build_success_result()` and `_build_error_result()` will
add:

```json
"failure_analysis": {
  "question_id": "q001",
  "failure_type": "no_failure",
  "reason": "The case satisfied correctness, fallback, and evidence checks.",
  "suggestion": "No action required."
}
```

`evaluation.evaluate._summarize()` will add:

```json
"failure_type_counts": {
  "no_failure": 29,
  "retrieval_failure": 4,
  "citation_failure": 2,
  "fallback_failure": 1
}
```

For comparison mode, naive and agentic result rows already use the same
single-system result shape, so each side will carry its own `failure_analysis`.

### Ablation Artifacts and Report

`experiments.run_ablation` already stores the complete evaluation report per
variant. Because each result row will contain `failure_analysis`, the JSON
artifacts automatically retain per-case diagnostics.

`format_ablation_report()` will add:

```markdown
## Failed Case Analysis

| Failure Type | Count |
|---|---:|
| retrieval_failure | 4 |
| citation_failure | 2 |

## Representative Failed Cases

| Question ID | Type | Failure | Reason | Suggestion |
|---|---|---|---|---|
```

The report will show representative non-`no_failure` cases from completed runs.
It will cap rows per report section so the Markdown stays readable.

## Source Matching

The analyzer will use simple source matching compatible with existing
evaluation logic:

- exact source match
- basename match for local paths
- substring match in either direction

This keeps analyzer decisions consistent with `source_hit` and `citation_hit`
metrics.

## Testing

Add `tests/test_failure_analyzer.py` for deterministic analyzer rules:

- expected source absent everywhere -> `retrieval_failure`
- expected source retrieved but absent from relevant/citations ->
  `reranking_failure`
- answerable case falls back -> `fallback_failure`
- unanswerable case returns answer -> `fallback_failure`
- non-empty error -> `tool_failure`
- unsupported claims -> `citation_failure`
- rewrite-required case with no source hit and no retry -> `query_rewrite_failure`
- evidence hit but incorrect answer -> `generation_failure`
- correct case -> `no_failure`

Extend evaluation tests to verify:

- each result row includes `failure_analysis`
- summary includes `failure_type_counts`
- comparison artifacts retain failure analysis for naive and agentic rows

Extend ablation tests to verify:

- `format_ablation_report()` includes failure count and representative failed
  case tables

## Documentation

Update:

- `README.md`: describe deterministic failed-case analysis under Evaluation.
- `CHANGELOG.md`: add a new P4a entry.
- `docs/resume_bullets.md`: mention failed-case attribution as part of
  evaluation diagnostics.

The documentation must avoid implying automatic repair, autonomous debugging,
or benchmark-grade causal analysis. This feature classifies likely failure
causes using deterministic signals already present in evaluation artifacts.
