# Evaluation And Ablation Report

P0a establishes the evaluation infrastructure for the Reliability-oriented Agentic RAG Document QA System. Final metrics should be regenerated after P1/P2 algorithm upgrades and recorded as P0b results.

## Dataset

The evaluation dataset uses structured question records with IDs, question types, gold answers, expected sources, expected keywords, answerability labels, and expected behavior labels.

The default dataset currently contains 36 questions covering single-document QA, multi-chunk synthesis, ambiguous questions, unanswerable questions, distractor handling, comparisons, follow-up questions, citation-sensitive questions, cross-file reasoning, and false-premise correction.

## Baseline Vs Agentic

Run:

```bash
.venv/bin/python -m evaluation.evaluate \
  --questions evaluation/eval_questions.json \
  --output-dir experiments/results
```

The generated JSON artifacts in `experiments/results/` are the reproducible source of truth:

- `baseline_result.json`
- `agentic_result.json`
- `comparison_result.json`

Each artifact includes a sanitized `runtime_config` snapshot for reproducibility. It records LLM model/temperature, retriever settings, hybrid retrieval settings, reranker settings, and vector collection name without API keys, base URLs, or local persistence paths.

## Ablation Table

| Method | Correctness | Context Relevance | Citation Accuracy | Fallback Accuracy | Unsupported Claims | Avg Latency | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Naive RAG | pending | pending | pending | pending | pending | pending | infrastructure-ready |
| + Query Transformation / Multi-query Retrieval | pending | pending | pending | pending | pending | pending | implemented in P1c/P1d, pending P0b eval |
| + Structured Retrieval Grading | pending | pending | pending | pending | pending | pending | implemented in P1e, pending P0b eval |
| + Partial-Relevance Recovery | pending | pending | pending | pending | pending | pending | implemented in P1f, pending P0b eval |
| + Retry / Fallback | pending | pending | pending | pending | pending | pending | proxy until independent toggle |
| + Hybrid Retrieval | pending | pending | pending | pending | pending | pending | implemented in P1a, pending P0b eval |
| + Reranker | pending | pending | pending | pending | pending | pending | pending P1/P2 final run |
| + Citation Verification | pending | pending | pending | pending | pending | pending | pending P1/P2 final run |

## Metric Notes

- Correctness is heuristic in P0a and uses expected keywords plus gold-answer overlap.
- Context relevance and source hit metrics use expected source matches from retrieved, relevant, or cited documents depending on the metric.
- Citation hit rate measures citation source matches, not merely retrieval success.
- Token usage and cost are recorded only when the active model client exposes usage metadata.
- Unsupported claims are counted from claim-verification payloads when present.

## Limitations

- P0a is an infrastructure checkpoint, not the final algorithm comparison.
- Current v1-v3 ablation rows use the current full Agentic RAG workflow as a proxy because independent toggles are not implemented yet.
- Hybrid retrieval is implemented after P0a and should be evaluated in the next P0b run with `HYBRID_RETRIEVAL_ENABLED=true`.
- P1b reranker evaluation readiness is implemented after P0a. Future P0b runs should record `RERANKER_TOP_N`, `RERANKER_CANDIDATE_TOP_K`, and reranker model in each artifact.
- P1c/P1d structured query transformation and multi-query retrieval are implemented after P0a. Multi-query strategy executes expanded queries and merges deduplicated chunks; decomposition sub-question retrieval remains future work.
- P1e/P1f structured retrieval grading and partial-relevance recovery are implemented after P0a. Future P0b runs should record `relevant_document_count`, `partial_document_count`, `max_relevance_confidence`, `partial_relevance_recovery`, and chunk-level grading labels in failure analysis; dynamic top-k and reranker adjustment remain future work.
- Reranker and full claim-level citation-verification ablations should be regenerated after P1/P2 algorithm upgrades.
- The interactive evaluation dashboard is deferred until CLI and JSON artifacts are stable.

## P0b Plan

After P1/P2 upgrades:

1. Rebuild or refresh the sample document index.
2. Run naive vs agentic evaluation on the same 36-question dataset.
3. Run ablation with real module toggles where available.
4. Refresh JSON artifacts and this report with observed metrics.
5. Document trade-offs such as reranker latency, fallback answer-rate impact, and citation-verification cost.
