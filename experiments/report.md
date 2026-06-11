# Evaluation And Ablation Report

P0b replaces the earlier proxy table with executable cumulative V0-V6 variants. Every row resolves to distinct LangGraph feature flags or retrieval settings, and duplicate effective configurations are rejected before model calls.

## Dataset

The evaluation dataset uses structured question records with IDs, question types, gold answers, expected sources, expected keywords, answerability labels, and expected behavior labels.

The default dataset currently contains 36 questions covering single-document QA, multi-chunk synthesis, ambiguous questions, unanswerable questions, distractor handling, comparisons, follow-up questions, citation-sensitive questions, cross-file reasoning, and false-premise correction.

## Reproduction

Run a representative smoke matrix first:

```bash
.venv/bin/python -m experiments.run_ablation \
  --questions evaluation/eval_questions.json \
  --config-dir experiments/configs \
  --output-dir experiments/results/smoke \
  --question-ids q001,q016,q027,q030,q033
```

Run all 36 questions:

```bash
.venv/bin/python -m experiments.run_ablation \
  --questions evaluation/eval_questions.json \
  --config-dir experiments/configs \
  --output-dir experiments/results \
  --report experiments/results/ablation_report.md
```

The per-variant JSON files are the source of truth. V0 produces `baseline_result.json`; V6 produces `agentic_result.json`; `comparison_result.json` is derived from those existing results without rerunning either system.

## Runtime Configuration

The P0b run completed on June 11, 2026 with:

- LLM: `deepseek-v4-flash`
- Temperature: `0.0`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Dense/BM25/RRF candidate limits: `20/20/20`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Reranker output/candidate limits: `5/12`
- Dataset: 36 questions
- Result status: all seven variants `completed`, zero runner errors

## Ablation Results

| Method | Correctness | Context Relevance | Citation Accuracy | Fallback Accuracy | Unsupported Claims | Supported Claim Ratio | Avg Retry | Avg Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 Naive RAG | 0.6111 | 1.0000 | 0.9615 | 0.8611 | N/A | N/A | 0.0000 | 4.4576s |
| V1 + Query Transformation | 0.5833 | 1.0000 | 1.0000 | 0.8611 | N/A | N/A | 0.0000 | 8.5843s |
| V2 + Retrieval Grading | 0.5833 | 1.0000 | 0.7308 | 0.7778 | N/A | N/A | 0.0000 | 13.8411s |
| V3 + Conditional Retry / Fallback | 0.5833 | 0.9615 | 0.8077 | 0.7778 | N/A | N/A | 0.7778 | 25.6108s |
| V4 + Hybrid Retrieval | 0.5833 | 0.9615 | 0.8462 | 0.8056 | N/A | N/A | 0.6944 | 25.7927s |
| V5 + Reranker | 0.6389 | 0.9615 | 0.8077 | 0.8056 | N/A | N/A | 0.6667 | 28.6960s |
| V6 + Claim-level Citation Verification | 0.5833 | 0.9615 | 0.7692 | 0.7778 | 0 | 1.0000 | 0.7500 | 41.2485s |

## Observed Trade-offs

- Query transformation improved citation accuracy from `0.9615` to `1.0000`, but correctness fell by `0.0278` and average latency increased by `4.13s`.
- Retrieval grading was conservative in this run. V2 lowered answer rate from `0.8056` to `0.5556`, citation accuracy to `0.7308`, and fallback accuracy to `0.7778`.
- Conditional retry recovered answer rate to `0.6667` and citation accuracy to `0.8077`, but added `0.7778` retries per question and approximately `11.77s` over V2.
- Hybrid retrieval improved citation accuracy from `0.8077` to `0.8462` and fallback accuracy from `0.7778` to `0.8056` with only `0.18s` additional average latency over V3.
- Reranking produced the highest heuristic correctness, `0.6389`, while adding `2.90s` average latency over V4. Citation accuracy decreased by `0.0385`, so the result does not justify claiming that reranking improved every metric.
- Claim-level verification finished with zero unsupported claims and a `1.0000` supported-claim ratio among verified claim records. It also reduced correctness from `0.6389` to `0.5833` and increased average latency by `12.55s`, showing the cost of stricter answer acceptance and extra verifier calls.
- V6 did not outperform V0 on heuristic correctness or latency. Its measured contribution is stronger claim support diagnostics and stricter answer control, not a blanket quality improvement.

## Metric Notes

- Correctness is deterministic and uses expected keywords plus gold-answer overlap.
- Context relevance and source hit metrics use expected source matches from retrieved, relevant, or cited documents depending on the metric.
- Citation hit rate measures citation source matches, not merely retrieval success.
- Token usage and cost are recorded only when the active model client exposes usage metadata.
- Unsupported and supported claim metrics are computed from `claim_verification_results`, not extracted claims.
- Citation-verification metrics are `N/A` for V0-V5 because that capability is disabled.
- Follow-up cases pass chat history to Agentic variants; the naive baseline accepts but intentionally ignores it.

## Limitations

- Results are single runs and do not include confidence intervals.
- Correctness is not an LLM-as-judge or human evaluation score.
- The local 36-question dataset is project-specific rather than a public benchmark.
- The DeepSeek integration did not expose reliable token usage metadata in the current result payloads, so token usage and cost are reported as unavailable rather than estimated.
- Context relevance is source-level expected-source matching. It does not independently judge whether every retrieved chunk is semantically sufficient.
- Unsupported-claim count is measured after the revision/fallback workflow. A zero value means no unsupported claim reached the final evaluated payload, not that the initial drafts were always correct.
- The Approach B modular typed evaluator remains a later architecture upgrade.
- The interactive Gradio Evaluation Dashboard remains on the roadmap after CLI artifacts stabilize.
