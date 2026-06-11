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

## Ablation Matrix

| Version | Added capability | Independent control |
|---|---|---|
| V0 | Naive dense retrieve-generate RAG | baseline runner |
| V1 | Query transformation | graph feature flag |
| V2 | Structured retrieval grading | graph feature flag |
| V3 | Conditional retry and fallback | graph feature flag |
| V4 | BM25 + dense retrieval + RRF fusion | retriever settings |
| V5 | Reranking | retriever settings |
| V6 | Claim-level citation verification and revision | graph feature flag |

Measured metrics are generated in `experiments/results/ablation_report.md`. This file is refreshed with the final DeepSeek run after smoke validation succeeds.

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
- Token usage and cost remain unavailable when the model client does not expose reliable metadata.
- The Approach B modular typed evaluator remains a later architecture upgrade.
- The interactive Gradio Evaluation Dashboard remains on the roadmap after CLI artifacts stabilize.
