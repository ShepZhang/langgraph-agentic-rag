# Evaluation Notes

Default evaluation should run the same documents and the same questions for
each compared system. This keeps comparisons between naive RAG and agentic RAG
honest: different corpora or different question sets would make the results
hard to interpret.

The key metrics are answer rate, fallback rate, citation rate, source hit rate,
keyword hit rate, fallback correctness, context relevance, citation hit rate,
verification rate, unsupported claim count, supported claim ratio, average
retry count, retrieved document count, relevant document count, latency, token
usage, and estimated cost.

An ablation should remove one capability at a time, such as query rewriting,
retrieval grading, fallback logic, or citation verification. The ablation
report should explain which capability changed and should avoid implying that
missing artifacts were generated.

Evaluation artifacts are JSON files written for downstream inspection. In
comparison mode the expected artifacts are `baseline_result.json`,
`agentic_result.json`, and `comparison_result.json`. These files should include
summary metrics and per-question rows so regressions can be audited.

P0b regeneration means rerunning the evaluation after dataset, prompt, or
pipeline changes and replacing stale artifacts with fresh results. The goal is
for the report to reflect the exact documents, questions, and code that were
used for that run.

