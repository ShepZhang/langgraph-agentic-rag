# Changelog

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
- The project is reliability-oriented and production-oriented as an architecture exercise, not production-ready.
