# P1b Reranker Evaluation Readiness Design

## Goal

Make the existing optional reranker easier to configure, diagnose, and evaluate without changing the project into a heavier serving system or adding a new model dependency.

## Scope

P1b builds on P1a hybrid retrieval. The system already supports a sentence-transformers CrossEncoder reranker. This milestone improves the engineering surface around that capability:

- Add explicit `RERANKER_TOP_N` configuration for the post-rerank output count.
- Keep `RERANKER_CANDIDATE_TOP_K` as the pre-rerank candidate pool size.
- Add structured reranker result records for diagnostics and tests.
- Add safe runtime configuration snapshots to evaluation artifacts so P0b runs can record retriever and reranker settings without leaking API keys.

Out of scope:

- Downloading or benchmarking a new reranker model.
- Adding learned sparse retrieval or neural score calibration.
- Implementing independently toggleable ablation execution for every module. That remains a P0b/P1c concern.

## Design

### Configuration

`Settings` gains `reranker_top_n`. When reranker is enabled and `Retriever.retrieve(..., top_k=None)` is used, the final reranked output count comes from `RERANKER_TOP_N`. An explicit `top_k` argument still overrides this value for existing call sites and tests.

Validation rules:

- `RERANKER_TOP_N > 0`
- `RERANKER_CANDIDATE_TOP_K > 0`
- when reranker is enabled, `RERANKER_CANDIDATE_TOP_K >= RERANKER_TOP_N`
- when reranker is enabled, `RERANKER_MODEL` must be non-empty

### Structured Reranker Output

`rag.reranker` keeps the tuple-based `rerank()` method for compatibility. It also exposes a structured `rerank_as_records()` method returning records with:

- `document_id`
- `chunk_id`
- `content`
- `metadata`
- `vector_score`
- `rerank_score`
- `rank`

The record shape matches the resume/project requirement that reranker output be inspectable without decoding internal tuple positions.

### Retriever Integration

`Retriever` continues to normalize final chunks for existing Agent and UI consumers. The final top-k logic changes only when reranking is enabled and no explicit `top_k` is passed:

- reranker disabled: use `TOP_K`
- reranker enabled: use `RERANKER_TOP_N`
- explicit `top_k`: use the explicit value

Hybrid retrieval remains compatible: when both hybrid and reranker are enabled, the candidate pool is `max(RERANKER_CANDIDATE_TOP_K, FUSION_TOP_K, final_top_k)`.

### Evaluation Metadata

Evaluation artifacts include a `runtime_config` snapshot with:

- LLM provider, model, and temperature
- retriever top-k and hybrid retrieval settings
- reranker enabled flag, model, candidate top-k, and top-n
- vector collection name

The snapshot intentionally excludes API keys, base URLs, local paths, and user secrets.

## Testing

Tests cover:

- config parsing and validation for `RERANKER_TOP_N`
- retriever default final top-n behavior when reranker is enabled
- structured reranker records and ranking order
- evaluation artifact `runtime_config` output with no secret fields

## Risks

- Existing callers that relied on `TOP_K` while enabling reranker may see a different default count if `RERANKER_TOP_N` differs. Explicit `top_k` still preserves the old behavior.
- `runtime_config` is intentionally limited; it records enough for evaluation reproducibility but not deployment details.
