# P1a Hybrid Retrieval Design

## Context

P0a established the evaluation infrastructure: naive baseline, structured dataset, reliability metrics, JSON artifacts, and ablation scaffolding. P1a is the first algorithm upgrade from the roadmap. It should improve retrieval recall without changing the LangGraph workflow or answer-generation behavior.

The existing retrieval path is:

```text
Retriever.retrieve(query)
  -> VectorStoreManager.similarity_search(query, top_k)
  -> optional CrossEncoderReranker
  -> normalized chunks
```

## Goal

Add optional hybrid retrieval:

```text
query
  -> dense retriever top-k
  -> BM25 sparse retriever top-k
  -> RRF fusion top-k
  -> optional reranker final top-k
  -> normalized chunks
```

This supports both semantic similarity questions and exact keyword, identifier, command, filename, metric-name, and acronym-sensitive questions.

## Non-Goals

- Do not modify LangGraph nodes, edges, prompts, or state in P1a.
- Do not implement query transformation, multi-query expansion, or decomposition.
- Do not implement P2 claim-level citation revision loops.
- Do not run final P0b evaluation numbers yet. P0b is regenerated after P1/P2.

## Configuration

Add these fields to `Settings`:

- `hybrid_retrieval_enabled: bool`
- `dense_top_k: int`
- `bm25_top_k: int`
- `fusion_top_k: int`

Environment variables:

- `HYBRID_RETRIEVAL_ENABLED=false`
- `DENSE_TOP_K=20`
- `BM25_TOP_K=20`
- `FUSION_TOP_K=20`

Validation:

- all top-k values must be positive integers.
- existing `TOP_K` remains the final answer candidate count.

## Components

### `rag/fusion.py`

Implements Reciprocal Rank Fusion over ranked document lists. Deduplication uses `chunk_id` when present, otherwise source/page/content fallback. Fused documents carry `fusion_score` and per-source rank metadata.

### `rag/bm25_retriever.py`

Implements a lightweight local BM25 retriever with no new dependency. It tokenizes document text and query text using lowercased alphanumeric terms. It accepts a list of `Document` objects and returns `(Document, score)` pairs.

### `rag/hybrid_retriever.py`

Coordinates dense retrieval, sparse retrieval, and RRF. It accepts injected callables/managers for testing. Runtime sparse corpus is loaded from the vector store through a read-only `VectorStoreManager.get_all_documents()` method.

### `rag/retriever.py`

Keeps the public `Retriever.retrieve()` API unchanged. If hybrid retrieval is disabled, the current dense path remains unchanged. If enabled, it calls the hybrid retriever and then applies the existing optional reranker.

## Error Handling

- Empty sparse corpus falls back to dense-only candidates.
- BM25 receives no documents or non-positive top-k returns an empty list.
- RRF ignores empty ranked lists.
- If vector store loading fails, existing exceptions continue to surface through evaluation as per-question errors.

## Testing

Add focused tests:

- `tests/test_bm25_retriever.py`
  - keyword-sensitive query ranks exact-match documents first.
  - empty documents or non-positive top-k returns empty.
- `tests/test_fusion.py`
  - RRF deduplicates overlapping chunks and combines scores.
  - RRF respects final top-k.
- `tests/test_hybrid_retriever.py`
  - hybrid retrieval calls dense and sparse paths and fuses results.
  - empty sparse corpus still returns dense results.
- update `tests/test_retriever.py`
  - default path stays dense-only.
  - hybrid-enabled settings call hybrid retrieval with configured top-k values.
- update `tests/test_config.py`
  - default hybrid settings are disabled and positive.
  - invalid top-k values raise clear errors.

## Documentation

README should describe the hybrid pipeline as an implemented P1a feature while keeping P0b result regeneration as future work.
