# P1a Hybrid Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional hybrid retrieval with dense search, BM25 sparse search, RRF fusion, and existing optional reranking.

**Architecture:** Keep `Retriever.retrieve()` as the public entrypoint. Add focused modules for BM25 and RRF, then compose them in `HybridRetriever`. The LangGraph workflow remains unchanged.

**Tech Stack:** Python 3.11, LangChain `Document`, pytest, existing `Settings`, existing vector store manager and reranker.

---

## File Structure

- Create `rag/fusion.py`: RRF scoring and document deduplication.
- Create `rag/bm25_retriever.py`: dependency-free BM25 retriever over `Document` chunks.
- Create `rag/hybrid_retriever.py`: dense + BM25 + RRF orchestration.
- Modify `rag/vectorstore.py`: add `get_all_documents()` read-only corpus loading.
- Modify `rag/retriever.py`: choose dense or hybrid path based on settings.
- Modify `config.py`: add hybrid settings and validation.
- Modify `.env.example` and `README.md`: document P1a config and pipeline.
- Add tests: `tests/test_fusion.py`, `tests/test_bm25_retriever.py`, `tests/test_hybrid_retriever.py`.
- Update tests: `tests/test_config.py`, `tests/test_retriever.py`.

## Task 1: Hybrid Configuration

**Files:**
- Modify: `config.py`
- Modify: `.env.example`
- Modify: `tests/test_config.py`

- [ ] Add failing tests for default hybrid config and invalid top-k values.
- [ ] Add `hybrid_retrieval_enabled`, `dense_top_k`, `bm25_top_k`, `fusion_top_k` fields to `Settings`.
- [ ] Load `HYBRID_RETRIEVAL_ENABLED`, `DENSE_TOP_K`, `BM25_TOP_K`, and `FUSION_TOP_K`.
- [ ] Validate all new top-k fields are positive.
- [ ] Run `tests/test_config.py`.
- [ ] Commit with `feat: add hybrid retrieval configuration`.

## Task 2: BM25 And RRF Units

**Files:**
- Create: `rag/bm25_retriever.py`
- Create: `rag/fusion.py`
- Create: `tests/test_bm25_retriever.py`
- Create: `tests/test_fusion.py`

- [ ] Add failing BM25 tests for exact keyword ranking and empty input.
- [ ] Add failing RRF tests for dedupe, fused scores, and top-k.
- [ ] Implement dependency-free BM25 tokenization/scoring.
- [ ] Implement RRF with stable document keys and `fusion_score` metadata.
- [ ] Run the two new test files.
- [ ] Commit with `feat: add bm25 retriever and rrf fusion`.

## Task 3: Hybrid Retriever Composition

**Files:**
- Create: `rag/hybrid_retriever.py`
- Modify: `rag/vectorstore.py`
- Create: `tests/test_hybrid_retriever.py`

- [ ] Add failing tests for dense + sparse fusion and sparse-empty fallback.
- [ ] Add `VectorStoreManager.get_all_documents()`.
- [ ] Implement `HybridRetriever.retrieve()` with injected dense manager, sparse corpus, and RRF.
- [ ] Run `tests/test_hybrid_retriever.py`.
- [ ] Commit with `feat: add hybrid retriever pipeline`.

## Task 4: Integrate With Public Retriever

**Files:**
- Modify: `rag/retriever.py`
- Modify: `tests/test_retriever.py`

- [ ] Add failing tests showing hybrid-disabled path is unchanged and hybrid-enabled path uses configured top-k values.
- [ ] Route `Retriever.retrieve()` through `HybridRetriever` when `settings.hybrid_retrieval_enabled` is true.
- [ ] Keep reranker behavior after candidate retrieval.
- [ ] Run `tests/test_retriever.py`.
- [ ] Commit with `feat: route retrieval through hybrid pipeline`.

## Task 5: Docs And Verification

**Files:**
- Modify: `README.md`
- Modify: `.env.example`

- [ ] Update README with P1a hybrid retrieval pipeline and config.
- [ ] Run focused tests: `tests/test_config.py tests/test_bm25_retriever.py tests/test_fusion.py tests/test_hybrid_retriever.py tests/test_retriever.py`.
- [ ] Run full test suite: `.venv/bin/python -m pytest -q`.
- [ ] Commit with `docs: document hybrid retrieval configuration`.

## Acceptance Criteria

- Default behavior remains dense retrieval with optional reranker.
- Hybrid retrieval can be enabled by config without changing agent nodes.
- Dense and BM25 candidates are fused with RRF.
- Existing reranker runs after hybrid fusion when enabled.
- All tests pass.
