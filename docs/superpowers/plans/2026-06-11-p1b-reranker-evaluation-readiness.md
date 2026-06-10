# P1b Reranker Evaluation Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the optional reranker configurable, inspectable, and evaluation-ready after the P1a hybrid retrieval milestone.

**Architecture:** Add `RERANKER_TOP_N` to runtime settings, keep reranker tuple compatibility while adding structured diagnostic records, and write sanitized runtime config snapshots into evaluation artifacts. The existing retriever remains the integration point for dense-only and hybrid retrieval paths.

**Tech Stack:** Python 3.12 test runtime, dataclass settings in `config.py`, LangChain `Document`, pytest, JSON evaluation artifacts.

---

### Task 1: Reranker Top-N Configuration

**Files:**
- Modify: `config.py`
- Modify: `.env.example`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

Add tests that set `RERANKER_TOP_N`, verify it parses, and verify invalid values fail:

```python
def test_get_settings_accepts_reranker_top_n(monkeypatch):
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_TOP_N", "5")
    monkeypatch.setenv("RERANKER_CANDIDATE_TOP_K", "8")

    settings = get_settings()

    assert settings.reranker_top_n == 5
    assert settings.reranker_candidate_top_k == 8


def test_get_settings_rejects_invalid_reranker_top_n(monkeypatch):
    monkeypatch.setenv("RERANKER_TOP_N", "0")

    with pytest.raises(ValueError, match="RERANKER_TOP_N"):
        get_settings()
```

- [ ] **Step 2: Run tests and see failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_config.py -q
```

Expected: fails because `Settings` has no `reranker_top_n`.

- [ ] **Step 3: Implement config**

Add `reranker_top_n: int` to `Settings`, parse `RERANKER_TOP_N` in `get_settings()`, validate it is positive, and validate `RERANKER_CANDIDATE_TOP_K >= RERANKER_TOP_N` when reranker is enabled.

- [ ] **Step 4: Run config tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_config.py -q
```

Expected: all config tests pass.

- [ ] **Step 5: Commit**

```bash
git add config.py .env.example tests/test_config.py
git commit -m "feat: add reranker top-n configuration"
```

### Task 2: Structured Reranker Records

**Files:**
- Modify: `rag/reranker.py`
- Test: `tests/test_reranker.py`

- [ ] **Step 1: Write failing tests**

Add a test for `rerank_as_records()`:

```python
def test_cross_encoder_reranker_returns_structured_records():
    model = FakeCrossEncoder([0.2, 0.9])
    reranker = CrossEncoderReranker("fake-model", model=model)
    docs = [
        (Document(page_content="first", metadata={"source": "a.md", "chunk_id": "a-1"}), 0.8),
        (Document(page_content="second", metadata={"source": "b.md", "chunk_id": "b-1", "document_id": "doc-b"}), 0.7),
    ]

    records = reranker.rerank_as_records("question", docs, top_k=2)

    assert records[0]["document_id"] == "doc-b"
    assert records[0]["chunk_id"] == "b-1"
    assert records[0]["content"] == "second"
    assert records[0]["vector_score"] == 0.7
    assert records[0]["rerank_score"] == 0.9
    assert records[0]["rank"] == 1
```

- [ ] **Step 2: Run tests and see failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_reranker.py -q
```

Expected: fails because `rerank_as_records` does not exist.

- [ ] **Step 3: Implement structured output**

Add `RerankerRecord` TypedDict and `rerank_as_records()` that calls existing `rerank()` and formats records without changing tuple compatibility.

- [ ] **Step 4: Run reranker tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_reranker.py -q
```

Expected: all reranker tests pass.

- [ ] **Step 5: Commit**

```bash
git add rag/reranker.py tests/test_reranker.py
git commit -m "feat: add structured reranker records"
```

### Task 3: Retriever Final Top-N Semantics

**Files:**
- Modify: `rag/retriever.py`
- Test: `tests/test_retriever.py`

- [ ] **Step 1: Write failing test**

Add a test showing `RERANKER_TOP_N` controls default output count only when no explicit `top_k` is passed.

- [ ] **Step 2: Run test and see failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_retriever.py -q
```

Expected: fails because retriever currently uses `TOP_K` as the default final count.

- [ ] **Step 3: Implement default top-n**

In `Retriever.retrieve()`, compute final top-k with:

```python
if top_k is not None:
    final_top_k = top_k
elif self.reranker:
    final_top_k = self.settings.reranker_top_n
else:
    final_top_k = self.settings.top_k
```

- [ ] **Step 4: Run retriever tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_retriever.py -q
```

Expected: all retriever tests pass.

- [ ] **Step 5: Commit**

```bash
git add rag/retriever.py tests/test_retriever.py
git commit -m "feat: use reranker top-n in retriever"
```

### Task 4: Runtime Config Snapshot For Evaluation

**Files:**
- Create: `evaluation/runtime_config.py`
- Modify: `evaluation/evaluate.py`
- Modify: `experiments/run_ablation.py`
- Test: `tests/test_evaluation_runner.py` or `tests/test_evaluate.py`

- [ ] **Step 1: Write failing tests**

Add tests asserting evaluation artifacts include `runtime_config` and exclude API keys.

- [ ] **Step 2: Run tests and see failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluate.py tests/test_ablation.py -q
```

Expected: fails because no runtime config snapshot exists.

- [ ] **Step 3: Implement snapshot utility**

Create `build_runtime_config_snapshot(settings=None)` returning LLM, retriever, reranker, and vectorstore config without API keys or base URLs.

- [ ] **Step 4: Wire artifacts**

Add `runtime_config` to `baseline_result.json`, `agentic_result.json`, `comparison_result.json`, and `ablation_result.json` payloads.

- [ ] **Step 5: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluate.py tests/test_ablation.py -q
```

Expected: tests pass.

- [ ] **Step 6: Commit**

```bash
git add evaluation/runtime_config.py evaluation/evaluate.py experiments/run_ablation.py tests/test_evaluate.py tests/test_ablation.py
git commit -m "feat: record runtime config in evaluation artifacts"
```

### Task 5: Documentation And Full Verification

**Files:**
- Modify: `README.md`
- Modify: `experiments/report.md`
- Modify: `docs/resume_bullets.md`

- [ ] **Step 1: Update docs**

Document `RERANKER_TOP_N`, structured reranker diagnostics, and runtime config snapshots.

- [ ] **Step 2: Run full tests**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add README.md experiments/report.md docs/resume_bullets.md
git commit -m "docs: document reranker evaluation readiness"
```
