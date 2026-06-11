# P1f Partial-Relevance Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight recovery path when retrieval grading finds partially relevant chunks but no directly relevant evidence.

**Architecture:** Keep the existing LangGraph topology. Add a structured `partial_relevance_recovery` state field, populate it in `grade_documents_node`, include it in `run_agent()` results, and feed partial evidence into retry query rewriting.

**Tech Stack:** Python TypedDicts, LangGraph state, existing prompt templates, pytest fake LLM tests.

---

### Task 1: State And Result Payload

**Files:**
- Modify: `agent/state.py`
- Modify: `agent/graph.py`
- Modify: `tests/test_agent_state_prompts.py`
- Modify: `tests/test_agent_graph.py`

- [ ] **Step 1: Write failing state and graph tests**

Add assertions that `create_initial_state()` contains inactive recovery metadata and that `run_agent()` exposes `partial_relevance_recovery`.

Expected inactive value:

```python
{
    "triggered": False,
    "action": "none",
    "reason": "",
    "partial_document_indices": [],
}
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_state_prompts.py tests/test_agent_graph.py -q
```

Expected: fail with missing `partial_relevance_recovery`.

- [ ] **Step 3: Implement minimal state and result fields**

Add a `PartialRelevanceRecovery` TypedDict, set the default value in `create_initial_state()`, and return the field in `run_agent()`.

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_state_prompts.py tests/test_agent_graph.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/state.py agent/graph.py tests/test_agent_state_prompts.py tests/test_agent_graph.py
git commit -m "feat: expose partial relevance recovery state"
```

### Task 2: Grade Node Recovery Trigger

**Files:**
- Modify: `agent/nodes.py`
- Modify: `tests/test_agent_nodes.py`

- [ ] **Step 1: Write failing grade-node tests**

Add a test where the grading LLM returns only `partially_relevant` chunks:

```json
{
  "grades": [
    {
      "document_index": 1,
      "relevance": "partially_relevant",
      "confidence": 0.72,
      "reason": "Related to the topic but missing the requested comparison."
    }
  ],
  "reason": "Only partial evidence found."
}
```

Assert:

```python
assert update["is_relevant"] is False
assert update["relevant_documents"] == []
assert update["partial_relevance_recovery"] == {
    "triggered": True,
    "action": "query_refinement",
    "reason": "Only partially relevant chunks were found; refine query to target missing evidence.",
    "partial_document_indices": [1],
}
```

Also assert a `relevant` chunk returns inactive recovery metadata.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_nodes.py -q
```

Expected: fail because `grade_documents_node()` does not return `partial_relevance_recovery`.

- [ ] **Step 3: Implement recovery trigger**

In `grade_documents_node()`, build active recovery metadata when `relevant_documents` is empty and `grading["partially_relevant_indices"]` is non-empty. Return inactive metadata for empty docs, relevant docs, and fully irrelevant docs.

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_nodes.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/nodes.py tests/test_agent_nodes.py
git commit -m "feat: trigger partial relevance recovery"
```

### Task 3: Retry Prompt Uses Partial Evidence

**Files:**
- Modify: `agent/prompts.py`
- Modify: `agent/nodes.py`
- Modify: `tests/test_agent_nodes.py`
- Modify: `tests/test_agent_state_prompts.py`

- [ ] **Step 1: Write failing retry prompt tests**

Add a retry rewrite test where `state["partial_relevance_recovery"]["triggered"]` is true and `state["document_grades"]` includes a partial chunk. Assert the prompt contains:

```python
assert "Partial relevance recovery" in llm.prompts[0]
assert "Only partially relevant chunks were found" in llm.prompts[0]
assert "Related but missing the requested comparison" in llm.prompts[0]
assert "Partially related context" in llm.prompts[0]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_nodes.py tests/test_agent_state_prompts.py -q
```

Expected: fail because the retry prompt has no recovery section.

- [ ] **Step 3: Implement prompt enrichment**

Add a `{partial_relevance_context}` placeholder to `RETRY_QUERY_REWRITE_PROMPT`. In `rewrite_query_node()`, format active recovery metadata plus partial chunk snippets; otherwise pass `No partial relevance recovery was triggered.`

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_nodes.py tests/test_agent_state_prompts.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/prompts.py agent/nodes.py tests/test_agent_nodes.py tests/test_agent_state_prompts.py
git commit -m "feat: guide retry rewrites with partial evidence"
```

### Task 4: Documentation And Full Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/resume_bullets.md`
- Modify: `experiments/report.md`

- [ ] **Step 1: Update documentation**

Document P1f as implemented lightweight partial-relevance recovery. Keep dynamic top-k expansion and reranker adjustment in the future roadmap.

- [ ] **Step 2: Run full test suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/resume_bullets.md experiments/report.md
git commit -m "docs: document partial relevance recovery milestone"
```

## Self-Review

- The plan covers every P1f spec requirement.
- The plan keeps recovery lightweight and does not add dynamic top-k, reranker adjustment, or new graph nodes.
- Each implementation task has a failing-test step before production code.
- The public result payload is updated for evaluation and future trace logging.
