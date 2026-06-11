# P1e Structured Retrieval Grading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace coarse retrieval grading with structured chunk-level relevance labels, confidence scores, and reasons.

**Architecture:** Add `agent/retrieval_grading.py` as a focused parser/normalizer. Keep `grade_documents_node` and graph routing compatible, but write richer grading diagnostics into state and result payloads.

**Tech Stack:** Python TypedDicts, existing LangGraph state, pytest, existing fake LLM test pattern.

---

### Task 1: Structured Grading Parser

**Files:**
- Create: `agent/retrieval_grading.py`
- Test: `tests/test_retrieval_grading.py`

- [ ] **Step 1: Write failing parser tests**

Cover structured schema, legacy schema, invalid JSON, invalid labels, confidence clamping, and out-of-range indexes.

- [ ] **Step 2: Run tests and see failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_retrieval_grading.py -q
```

Expected: import failure because `agent.retrieval_grading` does not exist.

- [ ] **Step 3: Implement parser**

Implement `DocumentGrade`, `GradingResult`, `parse_retrieval_grading_response()`, and helper functions.

- [ ] **Step 4: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_retrieval_grading.py -q
```

Expected: parser tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/retrieval_grading.py tests/test_retrieval_grading.py
git commit -m "feat: add structured retrieval grading parser"
```

### Task 2: Agent State And Grade Node Integration

**Files:**
- Modify: `agent/state.py`
- Modify: `agent/nodes.py`
- Modify: `tests/test_agent_state_prompts.py`
- Modify: `tests/test_agent_nodes.py`

- [ ] **Step 1: Write failing state/node tests**

Add defaults for `document_grades`, `relevant_document_count`, `partial_document_count`, and `max_relevance_confidence`. Add tests showing structured grading selects only `relevant` chunks and records partial counts.

- [ ] **Step 2: Run tests and see failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_state_prompts.py tests/test_agent_nodes.py -q
```

Expected: fails because state fields and node updates do not exist.

- [ ] **Step 3: Implement node integration**

Use `parse_retrieval_grading_response()` in `grade_documents_node()`. Populate `document_grades`, counts, max confidence, and `relevant_documents`.

- [ ] **Step 4: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_state_prompts.py tests/test_agent_nodes.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/state.py agent/nodes.py tests/test_agent_state_prompts.py tests/test_agent_nodes.py
git commit -m "feat: wire structured grading into agent node"
```

### Task 3: Graph Result Payload

**Files:**
- Modify: `agent/graph.py`
- Modify: `tests/test_agent_graph.py`

- [ ] **Step 1: Write failing graph test**

Assert `run_agent()` returns `document_grades`, `relevant_document_count`, `partial_document_count`, and `max_relevance_confidence`.

- [ ] **Step 2: Run test and see failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_graph.py -q
```

Expected: fails because result payload lacks structured grading fields.

- [ ] **Step 3: Implement payload fields**

Return structured grading diagnostics from final state.

- [ ] **Step 4: Run graph tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_graph.py -q
```

Expected: graph tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/graph.py tests/test_agent_graph.py
git commit -m "feat: expose structured grading diagnostics"
```

### Task 4: Prompt And Documentation

**Files:**
- Modify: `agent/prompts.py`
- Modify: `tests/test_agent_state_prompts.py`
- Modify: `README.md`
- Modify: `docs/resume_bullets.md`
- Modify: `experiments/report.md`

- [ ] **Step 1: Update prompt and docs**

Update `RETRIEVAL_GRADING_PROMPT` to request the structured schema while documenting legacy parser compatibility only in code/tests.

- [ ] **Step 2: Run full tests**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add agent/prompts.py tests/test_agent_state_prompts.py README.md docs/resume_bullets.md experiments/report.md
git commit -m "docs: document structured retrieval grading milestone"
```
