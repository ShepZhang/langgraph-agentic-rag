# P1c Query Transformation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the initial query rewrite step into structured query transformation with rewrite, multi-query, and decomposition metadata.

**Architecture:** Add a focused `agent/query_transform.py` module for prompt construction and parsing. Keep the existing LangGraph `rewrite_query` node name and route edges, but store structured query transformation fields in state and result payloads for later multi-query retrieval.

**Tech Stack:** Python TypedDicts, LangGraph state dicts, pytest, existing fake LLM test pattern.

---

### Task 1: Query Transform Parser And Prompt

**Files:**
- Create: `agent/query_transform.py`
- Test: `tests/test_query_transform.py`

- [ ] **Step 1: Write failing parser tests**

Add tests for structured JSON, fenced JSON, plain-text fallback, invalid strategy fallback, and blank response fallback.

- [ ] **Step 2: Run tests and see failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_query_transform.py -q
```

Expected: import fails because `agent.query_transform` does not exist.

- [ ] **Step 3: Implement module**

Implement `QueryTransformResult`, `build_query_transform_prompt`, `parse_query_transform_response`, and `fallback_query_transform`.

- [ ] **Step 4: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_query_transform.py -q
```

Expected: all query transform tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/query_transform.py tests/test_query_transform.py
git commit -m "feat: add structured query transform parser"
```

### Task 2: Agent State And Rewrite Node Integration

**Files:**
- Modify: `agent/state.py`
- Modify: `agent/nodes.py`
- Modify: `tests/test_agent_state_prompts.py`
- Modify: `tests/test_agent_nodes.py`

- [ ] **Step 1: Write failing state and node tests**

Add assertions that initial state includes query transformation defaults and that `rewrite_query_node` records `standalone_question`, `query_transform_strategy`, `query_transform_reason`, `expanded_queries`, `sub_questions`, and `query_transform`.

- [ ] **Step 2: Run tests and see failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_state_prompts.py tests/test_agent_nodes.py -q
```

Expected: fails because state fields and node updates do not exist.

- [ ] **Step 3: Implement state and node integration**

Add state fields. In `rewrite_query_node`, use structured query transformation only on the first attempt; keep retry rewrite logic unchanged except for storing a conservative query transform record for the retry query.

- [ ] **Step 4: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_state_prompts.py tests/test_agent_nodes.py -q
```

Expected: all state and node tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/state.py agent/nodes.py tests/test_agent_state_prompts.py tests/test_agent_nodes.py
git commit -m "feat: wire query transformation into agent state"
```

### Task 3: Graph Result Payload

**Files:**
- Modify: `agent/graph.py`
- Modify: `tests/test_agent_graph.py`

- [ ] **Step 1: Write failing graph test**

Add assertions that `run_agent()` returns `standalone_question`, `query_transform`, `query_transform_strategy`, `query_transform_reason`, `expanded_queries`, and `sub_questions`.

- [ ] **Step 2: Run test and see failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_graph.py -q
```

Expected: fails because result payload lacks query transformation fields.

- [ ] **Step 3: Implement payload fields**

Return query transformation fields from final state in `run_agent()`.

- [ ] **Step 4: Run graph tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_graph.py -q
```

Expected: all graph tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/graph.py tests/test_agent_graph.py
git commit -m "feat: expose query transformation results"
```

### Task 4: Documentation And Full Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/resume_bullets.md`
- Modify: `experiments/report.md`

- [ ] **Step 1: Update docs**

Document P1c query transformation as implemented metadata, and state that multi-query retrieval execution is deferred to P1d.

- [ ] **Step 2: Run full tests**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/resume_bullets.md experiments/report.md
git commit -m "docs: document query transformation milestone"
```
