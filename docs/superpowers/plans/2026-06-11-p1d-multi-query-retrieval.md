# P1d Multi-query Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute P1c `expanded_queries` as real retrieval calls and merge retrieved evidence before grading.

**Architecture:** Keep the LangGraph topology unchanged. Add `agent/multi_query.py` for query list construction and dict-level document merging, then call it from `AgentNodes.retrieve_node()`.

**Tech Stack:** Python TypedDict-style dicts, existing LangGraph state, existing retriever tool, pytest.

---

### Task 1: Multi-query Helper Module

**Files:**
- Create: `agent/multi_query.py`
- Test: `tests/test_multi_query.py`

- [ ] **Step 1: Write failing helper tests**

Cover query construction, blank-query filtering, order-preserving de-duplication, and document merging with `matched_queries` metadata.

- [ ] **Step 2: Run tests and see failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_multi_query.py -q
```

Expected: import failure because `agent.multi_query` does not exist.

- [ ] **Step 3: Implement helpers**

Implement `build_retrieval_queries()` and `merge_retrieved_documents()`.

- [ ] **Step 4: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_multi_query.py -q
```

Expected: helper tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/multi_query.py tests/test_multi_query.py
git commit -m "feat: add multi-query retrieval helpers"
```

### Task 2: Agent State And Retrieve Node Integration

**Files:**
- Modify: `agent/state.py`
- Modify: `agent/nodes.py`
- Modify: `tests/test_agent_state_prompts.py`
- Modify: `tests/test_agent_nodes.py`

- [ ] **Step 1: Write failing state/node tests**

Add defaults for `retrieval_queries`, `multi_query_used`, and `multi_query_result_count`. Add tests proving `retrieve_node` calls the retriever once for rewrite and multiple times for multi-query.

- [ ] **Step 2: Run tests and see failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_state_prompts.py tests/test_agent_nodes.py -q
```

Expected: fails because state fields and multi-query retrieve behavior do not exist.

- [ ] **Step 3: Implement state and node updates**

Use `build_retrieval_queries()` and `merge_retrieved_documents()` in `retrieve_node`.

- [ ] **Step 4: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_state_prompts.py tests/test_agent_nodes.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/state.py agent/nodes.py tests/test_agent_state_prompts.py tests/test_agent_nodes.py
git commit -m "feat: execute multi-query retrieval in agent node"
```

### Task 3: Graph Result Payload

**Files:**
- Modify: `agent/graph.py`
- Modify: `tests/test_agent_graph.py`

- [ ] **Step 1: Write failing graph payload test**

Assert `run_agent()` returns `retrieval_queries`, `multi_query_used`, and `multi_query_result_count`.

- [ ] **Step 2: Run test and see failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_graph.py -q
```

Expected: fails because result payload lacks these fields.

- [ ] **Step 3: Implement payload fields**

Return multi-query diagnostics from final state.

- [ ] **Step 4: Run graph tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_graph.py -q
```

Expected: graph tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/graph.py tests/test_agent_graph.py
git commit -m "feat: expose multi-query retrieval diagnostics"
```

### Task 4: Documentation And Full Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/resume_bullets.md`
- Modify: `experiments/report.md`

- [ ] **Step 1: Update docs**

Document P1d multi-query retrieval execution and keep decomposition retrieval listed as future work.

- [ ] **Step 2: Run full tests**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/resume_bullets.md experiments/report.md
git commit -m "docs: document multi-query retrieval milestone"
```
