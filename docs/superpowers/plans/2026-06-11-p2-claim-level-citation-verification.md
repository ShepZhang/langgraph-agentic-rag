# P2 Claim-Level Citation Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current in-node lightweight claim verification with an explicit LangGraph claim extraction, citation verification, answer revision, and finalization workflow.

**Architecture:** Add `agent/citation_verification.py` for parser and normalization logic, extend `AgentState` with draft/verification diagnostics, split verification into `extract_claims`, `verify_citations`, `revise_answer`, and `finalize_answer` nodes, then update graph edges and compatibility payloads.

**Tech Stack:** Python TypedDicts, LangGraph `StateGraph`, existing fake LLM tests, pytest, existing prompt templates and node helpers.

---

## File Map

- Create `agent/citation_verification.py`: parse claim extraction and verification JSON, normalize labels, validate cited chunk ids, build summary payloads.
- Modify `agent/state.py`: add draft-answer, cited-documents, verification-results, unsupported-claims, revision, and skip fields.
- Modify `agent/prompts.py`: add `CLAIM_EXTRACTION_PROMPT`, `CITATION_VERIFICATION_PROMPT`, and `ANSWER_REVISION_PROMPT`.
- Modify `agent/nodes.py`: split answer generation from verification and add extraction, verification, revision, and finalization nodes.
- Modify `agent/edges.py`: add conditional routing after citation verification and revision.
- Modify `agent/graph.py`: add new LangGraph nodes and result payload fields.
- Modify tests in `tests/test_citation_verification.py`, `tests/test_agent_state_prompts.py`, `tests/test_agent_nodes.py`, and `tests/test_agent_graph.py`.
- Modify docs in `README.md`, `docs/resume_bullets.md`, and `experiments/report.md`.

---

### Task 1: Citation Verification Parser Module

**Files:**
- Create: `agent/citation_verification.py`
- Create: `tests/test_citation_verification.py`

- [ ] **Step 1: Write failing parser tests**

Create `tests/test_citation_verification.py` with tests for these behaviors:

```python
from agent.citation_verification import (
    build_claim_verification_summary,
    parse_claim_extraction_response,
    parse_citation_verification_response,
)


def test_parse_claim_extraction_response_keeps_valid_cited_chunk_ids():
    result = parse_claim_extraction_response(
        (
            '{"claims": ['
            '{"claim_id": "c001", "claim": "Agentic RAG uses grading.", '
            '"cited_chunk_ids": ["chunk-1", "missing"]}'
            '], "reason": "one claim"}'
        ),
        valid_chunk_ids=["chunk-1"],
    )

    assert result["claims"] == [
        {
            "claim_id": "c001",
            "claim": "Agentic RAG uses grading.",
            "cited_chunk_ids": ["chunk-1"],
        }
    ]
    assert result["reason"] == "one claim"


def test_parse_claim_extraction_response_returns_none_for_invalid_json():
    assert parse_claim_extraction_response("not json", valid_chunk_ids=["chunk-1"]) is None


def test_parse_citation_verification_response_normalizes_labels_and_confidence():
    result = parse_citation_verification_response(
        (
            '{"results": ['
            '{"claim_id": "c001", "claim": "A", "cited_chunk_ids": ["chunk-1"], '
            '"verification_label": "SUPPORTED", "confidence": 2, "reason": "ok"},'
            '{"claim_id": "c002", "claim": "B", "cited_chunk_ids": ["chunk-2"], '
            '"verification_label": "unknown", "confidence": -1, "reason": "bad"}'
            '], "reason": "checked"}'
        ),
        valid_chunk_ids=["chunk-1", "chunk-2"],
    )

    assert result["results"][0]["verification_label"] == "supported"
    assert result["results"][0]["confidence"] == 1.0
    assert result["results"][1]["verification_label"] == "unsupported"
    assert result["results"][1]["confidence"] == 0.0


def test_build_claim_verification_summary_counts_unsupported_claims():
    results = [
        {
            "claim_id": "c001",
            "claim": "A",
            "cited_chunk_ids": ["chunk-1"],
            "verification_label": "supported",
            "confidence": 0.9,
            "reason": "ok",
        },
        {
            "claim_id": "c002",
            "claim": "B",
            "cited_chunk_ids": [],
            "verification_label": "unsupported",
            "confidence": 0.2,
            "reason": "no support",
        },
    ]

    summary = build_claim_verification_summary(results, reason="checked")

    assert summary["verified"] is False
    assert summary["reason"] == "checked"
    assert summary["unsupported_claims"] == [results[1]]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_citation_verification.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'agent.citation_verification'`.

- [ ] **Step 3: Implement parser module**

Create `agent/citation_verification.py` with public functions named `parse_claim_extraction_response(raw_result, valid_chunk_ids)`, `parse_citation_verification_response(raw_result, valid_chunk_ids)`, and `build_claim_verification_summary(results, reason)`.

Use a JSON decoder helper that extracts the first JSON object from fenced or prefixed LLM text. Drop claims with blank `claim_id` or blank `claim`. Filter `cited_chunk_ids` to selected citation chunk ids. Normalize labels to `supported`, `partially_supported`, or `unsupported`; unknown labels become `unsupported`. Clamp confidence into `[0.0, 1.0]`.

- [ ] **Step 4: Run parser tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_citation_verification.py -q
```

Expected: all parser tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/citation_verification.py tests/test_citation_verification.py
git commit -m "feat: add citation verification parsers"
```

### Task 2: State And Prompt Additions

**Files:**
- Modify: `agent/state.py`
- Modify: `agent/prompts.py`
- Modify: `tests/test_agent_state_prompts.py`

- [ ] **Step 1: Write failing state and prompt tests**

Add assertions to `test_create_initial_state_sets_defaults()`:

```python
assert state["draft_answer"] == ""
assert state["used_citation_indices"] == []
assert state["cited_documents"] == []
assert state["claim_verification_results"] == []
assert state["unsupported_claims"] == []
assert state["citation_verification_passed"] is False
assert state["citation_revision_count"] == 0
assert state["max_citation_revision_count"] == 1
assert state["citation_verification_skipped"] is False
```

Add prompt guardrails:

```python
from agent.prompts import (
    ANSWER_REVISION_PROMPT,
    CITATION_VERIFICATION_PROMPT,
    CLAIM_EXTRACTION_PROMPT,
)

assert "claim_id" in CLAIM_EXTRACTION_PROMPT
assert "verification_label" in CITATION_VERIFICATION_PROMPT
assert "unsupported" in ANSWER_REVISION_PROMPT.lower()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_state_prompts.py -q
```

Expected: fail because state fields and prompt constants are missing.

- [ ] **Step 3: Implement state and prompts**

Add the new fields to `AgentState` and `create_initial_state()`. Add the three prompt templates to `agent/prompts.py`. Keep the existing `CLAIM_VERIFICATION_PROMPT` importable during migration.

- [ ] **Step 4: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_state_prompts.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/state.py agent/prompts.py tests/test_agent_state_prompts.py
git commit -m "feat: add citation verification state and prompts"
```

### Task 3: Generate Answer Produces Draft State

**Files:**
- Modify: `agent/nodes.py`
- Modify: `tests/test_agent_nodes.py`

- [ ] **Step 1: Write failing generate-node tests**

Update the existing normal-answer test so `generate_answer_node()` no longer consumes a verifier response. The fake LLM should provide only answer-generation JSON:

```python
llm = FakeLLM([
    '{"answer": "Grounded answer with [2].", "used_citation_indices": [2]}'
])
```

Assert:

```python
assert update["draft_answer"] == "Grounded answer with [2]."
assert update["answer"] == ""
assert update["used_citation_indices"] == [2]
assert update["cited_documents"] == [state["relevant_documents"][1]]
assert update["citations"][0]["chunk_id"] == "paper.pdf:p2:c1"
assert update["route"] == "extract_claims"
```

Add an unable-to-answer test:

```python
llm = FakeLLM([
    '{"answer": "I cannot answer from the current documents.", "used_citation_indices": []}'
])
```

Assert:

```python
assert update["draft_answer"] == "I cannot answer from the current documents."
assert update["citation_verification_skipped"] is True
assert update["citation_verification_passed"] is False
assert update["route"] == "finalize_answer"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_nodes.py -q
```

Expected: fail because `generate_answer_node()` still calls the old verifier and returns final `answer`.

- [ ] **Step 3: Modify `generate_answer_node()`**

Remove the direct `_verify_answer_claims()` call from `generate_answer_node()`. For normal cited answers, return draft/citation state and route `extract_claims`. For unable-to-answer responses, return draft state, empty citations, `citation_verification_skipped=True`, and route `finalize_answer`. Keep citation marker validation and fallback behavior unchanged.

- [ ] **Step 4: Run node tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_nodes.py -q
```

Expected: generate-node tests pass after migrating old verifier expectations.

- [ ] **Step 5: Commit**

```bash
git add agent/nodes.py tests/test_agent_nodes.py
git commit -m "feat: generate draft answers before citation verification"
```

### Task 4: Claim Extraction And Citation Verification Nodes

**Files:**
- Modify: `agent/nodes.py`
- Modify: `tests/test_agent_nodes.py`

- [ ] **Step 1: Write failing node tests**

Add tests named `test_extract_claims_node_writes_structured_claims`, `test_verify_citations_node_passes_supported_claims`, and `test_verify_citations_node_collects_unsupported_claims`.

Expected assertions:

```python
assert update["claims"][0]["claim_id"] == "c001"
assert update["claim_verification_results"][0]["verification_label"] == "supported"
assert update["citation_verification_passed"] is True
assert update["unsupported_claims"] == []
assert update["claim_verification"]["verified"] is True
assert update["route"] == "finalize_answer"
```

For unsupported claims:

```python
assert update["citation_verification_passed"] is False
assert update["unsupported_claims"][0]["verification_label"] == "unsupported"
assert update["route"] == "revise_answer"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_nodes.py -q
```

Expected: fail because the new node methods do not exist.

- [ ] **Step 3: Implement extraction and verification nodes**

Add `extract_claims_node()` and `verify_citations_node()` to `AgentNodes`. Use the parser functions from `agent.citation_verification`. Add a helper to compute selected citation chunk ids from `cited_documents`, using `chunk_id` when present and a stable fallback from source/page/index when missing.

- [ ] **Step 4: Run node tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_nodes.py -q
```

Expected: extraction and verification node tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/nodes.py tests/test_agent_nodes.py
git commit -m "feat: add claim extraction and citation verification nodes"
```

### Task 5: Revision And Finalization Nodes

**Files:**
- Modify: `agent/nodes.py`
- Modify: `tests/test_agent_nodes.py`

- [ ] **Step 1: Write failing revision and finalization tests**

Add tests named `test_revise_answer_node_updates_draft_and_increments_revision_count`, `test_finalize_answer_node_promotes_verified_draft_answer`, and `test_finalize_answer_node_allows_verification_skipped_refusal`.

Expected assertions for revision:

```python
assert update["draft_answer"] == "Revised grounded answer [1]."
assert update["used_citation_indices"] == [1]
assert update["citation_revision_count"] == 1
assert update["route"] == "extract_claims"
```

Expected assertions for finalization:

```python
assert update["answer"] == "Verified answer [1]."
assert update["is_verified"] is True
assert update["route"] == "end"
```

For skipped refusal:

```python
assert update["answer"] == "I cannot answer from the current documents."
assert update["is_verified"] is False
assert update["route"] == "end"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_nodes.py -q
```

Expected: fail because revision and finalization nodes do not exist.

- [ ] **Step 3: Implement revision and finalization nodes**

Add `revise_answer_node()` and `finalize_answer_node()`. Revision uses `ANSWER_REVISION_PROMPT`, parses answer-generation-compatible JSON with `_parse_answer_result()`, validates citation markers, rebuilds `citations` and `cited_documents`, increments `citation_revision_count`, and routes to `extract_claims`. Finalization sets public `answer`, compatibility `is_verified`, and route `end`.

- [ ] **Step 4: Run node tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_nodes.py -q
```

Expected: revision and finalization node tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/nodes.py tests/test_agent_nodes.py
git commit -m "feat: add answer revision and finalization nodes"
```

### Task 6: Graph Routing And Result Payload

**Files:**
- Modify: `agent/edges.py`
- Modify: `agent/graph.py`
- Modify: `tests/test_agent_graph.py`

- [ ] **Step 1: Write failing graph tests**

Add graph integration tests for:

- supported claims finalize without revision.
- unsupported first pass revises, second pass finalizes.
- unsupported after revision budget falls back.

The revised-success fake LLM response order should be:

```python
[
    "rewritten query",
    '{"relevant": true, "relevant_indices": [1], "reason": "matches"}',
    '{"answer": "Draft unsupported answer [1].", "used_citation_indices": [1]}',
    '{"claims": [{"claim_id": "c001", "claim": "Unsupported draft", "cited_chunk_ids": ["chunk-1"]}], "reason": "claim"}',
    '{"results": [{"claim_id": "c001", "claim": "Unsupported draft", "cited_chunk_ids": ["chunk-1"], "verification_label": "unsupported", "confidence": 0.2, "reason": "too strong"}], "reason": "unsupported"}',
    '{"answer": "Revised supported answer [1].", "used_citation_indices": [1]}',
    '{"claims": [{"claim_id": "c001", "claim": "Revised supported answer", "cited_chunk_ids": ["chunk-1"]}], "reason": "claim"}',
    '{"results": [{"claim_id": "c001", "claim": "Revised supported answer", "cited_chunk_ids": ["chunk-1"], "verification_label": "supported", "confidence": 0.9, "reason": "supported"}], "reason": "supported"}',
]
```

Assert:

```python
assert result["answer"] == "Revised supported answer [1]."
assert result["citation_revision_count"] == 1
assert result["citation_verification_passed"] is True
assert result["unsupported_claims"] == []
```

- [ ] **Step 2: Run graph tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_graph.py -q
```

Expected: fail because graph nodes and payload fields are missing.

- [ ] **Step 3: Implement graph routing**

Add nodes in `build_graph()`:

```python
workflow.add_node("extract_claims", nodes.extract_claims_node)
workflow.add_node("verify_citations", nodes.verify_citations_node)
workflow.add_node("revise_answer", nodes.revise_answer_node)
workflow.add_node("finalize_answer", nodes.finalize_answer_node)
```

Replace `generate_answer -> END` with conditional routing by `state["route"]`:

```text
generate_answer -> extract_claims / finalize_answer / fallback
extract_claims -> verify_citations / fallback
verify_citations -> finalize_answer / revise_answer / fallback
revise_answer -> extract_claims / fallback
finalize_answer -> END
```

Add result payload fields:

```python
"draft_answer": final_state["draft_answer"],
"used_citation_indices": final_state["used_citation_indices"],
"claim_verification_results": final_state["claim_verification_results"],
"unsupported_claims": final_state["unsupported_claims"],
"citation_verification_passed": final_state["citation_verification_passed"],
"citation_revision_count": final_state["citation_revision_count"],
"citation_verification_skipped": final_state["citation_verification_skipped"],
```

- [ ] **Step 4: Run graph tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_graph.py -q
```

Expected: graph tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/edges.py agent/graph.py tests/test_agent_graph.py
git commit -m "feat: route claim verification workflow in graph"
```

### Task 7: Evaluation Compatibility And Full Test Repair

**Files:**
- Modify tests that expect the old in-node verification behavior.
- Modify evaluation assertions only where result schema changed.

- [ ] **Step 1: Run focused compatibility tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluate.py tests/test_ablation.py tests/test_gradio_app.py -q
```

Expected: failures only where tests assume old claim shape or old verification timing.

- [ ] **Step 2: Update tests to new diagnostics**

Keep public keys stable. Update test fixtures so `claims` use `claim_id`, `claim`, and `cited_chunk_ids` where they model P2 output. Keep legacy fixture support in evaluation tests that intentionally exercise old fields.

- [ ] **Step 3: Run full tests**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests evaluation baseline app.py
git commit -m "test: update citation verification compatibility coverage"
```

Stage only files that changed. If `evaluation`, `baseline`, or `app.py` do not change, do not include them in the commit.

### Task 8: Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/resume_bullets.md`
- Modify: `experiments/report.md`

- [ ] **Step 1: Update docs**

Document P2 as implemented:

- Claim extraction.
- Per-claim citation verification.
- One revision loop.
- Fallback after unsupported claims remain.
- New diagnostics in `run_agent()` output.

Keep P0b artifact regeneration as a future step.

- [ ] **Step 2: Run full tests**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass after documentation changes.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/resume_bullets.md experiments/report.md
git commit -m "docs: document claim-level citation verification milestone"
```

## Self-Review

- The plan implements the approved P2 spec without adding Tool Registry or FastAPI work.
- Every production-code task starts with a failing test.
- Parser logic is isolated from LangGraph node logic.
- Public result keys remain compatible while adding P2 diagnostics.
- Unable-to-answer responses can finalize safely but do not count as verification passes.
- Revision budget defaults to one repair attempt before fallback.
