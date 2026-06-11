# P0b Real Ablation Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real cumulative V0-V6 feature ablations, correct history and citation-verification metrics, then generate reproducible DeepSeek baseline, Agentic RAG, and ablation artifacts.

**Architecture:** Introduce immutable Agent feature flags and compose one LangGraph from those flags. Build typed experiment variants that inject both feature flags and per-variant retriever settings, standardize evaluation runners on `(question, chat_history)`, and checkpoint each variant before deriving V0/V6 comparison artifacts and reports.

**Tech Stack:** Python dataclasses and TypedDicts, LangGraph `StateGraph`, Chroma, BM25/RRF hybrid retrieval, optional cross-encoder reranking, pytest, JSON and simple YAML-compatible config files.

---

## File Map

- Create `agent/features.py`: immutable graph capability flags and sanitized snapshots.
- Modify `agent/nodes.py`: add deterministic acceptance of retrieved documents when grading is disabled.
- Modify `agent/edges.py`: make retry and citation-verification routing feature-aware.
- Modify `agent/graph.py`: compose graph topology from feature flags and expose effective flags.
- Modify `baseline/naive_rag.py`: accept but intentionally ignore chat history and expose verification applicability.
- Modify `baseline/run_baseline.py`: use the two-argument evaluation runner contract.
- Modify `evaluation/evaluate.py`: pass chat history, aggregate verifier results, and represent unavailable metrics.
- Modify `evaluation/runtime_config.py`: include sanitized Agent feature flags.
- Create `experiments/variants.py`: parse, validate, and instantiate cumulative ablation variants.
- Rewrite `experiments/run_ablation.py`: run distinct variants, checkpoint artifacts, and derive V0/V6 comparison outputs.
- Replace configs in `experiments/configs/`: define real V0-V6 cumulative settings.
- Modify `experiments/report.md`: document generated P0b results and limitations.
- Modify `README.md`: update evaluation and reproduction commands.
- Modify tests in `tests/test_agent_edges.py`, `tests/test_agent_graph.py`, `tests/test_agent_nodes.py`, `tests/test_baseline.py`, `tests/test_evaluate.py`, and `tests/test_ablation.py`.

### Task 1: Typed Agent Feature Flags

**Files:**
- Create: `agent/features.py`
- Modify: `agent/edges.py`
- Test: `tests/test_agent_edges.py`

- [ ] **Step 1: Write failing feature and routing tests**

Add tests that define the expected defaults and retry-disabled behavior:

```python
from agent.features import AgentFeatureFlags
from agent.edges import route_after_grading


def test_agent_feature_flags_default_to_complete_workflow():
    flags = AgentFeatureFlags()

    assert flags.query_transformation_enabled is True
    assert flags.retrieval_grading_enabled is True
    assert flags.conditional_retry_enabled is True
    assert flags.citation_verification_enabled is True
    assert flags.to_dict() == {
        "query_transformation_enabled": True,
        "retrieval_grading_enabled": True,
        "conditional_retry_enabled": True,
        "citation_verification_enabled": True,
    }


def test_route_after_grading_falls_back_when_retry_is_disabled():
    state = {
        "relevant_documents": [],
        "retry_count": 0,
        "max_retry_count": 2,
    }

    route = route_after_grading(
        state,
        features=AgentFeatureFlags(conditional_retry_enabled=False),
    )

    assert route == "fallback"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_edges.py -q
```

Expected: fail because `agent.features` and the `features` routing argument do not exist.

- [ ] **Step 3: Implement immutable feature flags**

Create `agent/features.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AgentFeatureFlags:
    query_transformation_enabled: bool = True
    retrieval_grading_enabled: bool = True
    conditional_retry_enabled: bool = True
    citation_verification_enabled: bool = True

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)
```

Extend `route_after_grading()` with `features: AgentFeatureFlags | None = None`.
Return `fallback` immediately when no relevant documents exist and
`conditional_retry_enabled` is false. Preserve all existing behavior when
`features` is omitted.

- [ ] **Step 4: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_edges.py -q
```

Expected: all edge tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/features.py agent/edges.py tests/test_agent_edges.py
git commit -m "feat: add agent workflow feature flags"
```

### Task 2: Feature-Aware Graph Composition

**Files:**
- Modify: `agent/nodes.py`
- Modify: `agent/edges.py`
- Modify: `agent/graph.py`
- Test: `tests/test_agent_nodes.py`
- Test: `tests/test_agent_graph.py`

- [ ] **Step 1: Write failing deterministic document-acceptance test**

Add:

```python
def test_accept_retrieved_documents_node_prepares_generation_context():
    nodes = AgentNodes(llm=FakeLLM([]), retriever_fn=lambda query: [])
    state = create_initial_state("What is RAG?")
    state["documents"] = [
        {"content": "RAG retrieves evidence.", "source": "notes.md", "chunk_id": "c1"}
    ]

    update = nodes.accept_retrieved_documents_node(state)

    assert update["relevant_documents"] == state["documents"]
    assert update["relevant_document_count"] == 1
    assert update["is_relevant"] is True
    assert update["grading_reason"] == "Retrieval grading disabled."
    assert update["route"] == "generate_answer"
```

- [ ] **Step 2: Run the node test and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_nodes.py::test_accept_retrieved_documents_node_prepares_generation_context -q
```

Expected: fail because `accept_retrieved_documents_node` is missing.

- [ ] **Step 3: Implement deterministic acceptance**

Add `accept_retrieved_documents_node()` to `AgentNodes`. It must copy
`documents` to `relevant_documents`, set counts and grading diagnostics, and
route to `generate_answer`. With no documents, it must route to `fallback`
without making an LLM call.

- [ ] **Step 4: Write failing graph-path tests**

Add tests with `AgentFeatureFlags`:

```python
def test_run_agent_skips_query_transformation_and_grading_when_disabled():
    flags = AgentFeatureFlags(
        query_transformation_enabled=False,
        retrieval_grading_enabled=False,
        conditional_retry_enabled=False,
        citation_verification_enabled=False,
    )
    llm = FakeLLM([
        '{"answer": "RAG retrieves evidence [1].", "used_citation_indices": [1]}'
    ])
    queries = []

    result = run_agent(
        "What is RAG?",
        llm=llm,
        retriever_fn=lambda query: (
            queries.append(query)
            or [{"content": "RAG retrieves evidence.", "source": "notes.md", "chunk_id": "c1"}]
        ),
        settings=get_settings(),
        features=flags,
    )

    assert queries == ["What is RAG?"]
    assert len(llm.prompts) == 1
    assert result["answer"] == "RAG retrieves evidence [1]."
    assert result["query_transform"] == {}
    assert result["document_grades"] == []
    assert result["citation_verification_enabled"] is False
```

Also add:

```python
def test_run_agent_grades_but_does_not_retry_when_retry_feature_is_disabled():
    flags = AgentFeatureFlags(
        conditional_retry_enabled=False,
        citation_verification_enabled=False,
    )
    llm = FakeLLM([
        "standalone query",
        '{"relevant": false, "relevant_indices": [], "reason": "no evidence"}',
    ])

    result = run_agent(
        "Question?",
        llm=llm,
        retriever_fn=lambda query: [{"content": "Unrelated", "source": "x.md"}],
        settings=get_settings(),
        features=flags,
    )

    assert result["retry_count"] == 0
    assert result["fallback_reason"]
    assert len(llm.prompts) == 2
```

- [ ] **Step 5: Run graph tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_graph.py -q
```

Expected: fail because `run_agent()` and `build_graph()` do not accept feature flags.

- [ ] **Step 6: Compose graph topology from flags**

Update `build_graph(..., features: AgentFeatureFlags | None = None)`:

- start at `rewrite_query` only when query transformation is enabled
- otherwise start at `retrieve`
- route `retrieve -> grade_documents` when grading is enabled
- otherwise route `retrieve -> accept_documents -> generate_answer`
- pass flags into `route_after_grading`
- when citation verification is disabled, route a valid generated answer to
  `finalize_answer`
- otherwise preserve the P2 extraction, verification, and revision workflow

Initialize `current_query`, `rewritten_question`, and `standalone_question` with
the original question in `run_agent()` when transformation is disabled. Add
`features` to `run_agent()` and return:

```python
"feature_flags": resolved_features.to_dict(),
"citation_verification_enabled": resolved_features.citation_verification_enabled,
```

Make the answer-generation conditional route feature-aware without changing
the node's default P2 route.

- [ ] **Step 7: Run graph and node tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_nodes.py tests/test_agent_edges.py tests/test_agent_graph.py -q
```

Expected: all focused tests pass, including existing full-workflow tests.

- [ ] **Step 8: Commit**

```bash
git add agent/nodes.py agent/edges.py agent/graph.py tests/test_agent_nodes.py tests/test_agent_graph.py
git commit -m "feat: compose agent graph from feature flags"
```

### Task 3: History-Aware Evaluation Runner Contract

**Files:**
- Modify: `baseline/naive_rag.py`
- Modify: `baseline/run_baseline.py`
- Modify: `agent/graph.py`
- Modify: `evaluation/evaluate.py`
- Test: `tests/test_baseline.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write failing history propagation tests**

Add:

```python
def test_evaluate_questions_passes_chat_history_to_runner():
    calls = []

    def runner(question, chat_history):
        calls.append((question, chat_history))
        return {
            "answer": "Grounded [1].",
            "citations": [{"source": "notes.md"}],
            "retrieved_documents": [{"source": "notes.md"}],
            "relevant_documents": [{"source": "notes.md"}],
        }

    evaluate_questions(
        [{
            "question": "How does it help?",
            "chat_history": [{"role": "user", "content": "Discuss grading."}],
            "expected_sources": ["notes.md"],
        }],
        run_agent_fn=runner,
    )

    assert calls == [(
        "How does it help?",
        [{"role": "user", "content": "Discuss grading."}],
    )]
```

Add a baseline test that passes history and asserts the generated prompt still
uses only the explicit question.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluate.py::test_evaluate_questions_passes_chat_history_to_runner tests/test_baseline.py -q
```

Expected: fail because evaluators call one-argument runners and the baseline
does not accept history.

- [ ] **Step 3: Standardize the runner signature**

Define:

```python
EvaluationRunner = Callable[[str, list[ChatMessage]], dict[str, Any]]
```

Pass normalized `item["chat_history"]` from `_evaluate_single_system()` to every
runner. Update comparison mode and test fakes to use two arguments.

Change the baseline signature to:

```python
def run_naive_rag(
    question: str,
    chat_history: list[ChatMessage] | None = None,
    retriever_fn: RetrieverFn | None = None,
    llm: Any | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
```

Do not use `chat_history` in retrieval or generation. Add
`"chat_history_used": False` and `"citation_verification_enabled": False` to the
payload. Add `"chat_history_used": bool(chat_history)` to `run_agent()` output.
Each per-question evaluation result must also retain:

```python
"question_id": item["id"]
"question_type": item["question_type"]
"chat_history_supplied": bool(item["chat_history"])
```

These fields support stable result pairing and follow-up diagnostics.

- [ ] **Step 4: Run evaluation and baseline tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_baseline.py tests/test_baselines.py tests/test_evaluate.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/graph.py baseline/naive_rag.py baseline/run_baseline.py evaluation/evaluate.py tests/test_baseline.py tests/test_baselines.py tests/test_evaluate.py
git commit -m "fix: evaluate follow-up questions with chat history"
```

### Task 4: Citation Verification Metric Corrections

**Files:**
- Modify: `evaluation/evaluate.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write failing verifier-metric tests**

Add:

```python
def test_evaluation_counts_labels_from_verification_results():
    def runner(question, chat_history):
        return {
            "answer": "Two claims [1].",
            "citations": [{"source": "notes.md"}],
            "retrieved_documents": [{"source": "notes.md"}],
            "relevant_documents": [{"source": "notes.md"}],
            "claims": [{"claim_id": "c1"}, {"claim_id": "c2"}],
            "claim_verification_results": [
                {"claim_id": "c1", "verification_label": "supported"},
                {"claim_id": "c2", "verification_label": "unsupported"},
            ],
            "citation_verification_enabled": True,
            "citation_verification_passed": False,
        }

    report = evaluate_questions(
        [{"question": "Q?", "expected_sources": ["notes.md"]}],
        run_agent_fn=runner,
    )

    assert report["summary"]["unsupported_claim_count"] == 1
    assert report["summary"]["supported_claim_ratio"] == 0.5
    assert report["summary"]["citation_verification_pass_rate"] == 0.0
```

Add a citation-disabled test:

```python
assert report["summary"]["unsupported_claim_count"] is None
assert report["summary"]["supported_claim_ratio"] is None
assert report["summary"]["citation_verification_pass_rate"] is None
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluate.py -q
```

Expected: verifier-result and unavailable-metric assertions fail.

- [ ] **Step 3: Aggregate explicit verification results**

In `_build_success_result()`, read and validate
`claim_verification_results`. Count labels from those records, not from
`claims`. Store `citation_verification_applicable` from the explicit
`citation_verification_enabled` result field.

In `_summarize()`, aggregate claim metrics only across applicable results. When
there are no applicable results, return JSON `null` (`None`) for:

```python
"unsupported_claim_count"
"supported_claim_ratio"
"citation_verification_pass_rate"
```

Keep `claim_count` based on extracted claims and keep citation hit metrics
independent of verification applicability.

- [ ] **Step 4: Run evaluation tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluate.py -q
```

Expected: all evaluation tests pass.

- [ ] **Step 5: Commit**

```bash
git add evaluation/evaluate.py tests/test_evaluate.py
git commit -m "fix: aggregate claim verification metrics correctly"
```

### Task 5: Typed V0-V6 Ablation Variants

**Files:**
- Create: `experiments/variants.py`
- Replace: `experiments/configs/v0_naive.yaml`
- Replace: `experiments/configs/v1_query_rewrite.yaml`
- Replace: `experiments/configs/v2_retrieval_grading.yaml`
- Replace: `experiments/configs/v3_retry_fallback.yaml`
- Create: `experiments/configs/v4_hybrid_retrieval.yaml`
- Replace: `experiments/configs/v4_reranker.yaml` with `experiments/configs/v5_reranker.yaml`
- Replace: `experiments/configs/v5_citation_verification.yaml` with `experiments/configs/v6_citation_verification.yaml`
- Modify: `evaluation/runtime_config.py`
- Test: `tests/test_ablation.py`

- [ ] **Step 1: Write failing variant parsing and validation tests**

Add tests that load the repository configs and assert:

```python
variants = load_ablation_variants(CONFIG_DIR)

assert [variant.id for variant in variants] == [
    "v0_naive",
    "v1_query_rewrite",
    "v2_retrieval_grading",
    "v3_retry_fallback",
    "v4_hybrid_retrieval",
    "v5_reranker",
    "v6_citation_verification",
]
assert variants[0].runner == "naive"
assert variants[3].features.conditional_retry_enabled is True
assert variants[4].settings_overrides == {"hybrid_retrieval_enabled": True}
assert variants[5].settings_overrides == {
    "hybrid_retrieval_enabled": True,
    "reranker_enabled": True,
}
assert variants[6].features.citation_verification_enabled is True
validate_cumulative_variants(variants)
```

Add a duplicate-effective-config test expecting `ValueError`.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_ablation.py -q
```

Expected: fail because typed variants and new configs do not exist.

- [ ] **Step 3: Implement `AblationVariant`**

Create a frozen dataclass:

```python
@dataclass(frozen=True)
class AblationVariant:
    id: str
    method: str
    runner: Literal["naive", "agentic"]
    features: AgentFeatureFlags
    settings_overrides: dict[str, bool]

    def apply_settings(self, base: Settings) -> Settings:
        return replace(base, **self.settings_overrides)

    def effective_signature(self) -> tuple[object, ...]:
        return (
            self.runner,
            *self.features.to_dict().values(),
            self.settings_overrides.get("hybrid_retrieval_enabled", False),
            self.settings_overrides.get("reranker_enabled", False),
        )
```

Parse strict `true`/`false` values from the simple config files. Reject unknown
keys, missing IDs, non-cumulative feature removal, and duplicate effective
signatures.

- [ ] **Step 4: Define cumulative configs**

Each config explicitly contains:

```yaml
id: v4_hybrid_retrieval
method: + Hybrid Retrieval
runner: agentic
query_transformation_enabled: true
retrieval_grading_enabled: true
conditional_retry_enabled: true
hybrid_retrieval_enabled: true
reranker_enabled: false
citation_verification_enabled: false
```

V0 disables all Agent features and retrieval extensions. Each following file
adds exactly the capability named by its row. V6 enables all capabilities.

- [ ] **Step 5: Include feature flags in runtime snapshots**

Extend `build_runtime_config_snapshot(settings, features=None)` with:

```python
"agent_features": (features or AgentFeatureFlags()).to_dict()
```

Do not add credentials, base URLs, or local persistence paths.

- [ ] **Step 6: Run ablation and runtime snapshot tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_ablation.py tests/test_evaluate.py -q
```

Expected: typed variant and snapshot tests pass.

- [ ] **Step 7: Commit**

```bash
git add agent/features.py evaluation/runtime_config.py experiments/variants.py experiments/configs tests/test_ablation.py tests/test_evaluate.py
git commit -m "feat: define cumulative p0b ablation variants"
```

### Task 6: Variant Runner And Retriever Injection

**Files:**
- Modify: `experiments/variants.py`
- Modify: `experiments/run_ablation.py`
- Test: `tests/test_ablation.py`

- [ ] **Step 1: Write failing runner-factory tests**

Use fake factories to assert that variant settings reach the retriever and graph:

```python
def test_create_variant_runner_injects_variant_settings_and_features():
    captured = {}

    def retriever_factory(settings):
        captured["retriever_settings"] = settings
        return lambda query: []

    def agent_runner(question, chat_history, *, settings, features, retriever_fn):
        captured.update(
            settings=settings,
            features=features,
            retriever_fn=retriever_fn,
            history=chat_history,
        )
        return {"answer": ""}

    runner = create_variant_runner(
        variant,
        base_settings=get_settings(),
        retriever_factory=retriever_factory,
        agent_runner=agent_runner,
    )
    runner("Question?", [{"role": "user", "content": "Context"}])

    assert captured["settings"].hybrid_retrieval_enabled is True
    assert captured["features"] == variant.features
    assert captured["retriever_settings"] == captured["settings"]
```

- [ ] **Step 2: Run the test and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_ablation.py -q
```

Expected: fail because `create_variant_runner` is missing.

- [ ] **Step 3: Implement settings-aware runner construction**

For each variant:

```python
resolved_settings = variant.apply_settings(base_settings)
retriever = Retriever(settings=resolved_settings).retrieve
```

The naive closure calls:

```python
run_naive_rag(
    question,
    chat_history=chat_history,
    settings=resolved_settings,
    retriever_fn=retriever,
)
```

The Agentic closure calls:

```python
run_agent(
    question,
    chat_history=chat_history,
    settings=resolved_settings,
    features=variant.features,
    retriever_fn=retriever,
)
```

This explicit injection is required; changing `Settings` without constructing a
matching `Retriever` would leave module-level retrieval on the default config.

- [ ] **Step 4: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_ablation.py tests/test_retriever.py -q
```

Expected: runner injection and existing retriever tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/variants.py experiments/run_ablation.py tests/test_ablation.py
git commit -m "feat: inject variant retrieval settings into ablations"
```

### Task 7: Recoverable Ablation Artifacts And V0/V6 Derivation

**Files:**
- Modify: `experiments/run_ablation.py`
- Test: `tests/test_ablation.py`

- [ ] **Step 1: Write failing artifact lifecycle tests**

Test a three-variant temporary matrix with fake runners and assert:

```python
assert (output_dir / "variants" / "v0_naive.json").exists()
assert (output_dir / "variants" / "v6_citation_verification.json").exists()
assert payload["runs"][0]["status"] == "completed"
assert payload["runs"][1]["status"] == "completed_with_errors"
assert comparison["summary"]["mode"] == "comparison"
assert baseline["results"] == payload["runs"][0]["results"]
assert agentic["results"] == payload["runs"][-1]["results"]
assert calls == expected_question_count_per_variant
```

The last assertion proves comparison generation does not rerun V0 or V6.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_ablation.py -q
```

Expected: fail because per-variant artifacts, statuses, and derived comparison
outputs are missing.

- [ ] **Step 3: Add atomic JSON writes and statuses**

Implement:

```python
def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)
```

Before executing a variant, write an `incomplete` checkpoint. After all
questions are attempted, overwrite it with `completed` when `error_count == 0`
or `completed_with_errors` otherwise.

- [ ] **Step 4: Derive canonical artifacts**

Build `baseline_result.json` from V0 and `agentic_result.json` from V6. Pair
their already-computed per-question records by `question_id` to produce
`comparison_result.json`. Do not invoke either runner again.

Write `ablation_result.json` only after all configured variants have final
statuses. Include dataset metadata, runtime snapshots, feature flags, summaries,
results, and status.

- [ ] **Step 5: Add smoke filtering**

Add CLI option:

```text
--question-ids q001,q016,q027,q030,q033
```

Normalize comma-separated IDs, reject unknown IDs, and preserve dataset order.
This supports a representative smoke run before the full matrix.

- [ ] **Step 6: Run artifact tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_ablation.py -q
```

Expected: all artifact, no-rerun, status, and filtering tests pass.

- [ ] **Step 7: Commit**

```bash
git add experiments/run_ablation.py tests/test_ablation.py
git commit -m "feat: checkpoint p0b ablation artifacts"
```

### Task 8: Generated Report And Documentation

**Files:**
- Modify: `experiments/run_ablation.py`
- Modify: `experiments/report.md`
- Modify: `README.md`
- Test: `tests/test_ablation.py`

- [ ] **Step 1: Write failing report-format tests**

Assert the generated Markdown contains:

```python
assert "| V0 Naive RAG |" in report
assert "| V6 + Claim-level Citation Verification |" in report
assert "completed_with_errors" in report
assert "## Observed Trade-offs" in report
assert "## Limitations" in report
assert "N/A" in report
```

Use synthetic summaries where one metric improves and latency rises. Assert the
trade-off text names only those observed changes.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_ablation.py -q
```

Expected: fail because the current report is P0a-specific.

- [ ] **Step 3: Implement P0b report generation**

Generate a table with:

```text
Method
Correctness
Context Relevance
Citation Accuracy
Fallback Accuracy
Unsupported Claims
Supported Claim Ratio
Avg Retry
Avg Latency
Errors
Status
```

Render `None` as `N/A`. Derive trade-off sentences from adjacent completed
variants only. Do not make causal claims from failed or unavailable metrics.

- [ ] **Step 4: Update durable docs**

Update `README.md` and `experiments/report.md` with:

- the V0-V6 cumulative matrix
- exact smoke and full commands
- deterministic metric limitations
- Approach B typed evaluation framework in the roadmap
- interactive evaluation dashboard in the roadmap

Do not call the project autonomous or production-ready.

- [ ] **Step 5: Run report tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_ablation.py -q
```

Expected: all report tests pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/run_ablation.py experiments/report.md README.md tests/test_ablation.py
git commit -m "docs: add reproducible p0b ablation reporting"
```

### Task 9: Preflight Verification

**Files:**
- Modify only if tests identify a P0b regression.

- [ ] **Step 1: Run focused P0b tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_agent_edges.py \
  tests/test_agent_graph.py \
  tests/test_agent_nodes.py \
  tests/test_baseline.py \
  tests/test_baselines.py \
  tests/test_evaluate.py \
  tests/test_ablation.py \
  tests/test_retriever.py \
  tests/test_hybrid_retriever.py \
  tests/test_reranker.py -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the complete test suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass with no network calls.

- [ ] **Step 3: Validate local model configuration without exposing secrets**

Run:

```bash
.venv/bin/python -c "from config import get_settings; s=get_settings(); print({'provider': s.llm_provider, 'model': s.effective_llm_model, 'configured': s.has_llm_config})"
```

Expected: `configured` is `True`; output must not include an API key.

- [ ] **Step 4: Commit any test-only correction**

If preflight required a code correction, stage only its scoped files and commit:

```bash
git commit -m "fix: stabilize p0b evaluation preflight"
```

If no correction was needed, do not create an empty commit.

### Task 10: Rebuild Index And Run DeepSeek Experiments

**Files:**
- Generate: `experiments/results/variants/*.json`
- Generate: `experiments/results/baseline_result.json`
- Generate: `experiments/results/agentic_result.json`
- Generate: `experiments/results/comparison_result.json`
- Generate: `experiments/results/ablation_result.json`
- Generate: `experiments/results/ablation_report.md`
- Modify: `experiments/report.md`

- [ ] **Step 1: Rebuild the sample index**

Run:

```bash
.venv/bin/python -c "from pathlib import Path; from rag.loader import load_documents; from rag.chunker import split_documents; from rag.vectorstore import create_vectorstore; paths=sorted(p for p in Path('sample_docs').iterdir() if p.is_file()); docs=load_documents(paths); chunks=split_documents(docs); create_vectorstore(chunks); print({'documents': len(docs), 'chunks': len(chunks)})"
```

Expected: all five sample documents are loaded and at least one chunk is indexed
per document.

- [ ] **Step 2: Run representative smoke evaluation**

Run:

```bash
.venv/bin/python -m experiments.run_ablation \
  --questions evaluation/eval_questions.json \
  --config-dir experiments/configs \
  --output-dir experiments/results/smoke \
  --question-ids q001,q016,q027,q030,q033
```

Expected:

- seven variant files are produced
- every variant is `completed`
- each summary has `total_questions: 5`
- no artifact contains an API key

- [ ] **Step 3: Inspect smoke errors before full execution**

Run:

```bash
.venv/bin/python -c "import json; p=json.load(open('experiments/results/smoke/ablation_result.json')); print([(r['id'], r['status'], r['summary']['error_count']) for r in p['runs']])"
```

Expected: every tuple has status `completed` and error count `0`. Stop and debug
before the full run if this condition is not met.

- [ ] **Step 4: Run the full 36-question V0-V6 matrix**

Run:

```bash
.venv/bin/python -m experiments.run_ablation \
  --questions evaluation/eval_questions.json \
  --config-dir experiments/configs \
  --output-dir experiments/results \
  --report experiments/results/ablation_report.md
```

Expected: seven completed variant artifacts and canonical V0/V6 comparison
artifacts are generated.

- [ ] **Step 5: Validate generated artifacts**

Run:

```bash
.venv/bin/python -c "import json; p=json.load(open('experiments/results/ablation_result.json')); assert len(p['runs']) == 7; assert all(r['status'] == 'completed' for r in p['runs']); assert all(r['summary']['total_questions'] == 36 for r in p['runs']); print([(r['id'], r['summary']['correctness_score'], r['summary']['average_latency']) for r in p['runs']])"
```

Expected: validation succeeds and prints seven result rows.

- [ ] **Step 6: Refresh the durable experiment report**

Copy the generated metrics and evidence-based trade-off analysis into
`experiments/report.md`. Include the exact model name and sanitized retrieval
configuration from the artifacts. Mark token usage or cost as unavailable when
the response metadata did not provide it.

- [ ] **Step 7: Run final tests**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: the complete suite passes after report generation.

- [ ] **Step 8: Commit P0b results**

Stage the durable report and intended experiment artifacts only:

```bash
git add experiments/report.md
git add -f \
  experiments/results/variants \
  experiments/results/baseline_result.json \
  experiments/results/agentic_result.json \
  experiments/results/comparison_result.json \
  experiments/results/ablation_result.json \
  experiments/results/ablation_report.md
git commit -m "experiments: publish p0b ablation results"
```

Before committing, inspect staged files and verify that no credential or local
secret is present.
