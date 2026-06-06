# Resume Project Presentation Design

## Objective

Improve the project's portfolio and interview presentation without adding unrelated backend features. The work should make the existing Agentic RAG behavior easier to demonstrate, evaluate, reproduce, and discuss.

DeepSeek will be used for the formal benchmark because its structured-output quality is currently more stable than the configured local Ollama model. Ollama remains a supported local option but is not the primary source for published evaluation results.

## Deliverables

1. `docs/evaluation.md`
   - Describe the evaluation dataset and execution environment.
   - Compare Naive RAG, Agentic RAG without reranking, and Agentic RAG with reranking.
   - Report actual metrics and latency without inventing results.
   - Include representative success and failure cases.
   - Explain the tradeoffs introduced by retrieval grading, retries, claim verification, and reranking.

2. `docs/demo.md`
   - Provide reproducible setup and indexing commands.
   - List demonstration questions grouped by expected behavior.
   - Include direct answer, contextual follow-up, retry rewrite, citation verification, and unable-to-answer scenarios.
   - Explain which UI diagnostics the reviewer should inspect.

3. Formal architecture diagram
   - Replace the outdated `assets/architecture.png`.
   - Show document ingestion, deterministic indexing, retrieval, optional reranking, LangGraph nodes, citation consistency, claim verification, and fallback.
   - Include remote OpenAI-compatible/DeepSeek and local Ollama provider boundaries.
   - Avoid overcrowding and ensure all text is readable at README width.

4. More realistic sample corpus
   - Add fictional, non-sensitive documents representing an employee handbook, product specification, and security policy.
   - Keep facts unambiguous enough for expected-source and expected-keyword evaluation.
   - Include overlapping vocabulary across documents so reranking and grading have meaningful work to do.

5. Expanded evaluation matrix
   - Add direct factual questions.
   - Add vague or contextual questions that benefit from rewriting.
   - Add cross-document questions where appropriate.
   - Add unanswerable questions to measure fallback correctness.
   - Preserve expected sources, expected keywords, `should_answer`, and `requires_rewrite`.

6. README interview presentation
   - Link to the evaluation and demo documents.
   - Add concise interview talking points.
   - Explain key design distinctions and tradeoffs without production-readiness claims.

7. Engineering presentation
   - Add `pyproject.toml` configuration for pytest and Ruff.
   - Add a GitHub Actions workflow that installs dependencies and runs offline tests.
   - CI tests must not require an API key, model download, vector index, or network LLM call.

## Evaluation Design

The formal benchmark will compare:

| Variant | Rewrite | Grading | Retry | Claim Verification | Reranker |
|---|---|---|---|---|---|
| Naive RAG | No | No | No | No | Off |
| Agentic RAG | Yes | Yes | Yes | Yes | Off |
| Agentic RAG + Reranker | Yes | Yes | Yes | Yes | On |

The runner should use the same indexed corpus and question set for every variant. Reranker configuration must be applied explicitly rather than relying on ambient environment state.

Primary metrics:

- source hit rate
- keyword hit rate
- citation rate
- claim verification rate
- fallback correctness rate
- average latency
- average retrieved document count
- average relevant document count
- rewrite/retry count

The report must state that the dataset is small and project-specific. Results demonstrate behavior and regression trends, not general RAG superiority.

## Data And Execution Flow

1. Load and chunk all sample documents.
2. Rebuild one deterministic Chroma collection.
3. Run Naive RAG with reranking disabled.
4. Run Agentic RAG with reranking disabled.
5. Run Agentic RAG with reranking enabled.
6. Save or transcribe the actual report into `docs/evaluation.md`.
7. Record configuration, model name, date, question count, and limitations.

The evaluation runner may gain a controlled matrix mode or supporting functions. It should not duplicate the Agentic graph or introduce a second retrieval architecture.

## Error Handling

- Missing DeepSeek configuration should produce a clear error before a formal benchmark begins.
- Per-question runtime failures should remain visible in evaluation output.
- Reranker model loading failures should identify the configured model and avoid silently reporting a reranker-enabled result.
- Published metrics must not include fabricated replacements for failed runs.

## Testing

Tests will use fake runners, fake rerankers, and temporary documents.

Required coverage:

- evaluation matrix invokes all three variants with isolated settings
- reranker-off and reranker-on results remain separately labeled
- report formatting includes all variants and handles unavailable metrics
- expanded evaluation questions validate successfully
- sample documents load and retain expected metadata
- README/demo/evaluation links reference existing files
- pytest and Ruff configuration can be used without an API key

## Implementation Phases

1. Evaluation foundation
   - Add realistic sample documents.
   - Expand the evaluation set.
   - Add three-variant evaluation support and tests.

2. Real DeepSeek benchmark
   - Rebuild the sample index.
   - Run the complete evaluation matrix.
   - Write `docs/evaluation.md` from actual output.

3. Demonstration material
   - Create `docs/demo.md`.
   - Add README interview talking points and document links.

4. Architecture presentation
   - Replace and visually verify `assets/architecture.png`.

5. Engineering polish
   - Add `pyproject.toml`.
   - Add Ruff configuration.
   - Add GitHub Actions for offline tests.

Each phase should have focused tests and a separate Git commit where practical.

## Non-Goals

- FastAPI, authentication, databases, Docker, and deployment infrastructure
- claiming production readiness
- claiming that the small evaluation proves universal superiority
- full deterministic claim-level factual verification
- replacing LangGraph or the existing Gradio application
