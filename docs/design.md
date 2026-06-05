# Design Notes

This project is a lightweight Agentic RAG prototype for private document QA. It is designed to make the retrieval control flow explicit enough to inspect, test, and explain in interviews.

## Why It Is Not Plain RAG

Plain RAG usually follows one fixed path:

```text
question -> retrieve -> generate
```

This project keeps retrieval under a LangGraph workflow:

```text
question -> rewrite -> retrieve -> grade -> answer | retry | fallback
```

The workflow can inspect retrieved chunks before answer generation. If the evidence is weak, it rewrites the retrieval query and tries again. If evidence remains insufficient, it returns an unable-to-answer fallback.

The chat model layer is provider-aware but intentionally small. Remote OpenAI-compatible APIs and local Ollama both use the same LangChain `ChatOpenAI` client; Ollama is selected with `LLM_PROVIDER=ollama` and a local `OLLAMA_MODEL`.

## AgentState

`AgentState` is the working memory passed between LangGraph nodes.

Important fields:

- `question`: original user question.
- `current_query`: retrieval-optimized query used only for search.
- `previous_queries`: retrieval queries already attempted.
- `documents`: raw retrieved chunks.
- `relevant_documents`: chunks accepted by chunk-level retrieval grading.
- `grading_reason`: reason returned by the grading step.
- `retry_count`: failed-retrieval rewrites. Initial query normalization does not count as retry.
- `citations`: final selected evidence citations.
- `claims`: factual claims extracted during claim-level verification.
- `is_verified`: whether normal answer claims were verified against selected evidence.
- `fallback_reason`: why the system declined to answer.

The most important semantic boundary is that `current_query` does not replace `question`. The retrieval query may contain extra search terms, but grading and answer generation must still target the original user question.

## Nodes And Edges

`rewrite_query_node` performs initial query normalization on the first pass. On retry, it receives failure context: previous query, previous queries, grading reason, and retrieved snippets. This makes retry rewrite more agentic than simply rephrasing the same question.

`retrieve_node` calls the `retrieve_context` tool using `current_query`.

`grade_documents_node` receives the original user question, the current retrieval query, and raw retrieved chunks. It asks whether the chunks can answer the original question, then returns `relevant_indices`.

`generate_answer_node` receives the original user question, the current retrieval query, and `relevant_documents`. It answers the original question only.

The conditional edge routes based on evidence:

- `relevant_documents` non-empty -> `generate_answer`
- no relevant documents and `retry_count < max_retry_count` -> `rewrite_query`
- no relevant documents and retry budget exhausted -> `fallback`

## Citation Safety And Claim Verification

The implementation combines citation-aware generation with lightweight claim-level verification:

1. The LLM returns JSON with `answer` and `used_citation_indices`.
2. The program maps those indices back to `relevant_documents`.
3. Normal answers without valid supporting citation indices are rejected and converted to fallback.
4. Explicit unable-to-answer responses may have empty citations.
5. Normal cited answers are passed to a claim verifier.
6. The claim verifier splits the answer into factual claims and checks each claim against selected citation chunks.
7. If any important claim is unsupported, the system falls back instead of returning the answer.

This reduces unsupported answers, but the verification is still LLM-based. It is a practical prototype mechanism, not a formal proof system.

## Evaluation

Evaluation compares two flows:

- Naive RAG: `question -> retrieve -> generate`
- Agentic RAG: `question -> rewrite -> retrieve -> grade -> retry or answer`

The evaluation set includes direct questions, questions that benefit from rewrite, and questions that should not be answerable from the sample documents.

Metrics include source hit rate, keyword hit rate, citation rate, claim verification rate, fallback correctness, latency, retry count, retrieved document count, relevant document count, and relevant filtering rate.

The evaluation is intentionally lightweight. It supports project explanation and regression checks, but it is not a rigorous benchmark.

## LLM Provider Boundary

The project separates chat LLM configuration from the Agentic RAG graph:

- `LLM_PROVIDER=openai_compatible` uses `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL`.
- `LLM_PROVIDER=ollama` uses `OLLAMA_BASE_URL` and `OLLAMA_MODEL`, with no remote API key required.

Both modes create the same chat-model interface for query rewriting, retrieval grading, answer generation, and claim verification. This keeps the agent workflow testable without coupling nodes to a specific vendor.

## Future Work

- Stricter deterministic citation validation and human-reviewed claim labels.
- Reranking before grading.
- Deterministic chunk IDs for incremental indexing.
- Larger evaluation set with human-reviewed expected answers.
- Model-specific prompt tuning for smaller local Ollama models.
