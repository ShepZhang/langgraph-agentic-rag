# P1f Partial-Relevance Recovery Design

## Goal

Add a lightweight recovery path for cases where retrieval grading finds partially relevant evidence but no directly relevant chunks. The system should treat these cases differently from fully irrelevant retrieval failures by recording a recovery decision and guiding the next retry rewrite with the partially relevant evidence.

## Scope

Implemented behavior:

- Detect the condition `relevant_document_count == 0` and `partial_document_count > 0`.
- Record a structured `partial_relevance_recovery` object in Agent state and final result payload.
- Include partial-evidence context in retry query rewriting so the next query targets missing evidence instead of blindly rephrasing the original question.
- Preserve existing LangGraph nodes and conditional edges.
- Preserve max-retry and fallback semantics.

Deferred behavior:

- Do not dynamically increase retriever `top_k` in this milestone.
- Do not add a separate LangGraph recovery node.
- Do not rerun reranking with a different threshold.
- Do not execute decomposition sub-questions as separate retrieval hops.

## Recovery Schema

`AgentState.partial_relevance_recovery` uses this shape:

```json
{
  "triggered": true,
  "action": "query_refinement",
  "reason": "Only partially relevant chunks were found; refine query to target missing evidence.",
  "partial_document_indices": [1, 3]
}
```

When recovery is not active, the value is:

```json
{
  "triggered": false,
  "action": "none",
  "reason": "",
  "partial_document_indices": []
}
```

## Workflow

The existing workflow remains:

```text
rewrite_query -> retrieve -> grade_documents -> rewrite_query/generate_answer/fallback
```

The behavior changes inside existing nodes:

1. `grade_documents_node` parses structured grades.
2. If no chunks are `relevant` but some chunks are `partially_relevant`, it writes `partial_relevance_recovery.triggered=true`.
3. `route_after_grading` keeps routing to `rewrite_query` while retry budget remains.
4. `rewrite_query_node` includes the recovery reason and partial chunks in the retry rewrite prompt.
5. If retries are exhausted, the graph still routes to fallback.

## Prompt Behavior

`RETRY_QUERY_REWRITE_PROMPT` should tell the LLM when partial evidence exists:

- The retrieved chunks are related but insufficient.
- The rewrite should target the missing facts, entities, comparisons, or constraints.
- The rewrite should not broaden into unrelated topics.

When no partial evidence exists, the prompt should keep the existing generic failed-retrieval behavior.

## State Additions

`AgentState` gains:

- `partial_relevance_recovery`

The public `run_agent()` result includes the same field so evaluation, traces, and future failed-case analysis can inspect whether recovery was triggered.

## Compatibility

Existing tests and fake LLM flows should continue to work. Legacy grading responses produce no partial recovery because they only expose relevant or irrelevant indices.

## Testing

Tests cover:

- Initial state contains inactive recovery metadata.
- Structured grading with only `partially_relevant` chunks triggers recovery.
- Structured grading with a `relevant` chunk does not trigger recovery.
- Retry rewrite prompt includes recovery metadata and partial chunk content.
- `run_agent()` exposes `partial_relevance_recovery`.
- Retry and fallback behavior remains bounded by `max_retry_count`.

## Risks

The main risk is creating a false sense that partial evidence is enough for answer generation. P1f avoids that by never adding partial chunks to `relevant_documents`; partial evidence only influences query refinement. Dynamic retrieval expansion and reranker adjustment remain explicit future work.
