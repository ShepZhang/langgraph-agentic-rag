# P1d Multi-query Retrieval Design

## Goal

Make the `expanded_queries` produced by P1c query transformation execute real retrieval calls and merge their evidence before retrieval grading.

## Scope

P1d implements multi-query retrieval execution inside the existing `retrieve_node`. It does not change the LangGraph topology and does not introduce a new retriever public API.

Implemented behavior:

- If `query_transform_strategy == "multi_query"`, retrieve with `current_query` plus `expanded_queries`.
- If strategy is `rewrite` or `decomposition`, keep the current single-query retrieval behavior.
- Deduplicate retrieved chunks across queries.
- Record multi-query diagnostics in state and result payloads.

Deferred behavior:

- Decomposition sub-question retrieval remains future P2/P1e work.
- Independent ablation toggles remain future P0b/P1 work.
- Retriever-layer `retrieve_many()` remains optional future refactor after the agent-level behavior is proven.

## Design

### Module Boundary

Create `agent/multi_query.py` with:

- `build_retrieval_queries(current_query, expanded_queries, strategy)`
- `merge_retrieved_documents(query_results)`

`build_retrieval_queries()` filters blank queries and deduplicates while preserving order. It returns only `current_query` unless the strategy is `multi_query`.

`merge_retrieved_documents()` accepts a list of `(query, documents)` pairs. It deduplicates by:

1. `chunk_id`, when available.
2. fallback key using `source`, `page`, and `content`.

Merged documents keep existing document fields and add:

- `matched_queries`
- `retrieval_query_count`
- `multi_query_rank`

### Agent Node Integration

`AgentNodes.retrieve_node()` builds retrieval queries from current state. It calls the existing retriever tool once per query, merges results, increments `retrieval_attempt` once per graph node execution, and returns:

- `documents`
- `retrieval_queries`
- `multi_query_used`
- `multi_query_result_count`
- `retrieval_attempt`

### State And Payload

`AgentState` gains:

- `retrieval_queries`
- `multi_query_used`
- `multi_query_result_count`

`run_agent()` returns the same fields for UI/evaluation/trace readiness.

### Testing

Tests cover:

- query list construction for rewrite and multi-query strategies.
- document deduplication and matched query metadata.
- `retrieve_node` calls retriever multiple times only for multi-query.
- `retrieve_node` keeps single-query behavior for rewrite.
- `run_agent()` returns multi-query diagnostics.

## Risks

Because the agent tool returns normalized dictionaries rather than LangChain `Document` objects, P1d uses deterministic dict-level merge rather than retriever-layer RRF. This is acceptable for this milestone because P1a already provides RRF inside hybrid retrieval. A later retriever-layer `retrieve_many()` can make cross-query fusion deeper if needed.
