# P1c Query Transformation Design

## Goal

Upgrade the current plain query rewrite step into a structured query transformation layer while keeping the existing LangGraph node names, UI fields, and retry behavior compatible.

## Scope

P1c adds structure and routing metadata to the first retrieval query preparation step:

- `rewrite`: direct standalone question rewrite.
- `multi_query`: standalone rewrite plus expanded equivalent or complementary queries.
- `decomposition`: standalone rewrite plus sub-questions for complex comparison or multi-hop questions.

This milestone does not make retrieval execute every expanded query. `retrieve_node` continues to use `current_query`. The expanded queries and sub-questions are recorded in state and result payloads so P1d can implement multi-query retrieval without changing the query transformation contract again.

## Design

### Module Boundary

Create `agent/query_transform.py` as a focused module with:

- `QueryTransformResult` TypedDict.
- `build_query_transform_prompt(question, chat_history)`.
- `parse_query_transform_response(raw_text, original_question)`.
- `fallback_query_transform(question)`.

`agent.nodes.AgentNodes.rewrite_query_node()` keeps its public name but calls this module on the initial attempt. Retry rewrites keep the existing retry-specific prompt and text behavior because they depend on failed retrieval context.

### Prompt Contract

The initial query transformation prompt asks the model to return JSON only:

```json
{
  "strategy": "rewrite",
  "rewritten_query": "standalone retrieval query",
  "expanded_queries": [],
  "sub_questions": [],
  "reason": "short reason"
}
```

The prompt includes routing guidance:

- simple factual question -> `rewrite`
- ambiguous follow-up -> `multi_query`
- complex comparison or multi-hop question -> `decomposition`

### Compatibility

If the model returns plain text or invalid JSON, the parser treats the response as a direct rewrite:

- `strategy = "rewrite"`
- `rewritten_query = raw_text.strip() or original_question`
- empty expanded queries and sub-questions
- reason explains fallback parsing

This preserves existing tests and fake LLM behavior.

### State Additions

`AgentState` gains:

- `standalone_question`
- `query_transform`
- `query_transform_strategy`
- `query_transform_reason`
- `expanded_queries`
- `sub_questions`

`current_query` and `rewritten_question` continue to mirror `rewritten_query` for compatibility.

### Result Payload

`run_agent()` returns query transformation fields so evaluation, UI, and future trace logging can inspect how the retrieval query was prepared.

### Testing

Tests cover:

- parser accepts structured JSON and fenced JSON.
- parser falls back on plain text and invalid JSON.
- `rewrite_query_node` records strategy, standalone question, expanded queries, and sub-questions.
- retry rewrite still increments retry count and uses the retry prompt.
- graph result payload exposes query transformation fields.

## Risks

The main risk is overclaiming multi-query retrieval before it exists. README and roadmap must state that P1c records expanded queries and sub-questions, while P1d will make retrieval consume them.
