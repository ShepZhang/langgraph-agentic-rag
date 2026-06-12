# P3d Tool Registry Design

Date: 2026-06-12

## Goal

Introduce an internal, typed Tool Registry that gives Agent capabilities a
consistent registration, dependency injection, validation, execution, and
diagnostic boundary.

P3d must integrate the retriever and claim-level citation verifier into the
existing LangGraph workflow. It must also provide independently callable
document summary and calculator tools without adding autonomous tool
selection, a planning loop, or a generic public tool execution API.

## Scope

### Included

- Add a top-level `tools/` package with a typed base interface and registry.
- Register retriever, citation verifier, document summary, and calculator
  tools.
- Make the retrieval and citation verification LangGraph nodes invoke tools
  through the registry.
- Inject runtime dependencies through an explicit tool context.
- Standardize tool input validation, results, errors, metadata, and timing.
- Add compact tool-call diagnostics to trace records.
- Preserve compatibility for existing `agent.tools.create_retriever_tool()`
  consumers during migration.
- Add focused unit and workflow tests.
- Update README, changelog, roadmap, and resume-facing documentation.

### Excluded

- LLM-driven tool selection or autonomous planning.
- A ReAct or repeated tool-calling loop.
- `GET /tools` or generic `POST /tools/{name}/invoke` API endpoints.
- Authentication or authorization policy for tools.
- Remote tool execution, MCP integration, or distributed tool discovery.
- Adding summary or calculator tools to the primary document QA graph.

## Architecture

The new package will use the following structure:

```text
tools/
├── __init__.py
├── base.py
├── registry.py
├── factory.py
├── retriever_tool.py
├── citation_verifier_tool.py
├── document_summary_tool.py
└── calculator_tool.py
```

### Base Types

`tools/base.py` defines:

- `ToolContext`: runtime dependencies such as `workspace_id`, `llm`, and an
  optional retriever callable.
- `ToolErrorInfo`: structured runtime failure containing a stable `code` and
  concise `message`.
- `ToolResult[T]`: normalized execution result containing `tool_name`,
  `success`, `data`, `error`, and `metadata`.
- `BaseTool[ArgsT, ResultT]`: typed tool contract declaring `name`,
  `description`, `args_schema`, and `run()`.
- Tool-specific exceptions:
  - `ToolRegistrationError`
  - `ToolNotFoundError`
  - `ToolInputError`
  - `ToolExecutionError`

Pydantic models are used for tool argument validation. Tool implementations do
not read workspace identity or runtime clients from mutable module globals.

### Registry

`tools/registry.py` defines `ToolRegistry` with:

- `register(tool)`
- `get(name)`
- `list_tools()`
- `invoke(name, arguments)`
- `set_call_observer(observer)`

Registration rejects blank names and duplicate names. Lookup of an unknown
tool raises `ToolNotFoundError` because it indicates a configuration or
programming defect.

For a known tool, `invoke()` validates arguments, measures execution latency,
and returns a `ToolResult`. Input and execution failures are represented by a
failed result so workflow nodes can apply reliability-oriented retry or
fallback behavior.

The optional call observer receives one compact diagnostic record after every
known-tool invocation. Observer failures are logged but cannot change the tool
result or break the Agent workflow.

### Default Factory

`tools/factory.py` defines a default registry factory. It receives runtime
dependencies from `build_graph()` and constructs a registry containing:

- `retrieve_context`
- `verify_citations`
- `summarize_document`
- `calculator`

The factory is the only default composition root for tool implementations.
Tests and custom integrations can supply a prebuilt registry.

## Tool Contracts

### Retriever Tool

Name: `retrieve_context`

Input:

```json
{
  "query": "What improves RAG reliability?"
}
```

The tool calls an injected retriever when provided. Otherwise, it calls the
project retriever with the context `workspace_id`. The result contains the
existing normalized retrieved-document records so multi-query merging,
retrieval grading, reranking diagnostics, and citation generation remain
compatible.

The workspace boundary is injected by `ToolContext`; callers cannot override
it through tool arguments.

### Citation Verifier Tool

Name: `verify_citations`

Input:

```json
{
  "question": "How does retrieval grading improve reliability?",
  "answer": "Retrieval grading filters weak evidence [1].",
  "claims": [
    {
      "claim_id": "c001",
      "claim": "Retrieval grading filters weak evidence.",
      "cited_chunk_ids": ["chunk_001"]
    }
  ],
  "documents": []
}
```

The tool owns citation-verification prompt construction, LLM invocation,
response parsing, valid chunk ID filtering, and normalized verification
output. It reuses the existing prompt and parser contracts.

The LangGraph node remains responsible for state updates and routing to
finalization, answer revision, or fallback.

### Document Summary Tool

Name: `summarize_document`

Input fields:

- `content`: required document text.
- `title`: optional document title.
- `max_points`: bounded number of requested summary points.

The tool uses the injected LLM to produce a grounded summary. It is
independently callable through the registry but is not added to the primary QA
workflow in P3d.

### Calculator Tool

Name: `calculator`

Input:

```json
{
  "expression": "(12 + 8) / 5"
}
```

The calculator parses an expression with Python's AST and evaluates only a
whitelist of numeric constants and arithmetic operators:

- `+`, `-`, `*`, `/`, `//`, `%`, `**`
- unary `+` and `-`
- parentheses through AST structure

Names, attributes, indexing, function calls, comprehensions, strings, and
other executable syntax are rejected. Direct `eval()` is not used.

## LangGraph Integration

`build_graph()` gains an optional `tool_registry` argument. When no registry is
provided, it creates the default registry from the resolved LLM, retriever,
and workspace.

`AgentNodes` receives the registry rather than constructing a retriever tool.

### Retrieval Flow

```text
retrieve_node
  -> registry.invoke("retrieve_context", {"query": retrieval_query})
  -> RetrieverTool
  -> dense or hybrid retrieval and optional reranking
  -> ToolResult.data
  -> merge_retrieved_documents()
```

Each expanded query remains a separate tool invocation. Successful result data
is passed to the current de-duplication and matched-query merge logic.

If retrieval execution fails, the node records the tool failure as the grading
or fallback reason, returns no documents for that attempt, and lets the
existing retry/fallback policy decide the next route.

### Citation Verification Flow

```text
verify_citations_node
  -> registry.invoke("verify_citations", verification arguments)
  -> CitationVerifierTool
  -> verification prompt and LLM
  -> normalized claim verification
  -> ToolResult.data
  -> node route: finalize, revise, or fallback
```

If the verifier tool fails, the node must not finalize an unverified answer.
It returns a safe fallback update with a diagnostic reason.

## Responsibility Boundaries

- Tools own input validation, capability execution, and normalized output.
- LangGraph nodes own Agent state mutation, retry counters, revision counters,
  and conditional routing.
- The registry owns registration, discovery, invocation, timing, and standard
  result wrapping.
- The default factory owns runtime composition.
- The trace layer owns compact persistence of tool-call diagnostics.
- `agent/tools.py` remains a temporary compatibility adapter and delegates to
  the new retriever implementation.

No prompt, retry, or routing logic is duplicated between tools and nodes.

## Error Handling

### Startup and Programming Errors

- Invalid or duplicate registration raises `ToolRegistrationError`.
- Unknown tool lookup or invocation raises `ToolNotFoundError`.

These failures remain exceptions because they indicate an invalid application
composition and should fail early.

### Runtime Errors

- Pydantic validation failures become failed `ToolResult` values with error
  code `tool_input_error`.
- Retriever, LLM, parser, or calculator failures become failed `ToolResult`
  values with error code `tool_execution_error`.
- Error text is concise and must not contain API keys or credentials.

Workflow behavior:

- Retriever failure enters the existing evidence retry/fallback path.
- Citation verifier failure triggers fallback and never promotes the draft
  answer.
- Summary and calculator failures are returned directly to their callers.

## Trace Logging

`TraceRecorder` gains `record_tool_call(record)`. During graph construction,
the active recorder is installed through `ToolRegistry.set_call_observer()`.
The registry calls this observer after every known-tool invocation:

```json
{
  "tool_name": "retrieve_context",
  "success": true,
  "latency_ms": 24.8,
  "error": null,
  "metadata": {
    "workspace_id": "w001",
    "result_count": 5
  }
}
```

Trace records must not duplicate full document content, prompts, raw model
responses, or secrets. Existing node events and route decisions remain
unchanged.

Tool calls are appended to the existing `events` sequence with
`event_type="tool"` and are also available in a top-level `tool_calls` list in
the final trace record.

## Compatibility

`agent/tools.py` continues to export `create_retriever_tool()` and
`retrieve_context`. The adapter delegates to the new retriever tool while
preserving the current LangChain `StructuredTool` interface and existing
tests.

This compatibility layer avoids a breaking change for current consumers and
can be removed in a later major cleanup after all callers migrate.

## Testing

Add:

- `tests/test_tool_registry.py`
- `tests/test_retriever_tool.py`
- `tests/test_citation_verifier_tool.py`
- `tests/test_document_summary_tool.py`
- `tests/test_calculator_tool.py`

Update:

- `tests/test_agent_tools.py`
- `tests/test_agent_nodes.py`
- `tests/test_agent_graph.py`
- `tests/test_trace_logging.py`

Required coverage:

- Register, list, retrieve, and invoke tools.
- Reject duplicate registration and unknown tool names.
- Validate arguments and normalize success/failure results.
- Preserve workspace identity in retrieval without accepting caller override.
- Parse supported, partially supported, and unsupported claim results.
- Route retrieval failures through retry/fallback behavior.
- Route citation verifier failures directly to safe fallback.
- Reject unsafe calculator AST nodes.
- Verify that the main graph uses the supplied registry for retrieval and
  citation verification.
- Record compact tool-call trace diagnostics.
- Preserve the existing `agent.tools` compatibility API.

The full existing test suite must continue to pass.

## Documentation and Versioning

The implementation release is planned as `v0.3.3-p3d`.

Update:

- README architecture, completed milestones, limitations, and roadmap.
- CHANGELOG with the P3d capability and explicit non-autonomous scope.
- Resume bullets to describe an extensible typed tool boundary without
  claiming autonomous planning.

## Acceptance Criteria

P3d is complete when:

1. A typed internal registry composes all four required tools.
2. Retrieval and citation verification nodes invoke tools through the
   registry.
3. Workspace isolation remains enforced for retrieval.
4. Citation verifier failures cannot produce an unverified final answer.
5. Document summary and calculator tools can be invoked independently.
6. Tool calls expose compact timing and error diagnostics for trace logging.
7. Existing `agent.tools` consumers remain compatible.
8. Focused tests and the complete project test suite pass.
9. README and CHANGELOG accurately describe the feature as extensible tooling,
   not autonomous tool planning.
