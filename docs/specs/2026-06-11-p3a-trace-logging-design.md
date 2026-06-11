# P3a Trace Logging Design

## Goal

Add local trace logging for each Agentic RAG run so developers can inspect what
the LangGraph workflow did, which routes it took, what evidence it used, and why
it returned an answer or fallback.

## Chosen Approach

Use a graph-wrapper observability layer with JSONL persistence:

- Wrap LangGraph node callables in `build_graph()` and record node events after
  each node returns a state delta.
- Wrap conditional edge functions and record route decisions with a concise
  reason.
- Keep trace storage independent from Agent node business logic.
- Save one JSON record per run to a local JSONL file.
- Return `trace_id`, `trace_path`, and `latency_ms` from `run_agent()`.

## Trace Contents

Each trace record includes:

- `trace_id`, `session_id`, `workspace_id`
- original question
- node events and route decisions
- query transformation
- retrieved and relevant document summaries
- document grades
- final answer and citations
- claim verification results
- retry count and latency
- error message when the workflow fails

## Non-goals

- Do not add FastAPI trace endpoints in P3a.
- Do not add a Gradio dashboard in P3a.
- Do not introduce external observability services.
- Do not store API keys, base URLs, full local paths, or full document bodies in
  traces.

## Validation

P3a is complete when:

- Trace storage can append and read JSONL records by `trace_id`.
- `run_agent()` can write a trace with node events and route decisions.
- Trace logging can be disabled by configuration.
- The full test suite still passes.
