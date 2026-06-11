# P3b FastAPI Service Layer Design

## Goal

Add a FastAPI backend that exposes the existing Agentic RAG capabilities through
integration-friendly HTTP endpoints while preserving the Gradio UI and current
CLI workflows.

## Chosen Approach

Use a modular FastAPI layer:

- `api/main.py` creates the app and registers routers.
- `api/schemas.py` owns request and response models.
- `api/dependencies.py` exposes injectable services for tests and future API
  composition.
- `api/routes/` contains chat, document, and evaluation endpoints.
- `api/services/` contains thin orchestration services over existing project
  modules.

This milestone stays synchronous and local. It does not introduce background
workers, authentication, authorization, or database-backed state.

## Endpoints

- `POST /chat`: run Agentic RAG and return answer, citations, trace metadata,
  retry count, latency, and fallback status.
- `GET /chat/{session_id}/trace`: return a trace by `trace_id` or the latest
  trace for a session.
- `POST /documents/upload`: save uploaded files into the API upload directory
  and register document metadata.
- `POST /documents/index`: load, chunk, and index registered uploaded files.
- `GET /documents`: list registered documents, optionally filtered by
  workspace.
- `DELETE /documents/{document_id}`: delete an API-managed document and its
  known vector IDs when available.
- `POST /evaluation/run`: run the lightweight evaluator and persist a run
  artifact.
- `GET /evaluation/{run_id}`: read a persisted evaluation run.

## Non-goals

- No production authentication or tenant authorization.
- No full workspace-isolated vector collections yet.
- No async job queue.
- No visual dashboard.
- No replacement for the Gradio UI.

## Validation

P3b is complete when:

- FastAPI routes return stable schemas under `TestClient`.
- Routes can be tested through dependency overrides without real LLM calls.
- Document service can register, index, list, and delete API-managed documents.
- Evaluation service can write and read run artifacts.
- README explains how to start the API.
- The full pytest suite passes.
