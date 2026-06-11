# P3b FastAPI Service Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a modular FastAPI backend for chat, trace lookup, document management, and evaluation runs.

**Architecture:** Create a focused `api` package with schemas, dependency factories, route modules, and service modules. Keep the API synchronous and local, reuse `run_agent()`, vectorstore helpers, P3a trace storage, and the existing evaluation runner.

**Tech Stack:** FastAPI, Pydantic v2, Starlette TestClient, JSON registry files, existing LangGraph/RAG/evaluation modules, pytest.

---

## File Map

- Create `api/main.py`: app factory and router registration.
- Create `api/schemas.py`: Pydantic request and response schemas.
- Create `api/dependencies.py`: injectable runner and service factories.
- Create `api/routes/chat.py`: chat and trace routes.
- Create `api/routes/documents.py`: upload, index, list, delete routes.
- Create `api/routes/evaluation.py`: evaluation run and read routes.
- Create `api/services/documents.py`: local upload registry and indexing service.
- Create `api/services/evaluation.py`: synchronous evaluation run service.
- Create `api/services/traces.py`: JSONL trace lookup service.
- Modify `rag/vectorstore.py`: expose delete-by-id support for API-managed documents.
- Modify `requirements.txt`: add explicit FastAPI runtime dependencies.
- Modify `.env.example`, `config.py`, README, and CHANGELOG.
- Add `tests/test_fastapi_routes.py`.

## Tasks

- [x] Add failing FastAPI route tests using dependency overrides.
- [x] Add API schemas, dependencies, route modules, and app factory.
- [x] Implement document registry/index/delete service.
- [x] Implement trace lookup and evaluation services.
- [x] Add vectorstore delete-by-id support.
- [x] Update config, requirements, README, changelog, and resume bullets.
- [x] Run focused FastAPI tests and the full pytest suite.
