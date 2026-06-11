# P3c Workspace Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce workspace-aware retrieval across dense, BM25, hybrid, Agent, and API chat paths.

**Architecture:** Keep one Chroma collection and use metadata filters for workspace scoping. Thread `workspace_id` through vectorstore search, corpus export, hybrid retrieval, retriever normalization, retriever tool construction, and `run_agent()`.

**Tech Stack:** Python, LangChain Chroma metadata filters, LangGraph, FastAPI, pytest.

---

## File Map

- Modify `rag/vectorstore.py`: add optional `workspace_id` to dense search and corpus export.
- Modify `rag/hybrid_retriever.py`: pass workspace id to dense and BM25 branches.
- Modify `rag/retriever.py`: accept workspace id and expose workspace/document metadata.
- Modify `agent/tools.py`: support workspace-scoped default retriever tools.
- Modify `agent/nodes.py`: pass workspace id to the retriever tool factory.
- Modify `agent/graph.py`: pass `workspace_id` into graph construction.
- Modify README, CHANGELOG, and resume bullets.
- Add `tests/test_workspace_isolation.py`.

## Tasks

- [x] Add failing tests for dense, BM25, hybrid, retriever, and Agent workspace scoping.
- [x] Implement workspace filters in vectorstore search and corpus export.
- [x] Thread workspace id through hybrid and project-level retriever APIs.
- [x] Scope default Agent retriever tools by workspace id.
- [x] Update docs to mark P3c completed and describe the limitation of unscoped legacy data.
- [x] Run focused workspace tests and full pytest.
