# P3a Trace Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add local JSONL trace logging for Agentic RAG runs.

**Architecture:** Build an `observability` package with trace record builders and JSONL storage. Instrument LangGraph through node and conditional-edge wrappers in `agent.graph`, keeping trace concerns out of node business logic.

**Tech Stack:** Python dataclasses, TypedDict-compatible dictionaries, LangGraph `StateGraph`, JSONL files, pytest.

---

## File Map

- Create `observability/trace.py`: trace recorder, state summarization, route reason helpers.
- Create `observability/storage.py`: JSONL trace storage and read-by-id support.
- Create `observability/logger.py`: settings-based trace store factory.
- Modify `config.py`: add trace logging settings.
- Modify `.env.example`: document trace settings.
- Modify `agent/graph.py`: wrap nodes and conditional edges, return trace metadata.
- Add `tests/test_trace_logging.py`: cover storage and `run_agent()` integration.
- Modify README and changelog after implementation.

## Tasks

- [x] Add failing tests for JSONL storage and `run_agent()` trace output.
- [x] Implement `observability` storage and trace recorder.
- [x] Add trace settings to `Settings` and `.env.example`.
- [x] Instrument `agent.graph` with wrappers around nodes and conditional edges.
- [x] Update README and changelog with P3a trace usage.
- [x] Run focused trace tests and the full pytest suite.
