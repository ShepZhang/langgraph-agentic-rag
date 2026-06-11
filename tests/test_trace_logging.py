"""Tests for local Agent trace logging."""

from __future__ import annotations

from dataclasses import replace

from agent.features import AgentFeatureFlags
from config import get_settings


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


class MemoryTraceStore:
    def __init__(self):
        self.records = []

    def save(self, record):
        self.records.append(record)


def test_jsonl_trace_store_persists_and_reads_by_trace_id(tmp_path):
    from observability.storage import JsonlTraceStore

    store = JsonlTraceStore(tmp_path)
    record = {
        "trace_id": "trace_test",
        "session_id": "session_1",
        "workspace_id": "workspace_1",
        "events": [{"event_type": "node", "node": "retrieve"}],
    }

    store.save(record)

    assert store.path == tmp_path / "traces.jsonl"
    assert store.get("trace_test") == record
    assert store.get("missing") is None


def test_run_agent_writes_node_events_and_route_decisions_to_trace(tmp_path):
    from agent.graph import run_agent
    from observability.storage import JsonlTraceStore

    settings = replace(
        get_settings(),
        trace_logging_enabled=True,
        trace_log_dir=tmp_path,
    )
    flags = AgentFeatureFlags(
        query_transformation_enabled=False,
        retrieval_grading_enabled=False,
        conditional_retry_enabled=False,
        citation_verification_enabled=False,
    )
    llm = FakeLLM(
        [
            (
                '{"answer": "RAG retrieves supporting evidence [1].", '
                '"used_citation_indices": [1]}'
            )
        ]
    )
    documents = [
        {
            "content": "RAG retrieves supporting evidence from indexed chunks.",
            "source": "notes.md",
            "chunk_id": "notes.md:pNA:c1",
            "score": 0.91,
        }
    ]

    result = run_agent(
        "What is RAG?",
        session_id="session_1",
        workspace_id="workspace_1",
        llm=llm,
        retriever_fn=lambda query: documents,
        settings=settings,
        features=flags,
    )

    store = JsonlTraceStore(tmp_path)
    trace = store.get(result["trace_id"])

    assert trace is not None
    assert result["trace_path"] == str(store.path)
    assert result["latency_ms"] >= 0
    assert trace["trace_id"] == result["trace_id"]
    assert trace["session_id"] == "session_1"
    assert trace["workspace_id"] == "workspace_1"
    assert trace["original_question"] == "What is RAG?"
    assert trace["final_answer"] == "RAG retrieves supporting evidence [1]."
    assert trace["retrieved_documents"][0]["chunk_id"] == "notes.md:pNA:c1"
    assert trace["relevant_documents"][0]["chunk_id"] == "notes.md:pNA:c1"
    assert trace["citations"][0]["chunk_id"] == "notes.md:pNA:c1"
    assert trace["error"] is None

    node_events = [
        event for event in trace["events"] if event["event_type"] == "node"
    ]
    assert [event["node"] for event in node_events] == [
        "retrieve",
        "accept_documents",
        "generate_answer",
        "finalize_answer",
    ]
    assert all(event["elapsed_ms"] >= 0 for event in node_events)

    route_decisions = trace["route_decisions"]
    assert {
        "from": "accept_documents",
        "to": "generate_answer",
        "reason": "Retrieved documents accepted without grading.",
    } in route_decisions
    assert {
        "from": "generate_answer",
        "to": "finalize_answer",
        "reason": "Citation verification skipped.",
    } in route_decisions


def test_run_agent_accepts_trace_store_without_path(tmp_path):
    from agent.graph import run_agent

    settings = replace(
        get_settings(),
        trace_logging_enabled=True,
        trace_log_dir=tmp_path,
    )
    flags = AgentFeatureFlags(
        query_transformation_enabled=False,
        retrieval_grading_enabled=False,
        conditional_retry_enabled=False,
        citation_verification_enabled=False,
    )
    store = MemoryTraceStore()

    result = run_agent(
        "What is RAG?",
        llm=FakeLLM(
            [
                (
                    '{"answer": "RAG retrieves supporting evidence [1].", '
                    '"used_citation_indices": [1]}'
                )
            ]
        ),
        retriever_fn=lambda query: [
            {
                "content": "RAG retrieves supporting evidence.",
                "source": "notes.md",
                "chunk_id": "notes.md:pNA:c1",
            }
        ],
        settings=settings,
        features=flags,
        trace_store=store,
    )

    assert result["trace_path"] is None
    assert store.records[0]["trace_id"] == result["trace_id"]
