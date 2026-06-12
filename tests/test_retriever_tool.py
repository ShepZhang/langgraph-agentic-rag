from __future__ import annotations

import pytest
from pydantic import ValidationError

from tools import ToolContext, ToolRegistry
from tools.retriever_tool import RetrieverArgs, RetrieverTool


def test_retriever_tool_uses_injected_retriever_and_traces_allowed_metadata():
    observer_records: list[dict[str, object]] = []
    calls: list[str] = []

    def fake_retriever(query: str) -> list[dict[str, object]]:
        calls.append(query)
        return [{"content": "chunk", "source": "notes.md"}]

    registry = ToolRegistry(call_observer=observer_records.append)
    registry.register(
        RetrieverTool(
            ToolContext(retriever_fn=fake_retriever, workspace_id="ws-123"),
        )
    )

    result = registry.invoke("retrieve_context", {"query": "What is RAG?"})

    assert calls == ["What is RAG?"]
    assert result.success is True
    assert result.data == [{"content": "chunk", "source": "notes.md"}]
    assert result.metadata["workspace_id"] == "ws-123"
    assert result.metadata["result_count"] == 1
    assert observer_records[0]["metadata"] == {
        "workspace_id": "ws-123",
        "result_count": 1,
    }


def test_retriever_tool_rejects_workspace_id_in_tool_input():
    registry = ToolRegistry()
    registry.register(RetrieverTool(ToolContext(workspace_id="ws-123")))

    result = registry.invoke(
        "retrieve_context",
        {"query": "What is RAG?", "workspace_id": "ws-999"},
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "tool_input_error"


def test_retriever_tool_default_path_uses_context_workspace_id(monkeypatch):
    observed: dict[str, object] = {}

    def fake_retrieve(query: str, top_k: int | None = None, workspace_id: str | None = None):
        observed["query"] = query
        observed["workspace_id"] = workspace_id
        observed["top_k"] = top_k
        return [{"content": "chunk", "workspace_id": workspace_id}]

    monkeypatch.setattr("tools.retriever_tool.retrieve", fake_retrieve)

    registry = ToolRegistry()
    registry.register(RetrieverTool(ToolContext(workspace_id="ws-456")))

    result = registry.invoke("retrieve_context", {"query": "workspace scoped"})

    assert result.success is True
    assert result.data == [{"content": "chunk", "workspace_id": "ws-456"}]
    assert observed == {
        "query": "workspace scoped",
        "workspace_id": "ws-456",
        "top_k": None,
    }


def test_retriever_tool_rejects_empty_query():
    registry = ToolRegistry()
    registry.register(RetrieverTool(ToolContext()))

    result = registry.invoke("retrieve_context", {"query": ""})

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "tool_input_error"


def test_retriever_args_forbid_extra_fields():
    with pytest.raises(ValidationError):
        RetrieverArgs.model_validate({"query": "hello", "workspace_id": "ws-1"})
