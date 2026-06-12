"""Tests for Agent retriever tools."""

from __future__ import annotations

import pytest

from agent.tools import create_retriever_tool


def test_create_retriever_tool_invokes_retriever_function():
    calls = []

    def fake_retriever(query: str):
        calls.append(query)
        return [{"content": "context", "source": "notes.md"}]

    tool = create_retriever_tool(fake_retriever)
    result = tool.invoke({"query": "What is RAG?"})

    assert calls == ["What is RAG?"]
    assert result == [{"content": "context", "source": "notes.md"}]
    assert tool.name == "retrieve_context"
    assert "private knowledge base" in tool.description


def test_create_retriever_tool_raises_for_broken_retriever():
    def broken_retriever(query: str):
        raise RuntimeError("retrieval unavailable")

    tool = create_retriever_tool(broken_retriever)

    with pytest.raises(RuntimeError, match="retrieval unavailable"):
        tool.invoke({"query": "What is RAG?"})


def test_create_retriever_tool_rejects_extra_workspace_id_before_invoking_retriever():
    calls = []

    def fake_retriever(query: str):
        calls.append(query)
        return [{"content": "context", "source": "notes.md"}]

    tool = create_retriever_tool(fake_retriever)

    with pytest.raises(Exception):
        tool.invoke({"query": "hello", "workspace_id": "ws-999"})

    assert calls == []


def test_create_retriever_tool_preserves_falsy_retriever_injection(monkeypatch):
    calls = []
    default_calls = []

    class FalsyRetriever:
        def __bool__(self):
            return False

        def __call__(self, query: str):
            calls.append(query)
            return [{"content": "context", "source": "notes.md"}]

    def default_retrieve(query: str, top_k=None, workspace_id=None):
        default_calls.append((query, top_k, workspace_id))
        return [{"content": "default", "source": "fallback.md"}]

    monkeypatch.setattr("agent.tools.retrieve", default_retrieve)

    tool = create_retriever_tool(FalsyRetriever())
    result = tool.invoke({"query": "hello"})

    assert calls == ["hello"]
    assert default_calls == []
    assert result == [{"content": "context", "source": "notes.md"}]
