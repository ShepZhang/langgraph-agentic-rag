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
