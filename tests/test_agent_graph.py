"""Tests for the LangGraph Agent workflow entrypoint."""

from __future__ import annotations

from dataclasses import replace

import pytest

from config import get_settings


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_run_agent_generates_answer_when_retrieval_is_relevant():
    from agent.graph import run_agent

    llm = FakeLLM(
        [
            "rewritten agentic rag question",
            '{"relevant": true, "reason": "matches"}',
            "Agentic RAG uses retrieval and agent control flow.",
        ]
    )
    documents = [
        {
            "content": "Agentic RAG uses retrieval and agent control flow.",
            "source": "agentic-rag.md",
            "page": 3,
            "chunk_id": "agentic-rag.md:p3:c1",
            "score": 0.91,
        }
    ]

    result = run_agent(
        "How does it work?",
        chat_history=[{"role": "user", "content": "Tell me about Agentic RAG"}],
        llm=llm,
        retriever_fn=lambda query: documents,
        settings=get_settings(),
    )

    assert result["answer"] == "Agentic RAG uses retrieval and agent control flow."
    assert result["rewritten_question"] == "rewritten agentic rag question"
    assert result["rewrite_count"] == 1
    assert result["is_relevant"] is True
    assert result["citations"] == [
        {
            "source": "agentic-rag.md",
            "page": 3,
            "chunk_id": "agentic-rag.md:p3:c1",
            "score": 0.91,
        }
    ]
    assert result["retrieved_documents"] == documents


def test_run_agent_retries_then_falls_back_when_documents_are_irrelevant():
    from agent.graph import run_agent

    settings = replace(get_settings(), max_rewrite_attempts=2)
    llm = FakeLLM(
        [
            "first rewritten question",
            '{"relevant": false, "reason": "wrong topic"}',
            "second rewritten question",
            '{"relevant": false, "reason": "still wrong"}',
        ]
    )
    retriever_queries = []

    def fake_retriever(query):
        retriever_queries.append(query)
        return [{"content": "unrelated context", "source": "other.md"}]

    result = run_agent(
        "original question",
        llm=llm,
        retriever_fn=fake_retriever,
        settings=settings,
    )

    assert retriever_queries == ["first rewritten question", "second rewritten question"]
    assert result["rewrite_count"] == 2
    assert result["is_relevant"] is False
    assert "无法可靠回答" in result["answer"]
    assert result["citations"] == []


def test_build_graph_requires_llm_config_when_no_llm_is_injected():
    from agent.graph import build_graph

    settings = replace(get_settings(), openai_api_key="", openai_model="")

    with pytest.raises(RuntimeError, match="Missing LLM configuration"):
        build_graph(settings=settings)
