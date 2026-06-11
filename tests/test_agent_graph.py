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
            '{"relevant": true, "relevant_indices": [1], "reason": "matches"}',
            (
                '{"answer": "Agentic RAG uses retrieval and agent control flow [1].", '
                '"used_citation_indices": [1]}'
            ),
            (
                '{"verified": true, "claims": ['
                '{"claim": "Agentic RAG uses retrieval and agent control flow", '
                '"supported": true, "citation_indices": [1]}'
                '], "reason": "Supported by chunk 1."}'
            ),
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

    assert result["answer"] == "Agentic RAG uses retrieval and agent control flow [1]."
    assert result["rewritten_question"] == "rewritten agentic rag question"
    assert result["current_query"] == "rewritten agentic rag question"
    assert result["standalone_question"] == "rewritten agentic rag question"
    assert result["query_transform_strategy"] == "rewrite"
    assert result["expanded_queries"] == []
    assert result["sub_questions"] == []
    assert result["retrieval_queries"] == ["rewritten agentic rag question"]
    assert result["multi_query_used"] is False
    assert result["multi_query_result_count"] == 1
    assert result["query_transform"]["rewritten_query"] == "rewritten agentic rag question"
    assert result["rewrite_count"] == 0
    assert result["retry_count"] == 0
    assert result["retrieval_attempt"] == 1
    assert result["is_relevant"] is True
    assert result["grading_reason"] == "matches"
    assert result["document_grades"] == [
        {
            "document_index": 1,
            "relevance": "relevant",
            "confidence": 1.0,
            "reason": "matches",
        }
    ]
    assert result["relevant_document_count"] == 1
    assert result["partial_document_count"] == 0
    assert result["max_relevance_confidence"] == 1.0
    assert result["is_verified"] is True
    assert result["claim_verification_reason"] == "Supported by chunk 1."
    assert result["claims"] == [
        {
            "claim": "Agentic RAG uses retrieval and agent control flow",
            "supported": True,
            "citation_indices": [1],
        }
    ]
    assert result["citations"] == [
        {
            "source": "agentic-rag.md",
            "page": 3,
            "chunk_id": "agentic-rag.md:p3:c1",
            "score": 0.91,
            "snippet": "Agentic RAG uses retrieval and agent control flow.",
        }
    ]
    assert result["retrieved_documents"][0]["content"] == documents[0]["content"]
    assert result["retrieved_documents"][0]["source"] == documents[0]["source"]
    assert result["retrieved_documents"][0]["matched_queries"] == [
        "rewritten agentic rag question"
    ]
    assert result["relevant_documents"][0]["content"] == documents[0]["content"]


def test_run_agent_executes_multi_query_retrieval_and_exposes_diagnostics():
    from agent.graph import run_agent

    llm = FakeLLM(
        [
            (
                '{"strategy": "multi_query", '
                '"rewritten_query": "Agentic RAG reliability advantages", '
                '"expanded_queries": ["retrieval grading reliability", "fallback handling"], '
                '"sub_questions": [], '
                '"reason": "Expand reliability concepts."}'
            ),
            '{"relevant": true, "relevant_indices": [1], "reason": "matches"}',
            (
                '{"answer": "Agentic RAG uses retrieval grading [1].", '
                '"used_citation_indices": [1]}'
            ),
            (
                '{"verified": true, "claims": ['
                '{"claim": "Agentic RAG uses retrieval grading", '
                '"supported": true, "citation_indices": [1]}'
                '], "reason": "Supported."}'
            ),
        ]
    )
    retriever_queries = []

    def fake_retriever(query):
        retriever_queries.append(query)
        if query == "Agentic RAG reliability advantages":
            return [
                {
                    "content": "Agentic RAG uses retrieval grading.",
                    "source": "notes.md",
                    "chunk_id": "notes:c1",
                }
            ]
        return [
            {
                "content": "Agentic RAG uses retrieval grading.",
                "source": "notes.md",
                "chunk_id": "notes:c1",
            }
        ]

    result = run_agent(
        "What reliability advantages does it have?",
        llm=llm,
        retriever_fn=fake_retriever,
        settings=get_settings(),
    )

    assert retriever_queries == [
        "Agentic RAG reliability advantages",
        "retrieval grading reliability",
        "fallback handling",
    ]
    assert result["retrieval_queries"] == retriever_queries
    assert result["multi_query_used"] is True
    assert result["multi_query_result_count"] == 1
    assert result["retrieved_documents"][0]["matched_queries"] == retriever_queries


def test_run_agent_retries_then_falls_back_when_documents_are_irrelevant():
    from agent.graph import run_agent

    settings = replace(get_settings(), max_retry_count=2)
    llm = FakeLLM(
        [
            "first rewritten question",
            '{"relevant": false, "relevant_indices": [], "reason": "wrong topic"}',
            "second rewritten question",
            '{"relevant": false, "relevant_indices": [], "reason": "still wrong"}',
            "third rewritten question",
            '{"relevant": false, "relevant_indices": [], "reason": "still wrong again"}',
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

    assert retriever_queries == [
        "first rewritten question",
        "second rewritten question",
        "third rewritten question",
    ]
    assert result["rewrite_count"] == 2
    assert result["retry_count"] == 2
    assert result["retrieval_attempt"] == 3
    assert result["is_relevant"] is False
    assert result["grading_reason"] == "still wrong again"
    assert "无法可靠回答" in result["answer"]
    assert result["citations"] == []


def test_build_graph_requires_llm_config_when_no_llm_is_injected():
    from agent.graph import build_graph

    settings = replace(get_settings(), openai_api_key="", openai_model="")

    with pytest.raises(RuntimeError, match="Missing LLM configuration"):
        build_graph(settings=settings)
