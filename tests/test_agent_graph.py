"""Tests for the LangGraph Agent workflow entrypoint."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from pydantic import BaseModel

from agent.features import AgentFeatureFlags
from config import get_settings
from tools import ToolContext, ToolRegistry
from tools.base import BaseTool


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


class RetrieveArgs(BaseModel):
    query: str


class VerifyArgs(BaseModel):
    question: str
    answer: str
    claims: list[dict[str, Any]]
    documents: list[dict[str, Any]]


class RecordingRetrieverTool(BaseTool[RetrieveArgs, list[dict[str, Any]]]):
    name = "retrieve_context"
    description = "Return retrieval results."
    args_schema = RetrieveArgs

    def __init__(
        self,
        context: ToolContext,
        *,
        calls: list[str],
        results_by_query: dict[str, list[dict[str, Any]]],
    ) -> None:
        super().__init__(context)
        self.calls = calls
        self.results_by_query = results_by_query

    def run(self, arguments: RetrieveArgs) -> list[dict[str, Any]]:
        self.calls.append(arguments.query)
        return self.results_by_query.get(arguments.query, [])


class RecordingVerifierTool(BaseTool[VerifyArgs, dict[str, Any]]):
    name = "verify_citations"
    description = "Return claim verification results."
    args_schema = VerifyArgs

    def __init__(
        self,
        context: ToolContext,
        *,
        calls: list[dict[str, Any]],
        result: dict[str, Any],
    ) -> None:
        super().__init__(context)
        self.calls = calls
        self.result = result

    def run(self, arguments: VerifyArgs) -> dict[str, Any]:
        self.calls.append(arguments.model_dump())
        return self.result


class FalsyRetriever:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __bool__(self) -> bool:
        return False

    def __call__(self, query: str) -> list[dict[str, Any]]:
        self.calls.append(query)
        return [
            {
                "content": "Agentic RAG uses retrieval and agent control flow.",
                "source": "agentic-rag.md",
                "page": 3,
                "chunk_id": "agentic-rag.md:p3:c1",
                "score": 0.91,
            }
        ]


def test_run_agent_skips_query_transformation_and_grading_when_disabled():
    from agent.graph import run_agent

    flags = AgentFeatureFlags(
        query_transformation_enabled=False,
        retrieval_grading_enabled=False,
        conditional_retry_enabled=False,
        citation_verification_enabled=False,
    )
    llm = FakeLLM(
        [
            (
                '{"answer": "RAG retrieves evidence [1].", '
                '"used_citation_indices": [1]}'
            )
        ]
    )
    queries = []

    def fake_retriever(query):
        queries.append(query)
        return [
            {
                "content": "RAG retrieves evidence.",
                "source": "notes.md",
                "chunk_id": "c1",
            }
        ]

    result = run_agent(
        "What is RAG?",
        llm=llm,
        retriever_fn=fake_retriever,
        settings=get_settings(),
        features=flags,
    )

    assert queries == ["What is RAG?"]
    assert len(llm.prompts) == 1
    assert result["answer"] == "RAG retrieves evidence [1]."
    assert result["query_transform"] == {}
    assert result["document_grades"] == []
    assert result["citation_verification_enabled"] is False
    assert result["feature_flags"] == flags.to_dict()


def test_run_agent_grades_but_does_not_retry_when_retry_feature_is_disabled():
    from agent.graph import run_agent

    flags = AgentFeatureFlags(
        conditional_retry_enabled=False,
        citation_verification_enabled=False,
    )
    llm = FakeLLM(
        [
            "standalone query",
            '{"relevant": false, "relevant_indices": [], "reason": "no evidence"}',
        ]
    )

    result = run_agent(
        "Question?",
        llm=llm,
        retriever_fn=lambda query: [
            {"content": "Unrelated", "source": "x.md", "chunk_id": "x1"}
        ],
        settings=get_settings(),
        features=flags,
    )

    assert result["retry_count"] == 0
    assert result["fallback_reason"]
    assert len(llm.prompts) == 2


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
                '{"claims": ['
                '{"claim_id": "c001", '
                '"claim": "Agentic RAG uses retrieval and agent control flow", '
                '"cited_chunk_ids": ["agentic-rag.md:p3:c1"]}'
                '], "reason": "Extracted one claim."}'
            ),
            (
                '{"results": ['
                '{"claim_id": "c001", '
                '"claim": "Agentic RAG uses retrieval and agent control flow", '
                '"cited_chunk_ids": ["agentic-rag.md:p3:c1"], '
                '"verification_label": "supported", "confidence": 0.94, '
                '"reason": "Supported by chunk 1."}'
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
    assert result["partial_relevance_recovery"] == {
        "triggered": False,
        "action": "none",
        "reason": "",
        "partial_document_indices": [],
    }
    assert result["is_verified"] is True
    assert result["draft_answer"] == (
        "Agentic RAG uses retrieval and agent control flow [1]."
    )
    assert result["used_citation_indices"] == [1]
    assert result["citation_verification_passed"] is True
    assert result["citation_revision_count"] == 0
    assert result["citation_verification_skipped"] is False
    assert result["unsupported_claims"] == []
    assert result["claim_verification_reason"] == "Supported by chunk 1."
    assert result["claims"] == [
        {
            "claim_id": "c001",
            "claim": "Agentic RAG uses retrieval and agent control flow",
            "cited_chunk_ids": ["agentic-rag.md:p3:c1"],
        }
    ]
    assert result["claim_verification_results"] == [
        {
            "claim_id": "c001",
            "claim": "Agentic RAG uses retrieval and agent control flow",
            "cited_chunk_ids": ["agentic-rag.md:p3:c1"],
            "verification_label": "supported",
            "confidence": 0.94,
            "reason": "Supported by chunk 1.",
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
                '{"claims": ['
                '{"claim_id": "c001", "claim": "Agentic RAG uses retrieval grading", '
                '"cited_chunk_ids": ["notes:c1"]}'
                '], "reason": "Extracted."}'
            ),
            (
                '{"results": ['
                '{"claim_id": "c001", "claim": "Agentic RAG uses retrieval grading", '
                '"cited_chunk_ids": ["notes:c1"], "verification_label": "supported", '
                '"confidence": 0.9, "reason": "Supported."}'
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
    assert result["citation_verification_passed"] is True


def test_run_agent_uses_supplied_tool_registry_for_retrieval_and_verification():
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
                '{"claims": ['
                '{"claim_id": "c001", '
                '"claim": "Agentic RAG uses retrieval and agent control flow", '
                '"cited_chunk_ids": ["agentic-rag.md:p3:c1"]}'
                '], "reason": "Extracted one claim."}'
            ),
        ]
    )
    retriever_calls: list[str] = []
    verifier_calls: list[dict[str, Any]] = []
    registry = ToolRegistry()
    registry.register(
        RecordingRetrieverTool(
            ToolContext(),
            calls=retriever_calls,
            results_by_query={
                "rewritten agentic rag question": [
                    {
                        "content": "Agentic RAG uses retrieval and agent control flow.",
                        "source": "agentic-rag.md",
                        "page": 3,
                        "chunk_id": "agentic-rag.md:p3:c1",
                        "score": 0.91,
                    }
                ]
            },
        )
    )
    registry.register(
        RecordingVerifierTool(
            ToolContext(),
            calls=verifier_calls,
            result={
                "results": [
                    {
                        "claim_id": "c001",
                        "claim": "Agentic RAG uses retrieval and agent control flow",
                        "cited_chunk_ids": ["agentic-rag.md:p3:c1"],
                        "verification_label": "supported",
                        "confidence": 0.94,
                        "reason": "Supported by chunk 1.",
                    }
                ],
                "reason": "Supported by chunk 1.",
            },
        )
    )

    result = run_agent(
        "How does it work?",
        chat_history=[{"role": "user", "content": "Tell me about Agentic RAG"}],
        llm=llm,
        retriever_fn=lambda query: [],
        tool_registry=registry,
        settings=get_settings(),
    )

    assert retriever_calls == ["rewritten agentic rag question"]
    assert len(verifier_calls) == 1
    assert verifier_calls[0]["question"] == "How does it work?"
    assert verifier_calls[0]["answer"] == (
        "Agentic RAG uses retrieval and agent control flow [1]."
    )
    assert verifier_calls[0]["claims"] == [
        {
            "claim_id": "c001",
            "claim": "Agentic RAG uses retrieval and agent control flow",
            "cited_chunk_ids": ["agentic-rag.md:p3:c1"],
        }
    ]
    assert verifier_calls[0]["documents"][0]["content"] == (
        "Agentic RAG uses retrieval and agent control flow."
    )
    assert verifier_calls[0]["documents"][0]["matched_queries"] == [
        "rewritten agentic rag question"
    ]
    assert result["answer"] == "Agentic RAG uses retrieval and agent control flow [1]."
    assert result["is_verified"] is True
    assert result["citation_verification_passed"] is True
    assert len(llm.prompts) == 4


def test_run_agent_uses_explicit_falsy_retriever_fn():
    from agent.graph import run_agent

    retriever = FalsyRetriever()
    llm = FakeLLM(
        [
            "rewritten agentic rag question",
            '{"relevant": true, "relevant_indices": [1], "reason": "matches"}',
            (
                '{"answer": "Agentic RAG uses retrieval and agent control flow [1].", '
                '"used_citation_indices": [1]}'
            ),
            (
                '{"claims": ['
                '{"claim_id": "c001", '
                '"claim": "Agentic RAG uses retrieval and agent control flow", '
                '"cited_chunk_ids": ["agentic-rag.md:p3:c1"]}'
                '], "reason": "Extracted one claim."}'
            ),
            (
                '{"results": ['
                '{"claim_id": "c001", '
                '"claim": "Agentic RAG uses retrieval and agent control flow", '
                '"cited_chunk_ids": ["agentic-rag.md:p3:c1"], '
                '"verification_label": "supported", "confidence": 0.94, '
                '"reason": "Supported by chunk 1."}'
                '], "reason": "Supported by chunk 1."}'
            ),
        ]
    )

    result = run_agent(
        "How does it work?",
        chat_history=[{"role": "user", "content": "Tell me about Agentic RAG"}],
        llm=llm,
        retriever_fn=retriever,
        settings=get_settings(),
    )

    assert retriever.calls == ["rewritten agentic rag question"]
    assert result["answer"] == "Agentic RAG uses retrieval and agent control flow [1]."
    assert result["is_verified"] is True


def test_run_agent_revises_unsupported_claim_then_finalizes():
    from agent.graph import run_agent

    llm = FakeLLM(
        [
            "rewritten query",
            '{"relevant": true, "relevant_indices": [1], "reason": "matches"}',
            (
                '{"answer": "Agentic RAG eliminates hallucination [1].", '
                '"used_citation_indices": [1]}'
            ),
            (
                '{"claims": ['
                '{"claim_id": "c001", "claim": "Agentic RAG eliminates hallucination", '
                '"cited_chunk_ids": ["chunk-1"]}'
                '], "reason": "claim"}'
            ),
            (
                '{"results": ['
                '{"claim_id": "c001", "claim": "Agentic RAG eliminates hallucination", '
                '"cited_chunk_ids": ["chunk-1"], "verification_label": "unsupported", '
                '"confidence": 0.2, "reason": "too strong"}'
                '], "reason": "unsupported"}'
            ),
            (
                '{"answer": "Agentic RAG can reduce hallucination risk [1].", '
                '"used_citation_indices": [1]}'
            ),
            (
                '{"claims": ['
                '{"claim_id": "c001", "claim": "Agentic RAG can reduce hallucination risk", '
                '"cited_chunk_ids": ["chunk-1"]}'
                '], "reason": "claim"}'
            ),
            (
                '{"results": ['
                '{"claim_id": "c001", "claim": "Agentic RAG can reduce hallucination risk", '
                '"cited_chunk_ids": ["chunk-1"], "verification_label": "supported", '
                '"confidence": 0.9, "reason": "supported"}'
                '], "reason": "supported"}'
            ),
        ]
    )
    documents = [
        {
            "content": "Citation checks can reduce hallucination risk.",
            "source": "notes.md",
            "chunk_id": "chunk-1",
        }
    ]

    result = run_agent(
        "What does Agentic RAG guarantee?",
        llm=llm,
        retriever_fn=lambda query: documents,
        settings=get_settings(),
    )

    assert result["answer"] == "Agentic RAG can reduce hallucination risk [1]."
    assert result["citation_revision_count"] == 1
    assert result["citation_verification_passed"] is True
    assert result["unsupported_claims"] == []
    assert result["is_verified"] is True


def test_run_agent_falls_back_when_revision_does_not_fix_unsupported_claims():
    from agent.graph import run_agent

    llm = FakeLLM(
        [
            "rewritten query",
            '{"relevant": true, "relevant_indices": [1], "reason": "matches"}',
            (
                '{"answer": "Agentic RAG eliminates hallucination [1].", '
                '"used_citation_indices": [1]}'
            ),
            (
                '{"claims": ['
                '{"claim_id": "c001", "claim": "Agentic RAG eliminates hallucination", '
                '"cited_chunk_ids": ["chunk-1"]}'
                '], "reason": "claim"}'
            ),
            (
                '{"results": ['
                '{"claim_id": "c001", "claim": "Agentic RAG eliminates hallucination", '
                '"cited_chunk_ids": ["chunk-1"], "verification_label": "unsupported", '
                '"confidence": 0.2, "reason": "too strong"}'
                '], "reason": "unsupported"}'
            ),
            (
                '{"answer": "Agentic RAG eliminates hallucination [1].", '
                '"used_citation_indices": [1]}'
            ),
            (
                '{"claims": ['
                '{"claim_id": "c001", "claim": "Agentic RAG eliminates hallucination", '
                '"cited_chunk_ids": ["chunk-1"]}'
                '], "reason": "claim"}'
            ),
            (
                '{"results": ['
                '{"claim_id": "c001", "claim": "Agentic RAG eliminates hallucination", '
                '"cited_chunk_ids": ["chunk-1"], "verification_label": "unsupported", '
                '"confidence": 0.2, "reason": "still too strong"}'
                '], "reason": "still unsupported"}'
            ),
        ]
    )
    documents = [
        {
            "content": "Citation checks can reduce hallucination risk.",
            "source": "notes.md",
            "chunk_id": "chunk-1",
        }
    ]

    result = run_agent(
        "What does Agentic RAG guarantee?",
        llm=llm,
        retriever_fn=lambda query: documents,
        settings=get_settings(),
    )

    assert "无法可靠回答" in result["answer"]
    assert result["citation_revision_count"] == 1
    assert result["citation_verification_passed"] is False
    assert result["unsupported_claims"][0]["verification_label"] == "unsupported"
    assert result["fallback_reason"] == "still unsupported"


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
