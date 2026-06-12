from __future__ import annotations

import json

import pytest

from tools import ToolContext, ToolRegistry
from tools.citation_verifier_tool import CitationVerifierTool


class FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


def _base_arguments() -> dict[str, object]:
    return {
        "question": "What does the document say?",
        "answer": "The answer cites one fact [1].",
        "claims": [
            {
                "claim_id": "c001",
                "claim": "The answer cites one fact.",
                "cited_chunk_ids": ["chunk-1", "missing"],
            }
        ],
        "documents": [
            {
                "content": "The document says one fact.",
                "source": "notes.md",
                "chunk_id": "chunk-1",
            }
        ],
    }


def test_citation_verifier_returns_normalized_results_and_metadata():
    observer_records: list[dict[str, object]] = []
    llm = FakeLLM(
        [
            (
                '{"results": ['
                '{"claim_id": "c001", "claim": "The answer cites one fact.", '
                '"cited_chunk_ids": ["chunk-1", "missing"], '
                '"verification_label": "supported", "confidence": 0.95, '
                '"reason": "Direct support."}'
                '], "reason": "All claims supported."}'
            )
        ]
    )
    registry = ToolRegistry(call_observer=observer_records.append)
    registry.register(CitationVerifierTool(ToolContext(llm=llm)))

    arguments = _base_arguments()
    result = registry.invoke("verify_citations", arguments)

    assert result.success is True
    assert result.data == {
        "results": [
            {
                "claim_id": "c001",
                "claim": "The answer cites one fact.",
                "cited_chunk_ids": ["chunk-1"],
                "verification_label": "supported",
                "confidence": 0.95,
                "reason": "Direct support.",
            }
        ],
        "reason": "All claims supported.",
    }
    assert result.metadata["claim_count"] == 1
    assert result.metadata["unsupported_count"] == 0
    assert observer_records[0]["metadata"] == {
        "claim_count": 1,
        "unsupported_count": 0,
    }

    prompt = llm.prompts[0]
    assert "The document says one fact." in prompt
    assert json.dumps(arguments["claims"], ensure_ascii=False) in prompt
    assert "verify each extracted claim" in prompt.lower()


def test_citation_verifier_counts_partially_supported_and_unsupported_claims():
    llm = FakeLLM(
        [
            (
                '{"results": ['
                '{"claim_id": "c001", "claim": "A", "cited_chunk_ids": ["chunk-1"], '
                '"verification_label": "partially_supported", "confidence": 0.5, '
                '"reason": "Only partly supported."},'
                '{"claim_id": "c002", "claim": "B", "cited_chunk_ids": [], '
                '"verification_label": "supported", "confidence": 0.9, '
                '"reason": "No cited support."}'
                '], "reason": "Needs revision."}'
            )
        ]
    )
    registry = ToolRegistry()
    registry.register(CitationVerifierTool(ToolContext(llm=llm)))

    result = registry.invoke("verify_citations", _base_arguments())

    assert result.success is True
    assert result.metadata["claim_count"] == 2
    assert result.metadata["unsupported_count"] == 2


def test_citation_verifier_rejects_invalid_model_output():
    registry = ToolRegistry()
    registry.register(CitationVerifierTool(ToolContext(llm=FakeLLM(["not json"]))))

    result = registry.invoke("verify_citations", _base_arguments())

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "tool_execution_error"
    assert "invalid JSON" in result.error.message


def test_citation_verifier_requires_llm():
    registry = ToolRegistry()
    registry.register(CitationVerifierTool(ToolContext()))

    result = registry.invoke("verify_citations", _base_arguments())

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "tool_execution_error"
    assert "requires an LLM" in result.error.message


def test_citation_verifier_rejects_extra_input_fields():
    registry = ToolRegistry()
    registry.register(CitationVerifierTool(ToolContext(llm=FakeLLM(["{}"]))))

    result = registry.invoke("verify_citations", {**_base_arguments(), "extra": True})

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "tool_input_error"
