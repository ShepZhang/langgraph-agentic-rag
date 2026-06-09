"""Tests for the standalone baseline package."""

from __future__ import annotations

from baseline import run_naive_rag
from evaluation.baselines import run_naive_rag as compatibility_run_naive_rag


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_baseline_package_exports_run_naive_rag():
    assert run_naive_rag is compatibility_run_naive_rag


def test_standalone_naive_rag_payload_matches_evaluator_contract():
    llm = FakeLLM(
        [
            (
                '{"answer": "Naive RAG retrieves once and answers [1].", '
                '"used_citation_indices": [1]}'
            )
        ]
    )
    docs = [
        {
            "content": "Naive RAG retrieves once and answers.",
            "source": "notes.md",
            "page": None,
            "chunk_id": "notes.md:pNA:c1",
            "score": 0.42,
        }
    ]

    result = run_naive_rag("What is naive RAG?", retriever_fn=lambda query: docs, llm=llm)

    assert result["question"] == "What is naive RAG?"
    assert result["answer"] == "Naive RAG retrieves once and answers [1]."
    assert result["citations"][0]["source"] == "notes.md"
    assert result["retrieved_documents"] == docs
    assert result["relevant_documents"] == docs
    assert result["claims"] == []
    assert result["claim_verification"] == {}
    assert result["is_verified"] is False
    assert result["retry_count"] == 0
    assert result["fallback_reason"] == ""
    assert result["token_usage"] is None
    assert result["estimated_cost"] is None
