"""Tests for evaluation baselines."""

from __future__ import annotations

from evaluation.baselines import run_naive_rag


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_run_naive_rag_returns_agent_compatible_payload():
    llm = FakeLLM(
        [
            (
                '{"answer": "Naive RAG retrieves once and answers [1].", '
                '"used_citation_indices": [1]}'
            )
        ]
    )
    retrieved_documents = [
        {
            "content": "Naive RAG retrieves once and answers.",
            "source": "notes.md",
            "page": None,
            "chunk_id": "notes.md:pNA:c1",
            "score": 0.2,
        }
    ]

    result = run_naive_rag(
        "What is naive RAG?",
        retriever_fn=lambda query: retrieved_documents,
        llm=llm,
    )

    assert result["question"] == "What is naive RAG?"
    assert result["answer"] == "Naive RAG retrieves once and answers [1]."
    assert result["citations"] == [
        {
            "source": "notes.md",
            "page": None,
            "chunk_id": "notes.md:pNA:c1",
            "score": 0.2,
            "snippet": "Naive RAG retrieves once and answers.",
        }
    ]
    assert result["retrieved_documents"] == retrieved_documents
    assert result["relevant_documents"] == retrieved_documents
    assert result["retry_count"] == 0
    assert result["fallback_reason"] == ""
    assert "Original user question:\nWhat is naive RAG?" in llm.prompts[0]
    assert "Retrieval query:\nWhat is naive RAG?" in llm.prompts[0]


def test_run_naive_rag_falls_back_when_answer_lacks_citations():
    llm = FakeLLM(
        [
            (
                '{"answer": "Naive RAG retrieves once and answers.", '
                '"used_citation_indices": []}'
            )
        ]
    )
    retrieved_documents = [{"content": "context", "source": "notes.md"}]

    result = run_naive_rag(
        "What is naive RAG?",
        retriever_fn=lambda query: retrieved_documents,
        llm=llm,
    )

    assert "无法可靠回答" in result["answer"]
    assert result["citations"] == []
    assert "citation" in result["fallback_reason"].lower()


def test_run_naive_rag_allows_unable_to_answer_without_citations():
    llm = FakeLLM(
        [
            (
                '{"answer": "The provided documents do not contain enough information.", '
                '"used_citation_indices": []}'
            )
        ]
    )

    result = run_naive_rag(
        "What is unknown?",
        retriever_fn=lambda query: [{"content": "context", "source": "notes.md"}],
        llm=llm,
    )

    assert result["answer"] == "The provided documents do not contain enough information."
    assert result["citations"] == []
    assert result["fallback_reason"] == ""
