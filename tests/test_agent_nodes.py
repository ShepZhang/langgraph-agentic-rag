"""Tests for Agent workflow nodes."""

from __future__ import annotations

from agent.nodes import AgentNodes
from agent.state import create_initial_state


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


class FakeMessage:
    def __init__(self, content):
        self.content = content


def test_rewrite_query_node_rewrites_and_increments_count():
    llm = FakeLLM(["What is Agentic RAG?"])
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state(
        "What is it?",
        chat_history=[{"role": "user", "content": "Tell me about Agentic RAG"}],
    )

    update = nodes.rewrite_query_node(state)

    assert update["rewritten_question"] == "What is Agentic RAG?"
    assert update["rewrite_count"] == 1
    assert "Tell me about Agentic RAG" in llm.prompts[0]


def test_rewrite_query_node_uses_original_question_for_blank_response():
    llm = FakeLLM(["   "])
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("What is Agentic RAG?")

    update = nodes.rewrite_query_node(state)

    assert update["rewritten_question"] == "What is Agentic RAG?"
    assert update["rewrite_count"] == 1


def test_retrieve_node_uses_rewritten_question_when_present():
    calls = []

    def fake_retriever(query):
        calls.append(query)
        return [{"content": "context", "source": "notes.md"}]

    nodes = AgentNodes(llm=FakeLLM([]), retriever_fn=fake_retriever)
    state = create_initial_state("original")
    state["rewritten_question"] = "rewritten"

    update = nodes.retrieve_node(state)

    assert calls == ["rewritten"]
    assert update["documents"] == [{"content": "context", "source": "notes.md"}]


def test_grade_documents_node_marks_empty_docs_irrelevant_without_llm_call():
    llm = FakeLLM(['{"relevant": true}'])
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")

    update = nodes.grade_documents_node(state)

    assert update["is_relevant"] is False
    assert update["route"] == "rewrite_query"
    assert llm.prompts == []


def test_grade_documents_node_parses_relevance_json():
    llm = FakeLLM(['{"relevant": true, "reason": "enough context"}'])
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["documents"] = [{"content": "answer context", "source": "a.md"}]

    update = nodes.grade_documents_node(state)

    assert update["is_relevant"] is True
    assert update["route"] == "generate_answer"


def test_grade_documents_node_parses_fenced_relevance_json():
    llm = FakeLLM(['```json\n{"relevant": true, "reason": "enough context"}\n```'])
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["documents"] = [{"content": "answer context", "source": "a.md"}]

    update = nodes.grade_documents_node(state)

    assert update["is_relevant"] is True
    assert update["route"] == "generate_answer"


def test_grade_documents_node_parses_first_json_object_in_text():
    llm = FakeLLM(['I checked the chunks.\n{"relevant": true}\nUse them.'])
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["documents"] = [{"content": "answer context", "source": "a.md"}]

    update = nodes.grade_documents_node(state)

    assert update["is_relevant"] is True
    assert update["route"] == "generate_answer"


def test_grade_documents_node_treats_invalid_json_as_irrelevant():
    llm = FakeLLM(["not json"])
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["documents"] = [{"content": "context", "source": "a.md"}]

    update = nodes.grade_documents_node(state)

    assert update["is_relevant"] is False
    assert update["route"] == "rewrite_query"


def test_grade_documents_node_treats_missing_json_as_irrelevant():
    llm = FakeLLM(["Relevant: yes"])
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["documents"] = [{"content": "context", "source": "a.md"}]

    update = nodes.grade_documents_node(state)

    assert update["is_relevant"] is False
    assert update["route"] == "rewrite_query"


def test_generate_answer_node_returns_answer_and_grounded_citations():
    llm = FakeLLM(["Grounded answer."])
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["documents"] = [
        {
            "content": "context",
            "source": "paper.pdf",
            "page": 2,
            "chunk_id": "paper.pdf:p2:c1",
            "score": 0.8,
        }
    ]

    update = nodes.generate_answer_node(state)

    assert update["answer"] == "Grounded answer."
    assert update["citations"] == [
        {
            "source": "paper.pdf",
            "page": 2,
            "chunk_id": "paper.pdf:p2:c1",
            "score": 0.8,
        }
    ]


def test_generate_answer_node_extracts_text_from_content_blocks():
    llm = FakeLLM(
        [
            FakeMessage(
                [
                    {"type": "text", "text": "Grounded "},
                    {"type": "text", "text": "answer."},
                ]
            )
        ]
    )
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["documents"] = [{"content": "context", "source": "paper.pdf"}]

    update = nodes.generate_answer_node(state)

    assert update["answer"] == "Grounded answer."


def test_generate_answer_node_deduplicates_citations():
    llm = FakeLLM(["Grounded answer."])
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["documents"] = [
        {
            "content": "context",
            "source": "paper.pdf",
            "page": 2,
            "chunk_id": "paper.pdf:p2:c1",
            "score": 0.8,
        },
        {
            "content": "same context",
            "source": "paper.pdf",
            "page": 2,
            "chunk_id": "paper.pdf:p2:c1",
            "score": 0.8,
        },
    ]

    update = nodes.generate_answer_node(state)

    assert update["citations"] == [
        {
            "source": "paper.pdf",
            "page": 2,
            "chunk_id": "paper.pdf:p2:c1",
            "score": 0.8,
        }
    ]


def test_fallback_node_returns_clear_message():
    nodes = AgentNodes(llm=FakeLLM([]), retriever_fn=lambda query: [])

    update = nodes.fallback_node(create_initial_state("question"))

    assert "无法可靠回答" in update["answer"]
    assert update["citations"] == []
