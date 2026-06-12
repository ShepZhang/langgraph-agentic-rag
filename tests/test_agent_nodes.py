"""Tests for Agent workflow nodes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agent.nodes import AgentNodes
from agent.state import create_initial_state
from tools import ToolContext, ToolRegistry
from tools.base import BaseTool


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
        results_by_query: dict[str, list[dict[str, Any]]] | None = None,
        error_message: str | None = None,
    ) -> None:
        super().__init__(context)
        self.calls = calls
        self.results_by_query = results_by_query or {}
        self.error_message = error_message

    def run(self, arguments: RetrieveArgs) -> list[dict[str, Any]]:
        self.calls.append(arguments.query)
        if self.error_message:
            raise RuntimeError(self.error_message)
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
        result: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> None:
        super().__init__(context)
        self.calls = calls
        self.result = result or {"results": [], "reason": ""}
        self.error_message = error_message

    def run(self, arguments: VerifyArgs) -> dict[str, Any]:
        self.calls.append(arguments.model_dump())
        if self.error_message:
            raise RuntimeError(self.error_message)
        return self.result


class FalsyToolRegistry(ToolRegistry):
    def __bool__(self) -> bool:
        return False


def test_accept_retrieved_documents_node_prepares_generation_context():
    nodes = AgentNodes(llm=FakeLLM([]), retriever_fn=lambda query: [])
    state = create_initial_state("What is RAG?")
    state["documents"] = [
        {
            "content": "RAG retrieves evidence.",
            "source": "notes.md",
            "chunk_id": "c1",
        }
    ]

    update = nodes.accept_retrieved_documents_node(state)

    assert update["relevant_documents"] == state["documents"]
    assert update["relevant_document_count"] == 1
    assert update["is_relevant"] is True
    assert update["grading_reason"] == "Retrieval grading disabled."
    assert update["route"] == "generate_answer"


def test_initial_rewrite_normalizes_query_without_incrementing_retry_count():
    llm = FakeLLM(["What is Agentic RAG?"])
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state(
        "What is it?",
        chat_history=[{"role": "user", "content": "Tell me about Agentic RAG"}],
    )

    update = nodes.rewrite_query_node(state)

    assert update["rewritten_question"] == "What is Agentic RAG?"
    assert update["current_query"] == "What is Agentic RAG?"
    assert update["previous_queries"] == ["What is Agentic RAG?"]
    assert update["rewrite_count"] == 0
    assert update["retry_count"] == 0
    assert "Tell me about Agentic RAG" in llm.prompts[0]


def test_initial_rewrite_records_structured_query_transform():
    llm = FakeLLM(
        [
            (
                '{"strategy": "multi_query", '
                '"rewritten_query": "What advantages does Agentic RAG have compared with naive RAG?", '
                '"expanded_queries": ["Agentic RAG benefits", "Agentic RAG reliability controls"], '
                '"sub_questions": ["ignored"], '
                '"reason": "The follow-up needs context and expansion."}'
            )
        ]
    )
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state(
        "What advantages does it have?",
        chat_history=[{"role": "user", "content": "We are discussing Agentic RAG."}],
    )

    update = nodes.rewrite_query_node(state)

    assert update["current_query"] == (
        "What advantages does Agentic RAG have compared with naive RAG?"
    )
    assert update["rewritten_question"] == update["current_query"]
    assert update["standalone_question"] == update["current_query"]
    assert update["query_transform_strategy"] == "multi_query"
    assert update["expanded_queries"] == [
        "Agentic RAG benefits",
        "Agentic RAG reliability controls",
    ]
    assert update["sub_questions"] == []
    assert update["query_transform"] == {
        "strategy": "multi_query",
        "rewritten_query": update["current_query"],
        "expanded_queries": [
            "Agentic RAG benefits",
            "Agentic RAG reliability controls",
        ],
        "sub_questions": [],
        "reason": "The follow-up needs context and expansion.",
    }
    assert "Return JSON only" in llm.prompts[0]
    assert "multi_query" in llm.prompts[0]


def test_initial_rewrite_uses_original_question_for_blank_response():
    llm = FakeLLM(["   "])
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("What is Agentic RAG?")

    update = nodes.rewrite_query_node(state)

    assert update["rewritten_question"] == "What is Agentic RAG?"
    assert update["current_query"] == "What is Agentic RAG?"
    assert update["standalone_question"] == "What is Agentic RAG?"
    assert update["query_transform_strategy"] == "rewrite"
    assert update["rewrite_count"] == 0
    assert update["retry_count"] == 0


def test_retry_rewrite_uses_failure_context_and_increments_retry_count():
    llm = FakeLLM(["agentic rag retrieval grading"])
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("How does it improve reliability?")
    state["current_query"] = "agentic rag"
    state["previous_queries"] = ["agentic rag"]
    state["retrieval_attempt"] = 1
    state["retry_count"] = 0
    state["rewrite_count"] = 0
    state["grading_reason"] = "The retrieved chunk only defines RAG but not reliability."
    state["documents"] = [
        {
            "content": "RAG combines retrieval with generation.",
            "source": "notes.md",
            "chunk_id": "notes.md:c1",
        }
    ]

    update = nodes.rewrite_query_node(state)

    assert update["current_query"] == "agentic rag retrieval grading"
    assert update["previous_queries"] == ["agentic rag", "agentic rag retrieval grading"]
    assert update["retry_count"] == 1
    assert update["rewrite_count"] == 1
    assert "Previous retrieval query" in llm.prompts[0]
    assert "agentic rag" in llm.prompts[0]
    assert "The retrieved chunk only defines RAG" in llm.prompts[0]
    assert "RAG combines retrieval with generation" in llm.prompts[0]


def test_retry_rewrite_includes_partial_relevance_recovery_context():
    llm = FakeLLM(["agentic rag baseline comparison fallback"])
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("How is Agentic RAG better than naive RAG?")
    state["current_query"] = "agentic rag reliability"
    state["previous_queries"] = ["agentic rag reliability"]
    state["retrieval_attempt"] = 1
    state["retry_count"] = 0
    state["grading_reason"] = "Only partial evidence found."
    state["documents"] = [
        {
            "content": "Agentic RAG includes retrieval grading and fallback.",
            "source": "agentic.md",
            "chunk_id": "agentic:c1",
        },
        {
            "content": "Naive RAG retrieves once and generates directly.",
            "source": "baseline.md",
            "chunk_id": "baseline:c1",
        },
    ]
    state["document_grades"] = [
        {
            "document_index": 1,
            "relevance": "partially_relevant",
            "confidence": 0.73,
            "reason": "Related but missing the requested comparison.",
        }
    ]
    state["partial_relevance_recovery"] = {
        "triggered": True,
        "action": "query_refinement",
        "reason": (
            "Only partially relevant chunks were found; "
            "refine query to target missing evidence."
        ),
        "partial_document_indices": [1],
    }

    update = nodes.rewrite_query_node(state)

    assert update["current_query"] == "agentic rag baseline comparison fallback"
    assert "Partial relevance recovery" in llm.prompts[0]
    assert "Only partially relevant chunks were found" in llm.prompts[0]
    assert "Partially related context" in llm.prompts[0]
    assert "Related but missing the requested comparison" in llm.prompts[0]
    assert "Agentic RAG includes retrieval grading and fallback." in llm.prompts[0]


def test_retrieve_node_uses_current_query_and_increments_retrieval_attempt():
    calls = []

    def fake_retriever(query):
        calls.append(query)
        return [{"content": "context", "source": "notes.md"}]

    nodes = AgentNodes(llm=FakeLLM([]), retriever_fn=fake_retriever)
    state = create_initial_state("original")
    state["current_query"] = "rewritten"
    state["retrieval_attempt"] = 1

    update = nodes.retrieve_node(state)

    assert calls == ["rewritten"]
    assert update["documents"][0]["content"] == "context"
    assert update["documents"][0]["source"] == "notes.md"
    assert update["documents"][0]["matched_queries"] == ["rewritten"]
    assert update["documents"][0]["retrieval_query_count"] == 1
    assert update["documents"][0]["multi_query_rank"] == 1
    assert update["retrieval_queries"] == ["rewritten"]
    assert update["multi_query_used"] is False
    assert update["multi_query_result_count"] == 1
    assert update["retrieval_attempt"] == 2


def test_retrieve_node_executes_and_merges_multi_query_retrieval():
    calls = []

    def fake_retriever(query):
        calls.append(query)
        if query == "Agentic RAG advantages":
            return [
                {
                    "content": "Agentic RAG uses grading.",
                    "source": "notes.md",
                    "chunk_id": "notes:c1",
                },
                {
                    "content": "Hybrid retrieval combines signals.",
                    "source": "retrieval.md",
                    "chunk_id": "retrieval:c2",
                },
            ]
        return [
            {
                "content": "Agentic RAG uses grading.",
                "source": "notes.md",
                "chunk_id": "notes:c1",
            },
            {
                "content": "Fallback handles missing evidence.",
                "source": "notes.md",
                "chunk_id": "notes:c3",
            },
        ]

    nodes = AgentNodes(llm=FakeLLM([]), retriever_fn=fake_retriever)
    state = create_initial_state("original")
    state["current_query"] = "Agentic RAG advantages"
    state["query_transform_strategy"] = "multi_query"
    state["expanded_queries"] = ["reliability controls", "Agentic RAG advantages"]

    update = nodes.retrieve_node(state)

    assert calls == ["Agentic RAG advantages", "reliability controls"]
    assert update["retrieval_queries"] == [
        "Agentic RAG advantages",
        "reliability controls",
    ]
    assert update["multi_query_used"] is True
    assert update["multi_query_result_count"] == 3
    assert [document["chunk_id"] for document in update["documents"]] == [
        "notes:c1",
        "retrieval:c2",
        "notes:c3",
    ]
    assert update["documents"][0]["matched_queries"] == [
        "Agentic RAG advantages",
        "reliability controls",
    ]
    assert update["documents"][0]["multi_query_rank"] == 1


def test_retrieve_node_uses_supplied_tool_registry():
    calls: list[str] = []
    registry = ToolRegistry()
    registry.register(
        RecordingRetrieverTool(
            ToolContext(),
            calls=calls,
            results_by_query={
                "rewritten": [
                    {
                        "content": "context",
                        "source": "notes.md",
                        "chunk_id": "notes:c1",
                    }
                ]
            },
        )
    )
    nodes = AgentNodes(llm=FakeLLM([]), retriever_fn=lambda query: [], tool_registry=registry)
    state = create_initial_state("original")
    state["current_query"] = "rewritten"

    update = nodes.retrieve_node(state)

    assert calls == ["rewritten"]
    assert update["documents"] == [
        {
            "content": "context",
            "source": "notes.md",
            "chunk_id": "notes:c1",
            "matched_queries": ["rewritten"],
            "retrieval_query_count": 1,
            "multi_query_rank": 1,
        }
    ]


def test_retrieve_node_returns_grading_reason_when_registry_tool_fails():
    calls: list[str] = []
    registry = ToolRegistry()
    registry.register(
        RecordingRetrieverTool(
            ToolContext(),
            calls=calls,
            error_message="retriever backend unavailable",
        )
    )
    nodes = AgentNodes(llm=FakeLLM([]), retriever_fn=lambda query: [], tool_registry=registry)
    state = create_initial_state("original")
    state["current_query"] = "rewritten"
    state["retrieval_attempt"] = 2

    update = nodes.retrieve_node(state)

    assert calls == ["rewritten"]
    assert update["documents"] == []
    assert "Retriever tool failed: retriever backend unavailable" == update["grading_reason"]
    assert update["retrieval_attempt"] == 3


def test_retrieve_node_uses_supplied_falsy_tool_registry():
    calls: list[str] = []
    registry = FalsyToolRegistry()
    registry.register(
        RecordingRetrieverTool(
            ToolContext(),
            calls=calls,
            results_by_query={
                "rewritten": [{"content": "context", "source": "notes.md"}]
            },
        )
    )
    nodes = AgentNodes(llm=FakeLLM([]), tool_registry=registry)
    state = create_initial_state("original")
    state["current_query"] = "rewritten"

    update = nodes.retrieve_node(state)

    assert calls == ["rewritten"]
    assert update["documents"][0]["matched_queries"] == ["rewritten"]


def test_retrieve_node_falls_back_when_registry_returns_invalid_success_data():
    calls: list[str] = []
    registry = ToolRegistry()
    registry.register(
        RecordingRetrieverTool(
            ToolContext(),
            calls=calls,
            results_by_query={"rewritten": ["not-a-document"]},
        )
    )
    nodes = AgentNodes(llm=FakeLLM([]), tool_registry=registry)
    state = create_initial_state("original")
    state["current_query"] = "rewritten"

    update = nodes.retrieve_node(state)

    assert calls == ["rewritten"]
    assert update["documents"] == []
    assert update["grading_reason"] == (
        "Retriever tool failed: Retriever tool returned invalid data: expected list[dict]."
    )


def test_retrieve_node_falls_back_when_document_content_is_none():
    calls: list[str] = []
    registry = ToolRegistry()
    registry.register(
        RecordingRetrieverTool(
            ToolContext(),
            calls=calls,
            results_by_query={
                "rewritten": [{"content": None, "source": "notes.md", "chunk_id": "c1"}]
            },
        )
    )
    nodes = AgentNodes(llm=FakeLLM([]), tool_registry=registry)
    state = create_initial_state("original")
    state["current_query"] = "rewritten"

    update = nodes.retrieve_node(state)

    assert calls == ["rewritten"]
    assert update["documents"] == []
    assert update["grading_reason"] == (
        "Retriever tool failed: Retriever tool returned invalid data: expected list[dict]."
    )


def test_grade_documents_node_marks_empty_docs_irrelevant_without_llm_call():
    llm = FakeLLM(['{"relevant": true}'])
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")

    update = nodes.grade_documents_node(state)

    assert update["is_relevant"] is False
    assert update["relevant_documents"] == []
    assert update["document_grades"] == []
    assert update["relevant_document_count"] == 0
    assert update["partial_document_count"] == 0
    assert update["max_relevance_confidence"] == 0.0
    assert update["partial_relevance_recovery"] == {
        "triggered": False,
        "action": "none",
        "reason": "",
        "partial_document_indices": [],
    }
    assert "No documents" in update["grading_reason"]
    assert update["route"] == "rewrite_query"
    assert llm.prompts == []


def test_grade_documents_node_parses_relevant_indices_and_filters_documents():
    llm = FakeLLM(
        [
            (
                '{"relevant": true, "relevant_indices": [2], '
                '"reason": "Chunk 2 directly answers the question."}'
            )
        ]
    )
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["current_query"] = "rewritten question"
    state["documents"] = [
        {"content": "unrelated context", "source": "a.md"},
        {"content": "answer context", "source": "b.md"},
    ]

    update = nodes.grade_documents_node(state)

    assert update["is_relevant"] is True
    assert update["relevant_documents"] == [{"content": "answer context", "source": "b.md"}]
    assert update["document_grades"] == [
        {
            "document_index": 2,
            "relevance": "relevant",
            "confidence": 1.0,
            "reason": "Chunk 2 directly answers the question.",
        }
    ]
    assert update["relevant_document_count"] == 1
    assert update["partial_document_count"] == 0
    assert update["max_relevance_confidence"] == 1.0
    assert update["partial_relevance_recovery"] == {
        "triggered": False,
        "action": "none",
        "reason": "",
        "partial_document_indices": [],
    }
    assert update["grading_reason"] == "Chunk 2 directly answers the question."
    assert update["route"] == "generate_answer"
    assert "Original user question:\nquestion" in llm.prompts[0]
    assert "Retrieval query:\nrewritten question" in llm.prompts[0]
    assert "grade the retrieved chunks against the original user question" in llm.prompts[0]


def test_grade_documents_node_records_structured_document_grades():
    llm = FakeLLM(
        [
            (
                '{"grades": ['
                '{"document_index": 1, "relevance": "partially_relevant", '
                '"confidence": 0.62, "reason": "Mentions reliability but lacks answer."},'
                '{"document_index": 2, "relevance": "relevant", '
                '"confidence": 0.87, "reason": "Directly explains grading."},'
                '{"document_index": 3, "relevance": "irrelevant", '
                '"confidence": 0.11, "reason": "Wrong topic."}'
                '], "reason": "Chunk 2 directly answers."}'
            )
        ]
    )
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("How does grading improve reliability?")
    state["current_query"] = "retrieval grading reliability"
    state["documents"] = [
        {"content": "Reliability overview", "source": "a.md"},
        {"content": "Grading filters weak chunks", "source": "b.md"},
        {"content": "Office policy", "source": "c.md"},
    ]

    update = nodes.grade_documents_node(state)

    assert update["is_relevant"] is True
    assert update["relevant_documents"] == [
        {"content": "Grading filters weak chunks", "source": "b.md"}
    ]
    assert update["document_grades"] == [
        {
            "document_index": 1,
            "relevance": "partially_relevant",
            "confidence": 0.62,
            "reason": "Mentions reliability but lacks answer.",
        },
        {
            "document_index": 2,
            "relevance": "relevant",
            "confidence": 0.87,
            "reason": "Directly explains grading.",
        },
        {
            "document_index": 3,
            "relevance": "irrelevant",
            "confidence": 0.11,
            "reason": "Wrong topic.",
        },
    ]
    assert update["relevant_document_count"] == 1
    assert update["partial_document_count"] == 1
    assert update["max_relevance_confidence"] == 0.87
    assert update["partial_relevance_recovery"] == {
        "triggered": False,
        "action": "none",
        "reason": "",
        "partial_document_indices": [],
    }
    assert update["route"] == "generate_answer"


def test_grade_documents_node_retries_when_only_partially_relevant():
    llm = FakeLLM(
        [
            (
                '{"grades": ['
                '{"document_index": 1, "relevance": "partially_relevant", '
                '"confidence": 0.91, "reason": "Mentions the topic only."}'
                '], "reason": "No direct answer."}'
            )
        ]
    )
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["documents"] = [{"content": "related context", "source": "a.md"}]

    update = nodes.grade_documents_node(state)

    assert update["is_relevant"] is False
    assert update["relevant_documents"] == []
    assert update["relevant_document_count"] == 0
    assert update["partial_document_count"] == 1
    assert update["max_relevance_confidence"] == 0.91
    assert update["partial_relevance_recovery"] == {
        "triggered": True,
        "action": "query_refinement",
        "reason": (
            "Only partially relevant chunks were found; "
            "refine query to target missing evidence."
        ),
        "partial_document_indices": [1],
    }
    assert update["route"] == "rewrite_query"


def test_grade_documents_node_parses_fenced_relevance_json():
    llm = FakeLLM(
        [
            (
                "```json\n"
                '{"relevant": true, "relevant_indices": [1], "reason": "enough context"}'
                "\n```"
            )
        ]
    )
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["documents"] = [{"content": "answer context", "source": "a.md"}]

    update = nodes.grade_documents_node(state)

    assert update["is_relevant"] is True
    assert update["relevant_documents"] == [{"content": "answer context", "source": "a.md"}]
    assert update["route"] == "generate_answer"


def test_grade_documents_node_parses_first_json_object_in_text():
    llm = FakeLLM(
        [
            (
                'I checked the chunks.\n{"relevant": true, "relevant_indices": [1], '
                '"reason": "direct"}\nUse them.'
            )
        ]
    )
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["documents"] = [{"content": "answer context", "source": "a.md"}]

    update = nodes.grade_documents_node(state)

    assert update["is_relevant"] is True
    assert update["relevant_documents"] == [{"content": "answer context", "source": "a.md"}]
    assert update["route"] == "generate_answer"


def test_grade_documents_node_ignores_out_of_range_relevant_indices():
    llm = FakeLLM(
        [
            (
                '{"relevant": true, "relevant_indices": [3], '
                '"reason": "index does not exist"}'
            )
        ]
    )
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["documents"] = [{"content": "context", "source": "a.md"}]

    update = nodes.grade_documents_node(state)

    assert update["is_relevant"] is False
    assert update["relevant_documents"] == []
    assert update["route"] == "rewrite_query"


def test_grade_documents_node_treats_invalid_json_as_irrelevant():
    llm = FakeLLM(["not json"])
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["documents"] = [{"content": "context", "source": "a.md"}]

    update = nodes.grade_documents_node(state)

    assert update["is_relevant"] is False
    assert update["relevant_documents"] == []
    assert "parse" in update["grading_reason"].lower()
    assert update["route"] == "rewrite_query"


def test_grade_documents_node_treats_missing_json_as_irrelevant():
    llm = FakeLLM(["Relevant: yes"])
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["documents"] = [{"content": "context", "source": "a.md"}]

    update = nodes.grade_documents_node(state)

    assert update["is_relevant"] is False
    assert update["relevant_documents"] == []
    assert update["route"] == "rewrite_query"


def test_generate_answer_node_uses_relevant_documents_and_selected_citations_only():
    llm = FakeLLM(
        [
            (
                '{"answer": "Grounded answer with [2].", '
                '"used_citation_indices": [2]}'
            )
        ]
    )
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("What reliability controls does Agentic RAG use?")
    state["current_query"] = "agentic rag retrieval grading fallback"
    state["documents"] = [
        {
            "content": "irrelevant context",
            "source": "other.pdf",
            "page": 9,
            "chunk_id": "other.pdf:p9:c1",
            "score": 0.7,
        }
    ]
    state["relevant_documents"] = [
        {
            "content": "first relevant context",
            "source": "paper.pdf",
            "page": 1,
            "chunk_id": "paper.pdf:p1:c1",
            "score": 0.7,
        },
        {
            "content": "second relevant context",
            "source": "paper.pdf",
            "page": 2,
            "chunk_id": "paper.pdf:p2:c1",
            "score": 0.8,
        }
    ]

    update = nodes.generate_answer_node(state)

    assert update["draft_answer"] == "Grounded answer with [2]."
    assert update["answer"] == ""
    assert update["used_citation_indices"] == [2]
    assert update["cited_documents"] == [state["relevant_documents"][1]]
    assert update["citations"] == [
        {
            "source": "paper.pdf",
            "page": 2,
            "chunk_id": "paper.pdf:p2:c1",
            "score": 0.8,
            "snippet": "second relevant context",
        }
    ]
    assert "irrelevant context" not in llm.prompts[0]
    assert "second relevant context" in llm.prompts[0]
    assert (
        "Original user question:\nWhat reliability controls does Agentic RAG use?"
        in llm.prompts[0]
    )
    assert "Retrieval query:\nagentic rag retrieval grading fallback" in llm.prompts[0]
    assert "answer the original user question" in llm.prompts[0]
    assert len(llm.prompts) == 1
    assert update["route"] == "extract_claims"


def test_generate_answer_node_extracts_text_from_content_blocks():
    llm = FakeLLM(
        [
            FakeMessage(
                [
                    {
                        "type": "text",
                        "text": '{"answer": "Grounded ',
                        },
                        {
                            "type": "text",
                            "text": 'answer [1].", "used_citation_indices": [1]}',
                        },
                ]
            ),
        ]
    )
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["relevant_documents"] = [{"content": "context", "source": "paper.pdf"}]

    update = nodes.generate_answer_node(state)

    assert update["draft_answer"] == "Grounded answer [1]."
    assert update["route"] == "extract_claims"


def test_generate_answer_node_deduplicates_selected_citations():
    llm = FakeLLM(
        [
            (
                '{"answer": "Grounded answer [1] [2].", '
                '"used_citation_indices": [1, 2]}'
            )
        ]
    )
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["relevant_documents"] = [
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
            "snippet": "context",
        }
    ]
    assert update["route"] == "extract_claims"


def test_generate_answer_node_keeps_valid_citation_and_ignores_invalid_indices():
    llm = FakeLLM(
        [
            (
                '{"answer": "Grounded answer [1].", '
                '"used_citation_indices": [1, 3]}'
            )
        ]
    )
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["relevant_documents"] = [{"content": "context", "source": "paper.pdf"}]

    update = nodes.generate_answer_node(state)

    assert update["draft_answer"] == "Grounded answer [1]."
    assert update["citations"] == [
        {
            "source": "paper.pdf",
            "page": None,
            "chunk_id": None,
            "score": None,
            "snippet": "context",
        }
    ]
    assert update["route"] == "extract_claims"


def test_generate_answer_node_defers_unsupported_claims_to_verification_nodes():
    llm = FakeLLM(
        [
            (
                '{"answer": "Agentic RAG eliminates hallucination [1].", '
                '"used_citation_indices": [1]}'
            ),
            (
                '{"verified": false, "claims": ['
                '{"claim": "Agentic RAG eliminates hallucination", '
                '"supported": false, "citation_indices": []}'
                '], "reason": "The cited chunk only says it reduces hallucination risk."}'
            ),
        ]
    )
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["relevant_documents"] = [
        {"content": "Citation-aware generation helps reduce hallucination risk.", "source": "paper.pdf"}
    ]

    update = nodes.generate_answer_node(state)

    assert update["draft_answer"] == "Agentic RAG eliminates hallucination [1]."
    assert update["citations"] != []
    assert "fallback_reason" not in update
    assert len(llm.prompts) == 1
    assert update["route"] == "extract_claims"


def test_generate_answer_node_does_not_call_legacy_claim_verifier():
    llm = FakeLLM(
        [
            (
                '{"answer": "Agentic RAG uses retrieval grading [1].", '
                '"used_citation_indices": [1]}'
            ),
            "not json",
        ]
    )
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["relevant_documents"] = [
        {"content": "Agentic RAG uses retrieval grading.", "source": "paper.pdf"}
    ]

    update = nodes.generate_answer_node(state)

    assert update["draft_answer"] == "Agentic RAG uses retrieval grading [1]."
    assert update["citations"] != []
    assert "fallback_reason" not in update
    assert len(llm.prompts) == 1
    assert update["route"] == "extract_claims"


def test_generate_answer_node_falls_back_for_normal_answer_without_citations():
    llm = FakeLLM(
        [
            (
                '{"answer": "Agentic RAG uses retrieval grading to reduce hallucination.", '
                '"used_citation_indices": []}'
            )
        ]
    )
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["relevant_documents"] = [{"content": "context", "source": "paper.pdf"}]

    update = nodes.generate_answer_node(state)

    assert "无法可靠回答" in update["answer"]
    assert update["citations"] == []
    assert "citation" in update["fallback_reason"].lower()


def test_generate_answer_node_falls_back_when_answer_markers_do_not_match_used_indices():
    llm = FakeLLM(
        [
            (
                '{"answer": "Agentic RAG uses retrieval grading [1].", '
                '"used_citation_indices": [1, 2]}'
            )
        ]
    )
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["relevant_documents"] = [
        {"content": "retrieval grading context", "source": "paper.pdf"},
        {"content": "fallback context", "source": "paper.pdf"},
    ]

    update = nodes.generate_answer_node(state)

    assert "无法可靠回答" in update["answer"]
    assert update["citations"] == []
    assert "citation marker" in update["fallback_reason"].lower()
    assert len(llm.prompts) == 1


def test_generate_answer_node_falls_back_when_answer_marker_is_out_of_range():
    llm = FakeLLM(
        [
            (
                '{"answer": "Agentic RAG uses retrieval grading [3].", '
                '"used_citation_indices": [1]}'
            )
        ]
    )
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["relevant_documents"] = [{"content": "context", "source": "paper.pdf"}]

    update = nodes.generate_answer_node(state)

    assert "无法可靠回答" in update["answer"]
    assert update["citations"] == []
    assert "citation marker" in update["fallback_reason"].lower()
    assert len(llm.prompts) == 1


def test_generate_answer_node_falls_back_when_all_citation_indices_are_invalid():
    llm = FakeLLM(
        [
            (
                '{"answer": "Agentic RAG uses retrieval grading [9].", '
                '"used_citation_indices": [9]}'
            )
        ]
    )
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["relevant_documents"] = [{"content": "context", "source": "paper.pdf"}]

    update = nodes.generate_answer_node(state)

    assert "无法可靠回答" in update["answer"]
    assert update["citations"] == []
    assert "citation" in update["fallback_reason"].lower()


def test_generate_answer_node_allows_unable_to_answer_without_citations():
    llm = FakeLLM(
        [
            (
                '{"answer": "I cannot answer from the current documents.", '
                '"used_citation_indices": []}'
            )
        ]
    )
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["relevant_documents"] = [{"content": "context", "source": "paper.pdf"}]

    update = nodes.generate_answer_node(state)

    assert update["draft_answer"] == "I cannot answer from the current documents."
    assert update["citations"] == []
    assert update["citation_verification_skipped"] is True
    assert update["citation_verification_passed"] is False
    assert update["route"] == "finalize_answer"
    assert "fallback_reason" not in update


def test_generate_answer_node_falls_back_for_invalid_answer_json():
    llm = FakeLLM(["Grounded answer without JSON."])
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["relevant_documents"] = [{"content": "context", "source": "paper.pdf"}]

    update = nodes.generate_answer_node(state)

    assert "无法可靠回答" in update["answer"]
    assert update["citations"] == []
    assert update["fallback_reason"] == "Answer generation returned invalid JSON."


def test_generate_answer_node_falls_back_without_relevant_documents():
    nodes = AgentNodes(llm=FakeLLM([]), retriever_fn=lambda query: [])

    update = nodes.generate_answer_node(create_initial_state("question"))

    assert "无法可靠回答" in update["answer"]
    assert update["citations"] == []
    assert update["fallback_reason"] == "No relevant documents available for answer generation."


def test_extract_claims_node_writes_structured_claims():
    llm = FakeLLM(
        [
            (
                '{"claims": ['
                '{"claim_id": "c001", "claim": "Agentic RAG uses retrieval grading.", '
                '"cited_chunk_ids": ["chunk-1"]}'
                '], "reason": "one claim"}'
            )
        ]
    )
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("What does Agentic RAG use?")
    state["draft_answer"] = "Agentic RAG uses retrieval grading [1]."
    state["cited_documents"] = [
        {
            "content": "Agentic RAG uses retrieval grading.",
            "source": "paper.pdf",
            "chunk_id": "chunk-1",
        }
    ]

    update = nodes.extract_claims_node(state)

    assert update["claims"] == [
        {
            "claim_id": "c001",
            "claim": "Agentic RAG uses retrieval grading.",
            "cited_chunk_ids": ["chunk-1"],
        }
    ]
    assert update["claim_verification_reason"] == "one claim"
    assert update["route"] == "verify_citations"
    assert "claim_id" in llm.prompts[0]
    assert "Agentic RAG uses retrieval grading [1]." in llm.prompts[0]


def test_verify_citations_node_passes_supported_claims():
    llm = FakeLLM(
        [
            (
                '{"results": ['
                '{"claim_id": "c001", "claim": "Agentic RAG uses retrieval grading.", '
                '"cited_chunk_ids": ["chunk-1"], "verification_label": "supported", '
                '"confidence": 0.91, "reason": "Directly supported."}'
                '], "reason": "All claims supported."}'
            )
        ]
    )
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("What does Agentic RAG use?")
    state["draft_answer"] = "Agentic RAG uses retrieval grading [1]."
    state["claims"] = [
        {
            "claim_id": "c001",
            "claim": "Agentic RAG uses retrieval grading.",
            "cited_chunk_ids": ["chunk-1"],
        }
    ]
    state["cited_documents"] = [
        {
            "content": "Agentic RAG uses retrieval grading.",
            "source": "paper.pdf",
            "chunk_id": "chunk-1",
        }
    ]

    update = nodes.verify_citations_node(state)

    assert update["claim_verification_results"] == [
        {
            "claim_id": "c001",
            "claim": "Agentic RAG uses retrieval grading.",
            "cited_chunk_ids": ["chunk-1"],
            "verification_label": "supported",
            "confidence": 0.91,
            "reason": "Directly supported.",
        }
    ]
    assert update["unsupported_claims"] == []
    assert update["claim_verification"]["verified"] is True
    assert update["claim_verification_reason"] == "All claims supported."
    assert update["citation_verification_passed"] is True
    assert update["route"] == "finalize_answer"


def test_verify_citations_node_collects_unsupported_claims():
    llm = FakeLLM(
        [
            (
                '{"results": ['
                '{"claim_id": "c001", "claim": "Agentic RAG eliminates hallucination.", '
                '"cited_chunk_ids": ["chunk-1"], "verification_label": "unsupported", '
                '"confidence": 0.2, "reason": "The chunk only says reduce risk."}'
                '], "reason": "Unsupported claim found."}'
            )
        ]
    )
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("What does Agentic RAG guarantee?")
    state["draft_answer"] = "Agentic RAG eliminates hallucination [1]."
    state["claims"] = [
        {
            "claim_id": "c001",
            "claim": "Agentic RAG eliminates hallucination.",
            "cited_chunk_ids": ["chunk-1"],
        }
    ]
    state["cited_documents"] = [
        {
            "content": "Citation checks can reduce hallucination risk.",
            "source": "paper.pdf",
            "chunk_id": "chunk-1",
        }
    ]

    update = nodes.verify_citations_node(state)

    assert update["citation_verification_passed"] is False
    assert update["unsupported_claims"] == [
        {
            "claim_id": "c001",
            "claim": "Agentic RAG eliminates hallucination.",
            "cited_chunk_ids": ["chunk-1"],
            "verification_label": "unsupported",
            "confidence": 0.2,
            "reason": "The chunk only says reduce risk.",
        }
    ]
    assert update["claim_verification"]["verified"] is False
    assert update["claim_verification_reason"] == "Unsupported claim found."
    assert update["route"] == "revise_answer"


def test_verify_citations_node_uses_supplied_tool_registry():
    calls: list[dict[str, Any]] = []
    registry = ToolRegistry()
    registry.register(
        RecordingVerifierTool(
            ToolContext(),
            calls=calls,
            result={
                "results": [
                    {
                        "claim_id": "c001",
                        "claim": "Agentic RAG uses retrieval grading.",
                        "cited_chunk_ids": ["chunk-1"],
                        "verification_label": "supported",
                        "confidence": 0.93,
                        "reason": "Directly supported.",
                    }
                ],
                "reason": "All claims supported.",
            },
        )
    )
    nodes = AgentNodes(llm=FakeLLM([]), retriever_fn=lambda query: [], tool_registry=registry)
    state = create_initial_state("What does Agentic RAG use?")
    state["draft_answer"] = "Agentic RAG uses retrieval grading [1]."
    state["claims"] = [
        {
            "claim_id": "c001",
            "claim": "Agentic RAG uses retrieval grading.",
            "cited_chunk_ids": ["chunk-1"],
        }
    ]
    state["cited_documents"] = [
        {
            "content": "Agentic RAG uses retrieval grading.",
            "source": "paper.pdf",
            "chunk_id": "chunk-1",
        }
    ]

    update = nodes.verify_citations_node(state)

    assert calls == [
        {
            "question": "What does Agentic RAG use?",
            "answer": "Agentic RAG uses retrieval grading [1].",
            "claims": state["claims"],
            "documents": state["cited_documents"],
        }
    ]
    assert update["citation_verification_passed"] is True
    assert update["claim_verification_reason"] == "All claims supported."
    assert update["route"] == "finalize_answer"


def test_verify_citations_node_falls_back_when_registry_tool_fails():
    calls: list[dict[str, Any]] = []
    registry = ToolRegistry()
    registry.register(
        RecordingVerifierTool(
            ToolContext(),
            calls=calls,
            error_message="verifier backend unavailable",
        )
    )
    nodes = AgentNodes(llm=FakeLLM([]), retriever_fn=lambda query: [], tool_registry=registry)
    state = create_initial_state("What does Agentic RAG use?")
    state["draft_answer"] = "Agentic RAG uses retrieval grading [1]."
    state["claims"] = [
        {
            "claim_id": "c001",
            "claim": "Agentic RAG uses retrieval grading.",
            "cited_chunk_ids": ["chunk-1"],
        }
    ]
    state["cited_documents"] = [
        {
            "content": "Agentic RAG uses retrieval grading.",
            "source": "paper.pdf",
            "chunk_id": "chunk-1",
        }
    ]

    update = nodes.verify_citations_node(state)

    assert len(calls) == 1
    assert update["route"] == "fallback"
    assert update["fallback_reason"] == (
        "Citation verification tool failed: verifier backend unavailable"
    )
    assert update["citation_verification_passed"] is False


def test_verify_citations_node_falls_back_when_registry_returns_invalid_success_data():
    calls: list[dict[str, Any]] = []
    registry = ToolRegistry()
    registry.register(
        RecordingVerifierTool(
            ToolContext(),
            calls=calls,
            result={"results": [{}], "reason": "bad"},
        )
    )
    nodes = AgentNodes(llm=FakeLLM([]), tool_registry=registry)
    state = create_initial_state("What does Agentic RAG use?")
    state["draft_answer"] = "Agentic RAG uses retrieval grading [1]."
    state["claims"] = [
        {
            "claim_id": "c001",
            "claim": "Agentic RAG uses retrieval grading.",
            "cited_chunk_ids": ["chunk-1"],
        }
    ]
    state["cited_documents"] = [
        {
            "content": "Agentic RAG uses retrieval grading.",
            "source": "paper.pdf",
            "chunk_id": "chunk-1",
        }
    ]

    update = nodes.verify_citations_node(state)

    assert len(calls) == 1
    assert update["route"] == "fallback"
    assert update["fallback_reason"] == (
        "Citation verification tool returned invalid data."
    )
    assert update["citation_verification_passed"] is False


def test_verify_citations_node_falls_back_for_unknown_chunk_supported_and_bad_confidence():
    calls: list[dict[str, Any]] = []
    registry = ToolRegistry()
    registry.register(
        RecordingVerifierTool(
            ToolContext(),
            calls=calls,
            result={
                "results": [
                    {
                        "claim_id": "c001",
                        "claim": "Agentic RAG uses retrieval grading.",
                        "cited_chunk_ids": ["unknown-chunk"],
                        "verification_label": "supported",
                        "confidence": 1.2,
                        "reason": "bad verifier output",
                    }
                ],
                "reason": "bad",
            },
        )
    )
    nodes = AgentNodes(llm=FakeLLM([]), tool_registry=registry)
    state = create_initial_state("What does Agentic RAG use?")
    state["draft_answer"] = "Agentic RAG uses retrieval grading [1]."
    state["claims"] = [
        {
            "claim_id": "c001",
            "claim": "Agentic RAG uses retrieval grading.",
            "cited_chunk_ids": ["chunk-1"],
        }
    ]
    state["cited_documents"] = [
        {
            "content": "Agentic RAG uses retrieval grading.",
            "source": "paper.pdf",
            "chunk_id": "chunk-1",
        }
    ]

    update = nodes.verify_citations_node(state)

    assert len(calls) == 1
    assert update["route"] == "fallback"
    assert update["fallback_reason"] == "Citation verification tool returned invalid data."


def test_verify_citations_node_falls_back_when_results_do_not_cover_all_claims():
    calls: list[dict[str, Any]] = []
    registry = ToolRegistry()
    registry.register(
        RecordingVerifierTool(
            ToolContext(),
            calls=calls,
            result={
                "results": [
                    {
                        "claim_id": "c001",
                        "claim": "Agentic RAG uses retrieval grading.",
                        "cited_chunk_ids": ["chunk-1"],
                        "verification_label": "supported",
                        "confidence": 0.9,
                        "reason": "supported",
                    }
                ],
                "reason": "missing one claim",
            },
        )
    )
    nodes = AgentNodes(llm=FakeLLM([]), tool_registry=registry)
    state = create_initial_state("What does Agentic RAG use?")
    state["draft_answer"] = "Agentic RAG uses retrieval grading and fallback [1]."
    state["claims"] = [
        {
            "claim_id": "c001",
            "claim": "Agentic RAG uses retrieval grading.",
            "cited_chunk_ids": ["chunk-1"],
        },
        {
            "claim_id": "c002",
            "claim": "Agentic RAG has fallback.",
            "cited_chunk_ids": ["chunk-1"],
        },
    ]
    state["cited_documents"] = [
        {
            "content": "Agentic RAG uses retrieval grading and fallback.",
            "source": "paper.pdf",
            "chunk_id": "chunk-1",
        }
    ]

    update = nodes.verify_citations_node(state)

    assert len(calls) == 1
    assert update["route"] == "fallback"
    assert update["fallback_reason"] == "Citation verification tool returned invalid data."


def test_revise_answer_node_updates_draft_and_increments_revision_count():
    llm = FakeLLM(
        [
            (
                '{"answer": "Agentic RAG can reduce hallucination risk [1].", '
                '"used_citation_indices": [1]}'
            )
        ]
    )
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("What does Agentic RAG guarantee?")
    state["draft_answer"] = "Agentic RAG eliminates hallucination [1]."
    state["citation_revision_count"] = 0
    state["cited_documents"] = [
        {
            "content": "Citation checks can reduce hallucination risk.",
            "source": "paper.pdf",
            "chunk_id": "chunk-1",
        }
    ]
    state["unsupported_claims"] = [
        {
            "claim_id": "c001",
            "claim": "Agentic RAG eliminates hallucination.",
            "cited_chunk_ids": ["chunk-1"],
            "verification_label": "unsupported",
            "confidence": 0.2,
            "reason": "The chunk only says reduce risk.",
        }
    ]

    update = nodes.revise_answer_node(state)

    assert update["draft_answer"] == "Agentic RAG can reduce hallucination risk [1]."
    assert update["used_citation_indices"] == [1]
    assert update["citation_revision_count"] == 1
    assert update["citations"] == [
        {
            "source": "paper.pdf",
            "page": None,
            "chunk_id": "chunk-1",
            "score": None,
            "snippet": "Citation checks can reduce hallucination risk.",
        }
    ]
    assert update["route"] == "extract_claims"
    assert "unsupported" in llm.prompts[0].lower()


def test_finalize_answer_node_promotes_verified_draft_answer():
    nodes = AgentNodes(llm=FakeLLM([]), retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["draft_answer"] = "Verified answer [1]."
    state["citation_verification_passed"] = True
    state["citations"] = [{"source": "paper.pdf", "snippet": "support"}]
    state["claims"] = [
        {
            "claim_id": "c001",
            "claim": "Verified answer.",
            "cited_chunk_ids": ["chunk-1"],
        }
    ]
    state["claim_verification"] = {
        "verified": True,
        "results": [],
        "reason": "All claims supported.",
    }

    update = nodes.finalize_answer_node(state)

    assert update["answer"] == "Verified answer [1]."
    assert update["citations"] == state["citations"]
    assert update["claims"] == state["claims"]
    assert update["claim_verification"] == state["claim_verification"]
    assert update["is_verified"] is True
    assert update["route"] == "end"


def test_finalize_answer_node_allows_verification_skipped_refusal():
    nodes = AgentNodes(llm=FakeLLM([]), retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["draft_answer"] = "I cannot answer from the current documents."
    state["citation_verification_skipped"] = True
    state["citation_verification_passed"] = False

    update = nodes.finalize_answer_node(state)

    assert update["answer"] == "I cannot answer from the current documents."
    assert update["is_verified"] is False
    assert update["citation_verification_skipped"] is True
    assert update["route"] == "end"


def test_fallback_node_returns_clear_message():
    nodes = AgentNodes(llm=FakeLLM([]), retriever_fn=lambda query: [])

    update = nodes.fallback_node(create_initial_state("question"))

    assert "无法可靠回答" in update["answer"]
    assert update["citations"] == []
    assert update["fallback_reason"]
