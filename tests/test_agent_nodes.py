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


def test_grade_documents_node_marks_empty_docs_irrelevant_without_llm_call():
    llm = FakeLLM(['{"relevant": true}'])
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")

    update = nodes.grade_documents_node(state)

    assert update["is_relevant"] is False
    assert update["relevant_documents"] == []
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
    assert update["grading_reason"] == "Chunk 2 directly answers the question."
    assert update["route"] == "generate_answer"
    assert "Original user question:\nquestion" in llm.prompts[0]
    assert "Retrieval query:\nrewritten question" in llm.prompts[0]
    assert "grade the retrieved chunks against the original user question" in llm.prompts[0]


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
            ),
            (
                '{"verified": true, "claims": ['
                '{"claim": "Grounded answer", "supported": true, "citation_indices": [1]}'
                '], "reason": "Supported by selected evidence."}'
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

    assert update["answer"] == "Grounded answer with [2]."
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
    assert "claim-level citation verifier" in llm.prompts[1].lower()
    assert "Answer to verify:\nGrounded answer with [2]." in llm.prompts[1]
    assert "Selected citation chunks" in llm.prompts[1]
    assert update["is_verified"] is True
    assert update["claims"] == [
        {
            "claim": "Grounded answer",
            "supported": True,
            "citation_indices": [1],
        }
    ]
    assert update["claim_verification_reason"] == "Supported by selected evidence."


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
            '{"verified": true, "claims": [{"claim": "Grounded answer", "supported": true, "citation_indices": [1]}], "reason": "supported"}',
        ]
    )
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["relevant_documents"] = [{"content": "context", "source": "paper.pdf"}]

    update = nodes.generate_answer_node(state)

    assert update["answer"] == "Grounded answer [1]."


def test_generate_answer_node_deduplicates_selected_citations():
    llm = FakeLLM(
        [
            (
                '{"answer": "Grounded answer [1] [2].", '
                '"used_citation_indices": [1, 2]}'
            ),
            (
                '{"verified": true, "claims": ['
                '{"claim": "Grounded answer", "supported": true, "citation_indices": [1]}'
                '], "reason": "supported"}'
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


def test_generate_answer_node_keeps_valid_citation_and_ignores_invalid_indices():
    llm = FakeLLM(
        [
            (
                '{"answer": "Grounded answer [1].", '
                '"used_citation_indices": [1, 3]}'
            ),
            (
                '{"verified": true, "claims": ['
                '{"claim": "Grounded answer", "supported": true, "citation_indices": [1]}'
                '], "reason": "supported"}'
            )
        ]
    )
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["relevant_documents"] = [{"content": "context", "source": "paper.pdf"}]

    update = nodes.generate_answer_node(state)

    assert update["answer"] == "Grounded answer [1]."
    assert update["citations"] == [
        {
            "source": "paper.pdf",
            "page": None,
            "chunk_id": None,
            "score": None,
            "snippet": "context",
        }
    ]
    assert update["is_verified"] is True


def test_generate_answer_node_falls_back_when_claim_verification_fails():
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

    assert "无法可靠回答" in update["answer"]
    assert update["citations"] == []
    assert update["is_verified"] is False
    assert "claim verification failed" in update["fallback_reason"].lower()


def test_generate_answer_node_falls_back_for_invalid_claim_verification_json():
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

    assert "无法可靠回答" in update["answer"]
    assert update["citations"] == []
    assert update["is_verified"] is False
    assert "claim verification" in update["fallback_reason"].lower()


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

    assert update["answer"] == "I cannot answer from the current documents."
    assert update["citations"] == []
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


def test_fallback_node_returns_clear_message():
    nodes = AgentNodes(llm=FakeLLM([]), retriever_fn=lambda query: [])

    update = nodes.fallback_node(create_initial_state("question"))

    assert "无法可靠回答" in update["answer"]
    assert update["citations"] == []
    assert update["fallback_reason"]
