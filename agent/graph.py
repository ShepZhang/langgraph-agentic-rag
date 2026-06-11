"""LangGraph build and run entrypoints for the Agentic RAG workflow."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.edges import (
    route_after_answer_generation,
    route_after_answer_revision,
    route_after_citation_verification,
    route_after_claim_extraction,
    route_after_grading,
)
from agent.llm import create_chat_model
from agent.nodes import AgentNodes
from agent.state import AgentState, ChatMessage, create_initial_state
from config import Settings, get_settings


def build_graph(
    llm: Any | None = None,
    retriever_fn: Any | None = None,
    settings: Settings | None = None,
) -> Any:
    """Build and compile the Agentic RAG LangGraph workflow."""

    resolved_settings = settings or get_settings()
    resolved_llm = llm or create_chat_model(resolved_settings)
    nodes = AgentNodes(llm=resolved_llm, retriever_fn=retriever_fn)

    workflow = StateGraph(AgentState)
    workflow.add_node("rewrite_query", nodes.rewrite_query_node)
    workflow.add_node("retrieve", nodes.retrieve_node)
    workflow.add_node("grade_documents", nodes.grade_documents_node)
    workflow.add_node("generate_answer", nodes.generate_answer_node)
    workflow.add_node("extract_claims", nodes.extract_claims_node)
    workflow.add_node("verify_citations", nodes.verify_citations_node)
    workflow.add_node("revise_answer", nodes.revise_answer_node)
    workflow.add_node("finalize_answer", nodes.finalize_answer_node)
    workflow.add_node("fallback", nodes.fallback_node)

    workflow.add_edge(START, "rewrite_query")
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        lambda state: route_after_grading(state, settings=resolved_settings),
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query",
            "fallback": "fallback",
        },
    )
    workflow.add_conditional_edges(
        "generate_answer",
        route_after_answer_generation,
        {
            "extract_claims": "extract_claims",
            "finalize_answer": "finalize_answer",
            "fallback": "fallback",
        },
    )
    workflow.add_conditional_edges(
        "extract_claims",
        route_after_claim_extraction,
        {
            "verify_citations": "verify_citations",
            "finalize_answer": "finalize_answer",
            "fallback": "fallback",
        },
    )
    workflow.add_conditional_edges(
        "verify_citations",
        route_after_citation_verification,
        {
            "finalize_answer": "finalize_answer",
            "revise_answer": "revise_answer",
            "fallback": "fallback",
        },
    )
    workflow.add_conditional_edges(
        "revise_answer",
        route_after_answer_revision,
        {
            "extract_claims": "extract_claims",
            "finalize_answer": "finalize_answer",
            "fallback": "fallback",
        },
    )
    workflow.add_edge("finalize_answer", END)
    workflow.add_edge("fallback", END)

    return workflow.compile()


def run_agent(
    question: str,
    chat_history: list[ChatMessage] | None = None,
    graph: Any | None = None,
    llm: Any | None = None,
    retriever_fn: Any | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run the Agentic RAG workflow and return the public result payload."""

    compiled_graph = graph or build_graph(
        llm=llm,
        retriever_fn=retriever_fn,
        settings=settings,
    )
    resolved_settings = settings or get_settings()
    final_state = compiled_graph.invoke(
        create_initial_state(
            question,
            chat_history,
            max_retry_count=resolved_settings.max_retry_count,
        )
    )

    return {
        "answer": final_state["answer"],
        "citations": final_state["citations"],
        "draft_answer": final_state["draft_answer"],
        "used_citation_indices": final_state["used_citation_indices"],
        "cited_documents": final_state["cited_documents"],
        "claims": final_state["claims"],
        "claim_verification": final_state["claim_verification"],
        "claim_verification_results": final_state["claim_verification_results"],
        "unsupported_claims": final_state["unsupported_claims"],
        "claim_verification_reason": final_state["claim_verification_reason"],
        "citation_verification_passed": final_state["citation_verification_passed"],
        "citation_revision_count": final_state["citation_revision_count"],
        "max_citation_revision_count": final_state["max_citation_revision_count"],
        "citation_verification_skipped": final_state["citation_verification_skipped"],
        "is_verified": final_state["is_verified"],
        "retrieved_documents": final_state["documents"],
        "relevant_documents": final_state["relevant_documents"],
        "current_query": final_state["current_query"],
        "rewritten_question": final_state["rewritten_question"],
        "standalone_question": final_state["standalone_question"],
        "query_transform": final_state["query_transform"],
        "query_transform_strategy": final_state["query_transform_strategy"],
        "query_transform_reason": final_state["query_transform_reason"],
        "expanded_queries": final_state["expanded_queries"],
        "sub_questions": final_state["sub_questions"],
        "retrieval_queries": final_state["retrieval_queries"],
        "multi_query_used": final_state["multi_query_used"],
        "multi_query_result_count": final_state["multi_query_result_count"],
        "rewrite_count": final_state["rewrite_count"],
        "retry_count": final_state["retry_count"],
        "retrieval_attempt": final_state["retrieval_attempt"],
        "is_relevant": final_state["is_relevant"],
        "grading_reason": final_state["grading_reason"],
        "document_grades": final_state["document_grades"],
        "relevant_document_count": final_state["relevant_document_count"],
        "partial_document_count": final_state["partial_document_count"],
        "max_relevance_confidence": final_state["max_relevance_confidence"],
        "partial_relevance_recovery": final_state["partial_relevance_recovery"],
        "fallback_reason": final_state["fallback_reason"],
        "route": final_state["route"],
    }
