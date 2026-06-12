"""LangGraph build and run entrypoints for the Agentic RAG workflow."""

from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, START, StateGraph

import agent.tools as legacy_agent_tools
from agent.edges import (
    route_after_answer_generation,
    route_after_answer_revision,
    route_after_citation_verification,
    route_after_claim_extraction,
    route_after_grading,
)
from agent.features import AgentFeatureFlags
from agent.llm import create_chat_model
from agent.nodes import AgentNodes
from agent.state import AgentState, ChatMessage, create_initial_state
from config import Settings, get_settings
from observability.logger import create_trace_store
from observability.trace import TraceRecorder
from tools import ToolRegistry, create_default_tool_registry


def build_graph(
    llm: Any | None = None,
    retriever_fn: Any | None = None,
    settings: Settings | None = None,
    features: AgentFeatureFlags | None = None,
    trace_recorder: TraceRecorder | None = None,
    workspace_id: str | None = None,
    tool_registry: ToolRegistry | None = None,
) -> Any:
    """Build and compile the Agentic RAG LangGraph workflow."""

    resolved_settings = settings or get_settings()
    resolved_features = features or AgentFeatureFlags()
    resolved_llm = llm or create_chat_model(resolved_settings)
    resolved_retriever_fn = (
        retriever_fn
        if retriever_fn is not None
        else _build_workspace_retriever_fn(workspace_id)
    )
    resolved_tool_registry = (
        tool_registry
        if tool_registry is not None
        else create_default_tool_registry(
            llm=resolved_llm,
            retriever_fn=resolved_retriever_fn,
            workspace_id=workspace_id,
        )
    )
    graph_tool_registry = _build_graph_scoped_tool_registry(
        resolved_tool_registry,
        trace_recorder,
    )
    nodes = AgentNodes(
        llm=resolved_llm,
        features=resolved_features,
        tool_registry=graph_tool_registry,
    )

    workflow = StateGraph(AgentState)
    workflow.add_node(
        "rewrite_query",
        _trace_node("rewrite_query", nodes.rewrite_query_node, trace_recorder),
    )
    workflow.add_node("retrieve", _trace_node("retrieve", nodes.retrieve_node, trace_recorder))
    workflow.add_node(
        "accept_documents",
        _trace_node(
            "accept_documents",
            nodes.accept_retrieved_documents_node,
            trace_recorder,
        ),
    )
    workflow.add_node(
        "grade_documents",
        _trace_node("grade_documents", nodes.grade_documents_node, trace_recorder),
    )
    workflow.add_node(
        "generate_answer",
        _trace_node("generate_answer", nodes.generate_answer_node, trace_recorder),
    )
    workflow.add_node(
        "extract_claims",
        _trace_node("extract_claims", nodes.extract_claims_node, trace_recorder),
    )
    workflow.add_node(
        "verify_citations",
        _trace_node("verify_citations", nodes.verify_citations_node, trace_recorder),
    )
    workflow.add_node(
        "revise_answer",
        _trace_node("revise_answer", nodes.revise_answer_node, trace_recorder),
    )
    workflow.add_node(
        "finalize_answer",
        _trace_node("finalize_answer", nodes.finalize_answer_node, trace_recorder),
    )
    workflow.add_node("fallback", _trace_node("fallback", nodes.fallback_node, trace_recorder))

    if resolved_features.query_transformation_enabled:
        workflow.add_edge(START, "rewrite_query")
        workflow.add_edge("rewrite_query", "retrieve")
    else:
        workflow.add_edge(START, "retrieve")

    if resolved_features.retrieval_grading_enabled:
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            _trace_route(
                "grade_documents",
                lambda state: route_after_grading(
                    state,
                    settings=resolved_settings,
                    features=resolved_features,
                ),
                trace_recorder,
            ),
            {
                "generate_answer": "generate_answer",
                "rewrite_query": "rewrite_query",
                "fallback": "fallback",
            },
        )
    else:
        workflow.add_edge("retrieve", "accept_documents")
        workflow.add_conditional_edges(
            "accept_documents",
            _trace_route(
                "accept_documents",
                lambda state: (
                    "generate_answer"
                    if state.get("relevant_documents")
                    else "fallback"
                ),
                trace_recorder,
            ),
            {
                "generate_answer": "generate_answer",
                "fallback": "fallback",
            },
        )
    workflow.add_conditional_edges(
        "generate_answer",
        _trace_route(
            "generate_answer",
            route_after_answer_generation,
            trace_recorder,
        ),
        {
            "extract_claims": "extract_claims",
            "finalize_answer": "finalize_answer",
            "fallback": "fallback",
        },
    )
    workflow.add_conditional_edges(
        "extract_claims",
        _trace_route(
            "extract_claims",
            route_after_claim_extraction,
            trace_recorder,
        ),
        {
            "verify_citations": "verify_citations",
            "finalize_answer": "finalize_answer",
            "fallback": "fallback",
        },
    )
    workflow.add_conditional_edges(
        "verify_citations",
        _trace_route(
            "verify_citations",
            route_after_citation_verification,
            trace_recorder,
        ),
        {
            "finalize_answer": "finalize_answer",
            "revise_answer": "revise_answer",
            "fallback": "fallback",
        },
    )
    workflow.add_conditional_edges(
        "revise_answer",
        _trace_route(
            "revise_answer",
            route_after_answer_revision,
            trace_recorder,
        ),
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
    session_id: str | None = None,
    workspace_id: str | None = None,
    graph: Any | None = None,
    llm: Any | None = None,
    retriever_fn: Any | None = None,
    settings: Settings | None = None,
    features: AgentFeatureFlags | None = None,
    trace_store: Any | None = None,
    tool_registry: ToolRegistry | None = None,
) -> dict[str, Any]:
    """Run the Agentic RAG workflow and return the public result payload."""

    resolved_settings = settings or get_settings()
    resolved_features = features or AgentFeatureFlags()
    trace_enabled = resolved_settings.trace_logging_enabled
    trace_recorder = (
        TraceRecorder(
            original_question=question,
            session_id=session_id,
            workspace_id=workspace_id,
        )
        if trace_enabled
        else None
    )
    resolved_trace_store = (
        trace_store
        if trace_store is not None
        else create_trace_store(resolved_settings)
        if trace_enabled
        else None
    )
    compiled_graph = graph or build_graph(
        llm=llm,
        retriever_fn=retriever_fn,
        settings=resolved_settings,
        features=resolved_features,
        trace_recorder=trace_recorder,
        workspace_id=workspace_id,
        tool_registry=tool_registry,
    )
    initial_state = create_initial_state(
        question,
        chat_history,
        max_retry_count=resolved_settings.max_retry_count,
    )
    if not resolved_features.query_transformation_enabled:
        initial_state["current_query"] = question
        initial_state["rewritten_question"] = question
        initial_state["standalone_question"] = question
        initial_state["previous_queries"] = [question]

    started_at = time.perf_counter()
    final_state: dict[str, Any] = {}
    try:
        final_state = compiled_graph.invoke(initial_state)
    except Exception as exc:
        latency_ms = (time.perf_counter() - started_at) * 1000
        if trace_recorder and resolved_trace_store:
            resolved_trace_store.save(
                trace_recorder.build_record(
                    final_state,
                    latency_ms=latency_ms,
                    error=str(exc),
                )
            )
        raise

    latency_ms = (time.perf_counter() - started_at) * 1000
    trace_path = None
    if trace_recorder and resolved_trace_store:
        resolved_trace_store.save(
            trace_recorder.build_record(final_state, latency_ms=latency_ms)
        )
        store_path = getattr(resolved_trace_store, "path", None)
        trace_path = str(store_path) if store_path is not None else None

    return {
        "trace_id": trace_recorder.trace_id if trace_recorder else None,
        "trace_path": trace_path,
        "latency_ms": round(max(latency_ms, 0.0), 4),
        "feature_flags": resolved_features.to_dict(),
        "citation_verification_enabled": (
            resolved_features.citation_verification_enabled
        ),
        "chat_history_used": bool(chat_history),
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


def _trace_node(
    node_name: str,
    node_fn: Any,
    trace_recorder: TraceRecorder | None,
) -> Any:
    """Wrap a graph node so observability stays outside node logic."""

    if trace_recorder is None:
        return node_fn

    def wrapped(state: AgentState) -> dict[str, Any]:
        started_at = time.perf_counter()
        try:
            update = node_fn(state)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            trace_recorder.record_node(
                node_name,
                state,
                {},
                elapsed_ms=elapsed_ms,
                error=str(exc),
            )
            raise
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        trace_recorder.record_node(
            node_name,
            state,
            update,
            elapsed_ms=elapsed_ms,
        )
        return update

    return wrapped


def _build_workspace_retriever_fn(
    workspace_id: str | None,
) -> Any:
    def retrieve_with_workspace(query: str) -> list[dict[str, Any]]:
        return legacy_agent_tools.retrieve(query, workspace_id=workspace_id)

    return retrieve_with_workspace


def _build_graph_scoped_tool_registry(
    source: ToolRegistry,
    trace_recorder: TraceRecorder | None,
) -> ToolRegistry:
    """Build an isolated registry wrapper so graph observers cannot conflict."""

    observer = trace_recorder.record_tool_call if trace_recorder is not None else None
    scoped = ToolRegistry(call_observer=observer)
    for tool_info in source.list_tools():
        scoped.register(source.get(tool_info["name"]))
    source.set_call_observer(None)
    return scoped


def _trace_route(
    from_node: str,
    route_fn: Any,
    trace_recorder: TraceRecorder | None,
) -> Any:
    """Wrap a conditional edge route function."""

    if trace_recorder is None:
        return route_fn

    def wrapped(state: AgentState) -> str:
        route = route_fn(state)
        trace_recorder.record_route(from_node, route, state)
        return route

    return wrapped
