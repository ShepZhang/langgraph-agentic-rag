"""Trace record construction for Agentic RAG runs."""

from __future__ import annotations

import time
import uuid
from copy import deepcopy
from typing import Any

MAX_SNIPPET_CHARS = 240


class TraceRecorder:
    """Collect node events and route decisions for one Agent run."""

    def __init__(
        self,
        original_question: str,
        session_id: str | None = None,
        workspace_id: str | None = None,
        trace_id: str | None = None,
        prompts: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.trace_id = trace_id or f"trace_{uuid.uuid4().hex}"
        self.session_id = session_id
        self.workspace_id = workspace_id
        self.original_question = original_question
        self.prompts = deepcopy(prompts or {})
        self.started_at = time.perf_counter()
        self.events: list[dict[str, Any]] = []
        self.route_decisions: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []

    def record_node(
        self,
        node: str,
        state_before: dict[str, Any],
        update: dict[str, Any] | None,
        elapsed_ms: float,
        error: str | None = None,
    ) -> None:
        """Record a LangGraph node execution."""

        after_state = dict(state_before)
        if update:
            after_state.update(update)

        event = {
            "event_type": "node",
            "node": node,
            "elapsed_ms": round(max(elapsed_ms, 0.0), 4),
            "output_keys": sorted((update or {}).keys()),
            "state_summary": summarize_state(after_state),
            "error": error,
        }
        self.events.append(_jsonable(event))

    def record_route(
        self,
        from_node: str,
        to_node: str,
        state: dict[str, Any],
    ) -> None:
        """Record a conditional edge decision."""

        decision = {
            "from": from_node,
            "to": to_node,
            "reason": build_route_reason(from_node, to_node, state),
        }
        jsonable_decision = _jsonable(decision)
        self.route_decisions.append(jsonable_decision)
        self.events.append(
            {
                "event_type": "route",
                **jsonable_decision,
            }
        )

    def record_tool_call(self, record: dict[str, Any]) -> None:
        """Record a compact internal tool invocation."""

        normalized = _jsonable(record)
        self.tool_calls.append(normalized)
        self.events.append(
            {
                "event_type": "tool",
                **normalized,
            }
        )

    def build_record(
        self,
        final_state: dict[str, Any] | None,
        latency_ms: float,
        error: str | None = None,
        token_usage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the persisted trace record."""

        state = final_state or {}
        return _jsonable(
            {
                "trace_id": self.trace_id,
                "session_id": self.session_id,
                "workspace_id": self.workspace_id,
                "original_question": self.original_question,
                "prompts": self.prompts,
                "query_transform": state.get("query_transform", {}),
                "retrieved_documents": summarize_documents(
                    state.get("documents", [])
                ),
                "relevant_documents": summarize_documents(
                    state.get("relevant_documents", [])
                ),
                "reranked_documents": summarize_documents(
                    [
                        document
                        for document in state.get("documents", [])
                        if isinstance(document, dict)
                        and document.get("rerank_score") is not None
                    ]
                ),
                "document_grades": state.get("document_grades", []),
                "route_decisions": self.route_decisions,
                "tool_calls": self.tool_calls,
                "events": self.events,
                "final_answer": state.get("answer", ""),
                "citations": state.get("citations", []),
                "claim_verification": state.get("claim_verification", {}),
                "claim_verification_results": state.get(
                    "claim_verification_results",
                    [],
                ),
                "unsupported_claims": state.get("unsupported_claims", []),
                "retry_count": state.get("retry_count", 0),
                "retrieval_attempt": state.get("retrieval_attempt", 0),
                "latency_ms": round(max(latency_ms, 0.0), 4),
                "token_usage": token_usage,
                "error": error,
            }
        )


def summarize_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return a compact state snapshot for trace events."""

    return {
        "current_query": state.get("current_query", ""),
        "route": state.get("route", ""),
        "retrieval_attempt": state.get("retrieval_attempt", 0),
        "retry_count": state.get("retry_count", 0),
        "document_count": len(state.get("documents", []) or []),
        "relevant_document_count": state.get("relevant_document_count", 0),
        "partial_document_count": state.get("partial_document_count", 0),
        "citation_verification_passed": state.get(
            "citation_verification_passed",
            False,
        ),
        "citation_revision_count": state.get("citation_revision_count", 0),
        "fallback_reason": state.get("fallback_reason", ""),
    }


def summarize_documents(
    documents: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Return trace-safe document summaries without full local paths."""

    summaries: list[dict[str, Any]] = []
    for index, document in enumerate(documents or [], start=1):
        if not isinstance(document, dict):
            continue
        summaries.append(
            _drop_none(
                {
                    "document_id": (
                        document.get("document_id")
                        or document.get("source")
                        or document.get("file_hash")
                        or document.get("chunk_id")
                        or f"document_{index}"
                    ),
                    "chunk_id": document.get("chunk_id"),
                    "source": document.get("source"),
                    "page": document.get("page"),
                    "score": document.get("score"),
                    "rerank_score": document.get("rerank_score"),
                    "matched_queries": document.get("matched_queries", []),
                    "retrieval_query_count": document.get("retrieval_query_count"),
                    "snippet": _snippet(str(document.get("content", ""))),
                }
            )
        )
    return summaries


def build_route_reason(
    from_node: str,
    to_node: str,
    state: dict[str, Any],
) -> str:
    """Build a concise human-readable route reason."""

    if from_node == "accept_documents":
        if to_node == "generate_answer":
            return "Retrieved documents accepted without grading."
        return "No retrieved documents available without grading."

    if from_node == "grade_documents":
        if to_node == "generate_answer":
            count = state.get("relevant_document_count", 0)
            return f"{count} relevant document(s) passed retrieval grading."
        if to_node == "rewrite_query":
            return (
                state.get("grading_reason")
                or "No relevant documents found and retry budget remains."
            )
        return (
            state.get("fallback_reason")
            or state.get("grading_reason")
            or "No relevant documents found and retry budget exhausted."
        )

    if from_node == "generate_answer":
        if state.get("citation_verification_skipped"):
            return "Citation verification skipped."
        return state.get("fallback_reason") or f"Answer generation routed to {to_node}."

    if from_node == "extract_claims":
        return state.get("claim_verification_reason") or f"Claim extraction routed to {to_node}."

    if from_node == "verify_citations":
        return (
            state.get("claim_verification_reason")
            or f"Citation verification routed to {to_node}."
        )

    if from_node == "revise_answer":
        return (
            state.get("claim_verification_reason")
            or f"Answer revision routed to {to_node}."
        )

    return f"{from_node} routed to {to_node}."


def _snippet(text: str) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= MAX_SNIPPET_CHARS:
        return normalized
    return f"{normalized[: MAX_SNIPPET_CHARS - 3]}..."


def _drop_none(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
