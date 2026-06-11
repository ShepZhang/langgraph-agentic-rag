"""Chat and trace routes."""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_agent_runner, get_trace_service
from api.schemas import ChatRequest, ChatResponse
from api.services.traces import TraceService


router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    runner: Callable[..., dict[str, Any]] = Depends(get_agent_runner),
) -> ChatResponse:
    """Run an Agentic RAG chat request."""

    result = runner(
        request.question,
        chat_history=[message.model_dump() for message in request.chat_history],
        session_id=request.session_id,
        workspace_id=request.workspace_id,
    )
    return ChatResponse(
        answer=str(result.get("answer", "")),
        citations=list(result.get("citations", []) or []),
        trace_id=result.get("trace_id"),
        retry_count=int(result.get("retry_count", 0) or 0),
        latency_ms=float(result.get("latency_ms", 0.0) or 0.0),
        fallback_triggered=bool(result.get("fallback_reason")),
    )


@router.get("/chat/{session_id}/trace")
def get_chat_trace(
    session_id: str,
    trace_id: str | None = Query(default=None),
    trace_service: TraceService = Depends(get_trace_service),
) -> dict[str, Any]:
    """Return a trace by id, or the latest trace for a session."""

    trace = trace_service.get_trace(session_id=session_id, trace_id=trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found.")
    return trace
