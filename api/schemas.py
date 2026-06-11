"""Request and response schemas for the FastAPI service layer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Chat history message supplied by clients."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Request body for Agentic RAG chat."""

    workspace_id: str = Field(default="default")
    session_id: str = Field(default="default")
    question: str = Field(min_length=1)
    chat_history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Response returned by the chat endpoint."""

    answer: str
    citations: list[dict[str, Any]]
    trace_id: str | None
    retry_count: int
    latency_ms: float
    fallback_triggered: bool


class DocumentRecord(BaseModel):
    """API-managed document registry record."""

    document_id: str
    workspace_id: str
    filename: str
    status: str
    chunk_count: int = 0
    vector_ids: list[str] = Field(default_factory=list)


class DocumentUploadResponse(BaseModel):
    """Uploaded document records."""

    documents: list[DocumentRecord]


class DocumentIndexRequest(BaseModel):
    """Request body for indexing uploaded documents."""

    workspace_id: str = Field(default="default")
    document_ids: list[str] | None = None
    reset_collection: bool = False


class DocumentIndexResponse(BaseModel):
    """Indexing result."""

    workspace_id: str
    indexed_documents: list[DocumentRecord]
    chunk_count: int
    reset_collection: bool


class DocumentListResponse(BaseModel):
    """Document listing response."""

    documents: list[DocumentRecord]


class DocumentDeleteResponse(BaseModel):
    """Document delete response."""

    document_id: str
    deleted: bool
    deleted_vector_count: int


class EvaluationRunRequest(BaseModel):
    """Request body for running evaluation."""

    workspace_id: str = Field(default="default")
    question_ids: list[str] | None = None
    include_baseline: bool = True


class EvaluationRunResponse(BaseModel):
    """Evaluation run metadata."""

    run_id: str
    workspace_id: str
    status: str
    summary: dict[str, Any]
    result_path: str
