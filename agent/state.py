"""LangGraph state definitions for the Agentic RAG workflow."""

from __future__ import annotations

from typing import TypedDict


class ChatMessage(TypedDict):
    """A minimal chat history message."""

    role: str
    content: str


class RetrievedDocument(TypedDict, total=False):
    """Retrieved document chunk passed through the agent state."""

    content: str
    source: str | None
    source_path: str | None
    file_hash: str | None
    page: int | None
    chunk_id: str | None
    score: float | None


class Citation(TypedDict, total=False):
    """Citation returned with the final answer."""

    source: str | None
    page: int | None
    chunk_id: str | None
    score: float | None
    snippet: str


class AgentState(TypedDict):
    """State passed between LangGraph nodes."""

    question: str
    current_query: str
    rewritten_question: str
    chat_history: list[ChatMessage]
    previous_queries: list[str]
    documents: list[RetrievedDocument]
    relevant_documents: list[RetrievedDocument]
    grading_reason: str
    answer: str
    citations: list[Citation]
    rewrite_count: int
    retry_count: int
    retrieval_attempt: int
    max_retry_count: int
    is_relevant: bool
    route: str
    fallback_reason: str


def create_initial_state(
    question: str,
    chat_history: list[ChatMessage] | None = None,
    max_retry_count: int = 2,
) -> AgentState:
    """Create the initial state for an Agentic RAG run."""

    return {
        "question": question,
        "current_query": "",
        "rewritten_question": "",
        "chat_history": chat_history or [],
        "previous_queries": [],
        "documents": [],
        "relevant_documents": [],
        "grading_reason": "",
        "answer": "",
        "citations": [],
        "rewrite_count": 0,
        "retry_count": 0,
        "retrieval_attempt": 0,
        "max_retry_count": max_retry_count,
        "is_relevant": False,
        "route": "",
        "fallback_reason": "",
    }
