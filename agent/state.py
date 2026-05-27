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
    page: int | None
    chunk_id: str | None
    score: float | None


class Citation(TypedDict, total=False):
    """Citation returned with the final answer."""

    source: str | None
    page: int | None
    chunk_id: str | None
    score: float | None


class AgentState(TypedDict):
    """State passed between LangGraph nodes."""

    question: str
    rewritten_question: str
    chat_history: list[ChatMessage]
    documents: list[RetrievedDocument]
    answer: str
    citations: list[Citation]
    rewrite_count: int
    is_relevant: bool
    route: str


def create_initial_state(
    question: str,
    chat_history: list[ChatMessage] | None = None,
) -> AgentState:
    """Create the initial state for an Agentic RAG run."""

    return {
        "question": question,
        "rewritten_question": "",
        "chat_history": chat_history or [],
        "documents": [],
        "answer": "",
        "citations": [],
        "rewrite_count": 0,
        "is_relevant": False,
        "route": "",
    }
