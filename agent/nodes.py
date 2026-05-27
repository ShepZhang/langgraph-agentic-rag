"""LangGraph node implementations for Agentic RAG."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import BaseMessage

from agent.prompts import (
    ANSWER_GENERATION_PROMPT,
    QUERY_REWRITE_PROMPT,
    RETRIEVAL_GRADING_PROMPT,
    format_chat_history,
    format_documents,
)
from agent.state import AgentState, Citation, RetrievedDocument
from agent.tools import create_retriever_tool


FALLBACK_ANSWER = "根据当前已索引文档，无法可靠回答这个问题。请补充相关文档，或换一种更具体的问法。"


class AgentNodes:
    """State transition nodes for the Agentic RAG graph."""

    def __init__(self, llm: Any, retriever_fn: Any | None = None) -> None:
        self.llm = llm
        self.retriever_tool = create_retriever_tool(retriever_fn)

    def rewrite_query_node(self, state: AgentState) -> dict[str, Any]:
        """Rewrite the user question for retrieval."""

        prompt = QUERY_REWRITE_PROMPT.format(
            chat_history=format_chat_history(state.get("chat_history", [])),
            question=state["question"],
        )
        rewritten_question = _coerce_llm_text(self.llm.invoke(prompt)).strip()
        if not rewritten_question:
            rewritten_question = state["question"]

        return {
            "rewritten_question": rewritten_question,
            "rewrite_count": state.get("rewrite_count", 0) + 1,
        }

    def retrieve_node(self, state: AgentState) -> dict[str, Any]:
        """Retrieve relevant chunks with the retriever tool."""

        query = state.get("rewritten_question") or state["question"]
        documents = self.retriever_tool.invoke({"query": query})
        return {"documents": documents}

    def grade_documents_node(self, state: AgentState) -> dict[str, Any]:
        """Grade whether retrieved chunks are relevant enough to answer."""

        documents = state.get("documents", [])
        if not documents:
            return {"is_relevant": False, "route": "rewrite_query"}

        prompt = RETRIEVAL_GRADING_PROMPT.format(
            question=state.get("rewritten_question") or state["question"],
            documents=format_documents(documents),
        )
        raw_result = _coerce_llm_text(self.llm.invoke(prompt))
        is_relevant = _parse_relevance(raw_result)
        return {
            "is_relevant": is_relevant,
            "route": "generate_answer" if is_relevant else "rewrite_query",
        }

    def generate_answer_node(self, state: AgentState) -> dict[str, Any]:
        """Generate a grounded answer and citations."""

        documents = state.get("documents", [])
        prompt = ANSWER_GENERATION_PROMPT.format(
            question=state.get("rewritten_question") or state["question"],
            documents=format_documents(documents),
        )
        answer = _coerce_llm_text(self.llm.invoke(prompt)).strip()
        if not answer:
            answer = FALLBACK_ANSWER

        return {
            "answer": answer,
            "citations": build_citations(documents),
        }

    def fallback_node(self, state: AgentState) -> dict[str, Any]:
        """Return a safe fallback answer when retrieval is insufficient."""

        return {
            "answer": FALLBACK_ANSWER,
            "citations": [],
            "is_relevant": False,
            "route": "fallback",
        }


def build_citations(documents: list[RetrievedDocument]) -> list[Citation]:
    """Build citations from retrieved document metadata."""

    citations: list[Citation] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for document in documents:
        citation: Citation = {
            "source": document.get("source"),
            "page": document.get("page"),
            "chunk_id": document.get("chunk_id"),
            "score": document.get("score"),
        }
        key = (
            citation.get("source"),
            citation.get("page"),
            citation.get("chunk_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        citations.append(citation)
    return citations


def _parse_relevance(raw_result: str) -> bool:
    """Parse a relevance grading JSON response."""

    try:
        parsed = json.loads(raw_result)
    except json.JSONDecodeError:
        return False

    return bool(parsed.get("relevant") is True)


def _coerce_llm_text(response: Any) -> str:
    """Convert LangChain or fake LLM responses into text."""

    if isinstance(response, str):
        return response
    if isinstance(response, BaseMessage):
        content = response.content
        if isinstance(content, str):
            return content
        return str(content)
    content = getattr(response, "content", None)
    if content is not None:
        return str(content)
    return str(response)
