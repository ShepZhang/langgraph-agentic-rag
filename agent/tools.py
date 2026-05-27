"""Agent tools for accessing the private knowledge base."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import StructuredTool

from rag.retriever import retrieve


RetrieverFn = Callable[[str], list[dict[str, Any]]]


def create_retriever_tool(retriever_fn: RetrieverFn | None = None) -> StructuredTool:
    """Create a retriever tool for Agent use."""

    resolved_retriever = retriever_fn or retrieve

    def _retrieve_context(query: str) -> list[dict[str, Any]]:
        """Retrieve relevant document chunks from the indexed private knowledge base."""

        return resolved_retriever(query)

    return StructuredTool.from_function(
        func=_retrieve_context,
        name="retrieve_context",
        description=(
            "Retrieve relevant document chunks from the indexed private knowledge base "
            "according to the user's question."
        ),
    )


retrieve_context = create_retriever_tool()
