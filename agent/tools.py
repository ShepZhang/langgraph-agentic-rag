"""Agent tools for accessing the private knowledge base."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from rag.retriever import retrieve


RetrieverFn = Any


def create_retriever_tool(
    retriever_fn: RetrieverFn | None = None,
    workspace_id: str | None = None,
) -> StructuredTool:
    """Create a retriever tool for Agent use."""

    def _retrieve_context(query: str) -> list[dict[str, Any]]:
        """Retrieve relevant document chunks from the indexed private knowledge base."""

        if retriever_fn is not None:
            return retriever_fn(query)
        return retrieve(query, workspace_id=workspace_id)

    return StructuredTool.from_function(
        func=_retrieve_context,
        name="retrieve_context",
        description=(
            "Retrieve relevant document chunks from the indexed private knowledge base "
            "according to the user's question."
        ),
    )


retrieve_context = create_retriever_tool()
