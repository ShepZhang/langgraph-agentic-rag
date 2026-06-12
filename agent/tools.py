"""Agent tools for accessing the private knowledge base."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import StructuredTool

from tools import ToolContext, ToolRegistry
from tools.retriever_tool import RetrieverTool
from rag.retriever import retrieve


RetrieverFn = Callable[[str], list[dict[str, Any]]]


def create_retriever_tool(
    retriever_fn: RetrieverFn | None = None,
    workspace_id: str | None = None,
) -> StructuredTool:
    """Create a retriever tool for Agent use."""

    def _retrieve_with_workspace(query: str) -> list[dict[str, Any]]:
        return retrieve(query, workspace_id=workspace_id)

    registry = ToolRegistry()
    registry.register(
        RetrieverTool(
            ToolContext(
                retriever_fn=retriever_fn or _retrieve_with_workspace,
                workspace_id=workspace_id,
            )
        )
    )
    registered_tool = registry.get("retrieve_context")

    def _retrieve_context(query: str) -> list[dict[str, Any]]:
        result = registry.invoke("retrieve_context", {"query": query})
        if not result.success:
            message = (
                result.error.message
                if result.error is not None and result.error.message
                else "Unknown tool failure"
            )
            raise RuntimeError(message)
        return result.data or []

    return StructuredTool.from_function(
        func=_retrieve_context,
        name=registered_tool.name,
        description=registered_tool.description,
    )


retrieve_context = create_retriever_tool()
