"""Default internal tool composition."""

from __future__ import annotations

from typing import Any, Callable

from tools.base import ToolContext
from tools.calculator_tool import CalculatorTool
from tools.citation_verifier_tool import CitationVerifierTool
from tools.document_summary_tool import DocumentSummaryTool
from tools.registry import ToolRegistry
from tools.retriever_tool import RetrieverTool


def create_default_tool_registry(
    *,
    llm: Any,
    retriever_fn: Callable[[str], list[dict[str, Any]]] | None = None,
    workspace_id: str | None = None,
) -> ToolRegistry:
    context = ToolContext(
        llm=llm,
        retriever_fn=retriever_fn,
        workspace_id=workspace_id,
    )
    registry = ToolRegistry()
    registry.register(RetrieverTool(context))
    registry.register(CitationVerifierTool(context))
    registry.register(DocumentSummaryTool(context))
    registry.register(CalculatorTool(context))
    return registry


__all__ = ["create_default_tool_registry"]
