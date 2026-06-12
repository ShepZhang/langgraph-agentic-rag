from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rag.retriever import retrieve
from tools.base import BaseTool, ToolContext


class RetrieverArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)


class RetrieverTool(BaseTool[RetrieverArgs, list[dict[str, Any]]]):
    name = "retrieve_context"
    description = (
        "Retrieve relevant document chunks from the indexed private knowledge base "
        "according to the user's question."
    )
    args_schema = RetrieverArgs
    trace_metadata_fields = frozenset({"workspace_id", "result_count"})

    def run(self, arguments: RetrieverArgs) -> list[dict[str, Any]]:
        if self.context.retriever_fn is not None:
            return self.context.retriever_fn(arguments.query)
        return retrieve(arguments.query, workspace_id=self.context.workspace_id)

    def build_metadata(
        self,
        arguments: RetrieverArgs,
        result: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "workspace_id": self.context.workspace_id,
            "result_count": len(result),
        }


__all__ = ["RetrieverArgs", "RetrieverTool"]
