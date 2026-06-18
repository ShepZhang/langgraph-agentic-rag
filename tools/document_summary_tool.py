"""Document summary tool."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prompting import render_prompt
from tools.base import BaseTool, ToolExecutionError, coerce_llm_text


class DocumentSummaryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1)
    title: str | None = None
    max_points: int = Field(default=5, ge=1, le=10)


class DocumentSummaryTool(BaseTool[DocumentSummaryArgs, str]):
    name = "summarize_document"
    description = "Summarize supplied document text without adding new facts."
    args_schema = DocumentSummaryArgs
    trace_metadata_fields = frozenset({"max_points"})

    def run(self, arguments: DocumentSummaryArgs) -> str:
        if self.context.llm is None:
            raise ToolExecutionError("Document summary requires an LLM.")

        title = arguments.title or "Untitled document"
        prompt = render_prompt(
            "tool.document_summary",
            max_points=arguments.max_points,
            title=title,
            content=arguments.content,
        )
        summary = coerce_llm_text(self.context.llm.invoke(prompt)).strip()
        if not summary:
            raise ToolExecutionError("Document summary returned empty text.")
        return summary

    def build_metadata(
        self,
        arguments: DocumentSummaryArgs,
        result: str,
    ) -> dict[str, Any]:
        return {"max_points": arguments.max_points}


__all__ = ["DocumentSummaryArgs", "DocumentSummaryTool"]
