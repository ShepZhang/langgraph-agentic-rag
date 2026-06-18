from __future__ import annotations

import pytest
from pydantic import ValidationError

from tools import ToolContext, ToolRegistry
import tools.document_summary_tool as document_summary_module
from tools.document_summary_tool import DocumentSummaryArgs, DocumentSummaryTool


class FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_document_summary_returns_summary_and_traces_max_points():
    observer_records: list[dict[str, object]] = []
    llm = FakeLLM(["  - first point\n  - second point  "])
    registry = ToolRegistry(call_observer=observer_records.append)
    registry.register(DocumentSummaryTool(ToolContext(llm=llm)))

    result = registry.invoke(
        "summarize_document",
        {
            "title": "Spec Notes",
            "content": "The document discusses retrieval quality and grounded answers.",
            "max_points": 2,
        },
    )

    assert result.success is True
    assert result.data == "- first point\n  - second point"
    assert result.metadata["max_points"] == 2
    assert observer_records[0]["metadata"] == {"max_points": 2}

    prompt = llm.prompts[0]
    assert "Spec Notes" in prompt
    assert "The document discusses retrieval quality and grounded answers." in prompt
    assert "at most 2" in prompt
    assert "unsupported facts" in prompt.lower()


def test_document_summary_uses_registered_prompt(monkeypatch):
    captured = {}

    def fake_render_prompt(prompt_id, **variables):
        captured["prompt_id"] = prompt_id
        captured["variables"] = variables
        return "registered document summary prompt"

    monkeypatch.setattr(document_summary_module, "render_prompt", fake_render_prompt)
    llm = FakeLLM(["- concise summary"])
    registry = ToolRegistry()
    registry.register(DocumentSummaryTool(ToolContext(llm=llm)))

    result = registry.invoke(
        "summarize_document",
        {
            "title": "Spec Notes",
            "content": "The document discusses retrieval quality.",
            "max_points": 3,
        },
    )

    assert captured == {
        "prompt_id": "tool.document_summary",
        "variables": {
            "max_points": 3,
            "title": "Spec Notes",
            "content": "The document discusses retrieval quality.",
        },
    }
    assert llm.prompts == ["registered document summary prompt"]
    assert result.success is True


def test_document_summary_uses_untitled_document_label():
    llm = FakeLLM(["summary"])
    registry = ToolRegistry()
    registry.register(DocumentSummaryTool(ToolContext(llm=llm)))

    result = registry.invoke(
        "summarize_document",
        {
            "content": "Standalone content.",
        },
    )

    assert result.success is True
    assert "Untitled document" in llm.prompts[0]


def test_document_summary_requires_llm():
    registry = ToolRegistry()
    registry.register(DocumentSummaryTool(ToolContext()))

    result = registry.invoke(
        "summarize_document",
        {"content": "Document text."},
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "tool_execution_error"
    assert "requires an LLM" in result.error.message


def test_document_summary_rejects_empty_llm_output():
    registry = ToolRegistry()
    registry.register(DocumentSummaryTool(ToolContext(llm=FakeLLM(["   "]))))

    result = registry.invoke(
        "summarize_document",
        {"content": "Document text."},
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "tool_execution_error"
    assert "returned empty text" in result.error.message


def test_document_summary_rejects_empty_content():
    registry = ToolRegistry()
    registry.register(DocumentSummaryTool(ToolContext(llm=FakeLLM(["summary"]))))

    result = registry.invoke(
        "summarize_document",
        {"content": ""},
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "tool_input_error"


def test_document_summary_args_forbid_extra_fields():
    with pytest.raises(ValidationError):
        DocumentSummaryArgs.model_validate(
            {"content": "Document text.", "unexpected": True},
        )
