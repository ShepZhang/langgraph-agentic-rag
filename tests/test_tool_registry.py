from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from tools.base import BaseTool, ToolContext, ToolErrorInfo, ToolInputError
from tools.registry import ToolNotFoundError, ToolRegistrationError, ToolRegistry


class EchoArgs(BaseModel):
    text: str = Field(min_length=1)


class EchoTool(BaseTool[EchoArgs, str]):
    name = "echo"
    description = "Return validated text."
    args_schema = EchoArgs

    def run(self, arguments: EchoArgs) -> str:
        return arguments.text

    def build_metadata(self, arguments: EchoArgs, result: str) -> dict[str, object]:
        return {"length": len(result)}


class ShoutTool(BaseTool[EchoArgs, str]):
    name = "shout"
    description = "Return upper-cased text."
    args_schema = EchoArgs

    def run(self, arguments: EchoArgs) -> str:
        return arguments.text.upper()


class RejectTool(BaseTool[EchoArgs, str]):
    name = "reject"
    description = "Rejects certain inputs."
    args_schema = EchoArgs

    def run(self, arguments: EchoArgs) -> str:
        raise ToolInputError(ToolErrorInfo(code="tool_input_error", message="bad input"))


def test_registry_registers_lists_and_invokes_typed_tool():
    calls: list[dict[str, object]] = []
    registry = ToolRegistry(call_observer=calls.append)
    registry.register(EchoTool(ToolContext()))
    registry.register(ShoutTool(ToolContext()))

    result = registry.invoke("echo", {"text": "hello"})

    assert result.success is True
    assert result.tool_name == "echo"
    assert result.data == "hello"
    assert result.error is None
    assert result.metadata["length"] == 5
    assert result.metadata["latency_ms"] >= 0
    assert registry.list_tools() == [
        {"name": "echo", "description": "Return validated text."},
        {"name": "shout", "description": "Return upper-cased text."},
    ]
    assert calls[0]["tool_name"] == "echo"
    assert calls[0]["success"] is True
    assert calls[0]["error"] is None
    assert calls[0]["metadata"] == {"length": 5}


def test_registry_rejects_duplicate_and_unknown_tools():
    registry = ToolRegistry()
    registry.register(EchoTool(ToolContext()))

    with pytest.raises(ToolRegistrationError):
        registry.register(EchoTool(ToolContext()))

    with pytest.raises(ToolNotFoundError):
        registry.invoke("missing", {})

    class BlankNameTool(EchoTool):
        name = " "

    with pytest.raises(ToolRegistrationError):
        registry.register(BlankNameTool(ToolContext()))


def test_registry_returns_structured_input_and_execution_failures():
    registry = ToolRegistry()
    registry.register(EchoTool(ToolContext()))
    registry.register(RejectTool(ToolContext()))

    invalid = registry.invoke("echo", {"text": ""})

    assert invalid.success is False
    assert invalid.error is not None
    assert invalid.error.code == "tool_input_error"
    assert invalid.data is None

    rejected = registry.invoke("reject", {"text": "hello"})

    assert rejected.success is False
    assert rejected.error is not None
    assert rejected.error.code == "tool_input_error"
    assert rejected.data is None

    class BrokenTool(EchoTool):
        name = "broken"

        def run(self, arguments: EchoArgs) -> str:
            raise RuntimeError("backend unavailable")

    registry.register(BrokenTool(ToolContext()))
    failed = registry.invoke("broken", {"text": "hello"})

    assert failed.success is False
    assert failed.error is not None
    assert failed.error.code == "tool_execution_error"
    assert "backend unavailable" in failed.error.message


def test_registry_redacts_secrets_and_ignores_observer_failure():
    def broken_observer(record: dict[str, object]) -> None:
        raise RuntimeError("observer unavailable")

    class SecretFailureTool(EchoTool):
        name = "secret_failure"

        def run(self, arguments: EchoArgs) -> str:
            raise RuntimeError("provider rejected Bearer token sk-secretvalue123")

    registry = ToolRegistry(call_observer=broken_observer)
    registry.register(SecretFailureTool(ToolContext()))

    result = registry.invoke("secret_failure", {"text": "hello"})

    assert result.success is False
    assert result.error is not None
    assert len(result.error.message) <= 500
    assert "sk-secretvalue123" not in result.error.message
    assert "Bearer" not in result.error.message
    assert "[REDACTED]" in result.error.message
