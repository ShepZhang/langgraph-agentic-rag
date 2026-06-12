from __future__ import annotations

import pytest
from pydantic import BaseModel, Field, field_validator

from tools import ToolContext, ToolError, ToolExecutionError, ToolInputError
from tools.base import BaseTool, ToolErrorInfo
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


class SecretArgs(BaseModel):
    api_key: str = Field(min_length=20)
    password: str = Field(min_length=20)


class SecretInputTool(BaseTool[SecretArgs, str]):
    name = "secret_input"
    description = "Validation failure with secret-bearing input."
    args_schema = SecretArgs

    def run(self, arguments: SecretArgs) -> str:
        return arguments.api_key


class MetadataArgs(BaseModel):
    text: str = Field(min_length=1)


class MetadataTool(BaseTool[MetadataArgs, str]):
    name = "metadata"
    description = "Return metadata with nested and sensitive fields."
    args_schema = MetadataArgs

    def run(self, arguments: MetadataArgs) -> str:
        return arguments.text

    def build_metadata(
        self,
        arguments: MetadataArgs,
        result: str,
    ) -> dict[str, object]:
        return {
            "nested": {"items": [1, {"flag": True}]},
            "api_key": "alpha-key",
            "password": "secret-pass",
            "token": "tok-123",
            "authorization": "Bearer hidden-token",
            "secret": "vault-secret",
            "content": "do not expose this body",
            "documents": [{"title": "doc"}],
            "prompt": "system prompt",
            "raw_response": {"text": "raw"},
            "business": {"count": 2},
        }


class ValidatorArgs(BaseModel):
    api_key: str = Field(min_length=1)
    password: str = Field(min_length=1)

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: str) -> str:
        raise ValueError(f"rejected {value}")


class ValidatorTool(BaseTool[ValidatorArgs, str]):
    name = "validator"
    description = "Validation error with echoed secret."
    args_schema = ValidatorArgs

    def run(self, arguments: ValidatorArgs) -> str:
        return arguments.api_key


class ComplexMetadataArgs(BaseModel):
    text: str = Field(min_length=1)


class ComplexMetadataTool(BaseTool[ComplexMetadataArgs, str]):
    name = "complex_metadata"
    description = "Nested metadata with bypass keys."
    args_schema = ComplexMetadataArgs

    def run(self, arguments: ComplexMetadataArgs) -> str:
        return arguments.text

    def build_metadata(
        self,
        arguments: ComplexMetadataArgs,
        result: str,
    ) -> dict[str, object]:
        return {
            "access_token": "token-123",
            "client_secret": "secret-456",
            "document_content": "body text",
            "rawResponse": {"text": "raw body"},
            "nested": [
                {
                    "access_token": "nested-token",
                    "client_secret": "nested-secret",
                    "document_content": "nested body",
                    "rawResponse": {"text": "nested raw body"},
                }
            ],
            "business": {"count": 3},
        }


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


def test_registry_hides_validation_inputs_from_errors_and_observer_records():
    observer_records: list[dict[str, object]] = []
    secret_api_key = "plain-secret"
    secret_password = "plain-secret"

    registry = ToolRegistry(call_observer=observer_records.append)
    registry.register(SecretInputTool(ToolContext()))

    result = registry.invoke(
        "secret_input",
        {"api_key": secret_api_key, "password": secret_password},
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "tool_input_error"
    assert secret_api_key not in result.error.message
    assert secret_password not in result.error.message
    assert "input=" not in result.error.message
    assert observer_records[0]["error"] is not None
    assert secret_api_key not in str(observer_records[0]["error"])
    assert secret_password not in str(observer_records[0]["error"])


def test_registry_sanitizes_validator_echoes_without_leaking_inputs():
    observer_records: list[dict[str, object]] = []
    secret_api_key = "validator-secret-api-key"
    secret_password = "validator-secret-password"

    registry = ToolRegistry(call_observer=observer_records.append)
    registry.register(ValidatorTool(ToolContext()))

    result = registry.invoke(
        "validator",
        {"api_key": secret_api_key, "password": secret_password},
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "tool_input_error"
    assert "api_key" in result.error.message
    assert "value_error" in result.error.message
    assert secret_api_key not in result.error.message
    assert secret_password not in result.error.message
    assert observer_records[0]["error"] is not None
    assert secret_api_key not in str(observer_records[0]["error"])
    assert secret_password not in str(observer_records[0]["error"])


def test_registry_isolates_observer_metadata_and_redacts_snapshot_fields():
    observer_records: list[dict[str, object]] = []

    def observer(record: dict[str, object]) -> None:
        observer_records.append(record)
        record["metadata"]["nested"]["items"][1]["flag"] = False

    registry = ToolRegistry(call_observer=observer)
    registry.register(MetadataTool(ToolContext()))

    result = registry.invoke("metadata", {"text": "hello"})

    assert result.success is True
    assert result.metadata["nested"]["items"][1]["flag"] is True
    assert result.metadata["api_key"] == "alpha-key"
    assert result.metadata["business"] == {"count": 2}

    record = observer_records[0]
    assert record["metadata"]["nested"]["items"][1]["flag"] is False
    assert record["metadata"]["api_key"] == "[REDACTED]"
    assert record["metadata"]["password"] == "[REDACTED]"
    assert record["metadata"]["token"] == "[REDACTED]"
    assert record["metadata"]["authorization"] == "[REDACTED]"
    assert record["metadata"]["secret"] == "[REDACTED]"
    assert "content" not in record["metadata"]
    assert "documents" not in record["metadata"]
    assert "prompt" not in record["metadata"]
    assert "raw_response" not in record["metadata"]


def test_registry_redacts_complex_metadata_keys_without_mutating_result():
    observer_records: list[dict[str, object]] = []

    def observer(record: dict[str, object]) -> None:
        observer_records.append(record)
        record["metadata"]["nested"][0]["business"] = "changed"

    registry = ToolRegistry(call_observer=observer)
    registry.register(ComplexMetadataTool(ToolContext()))

    result = registry.invoke("complex_metadata", {"text": "hello"})

    assert result.success is True
    assert result.metadata["access_token"] == "token-123"
    assert result.metadata["client_secret"] == "secret-456"
    assert result.metadata["document_content"] == "body text"
    assert result.metadata["rawResponse"] == {"text": "raw body"}
    assert result.metadata["nested"][0]["access_token"] == "nested-token"
    assert result.metadata["nested"][0]["document_content"] == "nested body"
    assert result.metadata["nested"][0]["rawResponse"] == {"text": "nested raw body"}
    assert "business" not in result.metadata["nested"][0]

    snapshot = observer_records[0]["metadata"]
    assert snapshot["access_token"] == "[REDACTED]"
    assert snapshot["client_secret"] == "[REDACTED]"
    assert "document_content" not in snapshot
    assert "rawResponse" not in snapshot
    assert snapshot["nested"][0]["access_token"] == "[REDACTED]"
    assert snapshot["nested"][0]["client_secret"] == "[REDACTED]"
    assert "document_content" not in snapshot["nested"][0]
    assert "rawResponse" not in snapshot["nested"][0]
    assert snapshot["nested"][0]["business"] == "changed"


def test_registry_forces_tool_input_error_code_even_when_exception_carries_custom_code():
    class CustomCodeInputErrorTool(EchoTool):
        name = "custom_input_error"

        def run(self, arguments: EchoArgs) -> str:
            raise ToolInputError(
                ToolErrorInfo(code="custom", message="invalid sk-secretvalue123")
            )

    registry = ToolRegistry()
    registry.register(CustomCodeInputErrorTool(ToolContext()))

    result = registry.invoke("custom_input_error", {"text": "hello"})

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "tool_input_error"
    assert result.error.message.endswith("[REDACTED]")


def test_registry_forces_tool_execution_error_code_even_when_exception_carries_custom_code():
    class CustomCodeExecutionErrorTool(EchoTool):
        name = "custom_execution_error"

        def run(self, arguments: EchoArgs) -> str:
            raise ToolExecutionError(
                ToolErrorInfo(code="custom", message="provider failed Bearer token x")
            )

    registry = ToolRegistry()
    registry.register(CustomCodeExecutionErrorTool(ToolContext()))

    result = registry.invoke("custom_execution_error", {"text": "hello"})

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "tool_execution_error"
    assert "Bearer" not in result.error.message
    assert "[REDACTED]" in result.error.message
