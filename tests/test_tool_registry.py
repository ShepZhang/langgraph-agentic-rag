from __future__ import annotations

import pytest
from pydantic import BaseModel, Field, field_validator, model_validator

from tools import ToolContext, ToolError, ToolExecutionError, ToolInputError
from tools.base import BaseTool, ToolErrorInfo
from tools.registry import ToolNotFoundError, ToolRegistrationError, ToolRegistry


class EchoArgs(BaseModel):
    text: str = Field(min_length=1)


class EchoTool(BaseTool[EchoArgs, str]):
    name = "echo"
    description = "Return validated text."
    args_schema = EchoArgs
    trace_metadata_fields = frozenset({"length"})

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
    trace_metadata_fields = frozenset(
        {"token_count", "api_key_count", "raw_response_code", "model_response_latency"}
    )

    def run(self, arguments: ComplexMetadataArgs) -> str:
        return arguments.text

    def build_metadata(
        self,
        arguments: ComplexMetadataArgs,
        result: str,
    ) -> dict[str, object]:
        return {
            "access_token_value": "token-123",
            "client_secret_value": "secret-456",
            "token_count": 11,
            "api_key_count": 9,
            "raw_response_code": 200,
            "model_response_latency": 0.25,
            "document_content": "body text",
            "retrieved_documents": ["doc-1"],
            "prompt": "system prompt",
            "rawResponse": {"text": "raw body"},
            "model_response": {"text": "model body"},
            "nested": [
                {
                    "access_token_value": "nested-token",
                    "client_secret_value": "nested-secret",
                    "token_count": 7,
                    "api_key_count": 4,
                    "raw_response_code": 201,
                    "model_response_latency": 0.5,
                    "document_content": "nested body",
                    "retrieved_documents": ["nested-doc"],
                    "prompt": "nested prompt",
                    "rawResponse": {"text": "nested raw body"},
                    "model_response": {"text": "nested model body"},
                }
            ],
            "business": {"count": 3},
        }


class DeterministicValidatorArgs(BaseModel):
    api_key: str = Field(min_length=1)
    password: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_model(self) -> "DeterministicValidatorArgs":
        raise ValueError(f"model rejected {self.api_key}")


class DeterministicValidatorTool(BaseTool[DeterministicValidatorArgs, str]):
    name = "deterministic_validator"
    description = "Model validator that echoes api_key."
    args_schema = DeterministicValidatorArgs

    def run(self, arguments: DeterministicValidatorArgs) -> str:
        return arguments.api_key


class ConstraintArgs(BaseModel):
    retry_count: int = Field(gt=10)


class ConstraintTool(BaseTool[ConstraintArgs, int]):
    name = "constraint"
    description = "Numeric constraint validation."
    args_schema = ConstraintArgs

    def run(self, arguments: ConstraintArgs) -> int:
        return arguments.retry_count


class CycleArgs(BaseModel):
    text: str = Field(min_length=1)


class CycleMetadataTool(BaseTool[CycleArgs, str]):
    name = "cycle_metadata"
    description = "Metadata with shared references and a cycle."
    args_schema = CycleArgs
    trace_metadata_fields = frozenset({"first", "second", "cycle"})

    def run(self, arguments: CycleArgs) -> str:
        return arguments.text

    def build_metadata(self, arguments: CycleArgs, result: str) -> dict[str, object]:
        shared = {"value": "shared"}
        cycle: dict[str, object] = {}
        cycle["self"] = cycle
        return {
            "first": shared,
            "second": shared,
            "cycle": cycle,
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


def test_registry_uses_deterministic_validation_messages_for_model_validators():
    observer_records: list[dict[str, object]] = []
    secret_api_key = "deterministic-secret-api-key"
    secret_password = "deterministic-secret-password"

    registry = ToolRegistry(call_observer=observer_records.append)
    registry.register(DeterministicValidatorTool(ToolContext()))

    result = registry.invoke(
        "deterministic_validator",
        {"api_key": secret_api_key, "password": secret_password},
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "tool_input_error"
    assert result.error.message == "loc=<model>; type=value_error; message=Value failed validation."
    assert secret_api_key not in result.error.message
    assert secret_password not in result.error.message
    assert observer_records[0]["error"]["message"] == result.error.message
    assert secret_api_key not in str(observer_records[0]["error"])
    assert secret_password not in str(observer_records[0]["error"])


def test_registry_uses_deterministic_validation_messages_for_numeric_constraints():
    registry = ToolRegistry()
    registry.register(ConstraintTool(ToolContext()))

    result = registry.invoke("constraint", {"retry_count": 1})

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "tool_input_error"
    assert result.error.message == "loc=retry_count; type=greater_than; message=Value failed validation."
    assert "[REDACTED]0" not in result.error.message


def test_registry_masks_dynamic_mapping_keys_in_validation_messages():
    class ValuesArgs(BaseModel):
        values: dict[str, int]

    class ValuesTool(BaseTool[ValuesArgs, dict[str, int]]):
        name = "values"
        description = "Dynamic mapping validation."
        args_schema = ValuesArgs

        def run(self, arguments: ValuesArgs) -> dict[str, int]:
            return arguments.values

    observer_records: list[dict[str, object]] = []
    registry = ToolRegistry(call_observer=observer_records.append)
    registry.register(ValuesTool(ToolContext()))

    result = registry.invoke("values", {"values": {"api_key": "secret"}})

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "tool_input_error"
    assert "values.<key>" in result.error.message
    assert "api_key" not in result.error.message
    assert "api_key" not in str(observer_records[0]["error"])


def test_registry_limits_validation_messages_and_reports_omitted_count():
    class BatchArgs(BaseModel):
        first: str = Field(min_length=2)
        second: str = Field(min_length=2)
        third: str = Field(min_length=2)
        fourth: str = Field(min_length=2)
        fifth: str = Field(min_length=2)
        sixth: str = Field(min_length=2)

    class BatchTool(BaseTool[BatchArgs, dict[str, str]]):
        name = "batch"
        description = "Batch validation."
        args_schema = BatchArgs

        def run(self, arguments: BatchArgs) -> dict[str, str]:
            return arguments.model_dump()

    registry = ToolRegistry()
    registry.register(BatchTool(ToolContext()))

    result = registry.invoke(
        "batch",
        {
            "first": "",
            "second": "",
            "third": "",
            "fourth": "",
            "fifth": "",
            "sixth": "",
        },
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "tool_input_error"
    assert result.error.message.count("loc=") == 5
    assert "omitted=1" in result.error.message
    assert len(result.error.message) <= 500


def test_registry_skips_unlisted_metadata_fields_by_default():
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
    assert observer_records[0]["metadata"] == {}


def test_registry_traces_only_explicit_allowlisted_metadata_fields():
    observer_records: list[dict[str, object]] = []

    def observer(record: dict[str, object]) -> None:
        observer_records.append(record)
        record["metadata"]["token_count"] = 99

    registry = ToolRegistry(call_observer=observer)
    registry.register(ComplexMetadataTool(ToolContext()))

    result = registry.invoke("complex_metadata", {"text": "hello"})

    assert result.success is True
    assert result.metadata["access_token_value"] == "token-123"
    assert result.metadata["client_secret_value"] == "secret-456"
    assert result.metadata["token_count"] == 11
    assert result.metadata["api_key_count"] == 9
    assert result.metadata["raw_response_code"] == 200
    assert result.metadata["model_response_latency"] == 0.25
    assert result.metadata["document_content"] == "body text"
    assert result.metadata["retrieved_documents"] == ["doc-1"]
    assert result.metadata["prompt"] == "system prompt"
    assert result.metadata["rawResponse"] == {"text": "raw body"}
    assert result.metadata["model_response"] == {"text": "model body"}
    assert result.metadata["nested"][0]["access_token_value"] == "nested-token"
    assert result.metadata["nested"][0]["client_secret_value"] == "nested-secret"
    assert result.metadata["nested"][0]["token_count"] == 7
    assert result.metadata["nested"][0]["api_key_count"] == 4
    assert result.metadata["nested"][0]["raw_response_code"] == 201
    assert result.metadata["nested"][0]["model_response_latency"] == 0.5
    assert result.metadata["nested"][0]["document_content"] == "nested body"
    assert result.metadata["nested"][0]["rawResponse"] == {"text": "nested raw body"}
    assert result.metadata["nested"][0]["model_response"] == {"text": "nested model body"}

    snapshot = observer_records[0]["metadata"]
    assert snapshot == {
        "token_count": 99,
        "api_key_count": 9,
        "raw_response_code": 200,
        "model_response_latency": 0.25,
    }
    assert result.metadata["token_count"] == 11
    assert result.metadata["api_key_count"] == 9
    assert result.metadata["raw_response_code"] == 200
    assert result.metadata["model_response_latency"] == 0.25


def test_registry_keeps_shared_metadata_references_and_handles_cycles():
    observer_records: list[dict[str, object]] = []

    def observer(record: dict[str, object]) -> None:
        observer_records.append(record)
        record["metadata"]["first"]["value"] = "observer-mutated"

    registry = ToolRegistry(call_observer=observer)
    registry.register(CycleMetadataTool(ToolContext()))

    result = registry.invoke("cycle_metadata", {"text": "hello"})

    assert result.success is True
    assert result.metadata["first"]["value"] == "shared"
    assert result.metadata["second"]["value"] == "shared"
    assert "self" in result.metadata["cycle"]

    snapshot = observer_records[0]["metadata"]
    assert snapshot["first"]["value"] == "observer-mutated"
    assert snapshot["second"]["value"] == "shared"
    assert snapshot["cycle"]["self"] == "[REDACTED]"


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
