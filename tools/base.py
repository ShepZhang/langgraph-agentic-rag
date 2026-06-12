from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from copy import deepcopy
import re
from typing import Any, Callable, ClassVar, Generic, Mapping, TypeVar

from pydantic import BaseModel


ArgsT = TypeVar("ArgsT", bound=BaseModel)
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class ToolContext:
    llm: Any | None = None
    retriever_fn: Callable[[str], list[dict[str, Any]]] | None = None
    workspace_id: str | None = None


@dataclass(frozen=True)
class ToolErrorInfo:
    code: str
    message: str


@dataclass(frozen=True)
class ToolResult(Generic[ResultT]):
    tool_name: str
    success: bool
    data: ResultT | None = None
    error: ToolErrorInfo | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolError(Exception):
    code = "tool_error"

    def __init__(self, message: str | ToolErrorInfo):
        if isinstance(message, ToolErrorInfo):
            self.info = message
            super().__init__(message.message)
        else:
            self.info = ToolErrorInfo(code=self.code, message=message)
            super().__init__(message)


class ToolRegistrationError(ToolError):
    code = "tool_registration_error"


class ToolNotFoundError(ToolError):
    code = "tool_not_found_error"


class ToolInputError(ToolError):
    code = "tool_input_error"


class ToolExecutionError(ToolError):
    code = "tool_execution_error"


class BaseTool(Generic[ArgsT, ResultT], ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    args_schema: type[ArgsT]
    trace_metadata_fields: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, context: ToolContext):
        self.context = context

    @abstractmethod
    def run(self, arguments: ArgsT) -> ResultT:
        raise NotImplementedError

    def build_metadata(self, arguments: ArgsT, result: ResultT) -> dict[str, Any]:
        return {}


def coerce_llm_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value

    content = getattr(value, "content", None)
    if content is not None:
        return _coerce_content(content)

    if isinstance(value, list):
        return _coerce_content(value)

    return str(value)


def _coerce_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_coerce_block(block) for block in content)
    return str(content)


def _coerce_block(block: Any) -> str:
    if isinstance(block, str):
        return block
    if isinstance(block, list):
        return "".join(_coerce_block(item) for item in block)
    if isinstance(block, Mapping):
        if block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                return text
            if isinstance(text, Mapping):
                for key in ("content", "value", "text"):
                    value = text.get(key)
                    if isinstance(value, str):
                        return value
        for key in ("content", "value", "text"):
            value = block.get(key)
            if isinstance(value, str):
                return value
        return ""
    nested_content = getattr(block, "content", None)
    if nested_content is not None:
        return _coerce_content(nested_content)
    return str(block)


_SK_REDACT_RE = re.compile(r"\bsk-[A-Za-z0-9._-]+\b")
_BEARER_REDACT_RE = re.compile(r"\bBearer\s+\S+", re.IGNORECASE)
_SENSITIVE_FIELD_RE = re.compile(
    r"(?i)(?P<prefix>\b(?:api[_-]?key|password|secret|token|authorization)\b\s*[:=]\s*)"
    r"(?P<value>(?:\"[^\"]*\"|'[^']*'|[^,\s\]\}]+))"
)
def redact_tool_message(message: str) -> str:
    redacted = _SK_REDACT_RE.sub("[REDACTED]", message)
    redacted = _BEARER_REDACT_RE.sub("[REDACTED]", redacted)
    redacted = _SENSITIVE_FIELD_RE.sub(r"\g<prefix>[REDACTED]", redacted)
    return redacted[:500]


def error_info_from_exception(
    exc: Exception,
    *,
    default_code: str,
) -> ToolErrorInfo:
    if isinstance(exc, ToolError) and hasattr(exc, "info"):
        info = exc.info
        message = info.message
    else:
        message = str(exc)
    return ToolErrorInfo(code=default_code, message=redact_tool_message(message))


def snapshot_observer_value(value: Any) -> Any:
    try:
        cloned = deepcopy(value)
    except Exception:
        cloned = value
    return _snapshot_observer_value(cloned, seen=set())


def _snapshot_observer_value(value: Any, *, seen: set[int]) -> Any:
    if isinstance(value, Mapping):
        value_id = id(value)
        if value_id in seen:
            return "[REDACTED]"
        seen.add(value_id)
        try:
            snapshot: dict[str, Any] = {}
            for key, item in value.items():
                snapshot[str(key)] = _snapshot_observer_value(item, seen=seen)
            return snapshot
        finally:
            seen.remove(value_id)
    if isinstance(value, list):
        value_id = id(value)
        if value_id in seen:
            return "[REDACTED]"
        seen.add(value_id)
        try:
            return [_snapshot_observer_value(item, seen=seen) for item in value]
        finally:
            seen.remove(value_id)
    if isinstance(value, tuple):
        value_id = id(value)
        if value_id in seen:
            return "[REDACTED]"
        seen.add(value_id)
        try:
            return [_snapshot_observer_value(item, seen=seen) for item in value]
        finally:
            seen.remove(value_id)
    if isinstance(value, set):
        value_id = id(value)
        if value_id in seen:
            return "[REDACTED]"
        seen.add(value_id)
        try:
            return [_snapshot_observer_value(item, seen=seen) for item in value]
        finally:
            seen.remove(value_id)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return redact_tool_message(value) if isinstance(value, str) else value
    return redact_tool_message(str(value))
