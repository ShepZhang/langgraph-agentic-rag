from __future__ import annotations

from collections.abc import Callable, Mapping
import logging
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from tools.base import (
    BaseTool,
    ToolErrorInfo,
    ToolExecutionError,
    ToolInputError,
    ToolNotFoundError,
    ToolRegistrationError,
    ToolResult,
    error_info_from_exception,
    snapshot_observer_value,
)

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(
        self,
        call_observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._tools: dict[str, BaseTool[Any, Any]] = {}
        self._call_observer = call_observer

    def register(self, tool: BaseTool[Any, Any]) -> None:
        name = tool.name.strip()
        if not name:
            raise ToolRegistrationError("Tool name must not be blank.")
        if name in self._tools:
            raise ToolRegistrationError(f"Tool '{name}' is already registered.")
        self._tools[name] = tool

    def get(self, name: str) -> BaseTool[Any, Any]:
        key = name.strip()
        try:
            return self._tools[key]
        except KeyError as exc:
            raise ToolNotFoundError(f"Tool '{name}' was not found.") from exc

    def list_tools(self) -> list[dict[str, str]]:
        return [
            {"name": tool.name.strip(), "description": tool.description}
            for tool in self._tools.values()
        ]

    def set_call_observer(
        self,
        observer: Callable[[dict[str, Any]], None] | None,
    ) -> None:
        self._call_observer = observer

    def invoke(self, name: str, arguments: Mapping[str, Any]) -> ToolResult[Any]:
        tool = self.get(name)
        start = perf_counter()
        try:
            validated = tool.args_schema.model_validate(arguments)
            data = tool.run(validated)
            metadata = dict(tool.build_metadata(validated, data) or {})
            latency_ms = self._latency_ms(start)
            metadata["latency_ms"] = latency_ms
            result = ToolResult(
                tool_name=tool.name.strip(),
                success=True,
                data=data,
                error=None,
                metadata=metadata,
            )
            self._notify(
                tool_name=result.tool_name,
                success=True,
                latency_ms=latency_ms,
                error=None,
                metadata=metadata,
            )
            return result
        except ValidationError as exc:
            return self._failure_result(
                tool=tool,
                start=start,
                error=self._validation_error_message(exc),
            )
        except ToolInputError as exc:
            return self._failure_result(
                tool=tool,
                start=start,
                error=error_info_from_exception(exc, default_code="tool_input_error"),
            )
        except ToolExecutionError as exc:
            return self._failure_result(
                tool=tool,
                start=start,
                error=error_info_from_exception(exc, default_code="tool_execution_error"),
            )
        except Exception as exc:
            return self._failure_result(
                tool=tool,
                start=start,
                error=error_info_from_exception(exc, default_code="tool_execution_error"),
            )

    def _failure_result(
        self,
        *,
        tool: BaseTool[Any, Any],
        start: float,
        error: ToolErrorInfo,
    ) -> ToolResult[Any]:
        latency_ms = self._latency_ms(start)
        metadata = {"latency_ms": latency_ms}
        result = ToolResult(
            tool_name=tool.name.strip(),
            success=False,
            data=None,
            error=error,
            metadata=metadata,
        )
        self._notify(
            tool_name=result.tool_name,
            success=False,
            latency_ms=latency_ms,
            error={"code": error.code, "message": error.message},
            metadata={},
        )
        return result

    def _notify(
        self,
        *,
        tool_name: str,
        success: bool,
        latency_ms: float,
        error: dict[str, Any] | None,
        metadata: dict[str, Any],
    ) -> None:
        if self._call_observer is None:
            return

        record = {
            "tool_name": tool_name,
            "success": success,
            "latency_ms": latency_ms,
            "error": snapshot_observer_value(error) if error is not None else None,
            "metadata": self._observer_metadata_snapshot(metadata),
        }
        try:
            self._call_observer(record)
        except Exception:
            logger.exception("tool call observer failed")

    @staticmethod
    def _latency_ms(start: float) -> float:
        return round((perf_counter() - start) * 1000, 3)

    @staticmethod
    def _validation_error_message(exc: ValidationError) -> ToolErrorInfo:
        details: list[str] = []
        for error in exc.errors(include_input=True, include_url=False):
            loc = error.get("loc", ())
            loc_text = ".".join(str(part) for part in loc) if loc else "<model>"
            err_type = str(error.get("type", "validation_error"))
            details.append(
                f"loc={loc_text}; type={err_type}; message=Value failed validation."
            )
        message = "; ".join(details) if details else "Invalid tool input"
        return ToolErrorInfo(code="tool_input_error", message=message)

    @staticmethod
    def _observer_metadata_snapshot(metadata: dict[str, Any]) -> dict[str, Any]:
        try:
            return snapshot_observer_value(
                {key: value for key, value in metadata.items() if key != "latency_ms"}
            )
        except Exception:
            return {}
