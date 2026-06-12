from __future__ import annotations

from collections.abc import Callable, Mapping
import logging
from time import perf_counter
from typing import Any, Annotated, Union, get_args, get_origin
from types import UnionType

from pydantic import BaseModel, ValidationError

from tools.base import (
    BaseTool,
    ToolErrorInfo,
    ToolExecutionError,
    ToolInputError,
    ToolNotFoundError,
    ToolRegistrationError,
    ToolResult,
    error_info_from_exception,
    redact_tool_message,
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
                metadata=self._observer_metadata_snapshot(tool, metadata),
            )
            return result
        except ValidationError as exc:
            return self._failure_result(
                tool=tool,
                start=start,
                error=self._validation_error_message(exc, tool.args_schema),
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
            "metadata": metadata,
        }
        try:
            self._call_observer(record)
        except Exception:
            logger.exception("tool call observer failed")

    @staticmethod
    def _latency_ms(start: float) -> float:
        return round((perf_counter() - start) * 1000, 3)

    @staticmethod
    def _validation_error_message(
        exc: ValidationError,
        schema: type[Any],
    ) -> ToolErrorInfo:
        all_errors = exc.errors(include_input=True, include_url=False)
        details: list[str] = []
        for error in all_errors[:5]:
            detail = ToolRegistry._format_validation_error(error, schema)
            omitted = len(all_errors) - (len(details) + 1)
            candidate = "; ".join(details + [detail])
            if omitted > 0:
                candidate = f"{candidate}; omitted={omitted}"
            if len(candidate) > 500:
                break
            details.append(detail)

        omitted = len(all_errors) - len(details)
        message = "; ".join(details) if details else "Invalid tool input"
        if omitted > 0:
            message = f"{message}; omitted={omitted}"
        return ToolErrorInfo(code="tool_input_error", message=redact_tool_message(message))

    @staticmethod
    def _format_validation_error(
        error: dict[str, Any],
        schema: type[Any],
    ) -> str:
        loc_text = ToolRegistry._safe_validation_location(schema, error.get("loc", ()))
        if len(loc_text) > 120:
            loc_text = f"{loc_text[:117]}..."
        err_type = str(error.get("type", "validation_error"))
        return f"loc={loc_text}; type={err_type}; message=Value failed validation."

    @staticmethod
    def _safe_validation_location(
        schema: type[Any] | Any,
        loc: tuple[Any, ...] | list[Any] | Any,
    ) -> str:
        if not loc:
            return "<model>"

        if not isinstance(loc, (tuple, list)):
            loc = (loc,)

        parts = ToolRegistry._safe_validation_location_parts(schema, list(loc))
        return ".".join(parts) if parts else "<model>"

    @staticmethod
    def _safe_validation_location_parts(
        annotation: type[Any] | Any,
        loc: list[Any],
    ) -> list[str]:
        if not loc:
            return []

        annotation = ToolRegistry._strip_annotated(annotation)
        annotation = ToolRegistry._resolve_union(annotation, loc[0])

        if ToolRegistry._is_model(annotation):
            field = ToolRegistry._model_field_for_segment(annotation, loc[0])
            if field is None:
                return ["<key>"] + ToolRegistry._safe_validation_location_parts(Any, loc[1:])
            return [
                str(loc[0]),
                *ToolRegistry._safe_validation_location_parts(field.annotation, loc[1:]),
            ]

        if ToolRegistry._is_mapping(annotation):
            return [
                "<key>",
                *ToolRegistry._safe_validation_location_parts(ToolRegistry._mapping_value_type(annotation), loc[1:]),
            ]

        if ToolRegistry._is_sequence(annotation):
            if isinstance(loc[0], int):
                next_annotation = ToolRegistry._sequence_item_type(annotation, loc[0])
                return [
                    str(loc[0]),
                    *ToolRegistry._safe_validation_location_parts(next_annotation, loc[1:]),
                ]
            return [
                "<key>",
                *ToolRegistry._safe_validation_location_parts(ToolRegistry._sequence_item_type(annotation, 0), loc[1:]),
            ]

        if isinstance(loc[0], int):
            return [str(loc[0]), *ToolRegistry._safe_validation_location_parts(Any, loc[1:])]

        return ["<key>", *ToolRegistry._safe_validation_location_parts(Any, loc[1:])]

    @staticmethod
    def _strip_annotated(annotation: Any) -> Any:
        while get_origin(annotation) is Annotated:
            annotation = get_args(annotation)[0]
        return annotation

    @staticmethod
    def _resolve_union(annotation: Any, segment: Any) -> Any:
        origin = get_origin(annotation)
        if origin not in (Union, UnionType):
            return annotation

        candidates = [arg for arg in get_args(annotation) if arg is not type(None)]
        if not candidates:
            return annotation

        if isinstance(segment, int):
            for candidate in candidates:
                candidate = ToolRegistry._strip_annotated(candidate)
                if ToolRegistry._is_sequence(candidate):
                    return candidate
            return candidates[0]

        for candidate in candidates:
            candidate = ToolRegistry._strip_annotated(candidate)
            if ToolRegistry._is_model(candidate) and ToolRegistry._model_field_for_segment(candidate, segment) is not None:
                return candidate

        for candidate in candidates:
            candidate = ToolRegistry._strip_annotated(candidate)
            if ToolRegistry._is_mapping(candidate):
                return candidate

        for candidate in candidates:
            candidate = ToolRegistry._strip_annotated(candidate)
            if ToolRegistry._is_sequence(candidate):
                return candidate

        return candidates[0]

    @staticmethod
    def _is_model(annotation: Any) -> bool:
        return isinstance(annotation, type) and issubclass(annotation, BaseModel)

    @staticmethod
    def _is_mapping(annotation: Any) -> bool:
        origin = get_origin(annotation)
        return origin in (dict, Mapping) or annotation in (dict, Mapping)

    @staticmethod
    def _is_sequence(annotation: Any) -> bool:
        origin = get_origin(annotation)
        return origin in (list, set, frozenset, tuple) or annotation in (list, set, frozenset, tuple)

    @staticmethod
    def _sequence_item_type(annotation: Any, index: int) -> Any:
        args = get_args(annotation)
        origin = get_origin(annotation)
        if origin is tuple and args:
            if len(args) == 2 and args[1] is Ellipsis:
                return args[0]
            if index < len(args):
                return args[index]
            return args[-1]
        if args:
            return args[0]
        return Any

    @staticmethod
    def _mapping_value_type(annotation: Any) -> Any:
        args = get_args(annotation)
        if len(args) >= 2:
            return args[1]
        return Any

    @staticmethod
    def _model_field_for_segment(model: type[BaseModel], segment: Any):
        segment_text = str(segment)
        for field_name, field in model.model_fields.items():
            if segment_text == field_name:
                return field
            alias = getattr(field, "alias", None)
            if isinstance(alias, str) and segment_text == alias:
                return field
            validation_alias = getattr(field, "validation_alias", None)
            if isinstance(validation_alias, str) and segment_text == validation_alias:
                return field
            choices = getattr(validation_alias, "choices", None)
            if choices and segment_text in {str(choice) for choice in choices}:
                return field
        return None

    def _observer_metadata_snapshot(
        self,
        tool: BaseTool[Any, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            allowed = {
                key: metadata[key]
                for key in tool.trace_metadata_fields
                if key in metadata
            }
            return snapshot_observer_value(allowed)
        except Exception:
            return {}
