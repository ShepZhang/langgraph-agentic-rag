from __future__ import annotations

import pytest
from pydantic import ValidationError

from tools import ToolContext, ToolRegistry
from tools.calculator_tool import CalculatorArgs, CalculatorTool


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(12 + 8) / 5", 4.0),
        ("2**4 + 3", 19),
        ("-7 // 2", -4),
    ],
)
def test_calculator_evaluates_bounded_arithmetic(expression: str, expected: int | float):
    registry = ToolRegistry()
    registry.register(CalculatorTool(ToolContext()))

    result = registry.invoke("calculator", {"expression": expression})

    assert result.success is True
    assert result.data == {"value": expected}


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('id')",
        "value + 1",
        "(1).__class__",
        "[1, 2][0]",
        '"abc"',
        "True",
        "2 ** 100",
    ],
)
def test_calculator_rejects_unsafe_or_unbounded_expressions(expression: str):
    registry = ToolRegistry()
    registry.register(CalculatorTool(ToolContext()))

    result = registry.invoke("calculator", {"expression": expression})

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "tool_execution_error"


def test_calculator_rejects_division_by_zero():
    registry = ToolRegistry()
    registry.register(CalculatorTool(ToolContext()))

    result = registry.invoke("calculator", {"expression": "1 / 0"})

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "tool_execution_error"


def test_calculator_rejects_extra_fields():
    registry = ToolRegistry()
    registry.register(CalculatorTool(ToolContext()))

    result = registry.invoke("calculator", {"expression": "1 + 1", "extra": True})

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "tool_input_error"


def test_calculator_rejects_overlong_expression():
    registry = ToolRegistry()
    registry.register(CalculatorTool(ToolContext()))

    result = registry.invoke("calculator", {"expression": "1+" * 101})

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "tool_input_error"


def test_calculator_args_forbid_extra_fields():
    with pytest.raises(ValidationError):
        CalculatorArgs.model_validate({"expression": "1 + 1", "extra": True})

