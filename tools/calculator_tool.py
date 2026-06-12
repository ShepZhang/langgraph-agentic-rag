"""Safe arithmetic calculator based on a bounded AST whitelist."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from tools.base import BaseTool, ToolExecutionError


class CalculatorArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expression: str = Field(min_length=1, max_length=200)


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorTool(BaseTool[CalculatorArgs, dict[str, int | float]]):
    name = "calculator"
    description = "Evaluate a bounded arithmetic expression."
    args_schema = CalculatorArgs

    def run(self, arguments: CalculatorArgs) -> dict[str, int | float]:
        try:
            tree = ast.parse(arguments.expression, mode="eval")
        except SyntaxError as exc:
            raise ToolExecutionError("Invalid arithmetic expression.") from exc

        if sum(1 for _ in ast.walk(tree)) > 64:
            raise ToolExecutionError("Arithmetic expression is too complex.")

        value = self._evaluate(tree.body)
        if isinstance(value, float) and not math.isfinite(value):
            raise ToolExecutionError("Arithmetic result must be finite.")
        return {"value": value}

    def _evaluate(self, node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ToolExecutionError("Only numeric constants are allowed.")
            return node.value

        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](self._evaluate(node.operand))

        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = self._evaluate(node.left)
            right = self._evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ToolExecutionError("Exponent magnitude must not exceed 10.")
            result = _BINARY_OPERATORS[type(node.op)](left, right)
            if isinstance(result, float) and not math.isfinite(result):
                raise ToolExecutionError("Arithmetic result must be finite.")
            return result

        raise ToolExecutionError("Expression contains a disallowed operation.")


__all__ = ["CalculatorArgs", "CalculatorTool"]
