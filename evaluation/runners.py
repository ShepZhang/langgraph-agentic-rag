"""Evaluation runner adaptation and single-question execution."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from typing import Any, Protocol

from agent.state import ChatMessage
from evaluation.metrics import (
    build_error_result,
    score_system_output,
)
from evaluation.schemas import EvaluationQuestion, EvaluationResult


class EvaluationRunner(Protocol):
    """History-aware evaluation runner interface."""

    def run(
        self,
        question: str,
        chat_history: list[ChatMessage],
    ) -> dict[str, Any]:
        """Run one evaluation question."""


class CallableRunnerAdapter:
    """Adapt legacy callables to the history-aware runner protocol."""

    def __init__(self, runner: Callable[..., dict[str, Any]]) -> None:
        self._runner = runner

    def run(
        self,
        question: str,
        chat_history: list[ChatMessage],
    ) -> dict[str, Any]:
        """Invoke a one-argument or history-aware callable."""

        try:
            parameters = inspect.signature(self._runner).parameters.values()
        except (TypeError, ValueError):
            return self._runner(question, chat_history)

        positional_parameters = [
            parameter
            for parameter in parameters
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ]
        accepts_varargs = any(
            parameter.kind is inspect.Parameter.VAR_POSITIONAL
            for parameter in parameters
        )
        if accepts_varargs or len(positional_parameters) >= 2:
            return self._runner(question, chat_history)
        return self._runner(question)


def evaluate_question(
    question: EvaluationQuestion,
    runner: EvaluationRunner,
    timer: Callable[[], float] = time.perf_counter,
) -> EvaluationResult:
    """Evaluate one normalized question and record execution failures as data."""

    started_at = timer()
    try:
        system_result = runner.run(question.question, question.chat_history)
        result = score_system_output(question, system_result)
        error = None
    except Exception as exc:  # noqa: BLE001 - evaluation records system failures.
        result = build_error_result(question)
        error = _format_error(exc)
    latency = timer() - started_at

    result.latency = latency
    result.error = error
    result.failure_analysis = {}
    return result


def _format_error(exc: Exception) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


__all__ = [
    "CallableRunnerAdapter",
    "EvaluationRunner",
    "evaluate_question",
]
