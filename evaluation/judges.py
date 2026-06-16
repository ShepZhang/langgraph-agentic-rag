"""Optional evaluation judge contract.

This module defines an extension boundary for future judge implementations.
The default judge is disabled and does not call external models.
"""

from __future__ import annotations

from typing import Protocol

from evaluation.schemas import EvaluationQuestion, EvaluationResult, JudgeResult


class Judge(Protocol):
    def evaluate(
        self,
        question: EvaluationQuestion,
        result: EvaluationResult,
    ) -> JudgeResult:
        ...


class DisabledJudge:
    def evaluate(
        self,
        question: EvaluationQuestion,
        result: EvaluationResult,
    ) -> JudgeResult:
        return JudgeResult.disabled()


def invoke_judge(
    judge: Judge,
    question: EvaluationQuestion,
    result: EvaluationResult,
) -> JudgeResult:
    try:
        return judge.evaluate(question, result)
    except Exception as exc:
        return JudgeResult.failed(f"{type(exc).__name__}: {exc}")
