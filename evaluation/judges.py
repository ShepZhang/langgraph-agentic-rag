"""Optional evaluation judge contract and DeepSeek semantic judge."""

from __future__ import annotations

import re
from typing import Any, Protocol

from evaluation.judge_config import (
    EvaluationJudgeSettings,
    create_evaluation_judge_model,
    load_evaluation_judge_settings,
)
from evaluation.judge_evidence import format_judge_citations, format_judge_evidence
from evaluation.judge_parsing import parse_semantic_judge_response
from evaluation.schemas import EvaluationQuestion, EvaluationResult, JudgeResult
from prompting import get_prompt_definition, render_prompt
from tools.base import coerce_llm_text


SEMANTIC_JUDGE_PROMPT_ID = "evaluation.semantic_judge"
SEMANTIC_JUDGE_PROMPT_VERSION = "v1"
MAX_JUDGE_ERROR_CHARS = 500
SEMANTIC_CORRECTNESS_UNAVAILABLE_REASON = (
    "Semantic correctness unavailable: gold answer was not provided."
)

_WHITESPACE_RE = re.compile(r"\s+")
_BEARER_TOKEN_RE = re.compile(r"\bBearer\s+\S+", re.IGNORECASE)
_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)"
    r"(?P<prefix>\b(?:api[_ -]?key|authorization)\b\s*[:=]\s*)"
    r"(?P<value>\"[^\"]*\"|'[^']*'|Bearer\s+\S+|[^\s,\]\}]+)"
)


class Judge(Protocol):
    def evaluate(
        self,
        question: EvaluationQuestion,
        result: EvaluationResult,
    ) -> JudgeResult:
        ...


class JudgeLLM(Protocol):
    def invoke(self, prompt: str) -> Any:
        ...


class DisabledJudge:
    def evaluate(
        self,
        question: EvaluationQuestion,
        result: EvaluationResult,
    ) -> JudgeResult:
        return JudgeResult.disabled()


class DeepSeekJudge:
    def __init__(
        self,
        llm: JudgeLLM,
        *,
        model: str,
        api_key: str,
        temperature: float = 0.0,
    ) -> None:
        self.llm = llm
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.prompt_definition = get_prompt_definition(
            SEMANTIC_JUDGE_PROMPT_ID,
            version=SEMANTIC_JUDGE_PROMPT_VERSION,
        )

    def evaluate(
        self,
        question: EvaluationQuestion,
        result: EvaluationResult,
    ) -> JudgeResult:
        try:
            citations = format_judge_citations(result)
            evidence = format_judge_evidence(result)
            prompt = render_prompt(
                self.prompt_definition.prompt_id,
                version=self.prompt_definition.version,
                question=question.question,
                gold_answer=question.gold_answer,
                should_answer=str(question.answerable).lower(),
                fallback_triggered=str(result.fallback_triggered).lower(),
                system_answer=result.answer,
                citations=citations,
                evidence=evidence,
            )
            raw_response = self.llm.invoke(prompt)
            raw_text = coerce_llm_text(raw_response)
            parsed = parse_semantic_judge_response(
                raw_text,
                fallback_triggered=result.fallback_triggered,
            )
            scores = dict(parsed.scores)
            raw_scores = dict(parsed.raw_scores)
            reasons = dict(parsed.reasons)
            if not question.gold_answer.strip():
                scores["semantic_correctness"] = None
                raw_scores["semantic_correctness"] = None
                reasons["semantic_correctness"] = (
                    SEMANTIC_CORRECTNESS_UNAVAILABLE_REASON
                )
            return JudgeResult.completed(
                scores,
                reason="Semantic Judge completed.",
                raw_scores=raw_scores,
                reasons=reasons,
                **self._metadata(),
            )
        except Exception as exc:
            return JudgeResult.failed(
                sanitize_judge_error(exc, api_key=self.api_key),
                **self._metadata(),
            )

    def _metadata(self) -> dict[str, str]:
        return {
            "model": self.model,
            "prompt_id": self.prompt_definition.prompt_id,
            "prompt_version": self.prompt_definition.version,
            "prompt_fingerprint": self.prompt_definition.fingerprint,
        }


def build_configured_judge(
    settings: EvaluationJudgeSettings | None = None,
    *,
    model_factory: Any | None = None,
) -> Judge:
    resolved_settings = (
        load_evaluation_judge_settings() if settings is None else settings
    )
    if not resolved_settings.enabled:
        return DisabledJudge()

    llm = (
        model_factory(resolved_settings)
        if model_factory is not None
        else create_evaluation_judge_model(resolved_settings)
    )
    return DeepSeekJudge(
        llm,
        model=resolved_settings.model,
        api_key=resolved_settings.api_key,
        temperature=resolved_settings.temperature,
    )


def describe_judge_runtime(
    judge: Judge,
    *,
    result_model: str | None = None,
) -> dict[str, bool | str | float | None]:
    """Return sanitized runtime metadata for a resolved Judge instance."""

    if isinstance(judge, DisabledJudge):
        return {
            "enabled": False,
            "provider": "openai_compatible",
            "model": None,
            "temperature": 0.0,
        }
    if isinstance(judge, DeepSeekJudge):
        return {
            "enabled": True,
            "provider": "openai_compatible",
            "model": judge.model,
            "temperature": judge.temperature,
        }

    model = result_model
    if model is None:
        candidate = getattr(judge, "model", None)
        if isinstance(candidate, str) and candidate.strip():
            model = candidate.strip()
    return {
        "enabled": True,
        "provider": "injected",
        "model": model,
        "temperature": None,
    }


def invoke_judge(
    judge: Judge,
    question: EvaluationQuestion,
    result: EvaluationResult,
) -> JudgeResult:
    try:
        return judge.evaluate(question, result)
    except Exception as exc:
        return JudgeResult.failed(sanitize_judge_error(exc))


def sanitize_judge_error(exc: Exception, *, api_key: str = "") -> str:
    message = str(exc)
    prefix = type(exc).__name__
    if not message:
        return prefix

    sanitized = message
    if api_key:
        sanitized = re.sub(re.escape(api_key), "[REDACTED]", sanitized)
    sanitized = _BEARER_TOKEN_RE.sub("Bearer [REDACTED]", sanitized)
    sanitized = _SENSITIVE_ASSIGNMENT_RE.sub(r"\g<prefix>[REDACTED]", sanitized)
    sanitized = _WHITESPACE_RE.sub(" ", sanitized).strip()

    if not sanitized:
        return prefix

    return f"{prefix}: {sanitized}"[:MAX_JUDGE_ERROR_CHARS]
