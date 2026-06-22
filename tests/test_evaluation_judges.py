"""Tests for optional evaluation judge contracts."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from evaluation.judge_config import EvaluationJudgeSettings
from evaluation.judge_evidence import format_judge_citations, format_judge_evidence
from evaluation.judges import (
    MAX_JUDGE_ERROR_CHARS,
    SEMANTIC_JUDGE_PROMPT_ID,
    SEMANTIC_JUDGE_PROMPT_VERSION,
    DeepSeekJudge,
    DisabledJudge,
    build_configured_judge,
    invoke_judge,
    sanitize_judge_error,
)
from evaluation.schemas import EvaluationQuestion, EvaluationResult
from prompting import get_prompt_definition, render_prompt


class FakeLLM:
    def __init__(self, response_or_exc):
        self._response_or_exc = response_or_exc
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        if isinstance(self._response_or_exc, Exception):
            raise self._response_or_exc
        return self._response_or_exc


class MessageResponse:
    def __init__(self, content):
        self.content = content


def _question(
    *,
    answerable: bool = True,
    gold_answer: str = "Retrieval augmented generation.",
) -> EvaluationQuestion:
    return EvaluationQuestion(
        id="q001",
        question="What is RAG?",
        question_type="single_doc",
        gold_answer=gold_answer,
        expected_sources=["notes.md"],
        expected_keywords=["retrieval"],
        source_match_mode="any",
        answerable=answerable,
        expected_behavior="answer_with_citation",
        chat_history=[],
        requires_rewrite=False,
    )


def _result(*, fallback_triggered: bool = False) -> EvaluationResult:
    result = EvaluationResult.empty(
        question_id="q001",
        question_type="single_doc",
        question="What is RAG?",
    )
    result.answer = "RAG combines retrieval with generation and cites supporting notes."
    result.fallback_triggered = fallback_triggered
    result.citations = [
        {
            "source": "/tmp/docs/notes.md",
            "page": 2,
            "chunk_id": "chunk-7",
            "snippet": "RAG combines retrieval with generation.",
        }
    ]
    result.relevant_documents = [
        {
            "source": "/tmp/docs/notes.md",
            "page": 2,
            "chunk_id": "chunk-7",
            "content": "RAG combines retrieval with generation and grounds the answer in retrieved evidence.",
        }
    ]
    return result


def _semantic_payload(*, fallback_triggered: bool = False) -> str:
    groundedness = (
        '{"applicable":false,"score":null,"reason":"Groundedness not applicable for fallback answers."}'
        if fallback_triggered
        else '{"applicable":true,"score":4,"reason":"The answer is fully supported by the supplied note."}'
    )
    return (
        '{"semantic_correctness":{"score":3,"reason":"The answer captures the core meaning with minor wording differences."},'
        '"groundedness":'
        + groundedness
        + "}"
    )


def test_disabled_judge_performs_no_scoring():
    judge = DisabledJudge()

    result = judge.evaluate(_question(), _result())

    assert result.status == "disabled"
    assert result.scores == {}
    assert result.reason == ""
    assert result.error is None


def test_deepseek_judge_calls_llm_once_and_returns_completed_scores_with_prompt_metadata():
    llm = FakeLLM(_semantic_payload())
    judge = DeepSeekJudge(llm, model="deepseek-reasoner", api_key="secret-key")
    question = _question()
    evaluation = _result()

    result = judge.evaluate(question, evaluation)

    expected_definition = get_prompt_definition(
        SEMANTIC_JUDGE_PROMPT_ID,
        version=SEMANTIC_JUDGE_PROMPT_VERSION,
    )
    expected_prompt = render_prompt(
        SEMANTIC_JUDGE_PROMPT_ID,
        version=SEMANTIC_JUDGE_PROMPT_VERSION,
        question=question.question,
        gold_answer=question.gold_answer,
        should_answer="true",
        fallback_triggered="false",
        system_answer=evaluation.answer,
        citations=format_judge_citations(evaluation),
        evidence=format_judge_evidence(evaluation),
    )

    assert llm.prompts == [expected_prompt]
    assert result.status == "completed"
    assert result.reason == "Semantic Judge completed."
    assert result.raw_scores == {"semantic_correctness": 3, "groundedness": 4}
    assert result.scores == {"semantic_correctness": 0.75, "groundedness": 1.0}
    assert result.reasons == {
        "semantic_correctness": "The answer captures the core meaning with minor wording differences.",
        "groundedness": "The answer is fully supported by the supplied note.",
    }
    assert result.error is None
    assert result.model == "deepseek-reasoner"
    assert result.prompt_id == expected_definition.prompt_id
    assert result.prompt_version == expected_definition.version
    assert result.prompt_fingerprint == expected_definition.fingerprint


@pytest.mark.parametrize(
    ("response", "label"),
    [
        pytest.param(
            MessageResponse(_semantic_payload()),
            "message-string",
        ),
        pytest.param(
            MessageResponse(
                [
                    {
                        "type": "text",
                        "text": {"content": _semantic_payload()},
                    }
                ]
            ),
            "message-list-content",
        ),
    ],
)
def test_deepseek_judge_accepts_message_content_shapes(response, label):
    llm = FakeLLM(response)
    judge = DeepSeekJudge(llm, model="deepseek-chat", api_key="secret-key")

    result = judge.evaluate(_question(), _result())

    assert result.status == "completed", label
    assert result.raw_scores["semantic_correctness"] == 3
    assert result.raw_scores["groundedness"] == 4


def test_deepseek_judge_returns_groundedness_none_for_fallback_answers():
    llm = FakeLLM(_semantic_payload(fallback_triggered=True))
    judge = DeepSeekJudge(llm, model="deepseek-chat", api_key="secret-key")

    result = judge.evaluate(_question(answerable=False), _result(fallback_triggered=True))

    assert result.status == "completed"
    assert result.raw_scores == {"semantic_correctness": 3, "groundedness": None}
    assert result.scores == {"semantic_correctness": 0.75, "groundedness": None}
    assert result.reasons["groundedness"] == "Groundedness not applicable for fallback answers."


def test_deepseek_judge_marks_semantic_correctness_unavailable_without_gold_answer():
    llm = FakeLLM(_semantic_payload())
    judge = DeepSeekJudge(llm, model="deepseek-chat", api_key="secret-key")

    result = judge.evaluate(_question(gold_answer="   "), _result())

    assert len(llm.prompts) == 1
    assert result.status == "completed"
    assert result.raw_scores == {"semantic_correctness": None, "groundedness": 4}
    assert result.scores == {"semantic_correctness": None, "groundedness": 1.0}
    assert result.reasons["semantic_correctness"] == (
        "Semantic correctness unavailable: gold answer was not provided."
    )


@pytest.mark.parametrize("response", ["", "not json"])
def test_deepseek_judge_returns_failed_result_for_blank_or_malformed_response(response):
    llm = FakeLLM(response)
    judge = DeepSeekJudge(llm, model="deepseek-chat", api_key="secret-key")

    result = judge.evaluate(_question(), _result())

    assert result.status == "failed"
    assert result.scores == {}
    assert result.raw_scores == {}
    assert result.reasons == {}
    assert result.model == "deepseek-chat"
    assert result.prompt_id == SEMANTIC_JUDGE_PROMPT_ID
    assert result.prompt_version == SEMANTIC_JUDGE_PROMPT_VERSION
    assert result.prompt_fingerprint == get_prompt_definition(
        SEMANTIC_JUDGE_PROMPT_ID,
        version=SEMANTIC_JUDGE_PROMPT_VERSION,
    ).fingerprint


def test_sanitize_judge_error_redacts_api_keys_bearer_tokens_and_normalizes_whitespace():
    api_key = "sk-live-exact-key"
    bearer_token = "Bearer top-secret-token"
    embedded_token = "authorization = Bearer another-secret-token"
    exc = RuntimeError(
        "network failed\n"
        f"  api_key={api_key}\n"
        f"api-key: {api_key}\n"
        f"api key = {api_key}\n"
        f"Authorization: {api_key}\n"
        f"{bearer_token}\n"
        f"{embedded_token}\n"
        " more   whitespace "
    )

    message = sanitize_judge_error(exc, api_key=api_key)

    assert message.startswith("RuntimeError:")
    assert api_key not in message
    assert "top-secret-token" not in message
    assert "another-secret-token" not in message
    assert "\n" not in message
    assert "  " not in message
    assert len(message) <= MAX_JUDGE_ERROR_CHARS


def test_sanitize_judge_error_truncates_long_messages():
    exc = RuntimeError("x" * (MAX_JUDGE_ERROR_CHARS * 3))

    message = sanitize_judge_error(exc)

    assert message.startswith("RuntimeError:")
    assert len(message) <= MAX_JUDGE_ERROR_CHARS


def test_build_configured_judge_returns_disabled_without_calling_model_factory():
    calls: list[EvaluationJudgeSettings] = []

    def model_factory(settings: EvaluationJudgeSettings):
        calls.append(settings)
        raise AssertionError("model_factory should not be called for disabled settings")

    judge = build_configured_judge(
        EvaluationJudgeSettings(enabled=False),
        model_factory=model_factory,
    )

    assert isinstance(judge, DisabledJudge)
    assert calls == []


def test_build_configured_judge_builds_deepseek_judge_with_injected_settings_once():
    settings = EvaluationJudgeSettings(
        enabled=True,
        api_key="judge-api-key",
        base_url="https://example.test/v1",
        model="deepseek-chat",
        temperature=0.0,
    )
    calls: list[EvaluationJudgeSettings] = []
    llm = FakeLLM(_semantic_payload())

    def model_factory(received_settings: EvaluationJudgeSettings):
        calls.append(received_settings)
        return llm

    judge = build_configured_judge(settings, model_factory=model_factory)

    assert isinstance(judge, DeepSeekJudge)
    assert calls == [settings]


def test_invoke_judge_records_failure_without_raising():
    class BrokenJudge:
        def evaluate(
            self,
            question: EvaluationQuestion,
            result: EvaluationResult,
        ):
            raise RuntimeError("judge unavailable")

    result = invoke_judge(BrokenJudge(), _question(), _result())

    assert result.status == "failed"
    assert result.error == "RuntimeError: judge unavailable"
    assert result.scores == {}


def test_invoke_judge_formats_empty_exception_message_as_class_name():
    class BrokenJudge:
        def evaluate(
            self,
            question: EvaluationQuestion,
            result: EvaluationResult,
        ):
            raise RuntimeError()

    result = invoke_judge(BrokenJudge(), _question(), _result())

    assert result.status == "failed"
    assert result.error == "RuntimeError"
    assert result.scores == {}


def test_judge_result_serialization_excludes_raw_prompt_and_response_fields():
    llm = FakeLLM(_semantic_payload())
    judge = DeepSeekJudge(llm, model="deepseek-chat", api_key="secret-key")

    payload = asdict(judge.evaluate(_question(), _result()))

    assert set(payload) == {
        "status",
        "scores",
        "reason",
        "error",
        "raw_scores",
        "reasons",
        "model",
        "prompt_id",
        "prompt_version",
        "prompt_fingerprint",
    }
    assert "raw_prompt" not in payload
    assert "prompt" not in payload
    assert "raw_response" not in payload
    assert "response" not in payload
