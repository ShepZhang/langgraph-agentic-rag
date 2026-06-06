"""Tests for the three-variant evaluation matrix."""

from __future__ import annotations

import importlib
import json
from dataclasses import replace

import pytest

from config import get_settings
from evaluation import evaluate as evaluator


def _matrix_module():
    return importlib.import_module("evaluation.matrix")


def _configured_settings():
    return replace(
        get_settings(),
        llm_provider="openai_compatible",
        openai_api_key="test-key",
        openai_base_url="https://example.test/v1",
        openai_model="test-model",
    )


def _system_result(
    answer: str,
    source: str = "notes.md",
) -> dict[str, object]:
    return {
        "answer": answer,
        "citations": [{"source": source}],
        "claims": [{"claim": answer}],
        "is_verified": True,
        "retrieved_documents": [{"source": source}],
        "relevant_documents": [{"source": source}],
        "retry_count": 0,
    }


class StepTimer:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        current = self.value
        self.value += 0.5
        return current


def test_public_wrappers_match_existing_evaluator_behavior():
    evaluate_single_system = getattr(evaluator, "evaluate_single_system", None)
    summarize_results = getattr(evaluator, "summarize_results", None)

    assert evaluate_single_system is not None
    assert summarize_results is not None

    item = {
        "question": "What is RAG?",
        "expected_keywords": ["retrieval"],
        "expected_sources": ["notes.md"],
    }

    def runner(_question):
        return _system_result("Retrieval augments generation.")

    public_result = evaluate_single_system(item, runner, StepTimer())
    private_result = evaluator._evaluate_single_system(item, runner, StepTimer())

    assert public_result == private_result
    assert summarize_results([public_result], [item]) == evaluator._summarize(
        [private_result],
        [item],
    )


def test_evaluate_matrix_preserves_question_and_runner_order_with_isolated_results():
    matrix = _matrix_module()
    calls = []
    questions = [
        {
            "question": "First?",
            "expected_sources": ["reranked.md"],
            "requires_rewrite": True,
        },
        {
            "question": "Second?",
            "expected_sources": ["reranked.md"],
            "requires_rewrite": False,
        },
    ]

    def make_runner(name, source):
        def runner(question):
            calls.append((question, name))
            return _system_result(f"{name}: {question}", source=source)

        return runner

    runners = {
        "naive": make_runner("naive", "plain.md"),
        "agentic": make_runner("agentic", "plain.md"),
        "agentic_reranker": make_runner("agentic_reranker", "reranked.md"),
    }

    report = matrix.evaluate_matrix(questions, runners, timer=StepTimer())

    assert calls == [
        ("First?", "naive"),
        ("First?", "agentic"),
        ("First?", "agentic_reranker"),
        ("Second?", "naive"),
        ("Second?", "agentic"),
        ("Second?", "agentic_reranker"),
    ]
    assert report["summary"]["mode"] == "matrix"
    assert report["summary"]["total_questions"] == 2
    assert list(report["summary"]["variants"]) == list(runners)
    assert [result["question"] for result in report["results"]] == [
        "First?",
        "Second?",
    ]
    assert report["results"][0]["requires_rewrite"] is True
    assert list(report["results"][0]["systems"]) == list(runners)
    assert report["results"][0]["systems"]["naive"]["answer"].startswith("naive:")
    assert report["results"][0]["systems"]["agentic"]["answer"].startswith(
        "agentic:"
    )
    assert (
        report["results"][0]["systems"]["agentic_reranker"]["source_hit"] is True
    )
    assert report["summary"]["variants"]["agentic_reranker"]["source_hit_rate"] == 1.0
    assert report["summary"]["variants"]["naive"]["source_hit_rate"] == 0.0


@pytest.mark.parametrize(
    "runners",
    [
        {"naive": lambda _question: {}},
        {
            "naive": lambda _question: {},
            "agentic": lambda _question: {},
            "agentic_reranker": lambda _question: {},
            "extra": lambda _question: {},
        },
    ],
)
def test_evaluate_matrix_rejects_missing_or_extra_runners(runners):
    matrix = _matrix_module()

    with pytest.raises(ValueError, match="exactly"):
        matrix.evaluate_matrix([], runners)


def test_evaluate_matrix_records_runner_errors_and_continues_other_variants():
    matrix = _matrix_module()
    calls = []

    def failing_runner(_question):
        calls.append("naive")
        raise RuntimeError("baseline failed")

    def successful_runner(name):
        def runner(_question):
            calls.append(name)
            return _system_result(name)

        return runner

    report = matrix.evaluate_matrix(
        [{"question": "Question?"}],
        {
            "naive": failing_runner,
            "agentic": successful_runner("agentic"),
            "agentic_reranker": successful_runner("agentic_reranker"),
        },
        timer=StepTimer(),
    )

    assert calls == ["naive", "agentic", "agentic_reranker"]
    assert report["results"][0]["systems"]["naive"]["error"] == (
        "RuntimeError: baseline failed"
    )
    assert report["summary"]["variants"]["naive"]["error_count"] == 1
    assert report["results"][0]["systems"]["agentic"]["answer"] == "agentic"
    assert (
        report["results"][0]["systems"]["agentic_reranker"]["answer"]
        == "agentic_reranker"
    )


def test_format_matrix_report_has_fixed_header_source_hit_row_and_na():
    matrix = _matrix_module()
    report = {
        "summary": {
            "variants": {
                "naive": {"source_hit_rate": 0.25},
                "agentic": {"source_hit_rate": 0.75},
                "agentic_reranker": {"source_hit_rate": 1.0},
            }
        }
    }

    text = matrix.format_matrix_report(report)

    assert "| Metric | Naive RAG | Agentic RAG | Agentic + Reranker |" in text
    assert "| Retrieval Source Hit Rate | 0.25 | 0.75 | 1.0 |" in text
    assert "| Keyword Hit Rate | N/A | N/A | N/A |" in text
    assert "| Error Count | N/A | N/A | N/A |" in text


def test_build_benchmark_runners_uses_independent_reranker_settings(monkeypatch):
    matrix = _matrix_module()
    retriever_settings = []
    calls = []

    class FakeRetriever:
        def __init__(self, settings):
            self.settings = settings
            retriever_settings.append(settings)

        def retrieve(self, question):
            return [
                {
                    "question": question,
                    "reranker_enabled": self.settings.reranker_enabled,
                }
            ]

    def fake_naive(question, retriever_fn=None, settings=None):
        calls.append(("naive", question, retriever_fn(question), settings))
        return {"variant": "naive"}

    def fake_agent(question, retriever_fn=None, settings=None):
        calls.append(("agentic", question, retriever_fn(question), settings))
        return {"variant": "agentic", "reranker": settings.reranker_enabled}

    monkeypatch.setattr(matrix, "Retriever", FakeRetriever)
    monkeypatch.setattr(matrix, "run_naive_rag", fake_naive)
    monkeypatch.setattr(matrix, "run_agent", fake_agent)
    base = _configured_settings()

    runners = matrix.build_benchmark_runners(base)
    outputs = {name: runner("Question?") for name, runner in runners.items()}

    assert list(runners) == ["naive", "agentic", "agentic_reranker"]
    assert [settings.reranker_enabled for settings in retriever_settings] == [
        False,
        True,
    ]
    assert retriever_settings[0] is not base
    assert retriever_settings[1] is not base
    assert retriever_settings[0] is not retriever_settings[1]
    assert outputs == {
        "naive": {"variant": "naive"},
        "agentic": {"variant": "agentic", "reranker": False},
        "agentic_reranker": {"variant": "agentic", "reranker": True},
    }
    assert calls[0][2][0]["reranker_enabled"] is False
    assert calls[1][2][0]["reranker_enabled"] is False
    assert calls[2][2][0]["reranker_enabled"] is True
    assert calls[0][3] is retriever_settings[0]
    assert calls[1][3] is retriever_settings[0]
    assert calls[2][3] is retriever_settings[1]


def test_build_benchmark_runners_requires_llm_config_before_retrievers(monkeypatch):
    matrix = _matrix_module()

    class MissingSettings:
        def require_llm_config(self):
            raise RuntimeError("Missing LLM configuration")

    def fail_retriever_construction(*_args, **_kwargs):
        raise AssertionError("retrievers must not be constructed")

    monkeypatch.setattr(matrix, "Retriever", fail_retriever_construction)

    with pytest.raises(RuntimeError, match="Missing LLM configuration"):
        matrix.build_benchmark_runners(MissingSettings())


def test_matrix_main_writes_nested_json_and_prints_markdown(tmp_path, capsys):
    matrix = _matrix_module()
    question_path = tmp_path / "questions.json"
    question_path.write_text(
        json.dumps(
            [
                {
                    "question": "What is RAG?",
                    "expected_sources": ["notes.md"],
                }
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "nested" / "reports" / "matrix.json"
    runners = {
        "naive": lambda _question: _system_result("朴素答案"),
        "agentic": lambda _question: _system_result("智能答案"),
        "agentic_reranker": lambda _question: _system_result("重排答案"),
    }

    exit_code = matrix.main(
        [
            "--questions",
            str(question_path),
            "--json-output",
            str(output_path),
        ],
        runner_builder=lambda: runners,
    )

    captured = capsys.readouterr()
    saved_text = output_path.read_text(encoding="utf-8")
    saved_report = json.loads(saved_text)
    assert exit_code == 0
    assert "| Metric | Naive RAG | Agentic RAG | Agentic + Reranker |" in captured.out
    assert saved_report["summary"]["mode"] == "matrix"
    assert "重排答案" in saved_text
    assert "\\u91cd" not in saved_text


def test_matrix_main_reports_runner_construction_errors_without_secret(
    tmp_path,
    capsys,
):
    matrix = _matrix_module()
    question_path = tmp_path / "questions.json"
    question_path.write_text(
        json.dumps([{"question": "What is RAG?"}]),
        encoding="utf-8",
    )

    def failing_builder():
        raise RuntimeError("Missing LLM configuration")

    with pytest.raises(SystemExit) as exc_info:
        matrix.main(
            ["--questions", str(question_path)],
            runner_builder=failing_builder,
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "Unable to build benchmark runners" in captured.err
    assert "Missing LLM configuration" in captured.err
    assert "test-key" not in captured.err
    assert "Traceback" not in captured.err
