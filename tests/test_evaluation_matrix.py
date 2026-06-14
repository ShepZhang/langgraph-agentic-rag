"""Tests for the three-variant evaluation matrix."""

from __future__ import annotations

import importlib
import inspect
import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any, get_type_hints

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


def test_public_wrappers_have_typed_signatures_and_docstrings():
    single_hints = get_type_hints(evaluator.evaluate_single_system)
    summary_hints = get_type_hints(evaluator.summarize_results)

    assert single_hints == {
        "item": dict[str, Any],
        "runner": Callable[[str], dict[str, Any]],
        "timer": Callable[[], float],
        "return": dict[str, Any],
    }
    assert summary_hints == {
        "results": list[dict[str, Any]],
        "questions": list[dict[str, Any]],
        "return": dict[str, Any],
    }
    assert inspect.signature(evaluator.evaluate_single_system).parameters[
        "timer"
    ].default is evaluator.time.perf_counter
    assert evaluator.evaluate_single_system.__doc__
    assert evaluator.summarize_results.__doc__


def test_evaluate_matrix_uses_canonical_variant_order_with_isolated_results():
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
        "agentic_reranker": make_runner("agentic_reranker", "reranked.md"),
        "naive": make_runner("naive", "plain.md"),
        "agentic": make_runner("agentic", "plain.md"),
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
    assert list(report["summary"]["variants"]) == [
        "naive",
        "agentic",
        "agentic_reranker",
    ]
    assert [result["question"] for result in report["results"]] == [
        "First?",
        "Second?",
    ]
    assert report["results"][0]["requires_rewrite"] is True
    assert list(report["results"][0]["systems"]) == [
        "naive",
        "agentic",
        "agentic_reranker",
    ]
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


def test_format_matrix_report_has_all_metrics_in_fixed_order():
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

    assert text.splitlines() == [
        "| Metric | Naive RAG | Agentic RAG | Agentic + Reranker |",
        "|---|---:|---:|---:|",
        "| Retrieval Source Hit Rate | 0.25 | 0.75 | 1.0 |",
        "| Keyword Hit Rate | N/A | N/A | N/A |",
        "| Citation Rate | N/A | N/A | N/A |",
        "| Claim Verification Rate | N/A | N/A | N/A |",
        "| Fallback Correctness | N/A | N/A | N/A |",
        "| Average Retry Count | N/A | N/A | N/A |",
        "| Average Retrieved Docs | N/A | N/A | N/A |",
        "| Average Relevant Docs | N/A | N/A | N/A |",
        "| Average Latency | N/A | N/A | N/A |",
        "| Error Count | N/A | N/A | N/A |",
    ]


def test_build_benchmark_runners_isolates_managers_and_retrievers(monkeypatch):
    matrix = _matrix_module()
    manager_settings = []
    manager_embeddings = []
    retriever_instances = []
    embedding_calls = []
    calls = []
    shared_embedding = object()

    class FakeVectorStoreManager:
        def __init__(self, settings, embedding_model):
            self.settings = settings
            self.embedding_model = embedding_model
            manager_settings.append(settings)
            manager_embeddings.append(embedding_model)

    class FakeRetriever:
        def __init__(self, vectorstore_manager, settings):
            self.vectorstore_manager = vectorstore_manager
            self.settings = settings
            retriever_instances.append(self)

        def retrieve(self, question):
            return [
                {
                    "question": question,
                    "reranker_enabled": self.settings.reranker_enabled,
                    "manager_id": id(self.vectorstore_manager),
                    "retriever_id": id(self),
                }
            ]

    def fake_naive(question, retriever_fn=None, settings=None):
        calls.append(("naive", question, retriever_fn(question), settings))
        return {"variant": "naive"}

    def fake_agent(question, retriever_fn=None, settings=None):
        calls.append(("agentic", question, retriever_fn(question), settings))
        return {"variant": "agentic", "reranker": settings.reranker_enabled}

    def fake_get_embedding_model(settings):
        embedding_calls.append(settings)
        return shared_embedding

    monkeypatch.setattr(matrix, "get_embedding_model", fake_get_embedding_model)
    monkeypatch.setattr(matrix, "VectorStoreManager", FakeVectorStoreManager)
    monkeypatch.setattr(matrix, "Retriever", FakeRetriever)
    monkeypatch.setattr(matrix, "run_naive_rag", fake_naive)
    monkeypatch.setattr(matrix, "run_agent", fake_agent)
    base = _configured_settings()

    runners = matrix.build_benchmark_runners(base)
    outputs = {name: runner("Question?") for name, runner in runners.items()}

    assert list(runners) == ["naive", "agentic", "agentic_reranker"]
    assert [settings.reranker_enabled for settings in manager_settings] == [
        False,
        False,
        True,
    ]
    assert embedding_calls == [manager_settings[0]]
    assert manager_embeddings == [shared_embedding] * 3
    assert len({id(instance.vectorstore_manager) for instance in retriever_instances}) == 3
    assert len({id(instance) for instance in retriever_instances}) == 3
    assert [
        instance.vectorstore_manager.settings is instance.settings
        for instance in retriever_instances
    ] == [True, True, True]
    assert outputs == {
        "naive": {"variant": "naive"},
        "agentic": {"variant": "agentic", "reranker": False},
        "agentic_reranker": {"variant": "agentic", "reranker": True},
    }
    assert calls[0][2][0]["reranker_enabled"] is False
    assert calls[1][2][0]["reranker_enabled"] is False
    assert calls[2][2][0]["reranker_enabled"] is True
    assert [call[2][0]["retriever_id"] for call in calls] == [
        id(instance) for instance in retriever_instances
    ]
    assert [call[2][0]["manager_id"] for call in calls] == [
        id(instance.vectorstore_manager) for instance in retriever_instances
    ]
    assert calls[0][3] is manager_settings[0]
    assert calls[1][3] is manager_settings[1]
    assert calls[2][3] is manager_settings[2]


def test_build_benchmark_runners_sanitizes_reranker_initialization_error(
    monkeypatch,
):
    matrix = _matrix_module()
    base = replace(
        _configured_settings(),
        reranker_model="safe-reranker-model",
    )

    class FakeVectorStoreManager:
        def __init__(self, settings, embedding_model):
            self.settings = settings
            self.embedding_model = embedding_model

    class FakeRetriever:
        def __init__(self, vectorstore_manager, settings):
            if settings.reranker_enabled:
                raise ImportError("loader failed with sk-sensitive-value")

        def retrieve(self, _question):
            return []

    monkeypatch.setattr(matrix, "get_embedding_model", lambda _settings: object())
    monkeypatch.setattr(matrix, "VectorStoreManager", FakeVectorStoreManager)
    monkeypatch.setattr(matrix, "Retriever", FakeRetriever)

    with pytest.raises(matrix.BenchmarkConfigurationError) as exc_info:
        matrix.build_benchmark_runners(base)

    assert "safe-reranker-model" in str(exc_info.value)
    assert "loader failed" not in str(exc_info.value)
    assert "sk-sensitive-value" not in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ImportError)


def test_matrix_main_reports_safe_reranker_configuration_error(
    tmp_path,
    capsys,
    monkeypatch,
):
    matrix = _matrix_module()
    base = replace(
        _configured_settings(),
        reranker_model="safe-reranker-model",
    )
    retriever_count = 0

    class FakeVectorStoreManager:
        def __init__(self, settings, embedding_model):
            self.settings = settings
            self.embedding_model = embedding_model

    class FakeRetriever:
        def __init__(self, vectorstore_manager, settings):
            nonlocal retriever_count
            retriever_count += 1
            if retriever_count == 3:
                raise ImportError("reranker failed with sk-sensitive-value")

        def retrieve(self, _question):
            return []

    question_path = tmp_path / "questions.json"
    question_path.write_text(
        json.dumps([{"question": "What is RAG?"}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(matrix, "get_settings", lambda: base)
    monkeypatch.setattr(matrix, "get_embedding_model", lambda _settings: object())
    monkeypatch.setattr(matrix, "VectorStoreManager", FakeVectorStoreManager)
    monkeypatch.setattr(matrix, "Retriever", FakeRetriever)

    with pytest.raises(SystemExit) as exc_info:
        matrix.main(["--questions", str(question_path)])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "safe-reranker-model" in captured.err
    assert "reranker failed" not in captured.err
    assert "sk-sensitive-value" not in captured.err
    assert "Traceback" not in captured.err


def test_matrix_main_generalizes_embedding_initialization_error(
    tmp_path,
    capsys,
    monkeypatch,
):
    matrix = _matrix_module()
    question_path = tmp_path / "questions.json"
    question_path.write_text(
        json.dumps([{"question": "What is RAG?"}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(matrix, "get_settings", _configured_settings)

    def failing_embedding(_settings):
        raise ImportError("embedding failed with sk-sensitive-value")

    monkeypatch.setattr(matrix, "get_embedding_model", failing_embedding)

    with pytest.raises(SystemExit) as exc_info:
        matrix.main(["--questions", str(question_path)])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert (
        "Unable to build benchmark runners. Check LLM and reranker configuration."
        in captured.err
    )
    assert "embedding failed" not in captured.err
    assert "sk-sensitive-value" not in captured.err
    assert "Traceback" not in captured.err


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


def test_matrix_main_sanitizes_runner_errors_in_json_without_mutating_report(
    tmp_path,
    monkeypatch,
):
    matrix = _matrix_module()
    questions = [{"question": "What is RAG?"}]

    def failing_runner(_question):
        raise RuntimeError("Authorization: Bearer sk-sensitive-value")

    runners = {
        "naive": failing_runner,
        "agentic": lambda _question: _system_result("Agentic answer"),
        "agentic_reranker": lambda _question: _system_result("Reranked answer"),
    }
    report = matrix.evaluate_matrix(questions, runners, timer=StepTimer())

    assert report["results"][0]["systems"]["naive"]["error"] == (
        "RuntimeError: Authorization: Bearer sk-sensitive-value"
    )

    question_path = tmp_path / "questions.json"
    question_path.write_text(json.dumps(questions), encoding="utf-8")
    output_path = tmp_path / "reports" / "matrix.json"
    monkeypatch.setattr(
        matrix,
        "evaluate_matrix",
        lambda _questions, _runners: report,
    )

    exit_code = matrix.main(
        [
            "--questions",
            str(question_path),
            "--json-output",
            str(output_path),
        ],
        runner_builder=lambda: runners,
    )

    saved_text = output_path.read_text(encoding="utf-8")
    saved_report = json.loads(saved_text)
    assert exit_code == 0
    assert report["results"][0]["systems"]["naive"]["error"] == (
        "RuntimeError: Authorization: Bearer sk-sensitive-value"
    )
    assert saved_report["results"][0]["systems"]["naive"]["error"] == "RuntimeError"
    assert saved_report["summary"]["variants"]["naive"]["error_count"] == 1
    assert "Authorization" not in saved_text
    assert "Bearer" not in saved_text
    assert "sk-sensitive-value" not in saved_text


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
        raise ImportError("reranker failed with sk-sensitive-value")

    with pytest.raises(SystemExit) as exc_info:
        matrix.main(
            ["--questions", str(question_path)],
            runner_builder=failing_builder,
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert (
        "Unable to build benchmark runners. Check LLM and reranker configuration."
        in captured.err
    )
    assert "reranker failed" not in captured.err
    assert "sk-sensitive-value" not in captured.err
    assert "Traceback" not in captured.err
