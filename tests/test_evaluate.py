"""Tests for lightweight evaluation runner."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import pytest

from agent.state import ChatMessage
import evaluation.evaluate as evaluator
from evaluation.evaluate import evaluate_questions, format_report, load_eval_questions, main
from evaluation.schemas import JudgeResult


def _disable_history_recording(monkeypatch: pytest.MonkeyPatch) -> None:
    class NoopHistoryService:
        def record_payload(self, *args: Any, **kwargs: Any) -> dict[str, str | None]:
            return {"status": "disabled", "run_id": None, "error": None}

    monkeypatch.setattr(evaluator, "EvaluationHistoryService", lambda: NoopHistoryService())


def test_public_facade_exports_owned_callables():
    public_names = [
        "load_eval_questions",
        "evaluate_questions",
        "evaluate_single_system",
        "summarize_results",
        "format_report",
        "write_evaluation_artifacts",
        "main",
    ]

    for name in public_names:
        assert callable(getattr(evaluator, name))


def test_public_facade_reexports_legacy_runner_alias():
    assert get_origin(evaluator.EvaluationRunner) is not None
    assert get_args(evaluator.EvaluationRunner) == (
        [str, list[ChatMessage]],
        dict[str, Any],
    )


def test_public_facade_type_hints_include_optional_judge() -> None:
    evaluate_questions_hints = get_type_hints(evaluator.evaluate_questions)
    evaluate_single_hints = get_type_hints(evaluator.evaluate_single_system)

    assert evaluate_questions_hints["judge"] == evaluator.Judge | None
    assert evaluate_single_hints["judge"] == evaluator.Judge | None


def test_summarize_results_ignores_unknown_result_fields_for_compatibility():
    result = {
        "question_id": "q001",
        "question_type": "single_doc",
        "question": "What is RAG?",
        "answer_returned": True,
        "correct": True,
        "context_relevant": True,
        "extra_metric": "external-diagnostic",
    }
    questions = [
        {
            "id": "q001",
            "question": "What is RAG?",
            "question_type": "single_doc",
        }
    ]

    summary = evaluator.summarize_results([result], questions)

    assert summary["total_questions"] == 1
    assert summary["answer_rate"] == 1.0
    assert summary["correctness_score"] == 1.0


def test_load_eval_questions_reads_new_schema_and_legacy_source(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question": "What is Agentic RAG?",
                    "expected_keywords": ["agent", "retrieval"],
                    "expected_source": "legacy.md",
                },
                {
                    "question": "What is not covered?",
                    "expected_sources": [],
                    "should_answer": False,
                    "requires_rewrite": True,
                },
            ]
        ),
        encoding="utf-8",
    )

    questions = load_eval_questions(path)

    assert questions == [
        {
            "question": "What is Agentic RAG?",
            "expected_keywords": ["agent", "retrieval"],
            "expected_source": "legacy.md",
            "id": "q001",
            "question_type": "unspecified",
            "gold_answer": "",
            "expected_sources": ["legacy.md"],
            "source_match_mode": "any",
            "answerable": True,
            "should_answer": True,
            "expected_behavior": "answer_with_citation",
            "chat_history": [],
            "requires_rewrite": False,
        },
        {
            "question": "What is not covered?",
            "expected_keywords": [],
            "expected_sources": [],
            "id": "q002",
            "question_type": "unspecified",
            "gold_answer": "",
            "source_match_mode": "any",
            "answerable": False,
            "should_answer": False,
            "expected_behavior": "fallback",
            "chat_history": [],
            "requires_rewrite": True,
        },
    ]


def test_load_eval_questions_defaults_requires_rewrite_false(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps([{"question": "What is RAG?"}]), encoding="utf-8")

    questions = load_eval_questions(path)

    assert questions[0]["requires_rewrite"] is False


def test_load_eval_questions_rejects_missing_question(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps([{"expected_sources": ["notes.md"]}]), encoding="utf-8")

    with pytest.raises(ValueError, match="question"):
        load_eval_questions(path)


def test_load_eval_questions_rejects_non_string_sources(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps([{"question": "What?", "expected_sources": ["valid", 5]}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected_sources"):
        load_eval_questions(path)


def test_load_eval_questions_normalizes_richer_schema(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "q001",
                    "question": "How does grading help?",
                    "question_type": "single_doc",
                    "gold_answer": "It checks retrieved evidence before answering.",
                    "expected_sources": ["notes.md"],
                    "expected_keywords": ["evidence"],
                    "answerable": True,
                    "expected_behavior": "answer_with_citation",
                    "chat_history": [{"role": "user", "content": "Discuss RAG."}],
                },
                {
                    "question": "Legacy?",
                    "expected_source": "legacy.md",
                    "should_answer": False,
                },
            ]
        ),
        encoding="utf-8",
    )

    questions = load_eval_questions(path)

    assert questions[0]["id"] == "q001"
    assert questions[0]["question_type"] == "single_doc"
    assert questions[0]["gold_answer"] == "It checks retrieved evidence before answering."
    assert questions[0]["answerable"] is True
    assert questions[0]["should_answer"] is True
    assert questions[0]["expected_behavior"] == "answer_with_citation"
    assert questions[0]["chat_history"] == [{"role": "user", "content": "Discuss RAG."}]
    assert questions[1]["id"] == "q002"
    assert questions[1]["expected_sources"] == ["legacy.md"]
    assert questions[1]["answerable"] is False
    assert questions[1]["should_answer"] is False
    assert questions[1]["expected_behavior"] == "fallback"


def test_load_eval_questions_rejects_conflicting_answerable_and_should_answer(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question": "Conflict?",
                    "answerable": True,
                    "should_answer": False,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="answerable"):
        load_eval_questions(path)


def test_load_eval_questions_rejects_malformed_chat_history(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question": "Bad history?",
                    "chat_history": ["user said this"],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="chat_history"):
        load_eval_questions(path)

    path.write_text(
        json.dumps(
            [
                {
                    "question": "Bad history?",
                    "chat_history": [{"role": "user"}],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="chat_history"):
        load_eval_questions(path)

    path.write_text(
        json.dumps(
            [
                {
                    "question": "Bad history?",
                    "chat_history": [{"role": "user", "content": 123}],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="chat_history"):
        load_eval_questions(path)


def test_load_eval_questions_rejects_invalid_expected_behavior(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question": "Bad behavior?",
                    "expected_behavior": "maybe",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected_behavior"):
        load_eval_questions(path)

    path.write_text(
        json.dumps(
            [
                {
                    "question": "Bad behavior?",
                    "answerable": False,
                    "expected_behavior": "answer_with_citation",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected_behavior"):
        load_eval_questions(path)

    path.write_text(
        json.dumps(
            [
                {
                    "question": "Bad behavior?",
                    "answerable": True,
                    "expected_behavior": "fallback",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected_behavior"):
        load_eval_questions(path)


def test_default_eval_dataset_has_required_coverage():
    questions = load_eval_questions()
    type_counts = Counter(question["question_type"] for question in questions)
    by_id = {question["id"]: question for question in questions}

    assert len(questions) == 36
    assert [question["id"] for question in questions] == [
        f"q{index:03d}" for index in range(1, 37)
    ]
    assert type_counts["single_doc"] >= 6
    assert type_counts["multi_chunk"] >= 4
    assert type_counts["ambiguous"] >= 4
    assert type_counts["unanswerable"] >= 5
    assert type_counts["distractor"] >= 3
    assert type_counts["comparison"] >= 4
    assert type_counts["follow_up"] >= 3
    assert type_counts["citation_sensitive"] >= 3
    assert type_counts["cross_file"] >= 3
    assert type_counts["false_premise"] >= 1
    assert all(question["expected_behavior"] for question in questions)
    assert all(
        (Path("sample_docs") / source).exists()
        for question in questions
        for source in question["expected_sources"]
        if source
    )
    assert by_id["q014"]["question"] == "Can you summarize it?"
    assert by_id["q014"]["answerable"] is False
    assert by_id["q036"]["answerable"] is True
    assert "retrieval_pipeline_notes.md" in by_id["q036"]["expected_sources"]


def test_evaluate_questions_passes_chat_history_to_runner():
    calls = []
    history = [{"role": "user", "content": "Discuss grading."}]

    def runner(question, chat_history):
        calls.append((question, chat_history))
        return {
            "answer": "Grounded answer [1].",
            "citations": [{"source": "notes.md"}],
            "retrieved_documents": [{"source": "notes.md"}],
            "relevant_documents": [{"source": "notes.md"}],
        }

    report = evaluate_questions(
        [
            {
                "id": "q-follow-up",
                "question": "How does it help?",
                "question_type": "follow_up",
                "chat_history": history,
                "expected_sources": ["notes.md"],
            }
        ],
        run_agent_fn=runner,
    )

    assert calls == [("How does it help?", history)]
    assert report["results"][0]["question_id"] == "q-follow-up"
    assert report["results"][0]["question_type"] == "follow_up"
    assert report["results"][0]["chat_history_supplied"] is True


def test_evaluation_counts_labels_from_verification_results():
    def runner(question, chat_history):
        return {
            "answer": "Two claims [1].",
            "citations": [{"source": "notes.md"}],
            "retrieved_documents": [{"source": "notes.md"}],
            "relevant_documents": [{"source": "notes.md"}],
            "claims": [{"claim_id": "c1"}, {"claim_id": "c2"}],
            "claim_verification_results": [
                {"claim_id": "c1", "verification_label": "supported"},
                {"claim_id": "c2", "verification_label": "unsupported"},
            ],
            "citation_verification_enabled": True,
            "citation_verification_passed": False,
        }

    report = evaluate_questions(
        [{"question": "Q?", "expected_sources": ["notes.md"]}],
        run_agent_fn=runner,
    )

    assert report["summary"]["unsupported_claim_count"] == 1
    assert report["summary"]["supported_claim_ratio"] == 0.5
    assert report["summary"]["citation_verification_pass_rate"] == 0.0


def test_evaluation_marks_verification_metrics_unavailable_when_disabled():
    def runner(question, chat_history):
        return {
            "answer": "Grounded answer [1].",
            "citations": [{"source": "notes.md"}],
            "retrieved_documents": [{"source": "notes.md"}],
            "relevant_documents": [{"source": "notes.md"}],
            "claims": [],
            "claim_verification_results": [],
            "citation_verification_enabled": False,
            "citation_verification_passed": False,
        }

    report = evaluate_questions(
        [{"question": "Q?", "expected_sources": ["notes.md"]}],
        run_agent_fn=runner,
    )

    assert report["summary"]["unsupported_claim_count"] is None
    assert report["summary"]["supported_claim_ratio"] is None
    assert report["summary"]["citation_verification_pass_rate"] is None


def test_evaluate_questions_computes_agentic_summary_metrics():
    questions = [
        {
            "question": "What is Agentic RAG?",
            "expected_keywords": ["agentic", "retrieval"],
            "expected_sources": ["notes.md"],
            "should_answer": True,
        },
        {
            "question": "What is missing?",
            "expected_keywords": [],
            "expected_sources": [],
            "should_answer": False,
        },
    ]
    timer_values = iter([0.0, 1.0, 1.0, 3.0])

    def fake_timer():
        return next(timer_values)

    def fake_run_agent(question):
        if question == "What is Agentic RAG?":
            return {
                "answer": "Agentic RAG uses retrieval.",
                "citations": [{"source": "notes.md"}],
                "retrieved_documents": [
                    {"source": "notes.md", "content": "Agentic RAG uses retrieval."},
                    {"source": "other.md", "content": "Other content."},
                ],
                "relevant_documents": [
                    {"source": "notes.md", "content": "Agentic RAG uses retrieval."}
                ],
                "retry_count": 1,
                "is_verified": True,
                "claims": [{"claim": "Agentic RAG uses retrieval", "supported": True}],
            }
        return {
            "answer": "根据当前已索引文档，无法可靠回答这个问题。",
            "citations": [],
            "retrieved_documents": [{"source": "other.md", "content": "Other content"}],
            "relevant_documents": [],
            "retry_count": 2,
            "fallback_reason": "No relevant chunks found.",
        }

    report = evaluate_questions(questions, run_agent_fn=fake_run_agent, timer=fake_timer)

    assert report["summary"] == {
        "total_questions": 2,
        "answer_rate": 0.5,
        "fallback_rate": 0.5,
        "citation_rate": 0.5,
        "verification_rate": 0.5,
        "average_claim_count": 0.5,
        "correctness_score": 1.0,
        "context_relevance_score": 1.0,
        "citation_hit_rate": 1.0,
        "fallback_accuracy": 1.0,
        "unsupported_claim_count": 0,
        "supported_claim_ratio": 1.0,
        "citation_verification_pass_rate": 0.5,
        "average_token_usage": 0.0,
        "estimated_cost": 0,
        "source_hit_rate": 1.0,
        "keyword_hit_rate": 1.0,
        "fallback_correctness_rate": 1.0,
        "average_retry_count": 1.5,
        "average_retrieved_docs": 1.5,
        "average_relevant_docs": 0.5,
        "relevant_filtering_rate": 0.6667,
        "average_latency": 1.5,
        "rewrite_triggered_count": 2,
        "error_count": 0,
        "judge_completed_count": 0,
        "judge_failed_count": 0,
        "judge_completion_rate": None,
        "average_semantic_correctness": None,
        "average_groundedness": None,
        "groundedness_applicable_count": 0,
        "failure_type_counts": {"no_failure": 2},
    }
    assert report["results"][0]["source_hit"] is True
    assert report["results"][0]["keyword_hit"] is True
    assert report["results"][1]["fallback_triggered"] is True
    assert report["results"][1]["fallback_correct"] is True


def test_evaluate_questions_attaches_failure_analysis_and_summary_counts():
    questions = [
        {
            "question": "What is Agentic RAG?",
            "expected_keywords": ["agentic", "retrieval"],
            "expected_sources": ["notes.md"],
            "answerable": True,
        },
        {
            "question": "Where is the policy documented?",
            "expected_keywords": ["policy"],
            "expected_sources": ["policy.md"],
            "answerable": True,
        },
    ]
    timer_values = iter([0.0, 0.5, 0.5, 1.0])

    def fake_timer():
        return next(timer_values)

    def fake_runner(question):
        if question == "What is Agentic RAG?":
            return {
                "answer": "Agentic RAG uses retrieval.",
                "citations": [{"source": "notes.md"}],
                "retrieved_documents": [{"source": "notes.md"}],
                "relevant_documents": [{"source": "notes.md"}],
            }
        return {
            "answer": "The policy is documented elsewhere.",
            "citations": [],
            "retrieved_documents": [{"source": "other.md"}],
            "relevant_documents": [],
        }

    report = evaluate_questions(questions, run_agent_fn=fake_runner, timer=fake_timer)

    failure_types = [
        result["failure_analysis"]["failure_type"]
        for result in report["results"]
    ]
    assert failure_types == ["no_failure", "retrieval_failure"]
    assert report["summary"]["failure_type_counts"] == {
        "no_failure": 1,
        "retrieval_failure": 1,
    }


def test_evaluate_questions_with_injected_judge_does_not_build_configured_judge(
    monkeypatch,
) -> None:
    class RecordingJudge:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def evaluate(self, question, result) -> JudgeResult:
            self.calls.append(question.question)
            return JudgeResult.completed(
                {"semantic_correctness": 0.9},
                reason="Injected judge.",
            )

    def fail_build() -> evaluator.Judge:
        raise AssertionError("build_configured_judge should not run")

    monkeypatch.setattr(evaluator, "build_configured_judge", fail_build)
    judge = RecordingJudge()

    report = evaluate_questions(
        [{"question": "What is Agentic RAG?"}],
        run_agent_fn=lambda _question: {"answer": "Agentic RAG."},
        timer=lambda: 0.0,
        judge=judge,
    )

    assert judge.calls == ["What is Agentic RAG?"]
    assert report["results"][0]["judge"]["status"] == "completed"


def test_injected_judge_runtime_metadata_is_preserved_in_written_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    _disable_history_recording(monkeypatch)

    class InjectedJudge:
        def evaluate(self, question, result) -> JudgeResult:
            return JudgeResult.completed(
                {"semantic_correctness": 1.0, "groundedness": 0.75},
                reason="Injected judge.",
                model="custom-judge-v2",
            )

    monkeypatch.setenv("EVALUATION_JUDGE_ENABLED", "true")
    monkeypatch.delenv("EVALUATION_JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("EVALUATION_JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("EVALUATION_JUDGE_MODEL", raising=False)
    report = evaluate_questions(
        [
            {
                "question": "What is Agentic RAG?",
                "gold_answer": "Agentic RAG adds control flow around retrieval.",
            }
        ],
        run_agent_fn=lambda _question: {"answer": "It adds retrieval control flow."},
        timer=lambda: 0.0,
        judge=InjectedJudge(),
    )

    evaluator.write_evaluation_artifacts(report, tmp_path)
    payload = json.loads((tmp_path / "agentic_result.json").read_text(encoding="utf-8"))

    expected_metadata = {
        "enabled": True,
        "provider": "injected",
        "model": "custom-judge-v2",
        "temperature": None,
    }
    assert report["runtime_config"]["judge"] == expected_metadata
    assert payload["runtime_config"]["judge"] == expected_metadata


def test_write_evaluation_artifacts_records_history_after_json_write(
    tmp_path, monkeypatch
):
    calls = []

    class SpyHistoryService:
        def record_payload(self, payload, *, source, result_path, run_id=None):
            calls.append(
                {
                    "payload": payload,
                    "source": source,
                    "result_path": result_path,
                    "run_id": run_id,
                }
            )
            return {"status": "stored", "run_id": "eval_spy", "error": None}

    monkeypatch.setattr(
        evaluator,
        "EvaluationHistoryService",
        lambda: SpyHistoryService(),
    )
    report = {
        "runtime_config": evaluator.build_runtime_config_snapshot(),
        "summary": {"total_questions": 1, "correctness_score": 1.0},
        "results": [{"question_id": "q001"}],
    }

    evaluator.write_evaluation_artifacts(report, tmp_path)

    assert (tmp_path / "agentic_result.json").exists()
    assert calls == [
        {
            "payload": report,
            "source": "cli",
            "result_path": str(tmp_path / "agentic_result.json"),
            "run_id": None,
        }
    ]


def test_write_evaluation_artifacts_records_comparison_artifact_path(
    tmp_path, monkeypatch
):
    calls = []

    class SpyHistoryService:
        def record_payload(self, payload, *, source, result_path, run_id=None):
            calls.append((source, result_path))
            return {"status": "stored", "run_id": "eval_spy", "error": None}

    monkeypatch.setattr(
        evaluator,
        "EvaluationHistoryService",
        lambda: SpyHistoryService(),
    )
    report = {
        "runtime_config": evaluator.build_runtime_config_snapshot(),
        "summary": {
            "mode": "comparison",
            "naive": {"total_questions": 1},
            "agentic": {"total_questions": 1},
        },
        "results": [
            {"naive": {"question_id": "q001"}, "agentic": {"question_id": "q001"}}
        ],
    }

    evaluator.write_evaluation_artifacts(report, tmp_path)

    assert calls == [("cli", str(tmp_path / "comparison_result.json"))]


def test_write_evaluation_artifacts_ignores_history_service_construction_failure(
    tmp_path, monkeypatch
):
    class BrokenHistoryService:
        def __init__(self) -> None:
            raise ValueError("history is misconfigured")

    monkeypatch.setattr(evaluator, "EvaluationHistoryService", BrokenHistoryService)
    report = {
        "runtime_config": evaluator.build_runtime_config_snapshot(),
        "summary": {"total_questions": 1, "correctness_score": 1.0},
        "results": [{"question_id": "q001"}],
    }

    evaluator.write_evaluation_artifacts(report, tmp_path)

    assert (tmp_path / "agentic_result.json").exists()


def test_evaluate_questions_builds_configured_judge_once_when_omitted(
    monkeypatch,
) -> None:
    build_calls = 0

    class RecordingJudge:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def evaluate(self, question, result) -> JudgeResult:
            self.calls.append(question.question)
            return JudgeResult.completed(
                {"semantic_correctness": 0.7},
                reason="Built judge.",
            )

    judge = RecordingJudge()

    def fake_build() -> evaluator.Judge:
        nonlocal build_calls
        build_calls += 1
        return judge

    monkeypatch.setattr(evaluator, "build_configured_judge", fake_build)

    report = evaluate_questions(
        [
            {"question": "First?"},
            {"question": "Second?"},
        ],
        run_agent_fn=lambda question: {"answer": question},
        timer=lambda: 0.0,
    )

    assert build_calls == 1
    assert judge.calls == ["First?", "Second?"]
    assert all(result["judge"]["status"] == "completed" for result in report["results"])


def test_public_evaluate_single_system_with_injected_judge_skips_builder(
    monkeypatch,
) -> None:
    class RecordingJudge:
        def evaluate(self, question, result) -> JudgeResult:
            return JudgeResult.completed(
                {"semantic_correctness": 0.6},
                reason="Injected single.",
            )

    monkeypatch.setattr(
        evaluator,
        "build_configured_judge",
        lambda: (_ for _ in ()).throw(
            AssertionError("build_configured_judge should not run")
        ),
    )

    result = evaluator.evaluate_single_system(
        {"question": "What is RAG?"},
        lambda _question: {"answer": "Retrieval augmented generation."},
        timer=lambda: 0.0,
        judge=RecordingJudge(),
    )

    assert result["judge"]["status"] == "completed"


def test_public_evaluate_single_system_builds_judge_once_when_omitted(
    monkeypatch,
) -> None:
    build_calls = 0

    class RecordingJudge:
        def evaluate(self, question, result) -> JudgeResult:
            return JudgeResult.completed(
                {"semantic_correctness": 0.6},
                reason="Built single.",
            )

    def fake_build() -> evaluator.Judge:
        nonlocal build_calls
        build_calls += 1
        return RecordingJudge()

    monkeypatch.setattr(evaluator, "build_configured_judge", fake_build)

    result = evaluator.evaluate_single_system(
        {"question": "What is RAG?"},
        lambda _question: {"answer": "Retrieval augmented generation."},
        timer=lambda: 0.0,
    )

    assert build_calls == 1
    assert result["judge"]["status"] == "completed"


def test_evaluate_questions_respects_answerable_without_should_answer():
    questions = [
        {
            "question": "Missing?",
            "answerable": False,
            "expected_keywords": [],
            "expected_sources": [],
        }
    ]
    timer_values = iter([0.0, 0.1])

    def fake_timer():
        return next(timer_values)

    def fake_runner(question):
        return {
            "answer": "The provided documents do not contain enough information.",
            "citations": [],
            "retrieved_documents": [],
            "relevant_documents": [],
            "retry_count": 0,
            "fallback_reason": "No evidence.",
        }

    report = evaluate_questions(questions, run_agent_fn=fake_runner, timer=fake_timer)

    assert report["results"][0]["fallback_correct"] is True
    assert report["summary"]["fallback_correctness_rate"] == 1.0


def test_evaluate_questions_normalizes_legacy_expected_source_without_loader():
    questions = [
        {
            "question": "Source?",
            "expected_source": "notes.md",
        }
    ]
    timer_values = iter([0.0, 0.1])

    def fake_timer():
        return next(timer_values)

    def fake_runner(question):
        return {
            "answer": "Retrieved from notes.md.",
            "citations": [{"source": "notes.md"}],
            "retrieved_documents": [{"source": "notes.md"}],
            "relevant_documents": [{"source": "notes.md"}],
            "retry_count": 0,
        }

    report = evaluate_questions(questions, run_agent_fn=fake_runner, timer=fake_timer)

    assert report["results"][0]["source_hit"] is True
    assert report["summary"]["source_hit_rate"] == 1.0


@pytest.mark.parametrize(
    ("question", "match"),
    [
        ({"question": "Bad chat", "chat_history": ["oops"]}, "chat_history"),
        (
            {"question": "Bad chat", "chat_history": [{"role": "user"}]},
            "chat_history",
        ),
        (
            {"question": "Bad chat", "chat_history": [{"role": "user", "content": 1}]},
            "chat_history",
        ),
        (
            {"question": "Bad behavior", "expected_behavior": "maybe"},
            "expected_behavior",
        ),
        (
            {
                "question": "Bad behavior",
                "answerable": False,
                "expected_behavior": "answer_with_citation",
            },
            "expected_behavior",
        ),
        (
            {
                "question": "Bad behavior",
                "answerable": True,
                "expected_behavior": "fallback",
            },
            "expected_behavior",
        ),
    ],
)
def test_evaluate_questions_rejects_invalid_raw_question_schema(question, match):
    with pytest.raises(ValueError, match=match):
        evaluate_questions([question], run_agent_fn=lambda _: {}, timer=lambda: 0.0)


def test_evaluate_questions_compares_naive_and_agentic_results():
    questions = [
        {
            "question": "How does it improve reliability?",
            "expected_keywords": ["grading"],
            "expected_sources": ["notes.md"],
            "should_answer": True,
            "requires_rewrite": True,
        },
        {
            "question": "What is the payroll policy?",
            "expected_keywords": [],
            "expected_sources": [],
            "should_answer": False,
            "requires_rewrite": False,
        },
    ]
    timer_values = iter([0.0, 1.0, 1.0, 3.0, 3.0, 4.0, 4.0, 6.0])

    def fake_timer():
        return next(timer_values)

    def fake_run_agent(question):
        if "reliability" in question:
            return {
                "answer": "Agentic RAG uses retrieval grading.",
                "citations": [{"source": "notes.md"}],
                "retrieved_documents": [{"source": "notes.md"}, {"source": "other.md"}],
                "relevant_documents": [{"source": "notes.md"}],
                "retry_count": 1,
            }
        return {
            "answer": "根据当前已索引文档，无法可靠回答这个问题。",
            "citations": [],
            "retrieved_documents": [{"source": "notes.md"}],
            "relevant_documents": [],
            "retry_count": 2,
            "fallback_reason": "No relevant chunks.",
        }

    def fake_run_naive(question):
        if "reliability" in question:
            return {
                "answer": "Naive answer without the expected term.",
                "citations": [],
                "retrieved_documents": [{"source": "other.md"}],
                "relevant_documents": [{"source": "other.md"}],
                "retry_count": 0,
                "fallback_reason": "Naive RAG answer generation did not return valid supporting citations.",
            }
        return {
            "answer": "The documents do not contain enough information.",
            "citations": [],
            "retrieved_documents": [{"source": "notes.md"}],
            "relevant_documents": [],
            "retry_count": 0,
        }

    report = evaluate_questions(
        questions,
        run_agent_fn=fake_run_agent,
        run_naive_fn=fake_run_naive,
        timer=fake_timer,
    )

    assert report["summary"]["mode"] == "comparison"
    assert report["summary"]["agentic"]["source_hit_rate"] == 1.0
    assert report["summary"]["naive"]["source_hit_rate"] == 0.0
    assert report["summary"]["comparison"]["agentic_source_hit_rate"] == 1.0
    assert report["summary"]["comparison"]["naive_source_hit_rate"] == 0.0
    assert report["summary"]["comparison"]["agentic_keyword_hit_rate"] == 1.0
    assert report["summary"]["comparison"]["naive_keyword_hit_rate"] == 0.0
    assert report["summary"]["comparison"]["agentic_fallback_correctness_rate"] == 1.0
    assert report["summary"]["comparison"]["naive_fallback_correctness_rate"] == 0.5
    assert report["summary"]["agentic"]["average_retry_count"] == 1.5
    assert report["summary"]["agentic"]["relevant_filtering_rate"] == 0.6667
    assert report["results"][0]["agentic"]["rewrite_triggered"] is True
    assert report["results"][0]["naive"]["rewrite_triggered"] is False


def test_evaluate_comparison_preserves_failure_analysis_for_each_system():
    questions = [
        {
            "question": "How does Agentic RAG use evidence?",
            "expected_keywords": ["evidence"],
            "expected_sources": ["notes.md"],
            "answerable": True,
        }
    ]
    timer_values = iter([0.0, 0.25, 0.25, 0.75])

    def fake_timer():
        return next(timer_values)

    def fake_agentic(question):
        return {
            "answer": "Agentic RAG uses evidence.",
            "citations": [{"source": "notes.md"}],
            "retrieved_documents": [{"source": "notes.md"}],
            "relevant_documents": [{"source": "notes.md"}],
        }

    def fake_naive(question):
        return {
            "answer": "Agentic RAG uses evidence.",
            "citations": [],
            "retrieved_documents": [{"source": "other.md"}],
            "relevant_documents": [],
        }

    report = evaluate_questions(
        questions,
        run_agent_fn=fake_agentic,
        run_naive_fn=fake_naive,
        timer=fake_timer,
    )

    paired_result = report["results"][0]
    assert paired_result["naive"]["failure_analysis"]["failure_type"] == (
        "retrieval_failure"
    )
    assert paired_result["agentic"]["failure_analysis"]["failure_type"] == "no_failure"
    assert report["summary"]["naive"]["failure_type_counts"] == {
        "retrieval_failure": 1
    }
    assert report["summary"]["agentic"]["failure_type_counts"] == {"no_failure": 1}


def test_evaluate_questions_records_agent_errors():
    questions = [{"question": "Broken?", "expected_sources": ["notes.md"]}]
    timer_values = iter([0.0, 0.25])

    def fake_timer():
        return next(timer_values)

    def failing_run_agent(question):
        raise RuntimeError("Missing LLM configuration")

    report = evaluate_questions(questions, run_agent_fn=failing_run_agent, timer=fake_timer)

    assert report["summary"]["answer_rate"] == 0.0
    assert report["summary"]["error_count"] == 1
    assert report["results"][0]["error"] == "RuntimeError: Missing LLM configuration"
    assert report["results"][0]["failure_analysis"]["failure_type"] == "tool_failure"


def test_evaluate_questions_records_malformed_agent_payload_errors():
    questions = [{"question": "Malformed?", "expected_keywords": ["x"]}]
    timer_values = iter([0.0, 0.5])

    def fake_timer():
        return next(timer_values)

    def malformed_run_agent(question):
        return {"answer": "x", "retry_count": "many"}

    report = evaluate_questions(
        questions,
        run_agent_fn=malformed_run_agent,
        timer=fake_timer,
    )

    assert report["summary"]["answer_rate"] == 0.0
    assert report["summary"]["error_count"] == 1
    assert report["results"][0]["answer_returned"] is False
    assert report["results"][0]["citation_returned"] is False
    assert report["results"][0]["source_hit"] is False
    assert report["results"][0]["keyword_hit"] is False
    assert report["results"][0]["rewrite_triggered"] is False
    assert report["results"][0]["latency"] == 0.5
    assert "ValueError" in report["results"][0]["error"]


def test_evaluate_questions_uses_expected_field_denominators_for_hit_rates():
    questions = [
        {
            "question": "Source expected and hit",
            "expected_sources": ["notes.md"],
        },
        {
            "question": "No expected source",
            "expected_keywords": ["present"],
        },
        {
            "question": "Keyword expected and missed",
            "expected_keywords": ["absent"],
        },
    ]
    timer_values = iter([0.0, 1.0, 1.0, 2.0, 2.0, 3.0])

    def fake_timer():
        return next(timer_values)

    def fake_run_agent(question):
        if question == "Source expected and hit":
            return {
                "answer": "No keyword expectation here.",
                "citations": [{"source": "notes.md"}],
                "retrieved_documents": [{"source": "notes.md"}],
            }
        if question == "No expected source":
            return {
                "answer": "present",
                "retrieved_documents": [{"source": "ignored.md"}],
            }
        return {"answer": "different"}

    report = evaluate_questions(questions, run_agent_fn=fake_run_agent, timer=fake_timer)

    assert report["summary"]["source_hit_rate"] == 1.0
    assert report["summary"]["keyword_hit_rate"] == 0.5


def test_evaluate_questions_error_message_includes_type_for_empty_message():
    questions = [{"question": "Broken?"}]
    timer_values = iter([0.0, 0.25])

    def fake_timer():
        return next(timer_values)

    def failing_run_agent(question):
        raise RuntimeError()

    report = evaluate_questions(questions, run_agent_fn=failing_run_agent, timer=fake_timer)

    assert report["results"][0]["error"] == "RuntimeError"


def test_format_report_includes_summary_and_question_rows():
    report = {
        "summary": {
            "mode": "comparison",
            "naive": {
                "source_hit_rate": 0.5,
                "keyword_hit_rate": 0.25,
                "citation_rate": 0.5,
                "verification_rate": 0.0,
                "fallback_correctness_rate": 0.75,
                "average_latency": 0.2,
            },
            "agentic": {
                "source_hit_rate": 1.0,
                "keyword_hit_rate": 1.0,
                "citation_rate": 1.0,
                "verification_rate": 1.0,
                "fallback_correctness_rate": 1.0,
                "average_latency": 0.25,
                "average_retry_count": 1.0,
                "rewrite_triggered_count": 1,
                "average_retrieved_docs": 2.0,
                "average_relevant_docs": 1.0,
                "relevant_filtering_rate": 0.5,
                "average_claim_count": 1.0,
            },
            "comparison": {
                "naive_source_hit_rate": 0.5,
                "agentic_source_hit_rate": 1.0,
                "naive_keyword_hit_rate": 0.25,
                "agentic_keyword_hit_rate": 1.0,
                "naive_citation_rate": 0.5,
                "agentic_citation_rate": 1.0,
                "naive_verification_rate": 0.0,
                "agentic_verification_rate": 1.0,
                "naive_fallback_correctness_rate": 0.75,
                "agentic_fallback_correctness_rate": 1.0,
                "naive_average_latency": 0.2,
                "agentic_average_latency": 0.25,
            },
        },
        "results": [
            {
                "question": "What is Agentic RAG?",
                "naive": {"answer_returned": True, "source_hit": False, "error": ""},
                "agentic": {
                    "answer_returned": True,
                    "source_hit": True,
                    "retry_count": 1,
                    "retrieved_doc_count": 2,
                    "relevant_doc_count": 1,
                    "latency": 0.25,
                    "error": "",
                },
            }
        ],
    }

    text = format_report(report)

    assert "Evaluation Report" in text
    assert "| Metric | Naive RAG | Agentic RAG |" in text
    assert "| Source Hit Rate | 0.5 | 1.0 |" in text
    assert "| Claim Verification Rate | 0.0 | 1.0 |" in text
    assert "Agentic-specific Metrics" in text
    assert "retry_count=1" in text
    assert "What is Agentic RAG?" in text


def test_main_prints_report_with_injected_runner(tmp_path, capsys):
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question": "What is Agentic RAG?",
                    "expected_sources": ["notes.md"],
                }
            ]
        ),
        encoding="utf-8",
    )

    def fake_run_agent(question):
        return {
            "answer": "Agentic RAG uses retrieval.",
            "citations": [{"source": "notes.md"}],
            "retrieved_documents": [
                {"source": "notes.md", "content": "Agentic RAG uses retrieval."}
            ],
            "relevant_documents": [
                {"source": "notes.md", "content": "Agentic RAG uses retrieval."}
            ],
            "retry_count": 0,
        }

    def fake_run_naive(question):
        return {
            "answer": "Naive RAG uses retrieval.",
            "citations": [{"source": "notes.md"}],
            "retrieved_documents": [
                {"source": "notes.md", "content": "Naive RAG uses retrieval."}
            ],
            "relevant_documents": [
                {"source": "notes.md", "content": "Naive RAG uses retrieval."}
            ],
            "retry_count": 0,
        }

    exit_code = main(
        ["--questions", str(path)],
        run_agent_fn=fake_run_agent,
        run_naive_fn=fake_run_naive,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Evaluation Report" in captured.out
    assert "| Metric | Naive RAG | Agentic RAG |" in captured.out


def test_main_writes_comparison_artifacts(tmp_path, monkeypatch):
    _disable_history_recording(monkeypatch)

    path = tmp_path / "eval.json"
    output_dir = tmp_path / "artifacts"
    for name in (
        "EVALUATION_JUDGE_ENABLED",
        "EVALUATION_JUDGE_API_KEY",
        "EVALUATION_JUDGE_BASE_URL",
        "EVALUATION_JUDGE_MODEL",
        "EVALUATION_JUDGE_TEMPERATURE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "secret-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://secret-base.example/v1")
    monkeypatch.setenv("HYBRID_RETRIEVAL_ENABLED", "true")
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_TOP_N", "6")
    monkeypatch.setenv("RERANKER_CANDIDATE_TOP_K", "9")
    path.write_text(
        json.dumps(
            [
                {
                    "question": "What is Agentic RAG?",
                    "expected_keywords": ["retrieval"],
                    "expected_sources": ["notes.md"],
                }
            ]
        ),
        encoding="utf-8",
    )

    def fake_agentic(question):
        return {
            "answer": "Agentic RAG uses retrieval.",
            "citations": [{"source": "notes.md"}],
            "retrieved_documents": [{"source": "notes.md"}],
            "relevant_documents": [{"source": "notes.md"}],
        }

    def fake_naive(question):
        return {
            "answer": "Naive RAG uses retrieval.",
            "citations": [{"source": "notes.md"}],
            "retrieved_documents": [{"source": "notes.md"}],
            "relevant_documents": [{"source": "notes.md"}],
        }

    exit_code = main(
        ["--questions", str(path), "--output-dir", str(output_dir)],
        run_agent_fn=fake_agentic,
        run_naive_fn=fake_naive,
    )

    baseline_path = output_dir / "baseline_result.json"
    agentic_path = output_dir / "agentic_result.json"
    comparison_path = output_dir / "comparison_result.json"
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    agentic_payload = json.loads(agentic_path.read_text(encoding="utf-8"))
    comparison_payload = json.loads(comparison_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert baseline_path.exists()
    assert agentic_path.exists()
    assert comparison_path.exists()
    assert baseline_payload["system"] == "naive_rag"
    assert agentic_payload["system"] == "agentic_rag"
    assert comparison_payload["summary"]["mode"] == "comparison"
    assert baseline_payload["runtime_config"]["retriever"]["hybrid_retrieval_enabled"] is True
    assert agentic_payload["runtime_config"]["reranker"]["top_n"] == 6
    assert comparison_payload["runtime_config"]["reranker"]["candidate_top_k"] == 9
    runtime_configs = [
        baseline_payload["runtime_config"],
        agentic_payload["runtime_config"],
        comparison_payload["runtime_config"],
    ]
    assert all(runtime_config["schema_version"] == 4 for runtime_config in runtime_configs)
    assert all(
        runtime_config["evaluator_version"] == "p5b"
        for runtime_config in runtime_configs
    )
    assert all(
        runtime_config["judge"]
        == {
            "enabled": False,
            "provider": "openai_compatible",
            "model": None,
            "temperature": 0.0,
        }
        for runtime_config in runtime_configs
    )
    assert all(
        "evaluation.semantic_judge" in runtime_config["prompts"]
        for runtime_config in runtime_configs
    )
    serialized_payloads = json.dumps(
        [baseline_payload, agentic_payload, comparison_payload],
        ensure_ascii=False,
    )
    assert "secret-key" not in serialized_payloads
    assert "secret-base" not in serialized_payloads
    assert "OPENAI_API_KEY" not in serialized_payloads
    assert len(baseline_payload["results"]) == 1
    assert len(agentic_payload["results"]) == 1
    assert len(comparison_payload["results"]) == 1


def test_main_writes_single_system_agentic_artifact_schema(tmp_path, monkeypatch):
    _disable_history_recording(monkeypatch)

    path = tmp_path / "eval.json"
    output_dir = tmp_path / "artifacts"
    for name in (
        "EVALUATION_JUDGE_ENABLED",
        "EVALUATION_JUDGE_API_KEY",
        "EVALUATION_JUDGE_BASE_URL",
        "EVALUATION_JUDGE_MODEL",
        "EVALUATION_JUDGE_TEMPERATURE",
    ):
        monkeypatch.delenv(name, raising=False)
    path.write_text(
        json.dumps(
            [
                {
                    "question": "What is Agentic RAG?",
                    "expected_keywords": ["retrieval"],
                    "expected_sources": ["notes.md"],
                }
            ]
        ),
        encoding="utf-8",
    )

    def fake_agentic(question):
        return {
            "answer": "Agentic RAG uses retrieval.",
            "citations": [{"source": "notes.md"}],
            "retrieved_documents": [{"source": "notes.md"}],
            "relevant_documents": [{"source": "notes.md"}],
        }

    exit_code = main(
        ["--questions", str(path), "--output-dir", str(output_dir)],
        run_agent_fn=fake_agentic,
        run_naive_fn=None,
    )

    agentic_path = output_dir / "agentic_result.json"
    payload = json.loads(agentic_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert agentic_path.exists()
    assert payload["system"] == "agentic_rag"
    assert "runtime_config" in payload
    assert payload["runtime_config"]["schema_version"] == 4
    assert payload["runtime_config"]["evaluator_version"] == "p5b"
    assert payload["runtime_config"]["judge"] == {
        "enabled": False,
        "provider": "openai_compatible",
        "model": None,
        "temperature": 0.0,
    }
    assert "evaluation.semantic_judge" in payload["runtime_config"]["prompts"]
    assert "llm" in payload["runtime_config"]
    assert "retriever" in payload["runtime_config"]
    assert "reranker" in payload["runtime_config"]
    assert "summary" in payload
    assert "results" in payload
    assert len(payload["results"]) == 1


def test_main_reports_question_load_errors_without_traceback(tmp_path, capsys):
    missing_path = tmp_path / "missing.json"

    def fake_run_agent(question):
        return {"answer": "unused"}

    with pytest.raises(SystemExit) as exc_info:
        main(["--questions", str(missing_path)], run_agent_fn=fake_run_agent)

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "Unable to load evaluation questions" in captured.err
    assert "Traceback" not in captured.err


def test_evaluate_questions_computes_reliability_metrics():
    questions = [
        {
            "id": "q001",
            "question": "Supported?",
            "gold_answer": "The answer mentions supported evidence.",
            "expected_keywords": ["supported", "evidence"],
            "expected_sources": ["notes.md"],
            "answerable": True,
            "should_answer": True,
            "expected_behavior": "answer_with_citation",
        },
        {
            "id": "q002",
            "question": "Missing?",
            "expected_keywords": [],
            "expected_sources": [],
            "answerable": False,
            "should_answer": False,
            "expected_behavior": "fallback",
        },
    ]
    timer_values = iter([0.0, 1.0, 1.0, 2.0])

    def fake_timer():
        return next(timer_values)

    def fake_runner(question):
        if question == "Supported?":
            return {
                "answer": "The answer mentions supported evidence [1].",
                "citations": [{"source": "notes.md"}],
                "retrieved_documents": [{"source": "notes.md"}],
                "relevant_documents": [{"source": "notes.md"}],
                "claims": [
                    {"claim": "supported evidence", "supported": True},
                    {"claim": "extra claim", "supported": False},
                ],
                "claim_verification": {"verified": False},
                "is_verified": False,
                "retry_count": 1,
                "token_usage": {"total_tokens": 120},
                "estimated_cost": 0.0003,
            }
        return {
            "answer": "The provided documents do not contain enough information.",
            "citations": [],
            "retrieved_documents": [],
            "relevant_documents": [],
            "claims": [],
            "is_verified": False,
            "retry_count": 0,
            "fallback_reason": "No evidence.",
        }

    report = evaluate_questions(questions, run_agent_fn=fake_runner, timer=fake_timer)
    summary = report["summary"]

    assert summary["correctness_score"] == 1.0
    assert summary["context_relevance_score"] == 1.0
    assert summary["citation_hit_rate"] == 1.0
    assert summary["fallback_accuracy"] == 1.0
    assert summary["unsupported_claim_count"] == 1
    assert summary["supported_claim_ratio"] == 0.5
    assert summary["citation_verification_pass_rate"] == 0.0
    assert summary["average_token_usage"] == 60.0
    assert summary["estimated_cost"] == 0.0003


def test_evaluate_questions_treats_source_hit_as_combined_evidence_and_citation_hit_as_citations_only():
    questions = [
        {
            "question": "Source mismatch?",
            "expected_sources": ["notes.md"],
        }
    ]
    timer_values = iter([0.0, 0.25])

    def fake_timer():
        return next(timer_values)

    def fake_runner(question):
        return {
            "answer": "Answer with a wrong citation.",
            "citations": [{"source": "wrong.md"}],
            "retrieved_documents": [{"source": "notes.md"}],
            "relevant_documents": [{"source": "notes.md"}],
            "retry_count": 0,
        }

    report = evaluate_questions(questions, run_agent_fn=fake_runner, timer=fake_timer)

    assert report["results"][0]["source_hit"] is True
    assert report["results"][0]["citation_hit"] is False
    assert report["summary"]["source_hit_rate"] == 1.0
    assert report["summary"]["citation_hit_rate"] == 0.0


def test_evaluate_questions_filters_invalid_estimated_cost_values():
    questions = [
        {"question": "Bool cost?", "expected_sources": []},
        {"question": "NaN cost?", "expected_sources": []},
        {"question": "Inf cost?", "expected_sources": []},
        {"question": "Valid cost?", "expected_sources": []},
    ]
    timer_values = iter([0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4])

    def fake_timer():
        return next(timer_values)

    def fake_runner(question):
        if question == "Bool cost?":
            return {"answer": "ok", "estimated_cost": True}
        if question == "NaN cost?":
            return {"answer": "ok", "estimated_cost": float("nan")}
        if question == "Inf cost?":
            return {"answer": "ok", "estimated_cost": float("inf")}
        return {"answer": "ok", "estimated_cost": 0.125}

    report = evaluate_questions(questions, run_agent_fn=fake_runner, timer=fake_timer)

    assert report["summary"]["estimated_cost"] == 0.125


def test_evaluate_questions_uses_all_claims_as_supported_ratio_denominator():
    questions = [
        {
            "question": "Claim ratio?",
            "expected_sources": [],
        }
    ]
    timer_values = iter([0.0, 0.25])

    def fake_timer():
        return next(timer_values)

    def fake_runner(question):
        return {
            "answer": "ok",
            "claims": [
                {"claim": "supported claim", "supported": True},
                {"claim": "unlabeled claim"},
            ],
        }

    report = evaluate_questions(questions, run_agent_fn=fake_runner, timer=fake_timer)

    assert report["results"][0]["supported_claim_count"] == 1
    assert report["results"][0]["total_claim_count"] == 2
    assert report["summary"]["supported_claim_ratio"] == 0.5


def test_evaluate_questions_ignores_malformed_total_tokens_and_keeps_valid_values():
    questions = [
        {"question": "Inf tokens?", "expected_sources": []},
        {"question": "NaN tokens?", "expected_sources": []},
        {"question": "Float tokens?", "expected_sources": []},
        {"question": "Bool tokens?", "expected_sources": []},
        {"question": "String tokens?", "expected_sources": []},
        {"question": "Valid tokens?", "expected_sources": []},
        {"question": "Zero tokens?", "expected_sources": []},
    ]
    timer_values = iter([0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7])

    def fake_timer():
        return next(timer_values)

    def fake_runner(question):
        if question == "Inf tokens?":
            return {"answer": "ok", "token_usage": {"total_tokens": float("inf")}}
        if question == "NaN tokens?":
            return {"answer": "ok", "token_usage": {"total_tokens": float("nan")}}
        if question == "Float tokens?":
            return {"answer": "ok", "token_usage": {"total_tokens": 1.9}}
        if question == "Bool tokens?":
            return {"answer": "ok", "token_usage": {"total_tokens": True}}
        if question == "String tokens?":
            return {"answer": "ok", "token_usage": {"total_tokens": "100"}}
        if question == "Valid tokens?":
            return {"answer": "ok", "token_usage": {"total_tokens": 120}}
        return {"answer": "ok", "token_usage": {"total_tokens": 0.0}}

    report = evaluate_questions(questions, run_agent_fn=fake_runner, timer=fake_timer)

    assert report["results"][0]["token_usage"]["total_tokens"] == float("inf")
    assert report["summary"]["average_token_usage"] == 17.1429
