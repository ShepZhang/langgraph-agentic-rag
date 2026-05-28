"""Tests for lightweight evaluation runner."""

from __future__ import annotations

import json

import pytest

from evaluation.evaluate import evaluate_questions, load_eval_questions


def test_load_eval_questions_reads_json_file(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question": "What is Agentic RAG?",
                    "expected_keywords": ["agent", "retrieval"],
                    "expected_source": "notes.md",
                }
            ]
        ),
        encoding="utf-8",
    )

    questions = load_eval_questions(path)

    assert questions == [
        {
            "question": "What is Agentic RAG?",
            "expected_keywords": ["agent", "retrieval"],
            "expected_source": "notes.md",
        }
    ]


def test_load_eval_questions_rejects_missing_question(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps([{"expected_source": "notes.md"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="question"):
        load_eval_questions(path)


def test_load_eval_questions_rejects_non_string_keywords(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps([{"question": "What?", "expected_keywords": ["valid", 5]}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected_keywords"):
        load_eval_questions(path)


def test_evaluate_questions_computes_summary_metrics():
    questions = [
        {
            "question": "What is Agentic RAG?",
            "expected_keywords": ["agentic", "retrieval"],
            "expected_source": "notes.md",
        },
        {
            "question": "What is missing?",
            "expected_keywords": ["missing"],
            "expected_source": "missing.md",
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
                    {"source": "notes.md", "content": "Agentic RAG uses retrieval."}
                ],
                "rewrite_count": 1,
            }
        return {
            "answer": "",
            "citations": [],
            "retrieved_documents": [{"source": "other.md", "content": "Other content"}],
            "rewrite_count": 0,
        }

    report = evaluate_questions(questions, run_agent_fn=fake_run_agent, timer=fake_timer)

    assert report["summary"] == {
        "total_questions": 2,
        "answer_rate": 0.5,
        "citation_rate": 0.5,
        "source_hit_rate": 0.5,
        "average_latency": 1.5,
        "rewrite_triggered_count": 1,
        "keyword_hit_rate": 0.5,
        "error_count": 0,
    }
    assert report["results"][0]["source_hit"] is True
    assert report["results"][0]["keyword_hit"] is True
    assert report["results"][1]["answer_returned"] is False


def test_evaluate_questions_records_agent_errors():
    questions = [{"question": "Broken?", "expected_source": "notes.md"}]
    timer_values = iter([0.0, 0.25])

    def fake_timer():
        return next(timer_values)

    def failing_run_agent(question):
        raise RuntimeError("Missing LLM configuration")

    report = evaluate_questions(questions, run_agent_fn=failing_run_agent, timer=fake_timer)

    assert report["summary"]["answer_rate"] == 0.0
    assert report["summary"]["error_count"] == 1
    assert report["results"][0]["error"] == "RuntimeError: Missing LLM configuration"


def test_evaluate_questions_records_malformed_agent_payload_errors():
    questions = [{"question": "Malformed?", "expected_keywords": ["x"]}]
    timer_values = iter([0.0, 0.5])

    def fake_timer():
        return next(timer_values)

    def malformed_run_agent(question):
        return {"answer": "x", "rewrite_count": "many"}

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
            "expected_source": "notes.md",
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
