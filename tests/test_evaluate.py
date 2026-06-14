"""Tests for lightweight evaluation runner."""

from __future__ import annotations

import json

import pytest

from evaluation.evaluate import (
    evaluate_questions,
    format_report,
    load_eval_questions,
    main,
)


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
            "expected_sources": ["legacy.md"],
            "source_match_mode": "any",
            "should_answer": True,
            "requires_rewrite": False,
        },
        {
            "question": "What is not covered?",
            "expected_keywords": [],
            "expected_sources": [],
            "source_match_mode": "any",
            "should_answer": False,
            "requires_rewrite": True,
        },
    ]


def test_load_eval_questions_defaults_requires_rewrite_false(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps([{"question": "What is RAG?"}]), encoding="utf-8")

    questions = load_eval_questions(path)

    assert questions[0]["requires_rewrite"] is False


def test_load_eval_questions_accepts_all_source_match_mode(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question": "What connects these documents?",
                    "expected_sources": ["product.md", "security.md"],
                    "source_match_mode": "all",
                }
            ]
        ),
        encoding="utf-8",
    )

    questions = load_eval_questions(path)

    assert questions[0]["source_match_mode"] == "all"


@pytest.mark.parametrize(
    "expected_sources",
    [
        ["product.md"],
        ["product.md", "product.md"],
        ["product.md", ""],
    ],
)
def test_load_eval_questions_rejects_invalid_all_expected_sources(
    tmp_path,
    expected_sources,
):
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question": "What connects these documents?",
                    "expected_sources": expected_sources,
                    "source_match_mode": "all",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source_match_mode"):
        load_eval_questions(path)


def test_load_eval_questions_rejects_invalid_source_match_mode(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question": "What connects these documents?",
                    "source_match_mode": "some",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source_match_mode"):
        load_eval_questions(path)


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
    }
    assert report["results"][0]["source_hit"] is True
    assert report["results"][0]["keyword_hit"] is True
    assert report["results"][1]["fallback_triggered"] is True
    assert report["results"][1]["fallback_correct"] is True


def test_source_hit_uses_all_retrieved_sources_even_when_answer_falls_back():
    questions = [
        {
            "question": "What connects product and security policy?",
            "expected_sources": ["product_specs.md", "security_policy.md"],
            "source_match_mode": "all",
            "should_answer": False,
        }
    ]

    def fake_run_agent(question):
        return {
            "answer": "The current documents do not contain enough information.",
            "citations": [],
            "retrieved_documents": [
                {"source": "product_specs.md"},
                {"source": "security_policy.md"},
            ],
            "fallback_reason": "Answer verification failed.",
        }

    report = evaluate_questions(questions, run_agent_fn=fake_run_agent)

    assert report["results"][0]["answer_returned"] is False
    assert report["results"][0]["fallback_triggered"] is True
    assert report["results"][0]["source_hit"] is True


def test_source_hit_all_requires_every_expected_retrieved_source():
    questions = [
        {
            "question": "What connects product and security policy?",
            "expected_sources": ["product_specs.md", "security_policy.md"],
            "source_match_mode": "all",
        }
    ]

    def fake_run_agent(question):
        return {
            "answer": "Atlas uses citation checks.",
            "retrieved_documents": [{"source": "product_specs.md"}],
        }

    report = evaluate_questions(questions, run_agent_fn=fake_run_agent)

    assert report["results"][0]["source_hit"] is False


def test_source_hit_ignores_citations_when_retrieval_misses_expected_source():
    questions = [
        {
            "question": "What does the product specification say?",
            "expected_sources": ["product_specs.md"],
            "source_match_mode": "any",
        }
    ]

    def fake_run_agent(question):
        return {
            "answer": "Atlas supports document QA.",
            "citations": [{"source": "product_specs.md"}],
            "retrieved_documents": [{"source": "security_policy.md"}],
        }

    report = evaluate_questions(questions, run_agent_fn=fake_run_agent)

    assert report["results"][0]["citation_returned"] is True
    assert report["results"][0]["source_hit"] is False


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
