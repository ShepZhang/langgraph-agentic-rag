"""Tests for the standalone baseline package."""

from __future__ import annotations

import importlib
import json
import sys
from unittest.mock import patch

from baseline import run_naive_rag
from baseline.run_baseline import main as baseline_main
import baseline.run_baseline as baseline_module
from evaluation.baselines import run_naive_rag as compatibility_run_naive_rag


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_baseline_package_exports_run_naive_rag():
    assert run_naive_rag is compatibility_run_naive_rag


def test_standalone_naive_rag_payload_matches_evaluator_contract():
    llm = FakeLLM(
        [
            (
                '{"answer": "Naive RAG retrieves once and answers [1].", '
                '"used_citation_indices": [1]}'
            )
        ]
    )
    docs = [
        {
            "content": "Naive RAG retrieves once and answers.",
            "source": "notes.md",
            "page": None,
            "chunk_id": "notes.md:pNA:c1",
            "score": 0.42,
        }
    ]

    result = run_naive_rag("What is naive RAG?", retriever_fn=lambda query: docs, llm=llm)

    assert result["question"] == "What is naive RAG?"
    assert result["answer"] == "Naive RAG retrieves once and answers [1]."
    assert result["citations"][0]["source"] == "notes.md"
    assert result["retrieved_documents"] == docs
    assert result["relevant_documents"] == docs
    assert result["claims"] == []
    assert result["claim_verification"] == {}
    assert result["is_verified"] is False
    assert result["retry_count"] == 0
    assert result["fallback_reason"] == ""
    assert result["token_usage"] is None
    assert result["estimated_cost"] is None


def test_baseline_cli_writes_output_json(tmp_path):
    questions_path = tmp_path / "questions.json"
    output_path = tmp_path / "nested" / "results" / "baseline_result.json"
    questions_path.write_text(
        json.dumps(
            [
                {
                    "id": "q001",
                    "question": "What is RAG?",
                    "expected_keywords": ["rag"],
                    "expected_sources": ["notes.md"],
                    "answerable": True,
                    "expected_behavior": "answer_with_citation",
                }
            ]
        ),
        encoding="utf-8",
    )

    def fake_runner(question):
        return {
            "answer": "RAG retrieves context [1].",
            "citations": [{"source": "notes.md"}],
            "retrieved_documents": [{"source": "notes.md"}],
            "relevant_documents": [{"source": "notes.md"}],
            "claims": [],
            "is_verified": False,
            "retry_count": 0,
            "fallback_reason": "",
            "token_usage": None,
            "estimated_cost": None,
        }

    exit_code = baseline_main(
        ["--questions", str(questions_path), "--output", str(output_path)],
        run_naive_fn=fake_runner,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert set(payload) == {"system", "summary", "results"}
    assert payload["system"] == "naive_rag"
    assert payload["summary"]["total_questions"] == 1
    assert isinstance(payload["results"], list)
    assert len(payload["results"]) == 1
    row = payload["results"][0]
    assert "naive" not in row
    assert "agentic" not in row
    assert row["question"] == "What is RAG?"
    assert row["answer"] == "RAG retrieves context [1]."
    assert row["answer_returned"] is True
    assert row["citation_returned"] is True
    assert row["source_hit"] is True
    assert row["keyword_hit"] is True
    assert row["rewrite_triggered"] is False
    assert row["retry_count"] == 0
    assert row["retrieved_doc_count"] == 1
    assert row["relevant_doc_count"] == 1


def test_baseline_cli_help_does_not_load_execution_dependencies():
    def fail_if_called():
        raise AssertionError("execution-only dependency loader should not run for --help")

    with patch.object(baseline_module, "_load_naive_rag_runner", side_effect=fail_if_called), patch.object(
        baseline_module, "_load_evaluation_tools", side_effect=fail_if_called
    ):
        try:
            baseline_main(["--help"])
        except SystemExit as exc:
            assert exc.code == 0
        else:
            raise AssertionError("--help should exit via SystemExit")


def test_baseline_cli_module_import_is_lazy():
    sys.modules.pop("baseline.run_baseline", None)
    module = importlib.import_module("baseline.run_baseline")

    assert "run_naive_rag" not in module.__dict__
    assert "evaluate_questions" not in module.__dict__
    assert "load_eval_questions" not in module.__dict__
