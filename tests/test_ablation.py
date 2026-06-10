"""Tests for lightweight ablation experiment runner."""

from __future__ import annotations

import json

from experiments.run_ablation import load_ablation_configs, main


def test_load_ablation_configs_reads_simple_yaml(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "v0_naive.yaml").write_text(
        "\n".join(
            [
                "# baseline config",
                "method: Naive RAG",
                "runner: naive",
                "status: supported",
            ]
        ),
        encoding="utf-8",
    )

    configs = load_ablation_configs(config_dir)

    assert configs == [
        {
            "id": "v0_naive",
            "method": "Naive RAG",
            "runner": "naive",
            "status": "supported",
        }
    ]


def test_ablation_main_writes_result_json_and_report(tmp_path, monkeypatch):
    questions_path = tmp_path / "questions.json"
    output_dir = tmp_path / "artifacts"
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    monkeypatch.setenv("OPENAI_API_KEY", "secret-key")
    monkeypatch.setenv("HYBRID_RETRIEVAL_ENABLED", "true")
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_TOP_N", "4")
    monkeypatch.setenv("RERANKER_CANDIDATE_TOP_K", "8")
    questions_path.write_text(
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
    (config_dir / "v0_naive.yaml").write_text(
        "\n".join(
            [
                "method: Naive RAG",
                "runner: naive",
                "status: supported",
                "runner_scope: baseline",
                "independent_ablation: true",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "v1_agentic.yaml").write_text(
        "\n".join(
            [
                "method: Agentic RAG",
                "runner: agentic",
                "status: supported",
                "runner_scope: current_agentic_workflow",
                "independent_ablation: false",
                "notes: Uses the current full agentic workflow, not an independent toggle.",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "v9_future.yaml").write_text(
        "\n".join(
            [
                "method: Future Method",
                "runner: pending",
                "status: pending",
                "runner_scope: pending",
                "independent_ablation: false",
            ]
        ),
        encoding="utf-8",
    )

    def fake_naive(question):
        return {
            "answer": f"Naive answer with retrieval for {question}",
            "citations": [{"source": "notes.md"}],
            "retrieved_documents": [{"source": "notes.md"}],
            "relevant_documents": [{"source": "notes.md"}],
        }

    def fake_agentic(question):
        return {
            "answer": f"Agentic answer with retrieval for {question}",
            "citations": [{"source": "notes.md"}],
            "retrieved_documents": [{"source": "notes.md"}],
            "relevant_documents": [{"source": "notes.md"}],
            "claims": [{"text": "retrieval helps", "supported": True}],
            "is_verified": True,
        }

    exit_code = main(
        [
            "--questions",
            str(questions_path),
            "--config-dir",
            str(config_dir),
            "--output-dir",
            str(output_dir),
        ],
        run_naive_fn=fake_naive,
        run_agent_fn=fake_agentic,
    )

    result_path = output_dir / "ablation_result.json"
    report_path = output_dir / "ablation_report.md"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")

    assert exit_code == 0
    assert result_path.exists()
    assert report_path.exists()
    assert payload["runtime_config"]["retriever"]["hybrid_retrieval_enabled"] is True
    assert payload["runtime_config"]["reranker"]["top_n"] == 4
    assert "secret-key" not in json.dumps(payload, ensure_ascii=False)
    assert [run["id"] for run in payload["runs"]] == [
        "v0_naive",
        "v1_agentic",
        "v9_future",
    ]
    assert payload["runs"][0]["status"] == "completed"
    assert payload["runs"][1]["status"] == "completed"
    assert payload["runs"][2]["status"] == "pending"
    assert payload["runs"][1]["runner_scope"] == "current_agentic_workflow"
    assert payload["runs"][1]["independent_ablation"] == "false"
    assert "Naive RAG" in report
    assert "Agentic RAG" in report
    assert "Future Method" in report
    assert "current_agentic_workflow" in report
    assert "not an independent toggle" in report
