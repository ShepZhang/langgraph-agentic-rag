"""Tests for the FastAPI service layer."""

from __future__ import annotations

from fastapi.testclient import TestClient


class FakeDocumentService:
    def __init__(self):
        self.deleted = []

    def upload_documents(self, workspace_id, files):
        return [
            {
                "document_id": "doc_1",
                "workspace_id": workspace_id,
                "filename": "notes.md",
                "status": "uploaded",
                "chunk_count": 0,
                "vector_ids": [],
            }
        ]

    def index_documents(self, workspace_id, document_ids=None, reset_collection=False):
        return {
            "workspace_id": workspace_id,
            "indexed_documents": [
                {
                    "document_id": "doc_1",
                    "workspace_id": workspace_id,
                    "filename": "notes.md",
                    "status": "indexed",
                    "chunk_count": 2,
                    "vector_ids": ["v1", "v2"],
                }
            ],
            "chunk_count": 2,
            "reset_collection": reset_collection,
        }

    def list_documents(self, workspace_id=None):
        return [
            {
                "document_id": "doc_1",
                "workspace_id": workspace_id or "workspace_1",
                "filename": "notes.md",
                "status": "indexed",
                "chunk_count": 2,
                "vector_ids": ["v1", "v2"],
            }
        ]

    def delete_document(self, document_id):
        self.deleted.append(document_id)
        return {
            "document_id": document_id,
            "deleted": True,
            "deleted_vector_count": 2,
        }


class FakeTraceService:
    def get_trace(self, session_id, trace_id=None):
        return {
            "trace_id": trace_id or "trace_1",
            "session_id": session_id,
            "workspace_id": "workspace_1",
            "events": [{"event_type": "node", "node": "retrieve"}],
            "route_decisions": [],
        }


class FakeEvaluationService:
    def run_evaluation(self, workspace_id, question_ids=None, include_baseline=True):
        return {
            "run_id": "eval_1",
            "workspace_id": workspace_id,
            "status": "completed",
            "summary": {"total_questions": 1, "judge_completion_rate": 1.0},
            "result_path": "data/evaluation_runs/eval_1.json",
        }

    def get_run(self, run_id):
        return {
            "run_id": run_id,
            "workspace_id": "workspace_1",
            "status": "completed",
            "summary": {"total_questions": 1, "judge_completion_rate": 1.0},
            "result_path": "data/evaluation_runs/eval_1.json",
        }

    def list_history_runs(self, limit=20):
        return [
            {
                "run_id": "eval_1",
                "created_at": "2026-06-27T12:00:00.000000Z",
                "source": "api",
                "workspace_id": "workspace_1",
                "status": "completed",
                "mode": "comparison",
                "schema_version": 4,
                "evaluator_version": "p5b",
                "prompt_manifest_hash": "sha256:abc",
                "question_count": 1,
                "result_path": "data/evaluation_runs/eval_1.json",
            }
        ]

    def query_history_trends(self, metric="correctness_score", system=None, limit=20):
        return [
            {
                "created_at": "2026-06-27T12:00:00.000000Z",
                "run_id": "eval_1",
                "system_id": system or "agentic",
                "system_label": "Agentic RAG",
                "evaluator_version": "p5b",
                "prompt_manifest_hash": "sha256:abc",
                "metric_name": metric,
                "metric_value": 0.75,
            }
        ]


def create_test_client():
    from api.dependencies import (
        get_agent_runner,
        get_document_service,
        get_evaluation_service,
        get_trace_service,
    )
    from api.main import create_app

    app = create_app()

    def fake_runner(question, chat_history=None, session_id=None, workspace_id=None):
        return {
            "answer": f"Answer for {question}",
            "citations": [{"source": "notes.md", "chunk_id": "c1"}],
            "trace_id": "trace_1",
            "retry_count": 1,
            "latency_ms": 12.5,
            "fallback_reason": "",
        }

    app.dependency_overrides[get_agent_runner] = lambda: fake_runner
    app.dependency_overrides[get_document_service] = lambda: FakeDocumentService()
    app.dependency_overrides[get_trace_service] = lambda: FakeTraceService()
    app.dependency_overrides[get_evaluation_service] = lambda: FakeEvaluationService()
    return TestClient(app)


def test_api_reports_p4b_version():
    from api.main import create_app

    assert create_app().version == "0.4.1-p4b"


def test_chat_route_returns_agent_response_schema():
    client = create_test_client()

    response = client.post(
        "/chat",
        json={
            "workspace_id": "workspace_1",
            "session_id": "session_1",
            "question": "What is Agentic RAG?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Answer for What is Agentic RAG?"
    assert payload["citations"] == [{"source": "notes.md", "chunk_id": "c1"}]
    assert payload["trace_id"] == "trace_1"
    assert payload["retry_count"] == 1
    assert payload["latency_ms"] == 12.5
    assert payload["fallback_triggered"] is False


def test_trace_route_returns_session_trace():
    client = create_test_client()

    response = client.get("/chat/session_1/trace", params={"trace_id": "trace_1"})

    assert response.status_code == 200
    assert response.json()["trace_id"] == "trace_1"
    assert response.json()["session_id"] == "session_1"


def test_document_routes_upload_index_list_and_delete():
    client = create_test_client()

    upload_response = client.post(
        "/documents/upload",
        data={"workspace_id": "workspace_1"},
        files={"files": ("notes.md", b"Agentic RAG notes", "text/markdown")},
    )
    assert upload_response.status_code == 200
    assert upload_response.json()["documents"][0]["status"] == "uploaded"

    index_response = client.post(
        "/documents/index",
        json={"workspace_id": "workspace_1", "document_ids": ["doc_1"]},
    )
    assert index_response.status_code == 200
    assert index_response.json()["chunk_count"] == 2

    list_response = client.get("/documents", params={"workspace_id": "workspace_1"})
    assert list_response.status_code == 200
    assert list_response.json()["documents"][0]["document_id"] == "doc_1"

    delete_response = client.delete("/documents/doc_1")
    assert delete_response.status_code == 200
    assert delete_response.json() == {
        "document_id": "doc_1",
        "deleted": True,
        "deleted_vector_count": 2,
    }


def test_evaluation_routes_run_and_read_results():
    client = create_test_client()

    run_response = client.post(
        "/evaluation/run",
        json={
            "workspace_id": "workspace_1",
            "question_ids": ["q001"],
            "include_baseline": True,
        },
    )
    assert run_response.status_code == 200
    assert run_response.json()["run_id"] == "eval_1"
    assert run_response.json()["summary"] == {
        "total_questions": 1,
        "judge_completion_rate": 1.0,
    }

    get_response = client.get("/evaluation/eval_1")
    assert get_response.status_code == 200
    assert get_response.json()["run_id"] == "eval_1"
    assert get_response.json()["summary"] == {
        "total_questions": 1,
        "judge_completion_rate": 1.0,
    }


def test_evaluation_history_routes_list_runs_and_trends_before_run_id_capture():
    client = create_test_client()

    history_response = client.get("/evaluation/history", params={"limit": 5})
    trends_response = client.get(
        "/evaluation/history/trends",
        params={"metric": "correctness_score", "system": "agentic", "limit": 5},
    )

    assert history_response.status_code == 200
    assert history_response.json()["runs"][0]["run_id"] == "eval_1"
    assert history_response.json()["runs"][0]["evaluator_version"] == "p5b"
    assert trends_response.status_code == 200
    assert trends_response.json() == {
        "metric": "correctness_score",
        "system": "agentic",
        "rows": [
            {
                "created_at": "2026-06-27T12:00:00.000000Z",
                "run_id": "eval_1",
                "system_id": "agentic",
                "system_label": "Agentic RAG",
                "evaluator_version": "p5b",
                "prompt_manifest_hash": "sha256:abc",
                "metric_name": "correctness_score",
                "metric_value": 0.75,
            }
        ],
    }
