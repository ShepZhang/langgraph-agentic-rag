"""Tests for Gradio UI helper functions."""

from __future__ import annotations

from langchain_core.documents import Document

from ui.gradio_app import answer_question, build_document_index


class UploadedFile:
    def __init__(self, name: str) -> None:
        self.name = name


def test_build_document_index_requires_files():
    status = build_document_index([])

    assert "Upload at least one" in status


def test_build_document_index_loads_splits_and_indexes_uploaded_files(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Agentic RAG notes", encoding="utf-8")
    calls = {}

    def fake_load(paths):
        calls["paths"] = paths
        return [
            Document(
                page_content="Agentic RAG notes",
                metadata={"source": "notes.txt", "page": None},
            )
        ]

    def fake_split(docs):
        calls["docs"] = docs
        return [
            Document(
                page_content="Agentic RAG chunk",
                metadata={
                    "source": "notes.txt",
                    "page": None,
                    "chunk_id": "notes.txt:pNA:c1",
                },
            )
        ]

    def fake_create_store(chunks):
        calls["chunks"] = chunks

    status = build_document_index(
        [UploadedFile(str(path))],
        load_fn=fake_load,
        split_fn=fake_split,
        create_store_fn=fake_create_store,
    )

    assert calls["paths"] == [str(path)]
    assert calls["docs"][0].page_content == "Agentic RAG notes"
    assert calls["chunks"][0].metadata["chunk_id"] == "notes.txt:pNA:c1"
    assert "Index built successfully" in status
    assert "Chunks: 1" in status


def test_answer_question_requires_non_empty_question():
    answer, citations, chunks, rewritten, rewrite_count, diagnostics, history = (
        answer_question("   ", [])
    )

    assert "Enter a question" in answer
    assert citations == []
    assert chunks == []
    assert rewritten == ""
    assert rewrite_count == 0
    assert "Rewrite triggered: No" in diagnostics
    assert history == []


def test_answer_question_returns_agent_payload_and_updates_history():
    def fake_run_agent(question, chat_history=None):
        return {
            "answer": "Agentic RAG uses query rewriting.",
            "citations": [
                {
                    "source": "notes.txt",
                    "page": None,
                    "chunk_id": "notes.txt:pNA:c1",
                    "score": 0.9,
                }
            ],
            "retrieved_documents": [{"content": "context", "source": "notes.txt"}],
            "rewritten_question": "What is Agentic RAG?",
            "rewrite_count": 1,
            "is_relevant": True,
        }

    result = answer_question(
        "What is it?",
        [{"role": "user", "content": "Tell me about Agentic RAG"}],
        run_agent_fn=fake_run_agent,
    )

    answer, citations, chunks, rewritten, rewrite_count, diagnostics, history = result
    assert answer == "Agentic RAG uses query rewriting."
    assert citations[0]["source"] == "notes.txt"
    assert chunks[0]["content"] == "context"
    assert rewritten == "What is Agentic RAG?"
    assert rewrite_count == 1
    assert "Rewrite triggered: Yes" in diagnostics
    assert "Relevant chunks accepted: Yes" in diagnostics
    assert history[-2]["role"] == "user"
    assert history[-1]["role"] == "assistant"


def test_answer_question_returns_clear_error_when_agent_fails():
    def failing_run_agent(question, chat_history=None):
        raise RuntimeError("Missing LLM configuration")

    answer, citations, chunks, rewritten, rewrite_count, diagnostics, history = (
        answer_question(
            "What is Agentic RAG?",
            [],
            run_agent_fn=failing_run_agent,
        )
    )

    assert "Unable to answer" in answer
    assert "Missing LLM configuration" in answer
    assert citations == []
    assert chunks == []
    assert rewritten == ""
    assert rewrite_count == 0
    assert "Rewrite triggered: No" in diagnostics
    assert history == []


def test_create_app_returns_gradio_blocks():
    import gradio as gr

    from ui.gradio_app import create_app

    app = create_app()

    assert isinstance(app, gr.Blocks)
