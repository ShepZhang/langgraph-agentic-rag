"""Gradio UI entrypoint for the Agentic RAG document QA system."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from agent.graph import run_agent
from config import get_settings
from rag.chunker import split_documents
from rag.loader import load_documents
from rag.vectorstore import create_vectorstore


def build_document_index(
    uploaded_files: list[Any] | None,
    load_fn=load_documents,
    split_fn=split_documents,
    create_store_fn=create_vectorstore,
) -> str:
    """Load uploaded files, split them, and build the local vector index."""

    paths = _normalize_uploaded_files(uploaded_files)
    if not paths:
        return "Upload at least one PDF, Markdown, or TXT file before building the index."

    try:
        documents = load_fn(paths)
        chunks = split_fn(documents)
        create_store_fn(chunks)
    except Exception as exc:
        return f"Unable to build index: {exc}"

    return (
        "Index built successfully. "
        f"Files: {len(paths)}. Documents: {len(documents)}. Chunks: {len(chunks)}."
    )


def answer_question(
    question: str,
    chat_history: list[dict[str, str]] | None = None,
    run_agent_fn=run_agent,
) -> tuple[str, list[Any], list[Any], str, int, str, list[dict[str, str]]]:
    """Return answer, citations, chunks, rewritten question, rewrite count, diagnostics, and history."""

    history = list(chat_history or [])
    normalized_question = (question or "").strip()

    if not normalized_question:
        return (
            "Enter a question to ask the indexed documents.",
            [],
            [],
            "",
            0,
            _format_diagnostics(rewrite_count=0, is_relevant=False),
            history,
        )

    try:
        result = run_agent_fn(normalized_question, chat_history=history)
    except Exception as exc:
        return (
            f"Unable to answer: {exc}",
            [],
            [],
            "",
            0,
            _format_diagnostics(rewrite_count=0, is_relevant=False),
            history,
        )

    answer = result.get("answer", "")
    citations = result.get("citations", [])
    chunks = result.get("retrieved_documents", [])
    rewritten_question = result.get("rewritten_question", "")
    rewrite_count = int(result.get("rewrite_count", 0) or 0)
    is_relevant = bool(result.get("is_relevant", False))
    updated_history = [
        *history,
        {"role": "user", "content": normalized_question},
        {"role": "assistant", "content": answer},
    ]

    return (
        answer,
        citations,
        chunks,
        rewritten_question,
        rewrite_count,
        _format_diagnostics(rewrite_count=rewrite_count, is_relevant=is_relevant),
        updated_history,
    )


def create_app() -> gr.Blocks:
    """Create the Gradio interface."""

    settings = get_settings()

    with gr.Blocks(title="Agentic RAG Document QA System") as demo:
        gr.Markdown(
            """
            # Agentic RAG Document QA System

            Upload PDF, Markdown, or TXT documents, build a local index, and ask
            citation-aware questions with Agentic RAG.
            """
        )

        chat_history = gr.State([])

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Document Indexing")
                uploaded_files = gr.File(
                    label="Upload documents",
                    file_count="multiple",
                    file_types=[".pdf", ".md", ".markdown", ".txt"],
                )
                build_button = gr.Button("Build Index", variant="primary")
                index_status = gr.Textbox(
                    label="Index status",
                    value="Upload documents, then build the index.",
                    interactive=False,
                )

            with gr.Column():
                gr.Markdown("## Question Answering")
                question = gr.Textbox(label="Question", lines=3)
                ask_button = gr.Button("Ask", variant="primary")
                answer = gr.Textbox(
                    label="Answer",
                    lines=6,
                    interactive=False,
                )

        with gr.Row():
            citations = gr.JSON(label="Citations")
            retrieved_chunks = gr.JSON(label="Retrieved chunks")

        with gr.Row():
            rewritten_question = gr.Textbox(
                label="Rewritten question",
                interactive=False,
            )
            rewrite_count = gr.Number(
                label="Rewrite count",
                value=0,
                precision=0,
                interactive=False,
            )

        diagnostics = gr.Markdown(
            label="Diagnostics",
            value=_format_diagnostics(rewrite_count=0, is_relevant=False),
        )

        build_button.click(
            fn=build_document_index,
            inputs=uploaded_files,
            outputs=index_status,
        )
        ask_button.click(
            fn=answer_question,
            inputs=[question, chat_history],
            outputs=[
                answer,
                citations,
                retrieved_chunks,
                rewritten_question,
                rewrite_count,
                diagnostics,
                chat_history,
            ],
        )

        gr.Markdown(
            f"""
            **Current configuration**

            - LLM model: `{settings.openai_model}`
            - Embedding model: `{settings.embedding_model}`
            - Chroma path: `{settings.chroma_persist_dir}`
            - Top K: `{settings.top_k}`
            - Max rewrite attempts: `{settings.max_rewrite_attempts}`
            """
        )

    return demo


def _normalize_uploaded_files(uploaded_files: list[Any] | None) -> list[str]:
    """Return filesystem paths from Gradio file values."""

    paths: list[str] = []
    for uploaded_file in uploaded_files or []:
        value = getattr(uploaded_file, "name", uploaded_file)
        if value:
            paths.append(str(Path(value)))
    return paths


def _format_diagnostics(rewrite_count: int, is_relevant: bool) -> str:
    """Format agent diagnostics for display in the UI."""

    rewrite_triggered = "Yes" if rewrite_count > 0 else "No"
    relevant_chunks = "Yes" if is_relevant else "No"
    return (
        f"Rewrite triggered: {rewrite_triggered}\n"
        f"Retry count: {rewrite_count}\n"
        f"Relevant chunks accepted: {relevant_chunks}"
    )
