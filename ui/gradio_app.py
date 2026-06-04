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
            _format_diagnostics(
                retry_count=0,
                retrieval_attempt=0,
                relevant_doc_count=0,
            ),
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
            _format_diagnostics(
                retry_count=0,
                retrieval_attempt=0,
                relevant_doc_count=0,
                fallback_reason=str(exc),
            ),
            history,
        )

    answer = result.get("answer", "")
    citations = result.get("citations", [])
    chunks = result.get("retrieved_documents", [])
    relevant_chunks = result.get("relevant_documents", [])
    rewritten_question = result.get("current_query") or result.get("rewritten_question", "")
    rewrite_count = int(result.get("rewrite_count", 0) or 0)
    retry_count = int(result.get("retry_count", rewrite_count) or 0)
    retrieval_attempt = int(result.get("retrieval_attempt", 0) or 0)
    grading_reason = str(result.get("grading_reason", "") or "")
    fallback_reason = str(result.get("fallback_reason", "") or "")
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
        _format_diagnostics(
            retry_count=retry_count,
            retrieval_attempt=retrieval_attempt,
            relevant_doc_count=len(relevant_chunks) if isinstance(relevant_chunks, list) else 0,
            grading_reason=grading_reason,
            fallback_reason=fallback_reason,
        ),
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
                label="Retry count",
                value=0,
                precision=0,
                interactive=False,
            )

        diagnostics = gr.Markdown(
            label="Diagnostics",
            value=_format_diagnostics(
                retry_count=0,
                retrieval_attempt=0,
                relevant_doc_count=0,
            ),
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
            - Max retry count: `{settings.max_retry_count}`
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


def _format_diagnostics(
    retry_count: int,
    retrieval_attempt: int,
    relevant_doc_count: int,
    grading_reason: str = "",
    fallback_reason: str = "",
) -> str:
    """Format agent diagnostics for display in the UI."""

    rewrite_triggered = "Yes" if retry_count > 0 else "No"
    lines = [
        f"Rewrite triggered: {rewrite_triggered}",
        f"Retry count: {retry_count}",
        f"Retrieval attempts: {retrieval_attempt}",
        f"Relevant chunks accepted: {relevant_doc_count}",
    ]
    if grading_reason:
        lines.append(f"Grading reason: {grading_reason}")
    if fallback_reason:
        lines.append(f"Fallback reason: {fallback_reason}")
    return "\n".join(lines)
