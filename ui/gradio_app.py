"""Gradio UI entrypoint for the Agentic RAG document QA system."""

from __future__ import annotations

import gradio as gr

from config import get_settings


def create_app() -> gr.Blocks:
    """Create the Gradio interface."""

    settings = get_settings()

    with gr.Blocks(title="Agentic RAG Document QA System") as demo:
        gr.Markdown(
            """
            # Agentic RAG Document QA System

            This project will support PDF, Markdown, and TXT upload, local
            vector indexing, LangGraph-based query rewriting, retrieval grading,
            conditional retries, and citation-aware answers.
            """
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Document Indexing")
                gr.File(
                    label="Upload documents",
                    file_count="multiple",
                    file_types=[".pdf", ".md", ".markdown", ".txt"],
                )
                gr.Button("Build Index", interactive=False)
                gr.Textbox(
                    label="Index status",
                    value="RAG indexing will be enabled in the RAG core phase.",
                    interactive=False,
                )

            with gr.Column():
                gr.Markdown("## Question Answering")
                gr.Textbox(label="Question", lines=3)
                gr.Button("Ask", interactive=False)
                gr.Textbox(
                    label="Answer",
                    value=(
                        "Agentic answering will be enabled after the LangGraph "
                        "workflow is implemented."
                    ),
                    lines=6,
                    interactive=False,
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
