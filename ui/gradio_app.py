"""Gradio UI entrypoint for the Agentic RAG document QA system."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import gradio as gr

from agent.graph import run_agent
from config import get_settings
from evaluation.dashboard_formatters import (
    build_failure_count_rows,
    failure_cases_to_table,
)
from evaluation.dashboard_models import SMOKE_QUESTION_IDS
from evaluation.dashboard_service import EvaluationDashboardService
from rag.chunker import split_documents
from rag.loader import load_documents
from rag.vectorstore import create_vectorstore


SYSTEM_MODE_VALUES = {
    "Naive RAG": "naive",
    "Agentic RAG": "agentic",
    "Compare Both": "comparison",
}


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


def question_selection(
    question_options: Sequence[Mapping[str, Any]],
    selection: str,
) -> list[str]:
    """Return smoke or complete question IDs in dataset order."""

    all_ids = [str(option["id"]) for option in question_options]
    if selection == "all":
        return all_ids

    smoke_ids = set(SMOKE_QUESTION_IDS)
    return [question_id for question_id in all_ids if question_id in smoke_ids]


def run_dashboard_evaluation(
    system_label: str,
    question_ids: Sequence[str] | None,
    previous_state: Mapping[str, Any] | None,
    service: EvaluationDashboardService | None = None,
) -> tuple[
    Mapping[str, Any],
    str,
    list[list[Any]],
    list[list[Any]],
    list[list[str]],
    dict[str, Any],
]:
    """Run quick evaluation and preserve the last successful view on failure."""

    resolved_service = service or EvaluationDashboardService()
    mode = SYSTEM_MODE_VALUES.get(system_label, "comparison")
    view = resolved_service.run_quick_evaluation(list(question_ids or []), mode)
    active_view = (
        view if view["status"] == "completed" else previous_state or view
    )
    cases = _dashboard_cases(active_view)

    return (
        active_view,
        str(view["message"]),
        list(active_view.get("summary_rows", [])),
        list(active_view.get("failure_count_rows", [])),
        failure_cases_to_table(cases),
        gr.update(choices=_failure_choices(cases), value=None),
    )


def filter_dashboard_failures(
    dashboard_state: Mapping[str, Any] | None,
    system: str | None,
    failure_type: str | None,
    service: EvaluationDashboardService | None = None,
) -> tuple[list[list[Any]], list[list[str]], dict[str, Any]]:
    """Filter the current dashboard state without rerunning evaluation."""

    resolved_service = service or EvaluationDashboardService()
    state = _dashboard_state(dashboard_state)
    cases = resolved_service.filter_failure_cases(
        state,
        system=system,
        failure_type=failure_type,
    )

    return (
        build_failure_count_rows(cases),
        failure_cases_to_table(cases),
        gr.update(choices=_failure_choices(cases), value=None),
    )


def format_failure_detail(
    dashboard_state: Mapping[str, Any] | None,
    case_key: str | None,
    service: EvaluationDashboardService | None = None,
) -> str:
    """Format one failure detail as concise Markdown."""

    resolved_service = service or EvaluationDashboardService()
    detail = resolved_service.get_failure_detail(
        _dashboard_state(dashboard_state),
        case_key,
    )
    if not detail["case_key"]:
        return "Select a failed case to inspect its diagnosis."

    return (
        f"### {detail['title']}\n\n"
        f"**Reason:** {detail['reason']}\n\n"
        f"**Suggestion:** {detail['suggestion']}\n\n"
        f"**Diagnostics source:** `{detail['diagnostics_source']}`"
    )


def load_ablation_dashboard(
    previous_state: Mapping[str, Any] | None,
    service: EvaluationDashboardService | None = None,
) -> tuple[
    Mapping[str, Any],
    str,
    list[list[Any]],
    list[list[Any]],
    list[list[str]],
    dict[str, Any],
    dict[str, Any],
]:
    """Load the saved ablation snapshot and preserve the last valid view."""

    resolved_service = service or EvaluationDashboardService()
    view = resolved_service.load_ablation_snapshot()
    active_view = (
        view if view["status"] == "completed" else previous_state or view
    )
    cases = _dashboard_cases(active_view)
    variant_choices = [("All variants", "all"), *_variant_choices(active_view)]

    return (
        active_view,
        str(view["message"]),
        list(active_view.get("summary_rows", [])),
        list(active_view.get("failure_count_rows", [])),
        failure_cases_to_table(cases),
        gr.update(choices=_failure_choices(cases), value=None),
        gr.update(choices=variant_choices, value="all"),
    )


def format_variant_runtime_config(
    dashboard_state: Mapping[str, Any] | None,
    variant_id: str | None,
    service: EvaluationDashboardService | None = None,
) -> dict[str, Any]:
    """Return saved runtime config for one selected ablation variant."""

    resolved_service = service or EvaluationDashboardService()
    return resolved_service.get_runtime_config(
        _dashboard_state(dashboard_state),
        variant_id,
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


def _dashboard_state(
    dashboard_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if dashboard_state is None:
        return _empty_dashboard_state()
    return dict(dashboard_state)


def _empty_dashboard_state() -> dict[str, Any]:
    return {
        "status": "unavailable",
        "run_id": "",
        "summary_rows": [],
        "failure_count_rows": [],
        "failure_cases": [],
        "raw_report": {},
        "message": "",
    }


def _dashboard_cases(
    dashboard_state: Mapping[str, Any] | None,
) -> list[Mapping[str, Any]]:
    state = dashboard_state or {}
    cases = state.get("failure_cases", [])
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
        return []
    return [case for case in cases if isinstance(case, Mapping)]


def _failure_choices(
    cases: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str]]:
    return [
        (
            f"{case['question_id']} / {case['failure_type']}",
            str(case["case_key"]),
        )
        for case in cases
    ]


def _variant_choices(
    dashboard_state: Mapping[str, Any] | None,
) -> list[tuple[str, str]]:
    state = dashboard_state or {}
    raw_report = state.get("raw_report", {})
    runs = raw_report.get("runs", []) if isinstance(raw_report, Mapping) else []
    choices: list[tuple[str, str]] = []
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        run_id = str(run.get("id") or "")
        if not run_id:
            continue
        method = str(run.get("method") or run_id)
        choices.append((f"{run_id} {method}", run_id))
    return choices


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
