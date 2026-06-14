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
from evaluation.dashboard_models import (
    FAILURE_CASE_COLUMNS,
    FAILURE_COUNT_COLUMNS,
    METRIC_COLUMNS,
    SMOKE_QUESTION_IDS,
)
from evaluation.dashboard_service import EvaluationDashboardService
from rag.chunker import split_documents
from rag.loader import load_documents
from rag.vectorstore import create_vectorstore


SYSTEM_MODE_VALUES = {
    "Naive RAG": "naive",
    "Agentic RAG": "agentic",
    "Compare Both": "comparison",
}
DEFAULT_FAILURE_DETAIL = "Select a failed case to inspect its diagnosis."
DEFAULT_QUICK_STATUS = "Select a system mode and questions."
APP_CSS = """
@media (max-width: 640px) {
    .dashboard-quick-controls {
        flex-direction: column !important;
    }

    .dashboard-quick-controls > * {
        width: 100% !important;
        min-width: 100% !important;
    }
}
"""


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
    previous_state: Any,
    service: EvaluationDashboardService | None = None,
) -> tuple[
    Mapping[str, Any],
    str,
    list[list[Any]],
    list[list[Any]],
    list[list[str]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    str,
]:
    """Run quick evaluation and preserve the last successful view on failure."""

    resolved_service = service or EvaluationDashboardService()
    mode = SYSTEM_MODE_VALUES.get(system_label, "comparison")
    view = resolved_service.run_quick_evaluation(list(question_ids or []), mode)
    previous_view = _dashboard_state(previous_state)
    active_view = (
        view if view["status"] == "completed" else previous_view or view
    )
    cases = _dashboard_cases(active_view)

    return (
        active_view,
        str(view["message"]),
        list(active_view.get("summary_rows", [])),
        list(active_view.get("failure_count_rows", [])),
        failure_cases_to_table(cases),
        gr.update(choices=_failure_choices(cases), value=None),
        gr.update(value="all"),
        gr.update(value="all"),
        DEFAULT_FAILURE_DETAIL,
    )


def filter_dashboard_failures(
    dashboard_state: Any,
    system: str | None,
    failure_type: str | None,
    service: EvaluationDashboardService | None = None,
) -> tuple[list[list[Any]], list[list[str]], dict[str, Any]]:
    """Filter the current dashboard state without rerunning evaluation."""

    resolved_service = service or EvaluationDashboardService()
    state = _dashboard_state(dashboard_state)
    if not state:
        return ([], [], gr.update(choices=[], value=None))

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
    dashboard_state: Any,
    case_key: str | None,
    service: EvaluationDashboardService | None = None,
) -> str:
    """Format one failure detail as concise Markdown."""

    resolved_service = service or EvaluationDashboardService()
    state = _dashboard_state(dashboard_state)
    if not state:
        return DEFAULT_FAILURE_DETAIL

    detail = resolved_service.get_failure_detail(state, case_key)
    if not detail["case_key"]:
        return DEFAULT_FAILURE_DETAIL

    return (
        f"### {detail['title']}\n\n"
        f"**Reason:** {detail['reason']}\n\n"
        f"**Suggestion:** {detail['suggestion']}\n\n"
        f"**Diagnostics source:** `{detail['diagnostics_source']}`"
    )


def load_ablation_dashboard(
    previous_state: Any,
    service: EvaluationDashboardService | None = None,
) -> tuple[
    Mapping[str, Any],
    str,
    list[list[Any]],
    list[list[Any]],
    list[list[str]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    str,
]:
    """Load the saved ablation snapshot and preserve the last valid view."""

    resolved_service = service or EvaluationDashboardService()
    view = resolved_service.load_ablation_snapshot()
    previous_view = _dashboard_state(previous_state)
    active_view = (
        view if view["status"] == "completed" else previous_view or view
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
        gr.update(value="all"),
        {},
        DEFAULT_FAILURE_DETAIL,
    )


def format_variant_runtime_config(
    dashboard_state: Any,
    variant_id: str | None,
    service: EvaluationDashboardService | None = None,
) -> dict[str, Any]:
    """Return saved runtime config for one selected ablation variant."""

    resolved_service = service or EvaluationDashboardService()
    state = _dashboard_state(dashboard_state)
    if not state:
        return {}
    return resolved_service.get_runtime_config(state, variant_id)


def create_app() -> gr.Blocks:
    """Create the Gradio interface."""

    settings = get_settings()
    dashboard_service = EvaluationDashboardService()
    evaluation_initial_status = DEFAULT_QUICK_STATUS
    try:
        question_options = dashboard_service.list_questions()
    except Exception as exc:
        question_options = []
        evaluation_initial_status = (
            "Unable to load evaluation questions: "
            f"{type(exc).__name__}: {exc}"
        )
    question_choices = [
        (option["label"], option["id"])
        for option in question_options
    ]
    smoke_ids = question_selection(question_options, "smoke")

    with gr.Blocks(title="Reliability-oriented Agentic RAG") as demo:
        gr.Markdown(
            "# Reliability-oriented Agentic RAG Document QA System"
        )
        with gr.Tabs():
            with gr.Tab("Document QA"):
                _build_document_qa_tab(settings)
            with gr.Tab("Evaluation"):
                _build_evaluation_tab(
                    dashboard_service,
                    question_options,
                    question_choices,
                    smoke_ids,
                    evaluation_initial_status,
                )

    return demo


def _build_document_qa_tab(settings: Any) -> None:
    """Build the existing upload, indexing, and QA workflow."""

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


def _build_evaluation_tab(
    service: EvaluationDashboardService,
    question_options: Sequence[Mapping[str, Any]],
    question_choices: list[tuple[str, str]],
    smoke_ids: list[str],
    initial_status: str = DEFAULT_QUICK_STATUS,
) -> None:
    """Build quick evaluation and saved ablation views."""

    quick_state = gr.State({})
    snapshot_state = gr.State({})
    failure_type_choices = [
        ("All failure types", "all"),
        "retrieval_failure",
        "reranking_failure",
        "query_rewrite_failure",
        "generation_failure",
        "citation_failure",
        "fallback_failure",
        "tool_failure",
    ]

    with gr.Tabs():
        with gr.Tab("Quick Compare"):
            with gr.Row(elem_classes="dashboard-quick-controls"):
                system_mode = gr.Radio(
                    choices=list(SYSTEM_MODE_VALUES),
                    value="Compare Both",
                    label="System mode",
                )
                questions = gr.Dropdown(
                    choices=question_choices,
                    value=smoke_ids,
                    multiselect=True,
                    label="Evaluation questions",
                )

            with gr.Row():
                smoke_button = gr.Button("Select smoke set")
                all_button = gr.Button("Select all 36")
                run_button = gr.Button("Run Evaluation", variant="primary")

            quick_status = gr.Markdown(
                initial_status,
                elem_classes="dashboard-status",
            )
            quick_metrics = gr.Dataframe(
                headers=METRIC_COLUMNS,
                value=[],
                label="Reliability metrics",
                interactive=False,
                wrap=True,
            )

            with gr.Row():
                quick_counts = gr.Dataframe(
                    headers=FAILURE_COUNT_COLUMNS,
                    value=[],
                    label="Failure type counts",
                    interactive=False,
                )
                with gr.Column():
                    quick_system_filter = gr.Dropdown(
                        choices=[
                            ("All systems", "all"),
                            ("Naive RAG", "naive"),
                            ("Agentic RAG", "agentic"),
                        ],
                        value="all",
                        label="Failure system",
                    )
                    quick_type_filter = gr.Dropdown(
                        choices=failure_type_choices,
                        value="all",
                        label="Failure type",
                    )

            quick_cases = gr.Dataframe(
                headers=FAILURE_CASE_COLUMNS,
                value=[],
                label="Failed cases",
                interactive=False,
                wrap=True,
                elem_classes="dashboard-table",
            )
            quick_case = gr.Dropdown(
                choices=[],
                label="Failure case",
            )
            quick_detail = gr.Markdown(
                DEFAULT_FAILURE_DETAIL,
                elem_classes="dashboard-detail",
            )

            smoke_button.click(
                fn=lambda: question_selection(question_options, "smoke"),
                outputs=questions,
                queue=False,
            )
            all_button.click(
                fn=lambda: question_selection(question_options, "all"),
                outputs=questions,
                queue=False,
            )
            run_event = run_button.click(
                fn=lambda: (
                    gr.update(interactive=False),
                    "Evaluation is running. A 36-question run may take several minutes.",
                ),
                outputs=[run_button, quick_status],
                queue=False,
            )
            run_event.then(
                fn=lambda mode, ids, state: run_dashboard_evaluation(
                    mode,
                    ids,
                    state,
                    service=service,
                ),
                inputs=[system_mode, questions, quick_state],
                outputs=[
                    quick_state,
                    quick_status,
                    quick_metrics,
                    quick_counts,
                    quick_cases,
                    quick_case,
                    quick_system_filter,
                    quick_type_filter,
                    quick_detail,
                ],
            ).then(
                fn=lambda: gr.update(interactive=True),
                outputs=run_button,
                queue=False,
            )
            for component in (quick_system_filter, quick_type_filter):
                component.change(
                    fn=lambda state, system, failure: filter_dashboard_failures(
                        state,
                        system,
                        failure,
                        service=service,
                    ),
                    inputs=[
                        quick_state,
                        quick_system_filter,
                        quick_type_filter,
                    ],
                    outputs=[quick_counts, quick_cases, quick_case],
                    queue=False,
                )
            quick_case.change(
                fn=lambda state, key: format_failure_detail(
                    state,
                    key,
                    service=service,
                ),
                inputs=[quick_state, quick_case],
                outputs=quick_detail,
                queue=False,
            )

        with gr.Tab("Ablation Snapshot"):
            gr.Markdown(
                "This view reads the saved V0-V6 artifact and does not rerun models."
            )
            refresh_snapshot = gr.Button("Refresh Snapshot", variant="primary")
            snapshot_status = gr.Markdown(
                "Load the saved ablation artifact.",
                elem_classes="dashboard-status",
            )
            snapshot_metrics = gr.Dataframe(
                headers=METRIC_COLUMNS,
                value=[],
                label="Ablation reliability metrics",
                interactive=False,
                wrap=True,
            )
            with gr.Row():
                snapshot_variant = gr.Dropdown(
                    choices=[("All variants", "all")],
                    value="all",
                    label="Ablation variant",
                )
                snapshot_type_filter = gr.Dropdown(
                    choices=failure_type_choices,
                    value="all",
                    label="Ablation failure type",
                )

            snapshot_config = gr.JSON(label="Runtime configuration")
            snapshot_counts = gr.Dataframe(
                headers=FAILURE_COUNT_COLUMNS,
                value=[],
                label="Ablation failure type counts",
                interactive=False,
            )
            snapshot_cases = gr.Dataframe(
                headers=FAILURE_CASE_COLUMNS,
                value=[],
                label="Ablation failed cases",
                interactive=False,
                wrap=True,
                elem_classes="dashboard-table",
            )
            snapshot_case = gr.Dropdown(
                choices=[],
                label="Ablation failure case",
            )
            snapshot_detail = gr.Markdown(
                "Select a failed case to inspect stored or derived diagnostics.",
                elem_classes="dashboard-detail",
            )

            refresh_snapshot.click(
                fn=lambda state: load_ablation_dashboard(
                    state,
                    service=service,
                ),
                inputs=snapshot_state,
                outputs=[
                    snapshot_state,
                    snapshot_status,
                    snapshot_metrics,
                    snapshot_counts,
                    snapshot_cases,
                    snapshot_case,
                    snapshot_variant,
                    snapshot_type_filter,
                    snapshot_config,
                    snapshot_detail,
                ],
            )
            snapshot_variant.change(
                fn=lambda state, variant: format_variant_runtime_config(
                    state,
                    variant,
                    service=service,
                ),
                inputs=[snapshot_state, snapshot_variant],
                outputs=snapshot_config,
                queue=False,
            )
            for component in (snapshot_variant, snapshot_type_filter):
                component.change(
                    fn=lambda state, variant, failure: filter_dashboard_failures(
                        state,
                        variant,
                        failure,
                        service=service,
                    ),
                    inputs=[
                        snapshot_state,
                        snapshot_variant,
                        snapshot_type_filter,
                    ],
                    outputs=[
                        snapshot_counts,
                        snapshot_cases,
                        snapshot_case,
                    ],
                    queue=False,
                )
            snapshot_case.change(
                fn=lambda state, key: format_failure_detail(
                    state,
                    key,
                    service=service,
                ),
                inputs=[snapshot_state, snapshot_case],
                outputs=snapshot_detail,
                queue=False,
            )

            gr.Markdown(
                "Future upgrade: background live V0-V6 runs with progress, "
                "cancel, checkpoint recovery, shared run IDs, and trace linkage."
            )


def _normalize_uploaded_files(uploaded_files: list[Any] | None) -> list[str]:
    """Return filesystem paths from Gradio file values."""

    paths: list[str] = []
    for uploaded_file in uploaded_files or []:
        value = getattr(uploaded_file, "name", uploaded_file)
        if value:
            paths.append(str(Path(value)))
    return paths


def _dashboard_state(
    dashboard_state: Any,
) -> dict[str, Any]:
    if not isinstance(dashboard_state, Mapping):
        return {}
    return dict(dashboard_state)


def _dashboard_cases(
    dashboard_state: Any,
) -> list[Mapping[str, Any]]:
    state = _dashboard_state(dashboard_state)
    cases = state.get("failure_cases", [])
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
        return []
    return [case for case in cases if isinstance(case, Mapping)]


def _failure_choices(
    cases: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str]]:
    return [
        (
            f"{case['system_label']} / {case['question_id']} / "
            f"{case['failure_type']}",
            str(case["case_key"]),
        )
        for case in cases
    ]


def _variant_choices(
    dashboard_state: Any,
) -> list[tuple[str, str]]:
    state = _dashboard_state(dashboard_state)
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
