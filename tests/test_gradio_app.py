"""Tests for Gradio UI helper functions."""

from __future__ import annotations

from langchain_core.documents import Document

from ui.gradio_app import (
    answer_question,
    build_document_index,
    filter_dashboard_failures,
    format_failure_detail,
    format_variant_runtime_config,
    load_ablation_dashboard,
    question_selection,
    run_dashboard_evaluation,
)


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
    assert "Retrieval attempts: 0" in diagnostics
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
                    "snippet": "context",
                }
            ],
            "retrieved_documents": [{"content": "context", "source": "notes.txt"}],
            "relevant_documents": [{"content": "context", "source": "notes.txt"}],
            "current_query": "What is Agentic RAG?",
            "rewritten_question": "What is Agentic RAG?",
            "rewrite_count": 0,
            "retry_count": 0,
            "retrieval_attempt": 1,
            "grading_reason": "Chunk 1 directly answers the question.",
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
    assert rewrite_count == 0
    assert "Rewrite triggered: No" in diagnostics
    assert "Retrieval attempts: 1" in diagnostics
    assert "Relevant chunks accepted: 1" in diagnostics
    assert "Chunk 1 directly answers" in diagnostics
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
    assert "Retrieval attempts: 0" in diagnostics
    assert history == []


def test_create_app_returns_gradio_blocks():
    import gradio as gr

    from ui.gradio_app import create_app

    app = create_app()

    assert isinstance(app, gr.Blocks)


def _component_labels(config: dict) -> set[str]:
    labels = set()
    for component in config.get("components", []):
        props = component.get("props", {})
        label = props.get("label")
        if isinstance(label, str):
            labels.add(label)
    return labels


def _component_props_by_label(config: dict, label: str) -> dict:
    for component in config.get("components", []):
        props = component.get("props", {})
        if props.get("label") == label:
            return props
    return {}


def _component_values(config: dict) -> list[str]:
    values = []
    for component in config.get("components", []):
        value = component.get("props", {}).get("value")
        if isinstance(value, str):
            values.append(value)
    return values


def _components_with_class(config: dict, class_name: str) -> list[dict]:
    return [
        component
        for component in config.get("components", [])
        if class_name in component.get("props", {}).get("elem_classes", [])
    ]


def test_create_app_contains_document_and_evaluation_tabs():
    from ui.gradio_app import create_app

    app = create_app()
    labels = _component_labels(app.get_config_file())

    assert "Document QA" in labels
    assert "Evaluation" in labels
    assert "Quick Compare" in labels
    assert "Ablation Snapshot" in labels


def test_create_app_contains_dashboard_tables_and_filters():
    from ui.gradio_app import create_app

    app = create_app()
    labels = _component_labels(app.get_config_file())

    assert "Evaluation questions" in labels
    assert "Reliability metrics" in labels
    assert "Failure type counts" in labels
    assert "Failed cases" in labels
    assert "Failure case" in labels
    assert "Ablation reliability metrics" in labels
    assert "Ablation variant" in labels
    assert "Runtime configuration" in labels


def test_create_app_stacks_quick_controls_on_narrow_viewports():
    from ui.gradio_app import APP_CSS, create_app

    config = create_app().get_config_file()
    quick_control_rows = _components_with_class(
        config,
        "dashboard-quick-controls",
    )

    assert len(quick_control_rows) == 1
    assert quick_control_rows[0]["type"] == "row"
    assert "@media (max-width: 640px)" in APP_CSS
    assert ".dashboard-quick-controls" in APP_CSS
    assert "flex-direction: column" in APP_CSS
    assert "min-width: 100%" in APP_CSS


def test_app_main_passes_responsive_css_to_gradio_launch(monkeypatch):
    from types import SimpleNamespace

    import app as app_module
    from ui.gradio_app import APP_CSS

    launch_kwargs = {}

    class FakeApp:
        def launch(self, **kwargs):
            launch_kwargs.update(kwargs)

    monkeypatch.setattr(
        app_module,
        "get_settings",
        lambda: SimpleNamespace(
            gradio_server_name="127.0.0.1",
            gradio_server_port=7860,
        ),
    )
    monkeypatch.setattr(app_module, "create_app", FakeApp)

    app_module.main()

    assert launch_kwargs == {
        "server_name": "127.0.0.1",
        "server_port": 7860,
        "css": APP_CSS,
    }


def test_create_app_survives_question_loading_failure(monkeypatch):
    import gradio as gr
    import ui.gradio_app as gradio_app

    class BrokenDashboardService:
        def list_questions(self):
            raise RuntimeError("questions unavailable")

    monkeypatch.setattr(
        gradio_app,
        "EvaluationDashboardService",
        BrokenDashboardService,
    )

    app = gradio_app.create_app()
    config = app.get_config_file()
    labels = _component_labels(config)
    question_props = _component_props_by_label(config, "Evaluation questions")

    assert isinstance(app, gr.Blocks)
    assert "Document QA" in labels
    assert "Evaluation" in labels
    assert question_props.get("choices") == []
    assert any(
        "questions unavailable" in value
        for value in _component_values(config)
    )


class FakeDashboardService:
    def __init__(self, view):
        self.view = view
        self.run_calls = []
        self.filter_calls = []
        self.detail_calls = []
        self.snapshot_calls = []
        self.runtime_calls = []

    def run_quick_evaluation(self, question_ids, system_mode):
        self.run_calls.append((question_ids, system_mode))
        return self.view

    def filter_failure_cases(self, view, system=None, failure_type=None):
        self.filter_calls.append((view, system, failure_type))
        return [
            case
            for case in view["failure_cases"]
            if (not system or system == "all" or case["system"] == system)
            and (
                not failure_type
                or failure_type == "all"
                or case["failure_type"] == failure_type
            )
        ]

    def get_failure_detail(self, view, case_key):
        self.detail_calls.append((view, case_key))
        for case in view["failure_cases"]:
            if case["case_key"] == case_key:
                return {
                    "case_key": case_key,
                    "title": (
                        f"{case['system_label']} / {case['question_id']} / "
                        f"{case['failure_type']}"
                    ),
                    "reason": case["reason"],
                    "suggestion": case["suggestion"],
                    "diagnostics_source": case["diagnostics_source"],
                }
        return {
            "case_key": "",
            "title": "No failed case selected",
            "reason": "",
            "suggestion": "",
            "diagnostics_source": "unavailable",
        }

    def load_ablation_snapshot(self):
        self.snapshot_calls.append(())
        return self.view

    def get_runtime_config(self, view, variant_id):
        self.runtime_calls.append((view, variant_id))
        raw_report = view.get("raw_report") if isinstance(view, dict) else {}
        for run in raw_report.get("runs", []):
            if run.get("id") == variant_id:
                return run.get("runtime_config", {})
        return {}


def _dashboard_view(status="completed"):
    return {
        "status": status,
        "run_id": "quick-fixed",
        "summary_rows": [["Agentic RAG", 0.8, 0.9, 0.7, 1.0, 0, 2.5, 0.4]],
        "failure_count_rows": [["Agentic RAG", "retrieval_failure", 1]],
        "failure_cases": [
            {
                "case_key": "agentic:q001",
                "system": "agentic",
                "system_label": "Agentic RAG",
                "question_id": "q001",
                "question_type": "single_doc",
                "question": "Question q001",
                "failure_type": "retrieval_failure",
                "reason": "Expected source missing.",
                "suggestion": "Tune retrieval.",
                "diagnostics_source": "stored",
            }
        ],
        "raw_report": {},
        "message": "Evaluation completed.",
    }


def test_question_selection_returns_smoke_and_all_ids():
    options = [{"id": "q001"}, {"id": "q002"}, {"id": "q016"}]

    assert question_selection(options, "smoke") == ["q001", "q016"]
    assert question_selection(options, "all") == ["q001", "q002", "q016"]
    assert question_selection(options, "unknown") == ["q001", "q016"]


def test_run_dashboard_evaluation_returns_visible_rows_and_state():
    view = _dashboard_view()
    service = FakeDashboardService(view)

    result = run_dashboard_evaluation(
        "Agentic RAG",
        ["q001"],
        {},
        service=service,
    )

    assert len(result) == 9
    (
        state,
        status,
        metrics,
        counts,
        cases,
        case_update,
        system_update,
        type_update,
        detail,
    ) = result
    assert state == view
    assert "completed" in status.lower()
    assert metrics == view["summary_rows"]
    assert counts == view["failure_count_rows"]
    assert cases[0][0] == "agentic:q001"
    assert case_update["choices"] == [
        ("Agentic RAG / q001 / retrieval_failure", "agentic:q001")
    ]
    assert system_update["value"] == "all"
    assert type_update["value"] == "all"
    assert detail == "Select a failed case to inspect its diagnosis."
    assert service.run_calls == [(["q001"], "agentic")]


def test_failed_run_preserves_previous_successful_state():
    previous = _dashboard_view()
    failed = _dashboard_view(status="failed")
    failed["message"] = "Evaluation failed: unavailable"
    failed["summary_rows"] = []
    failed["failure_count_rows"] = []
    failed["failure_cases"] = []
    service = FakeDashboardService(failed)

    (
        state,
        status,
        metrics,
        counts,
        cases,
        choices,
        system_update,
        type_update,
        detail,
    ) = run_dashboard_evaluation(
        "Agentic RAG",
        ["q001"],
        previous,
        service=service,
    )

    assert state == previous
    assert "failed" in status.lower()
    assert metrics == previous["summary_rows"]
    assert counts == previous["failure_count_rows"]
    assert cases[0][0] == "agentic:q001"
    assert choices["choices"] == [
        ("Agentic RAG / q001 / retrieval_failure", "agentic:q001")
    ]
    assert system_update["value"] == "all"
    assert type_update["value"] == "all"
    assert detail == "Select a failed case to inspect its diagnosis."


def test_failed_run_ignores_non_mapping_previous_state():
    failed = _dashboard_view(status="failed")
    failed["message"] = "Evaluation failed: unavailable"
    failed["summary_rows"] = []
    failed["failure_count_rows"] = []
    failed["failure_cases"] = []
    service = FakeDashboardService(failed)

    (
        state,
        status,
        metrics,
        counts,
        cases,
        choices,
        system_update,
        type_update,
        detail,
    ) = run_dashboard_evaluation(
        "Agentic RAG",
        ["q001"],
        ["bad"],
        service=service,
    )

    assert state == failed
    assert "failed" in status.lower()
    assert metrics == []
    assert counts == []
    assert cases == []
    assert choices["choices"] == []
    assert system_update["value"] == "all"
    assert type_update["value"] == "all"
    assert detail == "Select a failed case to inspect its diagnosis."


def test_filter_and_detail_helpers_do_not_run_evaluation_again():
    view = _dashboard_view()
    service = FakeDashboardService(view)

    counts, table, choices = filter_dashboard_failures(
        view,
        "agentic",
        "retrieval_failure",
        service=service,
    )
    detail = format_failure_detail(
        view,
        "agentic:q001",
        service=service,
    )
    empty_detail = format_failure_detail(
        view,
        None,
        service=service,
    )

    assert table[0][0] == "agentic:q001"
    assert counts == [["Agentic RAG", "retrieval_failure", 1]]
    assert choices["choices"] == [
        ("Agentic RAG / q001 / retrieval_failure", "agentic:q001")
    ]
    assert "### Agentic RAG / q001 / retrieval_failure" in detail
    assert "Expected source missing." in detail
    assert "Diagnostics source" in detail
    assert empty_detail == "Select a failed case to inspect its diagnosis."
    assert service.run_calls == []
    assert service.filter_calls == [(view, "agentic", "retrieval_failure")]


def test_dashboard_helpers_treat_non_mapping_state_as_empty():
    service = FakeDashboardService(_dashboard_view())

    counts, table, choices = filter_dashboard_failures(
        ["bad"],
        "agentic",
        "retrieval_failure",
        service=service,
    )
    detail = format_failure_detail(
        ["bad"],
        "agentic:q001",
        service=service,
    )
    runtime_config = format_variant_runtime_config(
        ["bad"],
        "v0_naive",
        service=service,
    )

    assert counts == []
    assert table == []
    assert choices["choices"] == []
    assert detail == "Select a failed case to inspect its diagnosis."
    assert runtime_config == {}
    assert service.filter_calls == []
    assert service.detail_calls == []
    assert service.runtime_calls == []


def test_snapshot_helpers_return_dropdown_updates_and_runtime_config():
    view = _dashboard_view()
    view["run_id"] = "snapshot-fixed"
    view["raw_report"] = {
        "runs": [
            {
                "id": "v0_naive",
                "method": "Naive RAG",
                "runtime_config": {"llm": {"model": "test-model"}},
            }
        ]
    }
    service = FakeDashboardService(view)

    result = load_ablation_dashboard({}, service=service)
    assert len(result) == 10
    (
        state,
        status,
        metrics,
        counts,
        cases,
        case_update,
        variant_update,
        type_update,
        runtime_config,
        detail,
    ) = result

    assert state == view
    assert "completed" in status.lower()
    assert metrics == view["summary_rows"]
    assert counts == view["failure_count_rows"]
    assert cases[0][0] == "agentic:q001"
    assert case_update["choices"][0][1] == "agentic:q001"
    assert variant_update["choices"] == [
        ("All variants", "all"),
        ("v0_naive Naive RAG", "v0_naive"),
    ]
    assert variant_update["value"] == "all"
    assert type_update["value"] == "all"
    assert runtime_config == {}
    assert detail == "Select a failed case to inspect its diagnosis."
    assert format_variant_runtime_config(
        view,
        "v0_naive",
        service=service,
    ) == {"llm": {"model": "test-model"}}
    assert format_variant_runtime_config(None, "v0_naive", service=service) == {}
    assert service.run_calls == []
    assert service.snapshot_calls == [()]


def test_snapshot_failure_choices_disambiguate_same_case_across_variants():
    view = _dashboard_view()
    first_case = {
        **view["failure_cases"][0],
        "case_key": "v0_naive:q004",
        "system": "v0_naive",
        "system_label": "v0_naive Naive RAG",
        "question_id": "q004",
        "failure_type": "generation_failure",
    }
    second_case = {
        **first_case,
        "case_key": "v1_query_rewrite:q004",
        "system": "v1_query_rewrite",
        "system_label": "v1_query_rewrite + Query Transformation",
    }
    view["failure_cases"] = [first_case, second_case]
    service = FakeDashboardService(view)

    result = load_ablation_dashboard({}, service=service)
    case_update = result[5]

    assert case_update["choices"] == [
        (
            "v0_naive Naive RAG / q004 / generation_failure",
            "v0_naive:q004",
        ),
        (
            "v1_query_rewrite + Query Transformation / q004 / "
            "generation_failure",
            "v1_query_rewrite:q004",
        ),
    ]


def test_snapshot_helper_preserves_previous_state_when_refresh_is_unavailable():
    previous = _dashboard_view()
    unavailable = _dashboard_view(status="unavailable")
    unavailable["message"] = "Ablation snapshot unavailable: missing artifact"
    unavailable["summary_rows"] = []
    unavailable["failure_count_rows"] = []
    unavailable["failure_cases"] = []
    service = FakeDashboardService(unavailable)

    (
        state,
        status,
        metrics,
        counts,
        cases,
        case_update,
        variant_update,
        type_update,
        runtime_config,
        detail,
    ) = load_ablation_dashboard(previous, service=service)

    assert state == previous
    assert "unavailable" in status.lower()
    assert metrics == previous["summary_rows"]
    assert counts == previous["failure_count_rows"]
    assert cases[0][0] == "agentic:q001"
    assert case_update["choices"][0][1] == "agentic:q001"
    assert variant_update["choices"][0] == ("All variants", "all")
    assert variant_update["value"] == "all"
    assert type_update["value"] == "all"
    assert runtime_config == {}
    assert detail == "Select a failed case to inspect its diagnosis."
