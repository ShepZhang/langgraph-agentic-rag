"""Lightweight evaluation runner for Agentic RAG."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

from agent.graph import run_agent


DEFAULT_EVAL_PATH = Path(__file__).with_name("eval_questions.json")


def load_eval_questions(path: str | Path = DEFAULT_EVAL_PATH) -> list[dict[str, Any]]:
    """Load and validate evaluation questions."""

    with Path(path).open(encoding="utf-8") as question_file:
        records = json.load(question_file)

    if not isinstance(records, list):
        raise ValueError("evaluation questions must be a list")

    questions: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"evaluation question at index {index} must be an object")

        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"evaluation question at index {index} requires question")

        should_answer = record.get("should_answer", True)
        if not isinstance(should_answer, bool):
            raise ValueError("should_answer must be a boolean")

        normalized = dict(record)
        normalized["expected_keywords"] = _normalize_string_list(
            record.get("expected_keywords"),
            field_name="expected_keywords",
        )
        normalized["expected_sources"] = _normalize_expected_sources(record)
        normalized["should_answer"] = should_answer
        questions.append(normalized)

    return questions


def evaluate_questions(
    questions: list[dict[str, Any]],
    run_agent_fn: Callable[[str], dict[str, Any]] = run_agent,
    timer: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Evaluate questions and return per-question results plus summary metrics."""

    results: list[dict[str, Any]] = []
    for item in questions:
        question = item["question"]
        started_at = timer()
        try:
            agent_result = run_agent_fn(question)
            result = _build_success_result(item, agent_result)
            error = None
        except Exception as exc:  # noqa: BLE001 - evaluation records agent failures.
            result = _build_error_result(item)
            error = _format_error(exc)
        latency = timer() - started_at

        result["latency"] = latency
        result["error"] = error
        results.append(result)

    return {"summary": _summarize(results, questions), "results": results}


def format_report(report: dict[str, Any]) -> str:
    """Format an evaluation report for terminal output."""

    lines = ["Evaluation Report", "", "Summary"]
    for key, value in report.get("summary", {}).items():
        lines.append(f"{key}: {value}")

    lines.extend(["", "Questions"])
    for index, result in enumerate(report.get("results", []), start=1):
        lines.append(
            (
                f"{index}. {result.get('question', '')} | "
                f"answer={_format_bool(result.get('answer_returned'))} | "
                f"fallback={_format_bool(result.get('fallback_triggered'))} | "
                f"citation={_format_bool(result.get('citation_returned'))} | "
                f"source_hit={_format_bool(result.get('source_hit'))} | "
                f"keyword_hit={_format_bool(result.get('keyword_hit'))} | "
                f"rewrite={_format_bool(result.get('rewrite_triggered'))} | "
                f"retry_count={result.get('retry_count', 0)} | "
                f"retrieved={result.get('retrieved_doc_count', 0)} | "
                f"relevant={result.get('relevant_doc_count', 0)} | "
                f"latency={float(result.get('latency', 0)):.4f}s | "
                f"error={result.get('error') or ''}"
            )
        )

    return "\n".join(lines)


def main(
    argv: list[str] | None = None,
    run_agent_fn: Callable[[str], dict[str, Any]] = run_agent,
) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Run Agentic RAG evaluations.")
    parser.add_argument(
        "--questions",
        default=DEFAULT_EVAL_PATH,
        type=Path,
        help="Path to evaluation questions JSON.",
    )
    args = parser.parse_args(argv)

    try:
        questions = load_eval_questions(args.questions)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(f"Unable to load evaluation questions: {exc}")

    report = evaluate_questions(questions, run_agent_fn=run_agent_fn)
    print(format_report(report))
    return 0


def _build_success_result(
    item: dict[str, Any],
    agent_result: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(agent_result, dict):
        raise ValueError("agent result must be a dictionary")

    answer = agent_result.get("answer", "")
    if not isinstance(answer, str):
        raise ValueError("agent answer must be a string")

    citations = _safe_list(agent_result.get("citations", []), "citations")
    retrieved_documents = _safe_list(
        agent_result.get("retrieved_documents", []),
        "retrieved_documents",
    )
    relevant_documents = _safe_list(
        agent_result.get("relevant_documents", []),
        "relevant_documents",
    )
    retry_count = _safe_int(
        agent_result.get("retry_count", agent_result.get("rewrite_count", 0)),
        "retry_count",
    )
    fallback_reason = agent_result.get("fallback_reason", "")
    if fallback_reason is None:
        fallback_reason = ""
    if not isinstance(fallback_reason, str):
        raise ValueError("fallback_reason must be a string")

    fallback_triggered = bool(fallback_reason.strip()) or _is_fallback_answer(answer)
    should_answer = bool(item.get("should_answer", True))
    answer_returned = bool(answer.strip()) and not fallback_triggered
    expected_keywords = item.get("expected_keywords", [])
    expected_sources = item.get("expected_sources", [])

    return {
        "question": item["question"],
        "answer_returned": answer_returned,
        "fallback_triggered": fallback_triggered,
        "fallback_correct": fallback_triggered is (not should_answer),
        "citation_returned": bool(citations),
        "source_hit": _has_expected_source(expected_sources, citations, retrieved_documents),
        "keyword_hit": (
            answer_returned and _has_expected_keywords(answer, expected_keywords)
        ),
        "rewrite_triggered": retry_count > 0,
        "retry_count": retry_count,
        "retrieved_doc_count": len(retrieved_documents),
        "relevant_doc_count": len(relevant_documents),
        "latency": 0,
        "error": None,
        "answer": answer,
        "citations": citations,
        "retrieved_documents": retrieved_documents,
        "relevant_documents": relevant_documents,
    }


def _build_error_result(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": item["question"],
        "answer_returned": False,
        "fallback_triggered": False,
        "fallback_correct": False,
        "citation_returned": False,
        "source_hit": False,
        "keyword_hit": False,
        "rewrite_triggered": False,
        "retry_count": 0,
        "retrieved_doc_count": 0,
        "relevant_doc_count": 0,
        "latency": 0,
        "error": None,
        "answer": "",
        "citations": [],
        "retrieved_documents": [],
        "relevant_documents": [],
    }


def _summarize(
    results: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    total_questions = len(results)
    if total_questions == 0:
        return {
            "total_questions": 0,
            "answer_rate": 0,
            "fallback_rate": 0,
            "citation_rate": 0,
            "source_hit_rate": 0,
            "keyword_hit_rate": 0,
            "fallback_correctness_rate": 0,
            "average_retry_count": 0,
            "average_retrieved_docs": 0,
            "average_relevant_docs": 0,
            "average_latency": 0,
            "rewrite_triggered_count": 0,
            "error_count": 0,
        }

    answer_count = sum(1 for result in results if result["answer_returned"])
    fallback_count = sum(1 for result in results if result["fallback_triggered"])
    citation_count = sum(1 for result in results if result["citation_returned"])
    source_hit_count = sum(1 for result in results if result["source_hit"])
    keyword_hit_count = sum(1 for result in results if result["keyword_hit"])
    fallback_correct_count = sum(1 for result in results if result["fallback_correct"])
    source_expected_count = sum(
        1 for item in questions if item.get("expected_sources", [])
    )
    keyword_expected_count = sum(
        1 for item in questions if item.get("expected_keywords", [])
    )
    rewrite_triggered_count = sum(1 for result in results if result["rewrite_triggered"])
    error_count = sum(1 for result in results if result["error"])

    return {
        "total_questions": total_questions,
        "answer_rate": _rate(answer_count, total_questions),
        "fallback_rate": _rate(fallback_count, total_questions),
        "citation_rate": _rate(citation_count, total_questions),
        "source_hit_rate": _rate(source_hit_count, source_expected_count),
        "keyword_hit_rate": _rate(keyword_hit_count, keyword_expected_count),
        "fallback_correctness_rate": _rate(fallback_correct_count, total_questions),
        "average_retry_count": _average(
            result["retry_count"] for result in results
        ),
        "average_retrieved_docs": _average(
            result["retrieved_doc_count"] for result in results
        ),
        "average_relevant_docs": _average(
            result["relevant_doc_count"] for result in results
        ),
        "average_latency": _average(result["latency"] for result in results),
        "rewrite_triggered_count": rewrite_triggered_count,
        "error_count": error_count,
    }


def _normalize_expected_sources(record: dict[str, Any]) -> list[str]:
    if "expected_sources" in record:
        return _normalize_string_list(
            record.get("expected_sources"),
            field_name="expected_sources",
        )
    return _normalize_string_list(
        record.get("expected_source"),
        field_name="expected_source",
    )


def _normalize_string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return value
        raise ValueError(f"{field_name} must contain only strings")
    raise ValueError(f"{field_name} must be a string or list of strings")


def _safe_list(value: Any, field_name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return value


def _safe_int(value: Any, field_name: str) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _has_expected_source(
    expected_sources: Any,
    citations: list[Any],
    retrieved_documents: list[Any],
) -> bool:
    if not expected_sources:
        return False

    evidence = citations if citations else retrieved_documents
    return any(
        isinstance(document, dict)
        and isinstance(document.get("source"), str)
        and document["source"] in expected_sources
        for document in evidence
    )


def _has_expected_keywords(answer: Any, expected_keywords: list[Any]) -> bool:
    if not expected_keywords or not isinstance(answer, str):
        return False

    lower_answer = answer.lower()
    return all(str(keyword).lower() in lower_answer for keyword in expected_keywords)


def _is_fallback_answer(answer: str) -> bool:
    lower_answer = answer.lower()
    fallback_markers = [
        "cannot answer from the current documents",
        "cannot answer based on the current documents",
        "i cannot answer",
        "无法可靠回答",
        "无法根据当前文档回答",
    ]
    return any(marker in lower_answer for marker in fallback_markers)


def _format_error(exc: Exception) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def _format_bool(value: Any) -> str:
    return "true" if value else "false"


def _rate(count: int, denominator: int) -> float:
    if denominator == 0:
        return 0
    return round(count / denominator, 4)


def _average(values: Any) -> float:
    values_list = list(values)
    if not values_list:
        return 0
    return round(sum(values_list) / len(values_list), 4)


if __name__ == "__main__":
    raise SystemExit(main())
