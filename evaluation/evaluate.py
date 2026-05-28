"""Lightweight evaluation runner for Agentic RAG."""

from __future__ import annotations

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

        normalized = dict(record)
        normalized["expected_keywords"] = _normalize_expected_keywords_for_loader(
            record.get("expected_keywords")
        )
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


def _normalize_expected_keywords_for_loader(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        if all(isinstance(keyword, str) for keyword in value):
            return value
        raise ValueError("expected_keywords must contain only strings")
    raise ValueError("expected_keywords must be a string or list of strings")


def _normalize_expected_keywords(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _build_success_result(
    item: dict[str, Any],
    agent_result: dict[str, Any],
) -> dict[str, Any]:
    answer = agent_result.get("answer", "")
    citations = agent_result.get("citations", [])
    retrieved_documents = agent_result.get("retrieved_documents", [])
    expected_keywords = _normalize_expected_keywords(item.get("expected_keywords", []))

    return {
        "question": item["question"],
        "answer_returned": bool(answer),
        "citation_returned": bool(citations),
        "source_hit": _has_expected_source(
            item.get("expected_source"),
            retrieved_documents,
        ),
        "keyword_hit": _has_expected_keywords(answer, expected_keywords),
        "rewrite_triggered": int(agent_result.get("rewrite_count", 0) or 0) > 0,
        "latency": 0,
        "error": None,
        "answer": answer,
        "citations": citations,
        "retrieved_documents": retrieved_documents,
    }


def _build_error_result(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": item["question"],
        "answer_returned": False,
        "citation_returned": False,
        "source_hit": False,
        "keyword_hit": False,
        "rewrite_triggered": False,
        "latency": 0,
        "error": None,
        "answer": "",
        "citations": [],
        "retrieved_documents": [],
    }


def _format_error(exc: Exception) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def _has_expected_source_value(expected_source: Any) -> bool:
    return isinstance(expected_source, str) and bool(expected_source.strip())


def _has_expected_source(
    expected_source: Any,
    retrieved_documents: Any,
) -> bool:
    if not _has_expected_source_value(expected_source) or not isinstance(
        retrieved_documents, list
    ):
        return False

    return any(
        isinstance(document, dict) and document.get("source") == expected_source
        for document in retrieved_documents
    )


def _has_expected_keywords(answer: Any, expected_keywords: list[Any]) -> bool:
    if not expected_keywords or not isinstance(answer, str):
        return False

    lower_answer = answer.lower()
    return all(str(keyword).lower() in lower_answer for keyword in expected_keywords)


def _summarize(
    results: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    total_questions = len(results)
    if total_questions == 0:
        return {
            "total_questions": 0,
            "answer_rate": 0,
            "citation_rate": 0,
            "source_hit_rate": 0,
            "average_latency": 0,
            "rewrite_triggered_count": 0,
            "keyword_hit_rate": 0,
            "error_count": 0,
        }

    answer_count = sum(1 for result in results if result["answer_returned"])
    citation_count = sum(1 for result in results if result["citation_returned"])
    source_hit_count = sum(1 for result in results if result["source_hit"])
    keyword_hit_count = sum(1 for result in results if result["keyword_hit"])
    source_expected_count = sum(
        1 for item in questions if _has_expected_source_value(item.get("expected_source"))
    )
    keyword_expected_count = sum(
        1
        for item in questions
        if _normalize_expected_keywords(item.get("expected_keywords", []))
    )
    rewrite_triggered_count = sum(1 for result in results if result["rewrite_triggered"])
    error_count = sum(1 for result in results if result["error"])
    total_latency = sum(result["latency"] for result in results)

    return {
        "total_questions": total_questions,
        "answer_rate": round(answer_count / total_questions, 4),
        "citation_rate": round(citation_count / total_questions, 4),
        "source_hit_rate": _rate(source_hit_count, source_expected_count),
        "average_latency": round(total_latency / total_questions, 4),
        "rewrite_triggered_count": rewrite_triggered_count,
        "keyword_hit_rate": _rate(keyword_hit_count, keyword_expected_count),
        "error_count": error_count,
    }


def _rate(count: int, denominator: int) -> float:
    if denominator == 0:
        return 0
    return round(count / denominator, 4)
