"""Lightweight evaluation runner for Agentic RAG."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

from agent.graph import run_agent
from evaluation.baselines import run_naive_rag


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
        requires_rewrite = record.get("requires_rewrite", False)
        if not isinstance(requires_rewrite, bool):
            raise ValueError("requires_rewrite must be a boolean")
        expected_sources = _normalize_expected_sources(record)
        source_match_mode = record.get("source_match_mode", "any")
        if (
            not isinstance(source_match_mode, str)
            or source_match_mode not in {"any", "all"}
        ):
            raise ValueError("source_match_mode must be 'any' or 'all'")
        normalized_source_names = [source.strip() for source in expected_sources]
        if source_match_mode == "all" and (
            len(normalized_source_names) < 2
            or any(not source for source in normalized_source_names)
            or len(set(normalized_source_names)) != len(normalized_source_names)
        ):
            raise ValueError(
                "source_match_mode 'all' requires at least two non-empty, "
                "unique expected_sources"
            )

        normalized = dict(record)
        normalized["expected_keywords"] = _normalize_string_list(
            record.get("expected_keywords"),
            field_name="expected_keywords",
        )
        normalized["expected_sources"] = expected_sources
        normalized["source_match_mode"] = source_match_mode
        normalized["should_answer"] = should_answer
        normalized["requires_rewrite"] = requires_rewrite
        questions.append(normalized)

    return questions


def evaluate_questions(
    questions: list[dict[str, Any]],
    run_agent_fn: Callable[[str], dict[str, Any]] = run_agent,
    run_naive_fn: Callable[[str], dict[str, Any]] | None = None,
    timer: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Evaluate questions and return per-question results plus summary metrics."""

    if run_naive_fn is not None:
        return _evaluate_comparison(
            questions=questions,
            run_agent_fn=run_agent_fn,
            run_naive_fn=run_naive_fn,
            timer=timer,
        )

    results: list[dict[str, Any]] = []
    for item in questions:
        results.append(_evaluate_single_system(item, run_agent_fn, timer))

    return {"summary": _summarize(results, questions), "results": results}


def format_report(report: dict[str, Any]) -> str:
    """Format an evaluation report for terminal output."""

    summary = report.get("summary", {})
    if summary.get("mode") == "comparison":
        return _format_comparison_report(report)

    lines = ["Evaluation Report", "", "Summary"]
    for key, value in summary.items():
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
    run_naive_fn: Callable[[str], dict[str, Any]] | None = run_naive_rag,
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

    report = evaluate_questions(
        questions,
        run_agent_fn=run_agent_fn,
        run_naive_fn=run_naive_fn,
    )
    print(format_report(report))
    return 0


def _evaluate_comparison(
    questions: list[dict[str, Any]],
    run_agent_fn: Callable[[str], dict[str, Any]],
    run_naive_fn: Callable[[str], dict[str, Any]],
    timer: Callable[[], float],
) -> dict[str, Any]:
    """Evaluate naive and agentic RAG on the same questions."""

    paired_results: list[dict[str, Any]] = []
    naive_results: list[dict[str, Any]] = []
    agentic_results: list[dict[str, Any]] = []

    for item in questions:
        naive_result = _evaluate_single_system(item, run_naive_fn, timer)
        agentic_result = _evaluate_single_system(item, run_agent_fn, timer)
        naive_results.append(naive_result)
        agentic_results.append(agentic_result)
        paired_results.append(
            {
                "question": item["question"],
                "requires_rewrite": item.get("requires_rewrite", False),
                "naive": naive_result,
                "agentic": agentic_result,
            }
        )

    naive_summary = _summarize(naive_results, questions)
    agentic_summary = _summarize(agentic_results, questions)
    return {
        "summary": {
            "mode": "comparison",
            "total_questions": len(questions),
            "naive": naive_summary,
            "agentic": agentic_summary,
            "comparison": _build_comparison_summary(
                naive_summary=naive_summary,
                agentic_summary=agentic_summary,
            ),
        },
        "results": paired_results,
    }


def _evaluate_single_system(
    item: dict[str, Any],
    runner: Callable[[str], dict[str, Any]],
    timer: Callable[[], float],
) -> dict[str, Any]:
    """Evaluate one system for one question and record errors as data."""

    question = item["question"]
    started_at = timer()
    try:
        system_result = runner(question)
        result = _build_success_result(item, system_result)
        error = None
    except Exception as exc:  # noqa: BLE001 - evaluation records system failures.
        result = _build_error_result(item)
        error = _format_error(exc)
    latency = timer() - started_at

    result["latency"] = latency
    result["error"] = error
    return result


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
    claims = _safe_list(agent_result.get("claims", []), "claims")
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
        "is_verified": bool(agent_result.get("is_verified", False)),
        "claim_count": len(claims),
        "source_hit": _has_expected_source(
            expected_sources,
            retrieved_documents,
            item.get("source_match_mode", "any"),
        ),
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
        "claims": claims,
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
        "is_verified": False,
        "claim_count": 0,
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
        "claims": [],
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
            "verification_rate": 0,
            "average_claim_count": 0,
            "average_retry_count": 0,
            "average_retrieved_docs": 0,
            "average_relevant_docs": 0,
            "relevant_filtering_rate": 0,
            "average_latency": 0,
            "rewrite_triggered_count": 0,
            "error_count": 0,
        }

    answer_count = sum(1 for result in results if result["answer_returned"])
    fallback_count = sum(1 for result in results if result["fallback_triggered"])
    citation_count = sum(1 for result in results if result["citation_returned"])
    verified_count = sum(1 for result in results if result["is_verified"])
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
    retrieved_doc_count = sum(result["retrieved_doc_count"] for result in results)
    relevant_doc_count = sum(result["relevant_doc_count"] for result in results)

    return {
        "total_questions": total_questions,
        "answer_rate": _rate(answer_count, total_questions),
        "fallback_rate": _rate(fallback_count, total_questions),
        "citation_rate": _rate(citation_count, total_questions),
        "verification_rate": _rate(verified_count, total_questions),
        "average_claim_count": _average(result["claim_count"] for result in results),
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
        "relevant_filtering_rate": _rate(
            retrieved_doc_count - relevant_doc_count,
            retrieved_doc_count,
        ),
        "average_latency": _average(result["latency"] for result in results),
        "rewrite_triggered_count": rewrite_triggered_count,
        "error_count": error_count,
    }


def _build_comparison_summary(
    naive_summary: dict[str, Any],
    agentic_summary: dict[str, Any],
) -> dict[str, Any]:
    """Flatten core comparison metrics for easy JSON/report consumption."""

    return {
        "naive_source_hit_rate": naive_summary.get("source_hit_rate", "N/A"),
        "agentic_source_hit_rate": agentic_summary.get("source_hit_rate", "N/A"),
        "naive_keyword_hit_rate": naive_summary.get("keyword_hit_rate", "N/A"),
        "agentic_keyword_hit_rate": agentic_summary.get("keyword_hit_rate", "N/A"),
        "naive_citation_rate": naive_summary.get("citation_rate", "N/A"),
        "agentic_citation_rate": agentic_summary.get("citation_rate", "N/A"),
        "naive_verification_rate": naive_summary.get("verification_rate", "N/A"),
        "agentic_verification_rate": agentic_summary.get("verification_rate", "N/A"),
        "naive_fallback_correctness_rate": naive_summary.get(
            "fallback_correctness_rate",
            "N/A",
        ),
        "agentic_fallback_correctness_rate": agentic_summary.get(
            "fallback_correctness_rate",
            "N/A",
        ),
        "naive_average_latency": naive_summary.get("average_latency", "N/A"),
        "agentic_average_latency": agentic_summary.get("average_latency", "N/A"),
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


def _format_comparison_report(report: dict[str, Any]) -> str:
    """Format a naive-vs-agentic report as readable markdown."""

    summary = report.get("summary", {})
    naive = summary.get("naive", {})
    agentic = summary.get("agentic", {})

    lines = [
        "Evaluation Report",
        "",
        "Comparison Summary",
        "",
        "| Metric | Naive RAG | Agentic RAG |",
        "|---|---:|---:|",
        (
            f"| Source Hit Rate | {naive.get('source_hit_rate', 'N/A')} | "
            f"{agentic.get('source_hit_rate', 'N/A')} |"
        ),
        (
            f"| Keyword Hit Rate | {naive.get('keyword_hit_rate', 'N/A')} | "
            f"{agentic.get('keyword_hit_rate', 'N/A')} |"
        ),
        (
            f"| Citation Rate | {naive.get('citation_rate', 'N/A')} | "
            f"{agentic.get('citation_rate', 'N/A')} |"
        ),
        (
            f"| Claim Verification Rate | {naive.get('verification_rate', 'N/A')} | "
            f"{agentic.get('verification_rate', 'N/A')} |"
        ),
        (
            f"| Fallback Correctness | "
            f"{naive.get('fallback_correctness_rate', 'N/A')} | "
            f"{agentic.get('fallback_correctness_rate', 'N/A')} |"
        ),
        (
            f"| Avg Latency | {naive.get('average_latency', 'N/A')} | "
            f"{agentic.get('average_latency', 'N/A')} |"
        ),
        "",
        "Agentic-specific Metrics",
        f"average_retry_count: {agentic.get('average_retry_count', 'N/A')}",
        f"rewrite_triggered_count: {agentic.get('rewrite_triggered_count', 'N/A')}",
        f"average_retrieved_docs: {agentic.get('average_retrieved_docs', 'N/A')}",
        f"average_relevant_docs: {agentic.get('average_relevant_docs', 'N/A')}",
        f"relevant_filtering_rate: {agentic.get('relevant_filtering_rate', 'N/A')}",
        f"verification_rate: {agentic.get('verification_rate', 'N/A')}",
        f"average_claim_count: {agentic.get('average_claim_count', 'N/A')}",
        "",
        "Questions",
    ]

    for index, result in enumerate(report.get("results", []), start=1):
        naive_result = result.get("naive", {})
        agentic_result = result.get("agentic", {})
        lines.append(
            (
                f"{index}. {result.get('question', '')} | "
                f"naive_answer={_format_bool(naive_result.get('answer_returned'))} | "
                f"agentic_answer={_format_bool(agentic_result.get('answer_returned'))} | "
                f"naive_source_hit={_format_bool(naive_result.get('source_hit'))} | "
                f"agentic_source_hit={_format_bool(agentic_result.get('source_hit'))} | "
                f"retry_count={agentic_result.get('retry_count', 0)} | "
                f"retrieved={agentic_result.get('retrieved_doc_count', 0)} | "
                f"relevant={agentic_result.get('relevant_doc_count', 0)} | "
                f"error={naive_result.get('error') or agentic_result.get('error') or ''}"
            )
        )

    return "\n".join(lines)


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
    retrieved_documents: list[Any],
    source_match_mode: str = "any",
) -> bool:
    if not expected_sources:
        return False

    expected_set = set(expected_sources)
    observed_set = {
        document["source"]
        for document in retrieved_documents
        if isinstance(document, dict)
        and isinstance(document.get("source"), str)
    }
    if source_match_mode == "all":
        return expected_set.issubset(observed_set)
    return bool(expected_set.intersection(observed_set))


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
        "provided documents do not contain enough information",
        "documents do not contain enough information",
        "do not contain enough information",
        "don't have enough evidence from the current documents",
        "do not have enough evidence from the current documents",
        "i cannot answer",
        "无法可靠回答",
        "无法根据当前文档回答",
        "当前文档无法回答",
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
