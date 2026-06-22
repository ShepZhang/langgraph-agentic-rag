"""Terminal report rendering for evaluation results."""

from __future__ import annotations

from typing import Any


def format_evaluation_report(report: dict[str, Any]) -> str:
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
            f"| Judge Completion Rate | "
            f"{_fmt_optional(naive.get('judge_completion_rate'))} | "
            f"{_fmt_optional(agentic.get('judge_completion_rate'))} |"
        ),
        (
            f"| Semantic Correctness | "
            f"{_fmt_optional(naive.get('average_semantic_correctness'))} | "
            f"{_fmt_optional(agentic.get('average_semantic_correctness'))} |"
        ),
        (
            f"| Groundedness | "
            f"{_fmt_optional(naive.get('average_groundedness'))} | "
            f"{_fmt_optional(agentic.get('average_groundedness'))} |"
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


def _fmt_optional(value: Any) -> str:
    """Format an optional numeric value: None → N/A, otherwise the value itself."""
    if value is None:
        return "N/A"
    return str(value)


def _format_bool(value: Any) -> str:
    return "true" if value else "false"
