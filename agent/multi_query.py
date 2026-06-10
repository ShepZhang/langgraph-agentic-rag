"""Helpers for executing and merging multi-query retrieval results."""

from __future__ import annotations

from typing import Any

from agent.state import RetrievedDocument


def build_retrieval_queries(
    current_query: str,
    expanded_queries: list[str],
    strategy: str,
) -> list[str]:
    """Build ordered retrieval queries for the current transform strategy."""

    if strategy != "multi_query":
        return _dedupe_queries([current_query])
    return _dedupe_queries([current_query, *expanded_queries])


def merge_retrieved_documents(
    query_results: list[tuple[str, list[RetrievedDocument]]],
) -> list[RetrievedDocument]:
    """Merge retrieved documents from multiple queries with deterministic de-dupe."""

    merged: list[RetrievedDocument] = []
    seen: dict[str, RetrievedDocument] = {}
    for query, documents in query_results:
        query_text = query.strip()
        if not query_text:
            continue
        for document in documents:
            document_key = _document_key(document)
            if document_key in seen:
                matched_queries = seen[document_key].setdefault("matched_queries", [])
                if isinstance(matched_queries, list) and query_text not in matched_queries:
                    matched_queries.append(query_text)
                continue

            merged_document: RetrievedDocument = dict(document)
            merged_document["matched_queries"] = [query_text]  # type: ignore[typeddict-unknown-key]
            seen[document_key] = merged_document
            merged.append(merged_document)

    total_query_count = len([query for query, _documents in query_results if query.strip()])
    for rank, document in enumerate(merged, start=1):
        document["retrieval_query_count"] = total_query_count  # type: ignore[typeddict-unknown-key]
        document["multi_query_rank"] = rank  # type: ignore[typeddict-unknown-key]
    return merged


def _dedupe_queries(queries: list[str]) -> list[str]:
    deduped: list[str] = []
    for query in queries:
        query_text = query.strip()
        if query_text and query_text not in deduped:
            deduped.append(query_text)
    return deduped


def _document_key(document: dict[str, Any]) -> str:
    chunk_id = document.get("chunk_id")
    if chunk_id:
        return f"chunk:{chunk_id}"
    return (
        f"fallback:{document.get('source', '')}:"
        f"{document.get('page', '')}:"
        f"{document.get('content', '')}"
    )
