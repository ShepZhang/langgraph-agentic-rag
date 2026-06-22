"""Bounded formatting helpers for semantic judge evidence."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping

from evaluation.schemas import EvaluationResult

MAX_JUDGE_EVIDENCE_CHUNKS = 8
MAX_JUDGE_EVIDENCE_CHARS = 1200


def select_judge_evidence(result: EvaluationResult) -> list[dict[str, Any]]:
    """Return bounded evidence records, preferring relevant documents."""

    documents = result.relevant_documents or result.retrieved_documents
    return deepcopy(list(documents[:MAX_JUDGE_EVIDENCE_CHUNKS]))


def format_judge_evidence(result: EvaluationResult) -> str:
    """Serialize bounded judge evidence as compact deterministic JSON."""

    evidence = [_format_evidence_record(record) for record in select_judge_evidence(result)]
    return json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))


def format_judge_citations(result: EvaluationResult) -> str:
    """Serialize bounded judge citations as compact deterministic JSON."""

    citations = [
        _format_citation_record(record)
        for record in deepcopy(result.citations[:MAX_JUDGE_EVIDENCE_CHUNKS])
    ]
    return json.dumps(citations, ensure_ascii=False, separators=(",", ":"))


def _format_evidence_record(record: Mapping[str, Any]) -> dict[str, Any]:
    formatted: dict[str, Any] = {}

    source = _select_source(record)
    if source:
        formatted["source"] = source

    page = record.get("page")
    if isinstance(page, int) and not isinstance(page, bool):
        formatted["page"] = page

    chunk_id = _clean_chunk_id(record.get("chunk_id"))
    if chunk_id:
        formatted["chunk_id"] = chunk_id

    formatted["content"] = _normalize_text(record.get("content"))
    return formatted


def _format_citation_record(record: Mapping[str, Any]) -> dict[str, Any]:
    formatted: dict[str, Any] = {}

    source = _select_source(record)
    if source:
        formatted["source"] = source

    page = record.get("page")
    if isinstance(page, int) and not isinstance(page, bool):
        formatted["page"] = page

    chunk_id = _clean_chunk_id(record.get("chunk_id"))
    if chunk_id:
        formatted["chunk_id"] = chunk_id

    formatted["snippet"] = _normalize_text(record.get("snippet"))
    return formatted


def _select_source(record: Mapping[str, Any]) -> str:
    for key in ("source", "source_path", "file_path"):
        candidate = _basename_value(record.get(key))
        if candidate:
            return candidate
    return ""


def _basename_value(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()
    if not cleaned:
        return ""
    normalized = cleaned.replace("\\", "/").rstrip("/")
    if not normalized:
        return ""
    return normalized.rsplit("/", maxsplit=1)[-1]


def _clean_chunk_id(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = " ".join(value.split())
    return normalized[:MAX_JUDGE_EVIDENCE_CHARS]
