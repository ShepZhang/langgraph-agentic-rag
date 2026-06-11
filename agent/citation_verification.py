"""Parsers for claim-level citation verification."""

from __future__ import annotations

import json
from typing import Any, Literal, TypedDict


VerificationLabel = Literal["supported", "partially_supported", "unsupported"]
VALID_VERIFICATION_LABELS = {
    "supported",
    "partially_supported",
    "unsupported",
}
DEFAULT_CLAIM_EXTRACTION_REASON = "No claim extraction reason provided."
DEFAULT_CITATION_VERIFICATION_REASON = "No citation verification reason provided."


class ExtractedClaim(TypedDict):
    """Atomic claim extracted from an answer."""

    claim_id: str
    claim: str
    cited_chunk_ids: list[str]


class ClaimExtractionResult(TypedDict):
    """Normalized claim extraction payload."""

    claims: list[ExtractedClaim]
    reason: str


class ClaimVerificationResult(TypedDict):
    """Normalized per-claim citation verification result."""

    claim_id: str
    claim: str
    cited_chunk_ids: list[str]
    verification_label: VerificationLabel
    confidence: float
    reason: str


class CitationVerificationResult(TypedDict):
    """Normalized citation verification payload."""

    results: list[ClaimVerificationResult]
    reason: str


def parse_claim_extraction_response(
    raw_result: str,
    valid_chunk_ids: list[str],
) -> ClaimExtractionResult | None:
    """Parse claim extraction JSON into validated claim records."""

    parsed = _extract_first_json_object(raw_result)
    if parsed is None:
        return None

    raw_claims = parsed.get("claims")
    if not isinstance(raw_claims, list):
        return None

    valid_ids = set(valid_chunk_ids)
    claims: list[ExtractedClaim] = []
    for raw_claim in raw_claims:
        if not isinstance(raw_claim, dict):
            continue
        claim_id = _normalize_text(raw_claim.get("claim_id"))
        claim = _normalize_text(raw_claim.get("claim"))
        if not claim_id or not claim:
            continue
        claims.append(
            {
                "claim_id": claim_id,
                "claim": claim,
                "cited_chunk_ids": _filter_chunk_ids(
                    raw_claim.get("cited_chunk_ids"),
                    valid_ids,
                ),
            }
        )

    return {
        "claims": claims,
        "reason": _normalize_reason(
            parsed.get("reason"),
            default=DEFAULT_CLAIM_EXTRACTION_REASON,
        ),
    }


def parse_citation_verification_response(
    raw_result: str,
    valid_chunk_ids: list[str],
) -> CitationVerificationResult | None:
    """Parse per-claim citation verification JSON."""

    parsed = _extract_first_json_object(raw_result)
    if parsed is None:
        return None

    raw_results = parsed.get("results")
    if not isinstance(raw_results, list):
        return None

    valid_ids = set(valid_chunk_ids)
    results: list[ClaimVerificationResult] = []
    for raw_result_record in raw_results:
        if not isinstance(raw_result_record, dict):
            continue
        claim_id = _normalize_text(raw_result_record.get("claim_id"))
        claim = _normalize_text(raw_result_record.get("claim"))
        if not claim_id or not claim:
            continue
        results.append(
            {
                "claim_id": claim_id,
                "claim": claim,
                "cited_chunk_ids": _filter_chunk_ids(
                    raw_result_record.get("cited_chunk_ids"),
                    valid_ids,
                ),
                "verification_label": _normalize_verification_label(
                    raw_result_record.get("verification_label"),
                ),
                "confidence": _clamp_confidence(
                    raw_result_record.get("confidence"),
                ),
                "reason": _normalize_reason(
                    raw_result_record.get("reason"),
                    default=DEFAULT_CITATION_VERIFICATION_REASON,
                ),
            }
        )

    return {
        "results": results,
        "reason": _normalize_reason(
            parsed.get("reason"),
            default=DEFAULT_CITATION_VERIFICATION_REASON,
        ),
    }


def build_claim_verification_summary(
    results: list[ClaimVerificationResult],
    reason: str,
) -> dict[str, object]:
    """Build a compatibility summary for claim verification."""

    unsupported_claims = [
        result
        for result in results
        if result["verification_label"] != "supported" or not result["cited_chunk_ids"]
    ]
    verified = bool(results) and not unsupported_claims
    normalized_reason = reason.strip() if isinstance(reason, str) else ""
    if not normalized_reason:
        normalized_reason = (
            "All claims supported."
            if verified
            else "One or more claims are unsupported."
        )
    return {
        "verified": verified,
        "results": results,
        "reason": normalized_reason,
        "unsupported_claims": unsupported_claims,
    }


def _extract_first_json_object(raw_result: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(raw_result):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw_result[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _filter_chunk_ids(raw_value: Any, valid_chunk_ids: set[str]) -> list[str]:
    if not isinstance(raw_value, list):
        return []
    chunk_ids: list[str] = []
    for raw_chunk_id in raw_value:
        chunk_id = _normalize_text(raw_chunk_id)
        if not chunk_id or chunk_id not in valid_chunk_ids:
            continue
        if chunk_id not in chunk_ids:
            chunk_ids.append(chunk_id)
    return chunk_ids


def _normalize_verification_label(value: Any) -> VerificationLabel:
    if not isinstance(value, str):
        return "unsupported"
    normalized = value.strip().lower()
    if normalized in VALID_VERIFICATION_LABELS:
        return normalized  # type: ignore[return-value]
    return "unsupported"


def _clamp_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _normalize_reason(value: Any, *, default: str) -> str:
    normalized = _normalize_text(value)
    return normalized or default


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()
