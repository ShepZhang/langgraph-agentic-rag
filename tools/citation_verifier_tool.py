"""Claim-level citation verification tool."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent.citation_verification import parse_citation_verification_response
from agent.prompts import format_documents
from prompting import render_prompt
from tools.base import BaseTool, ToolExecutionError, coerce_llm_text


class CitationVerifierArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    claims: list[dict[str, Any]] = Field(min_length=1)
    documents: list[dict[str, Any]] = Field(min_length=1)


class CitationVerifierTool(BaseTool[CitationVerifierArgs, dict[str, Any]]):
    name = "verify_citations"
    description = "Verify each answer claim against its cited document chunks."
    args_schema = CitationVerifierArgs
    trace_metadata_fields = frozenset({"claim_count", "unsupported_count"})

    def run(self, arguments: CitationVerifierArgs) -> dict[str, Any]:
        if self.context.llm is None:
            raise ToolExecutionError("Citation verifier requires an LLM.")

        valid_chunk_ids = []
        for document in arguments.documents:
            chunk_id = document.get("chunk_id")
            if chunk_id:
                valid_chunk_ids.append(str(chunk_id))

        prompt = render_prompt(
            "agent.citation_verification",
            question=arguments.question,
            answer=arguments.answer,
            claims=json.dumps(arguments.claims, ensure_ascii=False),
            documents=format_documents(arguments.documents),
        )
        raw_result = coerce_llm_text(self.context.llm.invoke(prompt))
        verification = parse_citation_verification_response(
            raw_result,
            valid_chunk_ids=valid_chunk_ids,
        )
        if verification is None:
            raise ToolExecutionError("Citation verification returned invalid JSON.")
        return verification

    def build_metadata(
        self,
        arguments: CitationVerifierArgs,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        records = result.get("results", [])
        if not isinstance(records, list):
            records = []

        unsupported_count = sum(
            record.get("verification_label") != "supported"
            or not record.get("cited_chunk_ids")
            for record in records
            if isinstance(record, dict)
        )
        return {
            "claim_count": len(records),
            "unsupported_count": unsupported_count,
        }


__all__ = ["CitationVerifierArgs", "CitationVerifierTool"]
