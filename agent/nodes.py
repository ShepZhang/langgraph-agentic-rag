"""LangGraph node implementations for Agentic RAG."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import BaseMessage

from agent.features import AgentFeatureFlags
from agent.multi_query import build_retrieval_queries, merge_retrieved_documents
from agent.prompts import (
    ANSWER_GENERATION_PROMPT,
    ANSWER_REVISION_PROMPT,
    CLAIM_EXTRACTION_PROMPT,
    RETRY_QUERY_REWRITE_PROMPT,
    RETRIEVAL_GRADING_PROMPT,
    format_documents,
)
from agent.citation_verification import (
    build_claim_verification_summary,
    parse_claim_extraction_response,
)
from agent.query_transform import (
    build_query_transform_prompt,
    fallback_query_transform,
    parse_query_transform_response,
)
from agent.retrieval_grading import parse_retrieval_grading_response
from agent.state import AgentState, Citation, RetrievedDocument
from tools import ToolRegistry, create_default_tool_registry


FALLBACK_ANSWER = "根据当前已索引文档，无法可靠回答这个问题。请补充相关文档，或换一种更具体的问法。"
PARTIAL_RELEVANCE_RECOVERY_REASON = (
    "Only partially relevant chunks were found; "
    "refine query to target missing evidence."
)
logger = logging.getLogger(__name__)


class AgentNodes:
    """State transition nodes for the Agentic RAG graph."""

    def __init__(
        self,
        llm: Any,
        retriever_fn: Any | None = None,
        features: AgentFeatureFlags | None = None,
        workspace_id: str | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.llm = llm
        self.features = features or AgentFeatureFlags()
        self.tool_registry = (
            tool_registry
            if tool_registry is not None
            else create_default_tool_registry(
                llm=llm,
                retriever_fn=retriever_fn,
                workspace_id=workspace_id,
            )
        )

    def rewrite_query_node(self, state: AgentState) -> dict[str, Any]:
        """Normalize the first query or rewrite after failed retrieval."""

        is_retry = state.get("retrieval_attempt", 0) > 0
        if is_retry:
            prompt = RETRY_QUERY_REWRITE_PROMPT.format(
                question=state["question"],
                current_query=state.get("current_query") or state["question"],
                previous_queries=_format_previous_queries(
                    state.get("previous_queries", [])
                ),
                grading_reason=state.get("grading_reason") or "No grading reason.",
                partial_relevance_context=_format_partial_relevance_context(state),
                documents=format_documents(state.get("documents", [])),
            )
            raw_result = _coerce_llm_text(self.llm.invoke(prompt))
            rewritten_question = raw_result.strip()
            if not rewritten_question:
                rewritten_question = state.get("current_query") or state["question"]
            query_transform = fallback_query_transform(
                rewritten_question,
                reason="Retry rewrite after failed retrieval.",
            )
        else:
            prompt = build_query_transform_prompt(
                question=state["question"],
                chat_history=state.get("chat_history", []),
            )
            raw_result = _coerce_llm_text(self.llm.invoke(prompt))
            query_transform = parse_query_transform_response(
                raw_result,
                original_question=state["question"],
            )
            rewritten_question = query_transform["rewritten_query"].strip()
            if not rewritten_question:
                rewritten_question = state.get("current_query") or state["question"]
                query_transform = fallback_query_transform(
                    rewritten_question,
                    reason="Blank rewritten query; using fallback question.",
                )

        previous_queries = list(state.get("previous_queries", []))
        if is_retry and rewritten_question in previous_queries:
            logger.warning("Retry rewrite repeated a previous query: %s", rewritten_question)
        if not previous_queries or previous_queries[-1] != rewritten_question:
            previous_queries.append(rewritten_question)

        retry_count = state.get("retry_count", 0) + 1 if is_retry else state.get(
            "retry_count",
            0,
        )
        logger.info(
            "Prepared retrieval query retry=%s retry_count=%s query=%s",
            is_retry,
            retry_count,
            rewritten_question,
        )

        return {
            "current_query": rewritten_question,
            "rewritten_question": rewritten_question,
            "standalone_question": rewritten_question,
            "query_transform": query_transform,
            "query_transform_strategy": query_transform["strategy"],
            "query_transform_reason": query_transform["reason"],
            "expanded_queries": query_transform["expanded_queries"],
            "sub_questions": query_transform["sub_questions"],
            "previous_queries": previous_queries,
            "retry_count": retry_count,
            "rewrite_count": retry_count,
            "route": "retrieve",
        }

    def retrieve_node(self, state: AgentState) -> dict[str, Any]:
        """Retrieve relevant chunks with the retriever tool."""

        query = state.get("current_query") or state.get("rewritten_question") or state[
            "question"
        ]
        retrieval_queries = build_retrieval_queries(
            current_query=query,
            expanded_queries=state.get("expanded_queries", []),
            strategy=state.get("query_transform_strategy", "rewrite"),
        )
        query_results: list[tuple[str, list[RetrievedDocument]]] = []
        tool_errors: list[str] = []
        for retrieval_query in retrieval_queries:
            result = self.tool_registry.invoke(
                "retrieve_context",
                {"query": retrieval_query},
            )
            if result.success:
                if not _is_valid_retriever_tool_data(result.data):
                    tool_errors.append("Retriever tool returned invalid data: expected list[dict].")
                    continue
                query_results.append((retrieval_query, result.data or []))
                continue
            tool_errors.append(
                (
                    result.error.message
                    if result.error is not None and result.error.message
                    else "Unknown retrieval failure"
                )
            )
        documents = merge_retrieved_documents(query_results)
        retrieval_attempt = state.get("retrieval_attempt", 0) + 1
        multi_query_used = len(retrieval_queries) > 1
        logger.info(
            "Retrieved documents count=%s attempt=%s query_count=%s query=%s",
            len(documents),
            retrieval_attempt,
            len(retrieval_queries),
            query,
        )
        update = {
            "documents": documents,
            "retrieval_queries": retrieval_queries,
            "multi_query_used": multi_query_used,
            "multi_query_result_count": len(documents),
            "retrieval_attempt": retrieval_attempt,
        }
        if not documents and tool_errors:
            update["grading_reason"] = f"Retriever tool failed: {'; '.join(tool_errors)}"
        return update

    def accept_retrieved_documents_node(self, state: AgentState) -> dict[str, Any]:
        """Use retrieved documents directly when retrieval grading is disabled."""

        documents = state.get("documents", [])
        if not documents:
            return _fallback_update(
                state.get("grading_reason")
                or "No documents retrieved while retrieval grading was disabled."
            )

        return {
            "is_relevant": True,
            "relevant_documents": documents,
            "document_grades": [],
            "relevant_document_count": len(documents),
            "partial_document_count": 0,
            "max_relevance_confidence": 0.0,
            "partial_relevance_recovery": _inactive_partial_relevance_recovery(),
            "grading_reason": "Retrieval grading disabled.",
            "route": "generate_answer",
        }

    def grade_documents_node(self, state: AgentState) -> dict[str, Any]:
        """Grade whether retrieved chunks are relevant enough to answer."""

        documents = state.get("documents", [])
        if not documents:
            reason = state.get("grading_reason") or "No documents retrieved."
            logger.info("Retrieval grading skipped: %s", reason)
            return {
                "is_relevant": False,
                "relevant_documents": [],
                "document_grades": [],
                "relevant_document_count": 0,
                "partial_document_count": 0,
                "max_relevance_confidence": 0.0,
                "partial_relevance_recovery": _inactive_partial_relevance_recovery(),
                "grading_reason": reason,
                "route": "rewrite_query",
            }

        prompt = RETRIEVAL_GRADING_PROMPT.format(
            question=state["question"],
            current_query=state.get("current_query") or state["question"],
            documents=format_documents(documents),
        )
        raw_result = _coerce_llm_text(self.llm.invoke(prompt))
        grading = parse_retrieval_grading_response(
            raw_result,
            document_count=len(documents),
        )
        relevant_documents = [
            documents[index - 1] for index in grading["relevant_indices"]
        ]
        is_relevant = bool(relevant_documents)
        partial_document_count = len(grading["partially_relevant_indices"])
        max_relevance_confidence = _max_grade_confidence(grading["grades"])
        partial_relevance_recovery = _build_partial_relevance_recovery(
            is_relevant=is_relevant,
            partial_document_indices=grading["partially_relevant_indices"],
        )
        logger.info(
            "Retrieval grading relevant=%s relevant_count=%s partial_count=%s reason=%s",
            is_relevant,
            len(relevant_documents),
            partial_document_count,
            grading["reason"],
        )
        return {
            "is_relevant": is_relevant,
            "relevant_documents": relevant_documents,
            "document_grades": grading["grades"],
            "relevant_document_count": len(relevant_documents),
            "partial_document_count": partial_document_count,
            "max_relevance_confidence": max_relevance_confidence,
            "partial_relevance_recovery": partial_relevance_recovery,
            "grading_reason": grading["reason"],
            "route": "generate_answer" if is_relevant else "rewrite_query",
        }

    def generate_answer_node(self, state: AgentState) -> dict[str, Any]:
        """Generate a grounded answer and citations."""

        documents = state.get("relevant_documents", [])
        if not documents:
            reason = "No relevant documents available for answer generation."
            logger.info("Answer generation skipped: %s", reason)
            return _fallback_update(reason)

        prompt = ANSWER_GENERATION_PROMPT.format(
            question=state["question"],
            current_query=state.get("current_query") or state["question"],
            documents=format_documents(documents),
        )
        raw_result = _coerce_llm_text(self.llm.invoke(prompt))
        parsed_answer = _parse_answer_result(raw_result, document_count=len(documents))
        if parsed_answer is None:
            logger.warning("Answer generation returned invalid JSON.")
            return _fallback_update("Answer generation returned invalid JSON.")

        answer = parsed_answer["answer"].strip()
        if not answer:
            return _fallback_update("Answer generation returned an empty answer.")

        marker_error = _validate_answer_citation_markers(
            answer=answer,
            used_citation_indices=parsed_answer["used_citation_indices"],
            document_count=len(documents),
        )
        if marker_error:
            logger.warning("Answer citation marker validation failed: %s", marker_error)
            return _fallback_update(marker_error)

        citations = build_citations(
            documents,
            used_citation_indices=parsed_answer["used_citation_indices"],
        )
        cited_documents = _select_documents_by_indices(
            documents,
            parsed_answer["used_citation_indices"],
        )
        if not citations and not is_unable_to_answer(answer):
            logger.warning("Answer generation returned a normal answer without citations.")
            return _fallback_update(
                "Answer generation did not return valid supporting citations."
            )
        if is_unable_to_answer(answer) and not citations:
            logger.info("Generated unable-to-answer draft; verification skipped.")
            return {
                "answer": "",
                "draft_answer": answer,
                "citations": [],
                "used_citation_indices": [],
                "cited_documents": [],
                "claims": [],
                "claim_verification": {
                    "verified": False,
                    "results": [],
                    "reason": "Unable-to-answer response; claim verification skipped.",
                    "unsupported_claims": [],
                },
                "claim_verification_results": [],
                "unsupported_claims": [],
                "claim_verification_reason": (
                    "Unable-to-answer response; claim verification skipped."
                ),
                "citation_verification_passed": False,
                "citation_verification_skipped": True,
                "is_verified": False,
                "route": "finalize_answer",
            }
        logger.info(
            "Generated draft answer citation_count=%s used_indices=%s",
            len(citations),
            parsed_answer["used_citation_indices"],
        )
        if not self.features.citation_verification_enabled:
            reason = "Citation verification disabled by workflow configuration."
            return {
                "answer": "",
                "draft_answer": answer,
                "citations": citations,
                "used_citation_indices": parsed_answer["used_citation_indices"],
                "cited_documents": cited_documents,
                "claims": [],
                "claim_verification": {
                    "verified": False,
                    "results": [],
                    "reason": reason,
                    "unsupported_claims": [],
                },
                "claim_verification_results": [],
                "unsupported_claims": [],
                "claim_verification_reason": reason,
                "citation_verification_passed": False,
                "citation_verification_skipped": True,
                "is_verified": False,
                "route": "finalize_answer",
            }

        return {
            "answer": "",
            "draft_answer": answer,
            "citations": citations,
            "used_citation_indices": parsed_answer["used_citation_indices"],
            "cited_documents": cited_documents,
            "claims": [],
            "claim_verification": {},
            "claim_verification_results": [],
            "unsupported_claims": [],
            "claim_verification_reason": "",
            "citation_verification_passed": False,
            "citation_verification_skipped": False,
            "is_verified": False,
            "route": "extract_claims",
        }

    def extract_claims_node(self, state: AgentState) -> dict[str, Any]:
        """Extract atomic claims from the draft answer."""

        if state.get("citation_verification_skipped"):
            return {
                "claims": [],
                "claim_verification_reason": (
                    "Unable-to-answer response; claim extraction skipped."
                ),
                "route": "finalize_answer",
            }

        draft_answer = state.get("draft_answer", "").strip()
        if not draft_answer:
            return _fallback_update("Claim extraction skipped because draft answer is empty.")

        cited_documents = state.get("cited_documents", [])
        valid_chunk_ids = _selected_citation_chunk_ids(cited_documents)
        prompt = CLAIM_EXTRACTION_PROMPT.format(
            question=state["question"],
            answer=draft_answer,
            documents=format_documents(cited_documents),
        )
        raw_result = _coerce_llm_text(self.llm.invoke(prompt))
        extraction = parse_claim_extraction_response(
            raw_result,
            valid_chunk_ids=valid_chunk_ids,
        )
        if extraction is None:
            logger.warning("Claim extraction returned invalid JSON.")
            return _fallback_update("Claim extraction returned invalid JSON.")

        return {
            "claims": extraction["claims"],
            "claim_verification_reason": extraction["reason"],
            "route": "verify_citations",
        }

    def verify_citations_node(self, state: AgentState) -> dict[str, Any]:
        """Verify extracted claims against selected citation chunks."""

        if state.get("citation_verification_skipped"):
            return {
                "citation_verification_passed": False,
                "route": "finalize_answer",
            }

        claims = state.get("claims", [])
        if not claims:
            return _fallback_update("Claim extraction returned no verifiable claims.")

        cited_documents = state.get("cited_documents", [])
        result = self.tool_registry.invoke(
            "verify_citations",
            {
                "question": state["question"],
                "answer": state.get("draft_answer", ""),
                "claims": claims,
                "documents": cited_documents,
            },
        )
        if not result.success:
            message = (
                result.error.message
                if result.error is not None and result.error.message
                else "Unknown citation verification failure"
            )
            return {
                **_fallback_update(f"Citation verification tool failed: {message}"),
                "citation_verification_passed": False,
            }

        verification = result.data
        if not _is_valid_citation_verification_tool_data(verification):
            logger.warning("Citation verification tool returned invalid data.")
            return {
                **_fallback_update("Citation verification tool returned invalid data."),
                "citation_verification_passed": False,
            }

        summary = build_claim_verification_summary(
            verification["results"],
            reason=verification["reason"],
        )
        passed = summary["verified"] is True
        unsupported_claims = summary["unsupported_claims"]
        if passed:
            route = "finalize_answer"
        elif state.get("citation_revision_count", 0) < state.get(
            "max_citation_revision_count",
            1,
        ):
            route = "revise_answer"
        else:
            route = "fallback"

        return {
            "claim_verification_results": verification["results"],
            "unsupported_claims": unsupported_claims,
            "claim_verification": summary,
            "claim_verification_reason": summary["reason"],
            "citation_verification_passed": passed,
            "is_verified": passed,
            "route": route,
        }

    def revise_answer_node(self, state: AgentState) -> dict[str, Any]:
        """Revise unsupported draft-answer claims using verifier feedback."""

        cited_documents = state.get("cited_documents", [])
        prompt = ANSWER_REVISION_PROMPT.format(
            question=state["question"],
            answer=state.get("draft_answer", ""),
            unsupported_claims=json.dumps(
                state.get("unsupported_claims", []),
                ensure_ascii=False,
            ),
            documents=format_documents(cited_documents),
        )
        raw_result = _coerce_llm_text(self.llm.invoke(prompt))
        parsed_answer = _parse_answer_result(
            raw_result,
            document_count=len(cited_documents),
        )
        if parsed_answer is None:
            logger.warning("Answer revision returned invalid JSON.")
            return _fallback_update("Answer revision returned invalid JSON.")

        answer = parsed_answer["answer"].strip()
        if not answer:
            return _fallback_update("Answer revision returned an empty answer.")

        marker_error = _validate_answer_citation_markers(
            answer=answer,
            used_citation_indices=parsed_answer["used_citation_indices"],
            document_count=len(cited_documents),
        )
        if marker_error:
            logger.warning(
                "Answer revision citation marker validation failed: %s",
                marker_error,
            )
            return _fallback_update(marker_error)

        citations = build_citations(
            cited_documents,
            used_citation_indices=parsed_answer["used_citation_indices"],
        )
        selected_documents = _select_documents_by_indices(
            cited_documents,
            parsed_answer["used_citation_indices"],
        )
        revision_count = state.get("citation_revision_count", 0) + 1
        if is_unable_to_answer(answer) and not citations:
            return {
                "answer": "",
                "draft_answer": answer,
                "citations": [],
                "used_citation_indices": [],
                "cited_documents": [],
                "claims": [],
                "claim_verification_results": [],
                "unsupported_claims": [],
                "citation_revision_count": revision_count,
                "citation_verification_passed": False,
                "citation_verification_skipped": True,
                "is_verified": False,
                "route": "finalize_answer",
            }
        if not citations:
            return _fallback_update(
                "Answer revision did not return valid supporting citations."
            )

        return {
            "answer": "",
            "draft_answer": answer,
            "citations": citations,
            "used_citation_indices": parsed_answer["used_citation_indices"],
            "cited_documents": selected_documents,
            "claims": [],
            "claim_verification": {},
            "claim_verification_results": [],
            "unsupported_claims": [],
            "claim_verification_reason": "",
            "citation_revision_count": revision_count,
            "citation_verification_passed": False,
            "citation_verification_skipped": False,
            "is_verified": False,
            "route": "extract_claims",
        }

    def finalize_answer_node(self, state: AgentState) -> dict[str, Any]:
        """Promote a verified or safely skipped draft answer to final answer."""

        draft_answer = state.get("draft_answer", "").strip()
        if not draft_answer:
            return _fallback_update("Finalization skipped because draft answer is empty.")
        if not state.get("citation_verification_passed") and not state.get(
            "citation_verification_skipped"
        ):
            return _fallback_update("Citation verification did not pass.")

        verified = state.get("citation_verification_passed", False)
        return {
            "answer": draft_answer,
            "citations": state.get("citations", []),
            "claims": state.get("claims", []),
            "claim_verification": state.get("claim_verification", {}),
            "claim_verification_results": state.get("claim_verification_results", []),
            "unsupported_claims": state.get("unsupported_claims", []),
            "claim_verification_reason": state.get("claim_verification_reason", ""),
            "citation_verification_passed": verified,
            "citation_verification_skipped": state.get(
                "citation_verification_skipped",
                False,
            ),
            "is_verified": verified,
            "route": "end",
        }

    def fallback_node(self, state: AgentState) -> dict[str, Any]:
        """Return a safe fallback answer when retrieval is insufficient."""

        reason = (
            state.get("fallback_reason")
            or state.get("claim_verification_reason")
            or state.get("grading_reason")
            or "No reliable supporting evidence found."
        )
        logger.info("Fallback answer returned: %s", reason)
        return _fallback_update(reason)


def build_citations(
    documents: list[RetrievedDocument],
    used_citation_indices: list[int] | None = None,
) -> list[Citation]:
    """Build citations from selected document metadata."""

    citations: list[Citation] = []
    seen: set[tuple[Any, Any, Any]] = set()
    if used_citation_indices is None:
        selected_documents = documents
    else:
        selected_documents = []
        for citation_index in used_citation_indices:
            if citation_index < 1 or citation_index > len(documents):
                logger.warning("Ignoring out-of-range citation index: %s", citation_index)
                continue
            selected_documents.append(documents[citation_index - 1])

    for document in selected_documents:
        citation: Citation = {
            "source": document.get("source"),
            "page": document.get("page"),
            "chunk_id": document.get("chunk_id"),
            "score": document.get("score"),
            "snippet": _make_snippet(document.get("content", "")),
        }
        key = (
            citation.get("source"),
            citation.get("page"),
            citation.get("chunk_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        citations.append(citation)
    return citations


def _select_documents_by_indices(
    documents: list[RetrievedDocument],
    used_citation_indices: list[int],
) -> list[RetrievedDocument]:
    """Return valid cited documents in citation order, deduplicated by metadata."""

    selected_documents: list[RetrievedDocument] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for citation_index in used_citation_indices:
        if citation_index < 1 or citation_index > len(documents):
            continue
        document = documents[citation_index - 1]
        key = (document.get("source"), document.get("page"), document.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        selected_documents.append(document)
    return selected_documents


def is_unable_to_answer(answer: str) -> bool:
    """Return True when an answer explicitly declines due to missing evidence."""

    normalized = " ".join(answer.lower().split())
    markers = [
        "cannot answer from the current documents",
        "cannot answer based on the current documents",
        "provided documents do not contain enough information",
        "do not contain enough information",
        "don't have enough evidence from the current documents",
        "do not have enough evidence from the current documents",
        "i cannot answer",
        "无法根据当前文档回答",
        "无法可靠回答",
        "当前文档无法回答",
    ]
    return any(marker in normalized for marker in markers)


def _validate_answer_citation_markers(
    answer: str,
    used_citation_indices: list[int],
    document_count: int,
) -> str:
    """Return an error message when answer markers and used indices disagree."""

    if is_unable_to_answer(answer):
        return ""

    citation_markers = _extract_answer_citation_markers(answer)
    if not citation_markers:
        return "Answer citation markers are missing for a normal answer."

    invalid_markers = [
        marker for marker in citation_markers if marker < 1 or marker > document_count
    ]
    if invalid_markers:
        return (
            "Answer citation markers reference out-of-range chunks: "
            f"{sorted(set(invalid_markers))}."
        )

    marker_set = set(citation_markers)
    used_index_set = set(used_citation_indices)
    if marker_set != used_index_set:
        return (
            "Answer citation markers do not match used_citation_indices: "
            f"markers={sorted(marker_set)}, used={sorted(used_index_set)}."
        )

    return ""


def _extract_answer_citation_markers(answer: str) -> list[int]:
    """Extract numeric citation markers like [1] and [2] from answer text."""

    markers: list[int] = []
    for raw_marker in re.findall(r"\[(\d+)\]", answer):
        marker = int(raw_marker)
        if marker not in markers:
            markers.append(marker)
    return markers


def _parse_grading_result(
    raw_result: str,
    document_count: int,
) -> dict[str, Any]:
    """Parse a chunk-level relevance grading JSON response."""

    parsed = _extract_first_json_object(raw_result)
    if parsed is None:
        logger.warning("Could not parse retrieval grading JSON: %s", raw_result)
        return {
            "relevant": False,
            "relevant_indices": [],
            "reason": "Could not parse retrieval grading JSON.",
        }

    reason = str(parsed.get("reason") or "").strip()
    raw_indices = parsed.get("relevant_indices", [])
    if not isinstance(raw_indices, list):
        raw_indices = []

    relevant_indices: list[int] = []
    for raw_index in raw_indices:
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            continue
        if raw_index < 1 or raw_index > document_count:
            logger.warning("Ignoring out-of-range relevant chunk index: %s", raw_index)
            continue
        if raw_index not in relevant_indices:
            relevant_indices.append(raw_index)

    relevant = parsed.get("relevant") is True and bool(relevant_indices)
    if not reason:
        reason = "Relevant chunks found." if relevant else "No relevant chunks found."
    if parsed.get("relevant") is True and not relevant_indices:
        reason = (
            reason
            if reason
            else "Model marked retrieval relevant but did not provide valid indices."
        )

    return {
        "relevant": relevant,
        "relevant_indices": relevant_indices if relevant else [],
        "reason": reason,
    }


def _parse_answer_result(
    raw_result: str,
    document_count: int,
) -> dict[str, Any] | None:
    """Parse answer generation JSON and validate citation indices."""

    parsed = _extract_first_json_object(raw_result)
    if parsed is None:
        return None

    answer = parsed.get("answer")
    raw_indices = parsed.get("used_citation_indices", [])
    if not isinstance(answer, str) or not isinstance(raw_indices, list):
        return None

    used_citation_indices: list[int] = []
    for raw_index in raw_indices:
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            continue
        if raw_index < 1 or raw_index > document_count:
            logger.warning("Ignoring out-of-range answer citation index: %s", raw_index)
            continue
        if raw_index not in used_citation_indices:
            used_citation_indices.append(raw_index)

    return {
        "answer": answer,
        "used_citation_indices": used_citation_indices,
    }


def _is_valid_retriever_tool_data(data: Any) -> bool:
    """Return True when retriever tool data matches list[dict]."""

    if not isinstance(data, list):
        return False
    return all(isinstance(document, dict) for document in data)


def _is_valid_citation_verification_tool_data(data: Any) -> bool:
    """Return True when verifier tool data provides the required summary fields."""

    if not isinstance(data, dict):
        return False
    if not isinstance(data.get("reason"), str):
        return False

    results = data.get("results")
    if not isinstance(results, list):
        return False

    for result in results:
        if not isinstance(result, dict):
            return False
        if not isinstance(result.get("claim_id"), str):
            return False
        if not isinstance(result.get("claim"), str):
            return False
        cited_chunk_ids = result.get("cited_chunk_ids")
        if not isinstance(cited_chunk_ids, list) or not all(
            isinstance(chunk_id, str) for chunk_id in cited_chunk_ids
        ):
            return False
        if not isinstance(result.get("verification_label"), str):
            return False
        confidence = result.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return False
        if not isinstance(result.get("reason"), str):
            return False

    return True


def _extract_first_json_object(raw_result: str) -> dict[str, Any] | None:
    """Extract the first JSON object from an LLM response."""

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


def _coerce_content_text(content: Any) -> str:
    """Extract text from LangChain message content."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _coerce_llm_text(response: Any) -> str:
    """Convert LangChain or fake LLM responses into text."""

    if isinstance(response, str):
        return response
    if isinstance(response, BaseMessage):
        return _coerce_content_text(response.content)
    content = getattr(response, "content", None)
    if content is not None:
        return _coerce_content_text(content)
    return str(response)


def _fallback_update(reason: str) -> dict[str, Any]:
    """Build a fallback node update with a consistent reason."""

    return {
        "answer": FALLBACK_ANSWER,
        "citations": [],
        "claims": [],
        "claim_verification": {},
        "claim_verification_reason": reason,
        "is_verified": False,
        "is_relevant": False,
        "route": "fallback",
        "fallback_reason": reason,
    }


def _max_grade_confidence(grades: list[dict[str, Any]]) -> float:
    """Return the highest normalized grade confidence."""

    if not grades:
        return 0.0
    return max(float(grade.get("confidence", 0.0)) for grade in grades)


def _build_partial_relevance_recovery(
    *,
    is_relevant: bool,
    partial_document_indices: list[int],
) -> dict[str, Any]:
    """Build recovery metadata for partially relevant evidence."""

    if is_relevant or not partial_document_indices:
        return _inactive_partial_relevance_recovery()
    return {
        "triggered": True,
        "action": "query_refinement",
        "reason": PARTIAL_RELEVANCE_RECOVERY_REASON,
        "partial_document_indices": list(partial_document_indices),
    }


def _inactive_partial_relevance_recovery() -> dict[str, Any]:
    """Return inactive partial-relevance recovery metadata."""

    return {
        "triggered": False,
        "action": "none",
        "reason": "",
        "partial_document_indices": [],
    }


def _format_previous_queries(previous_queries: list[str]) -> str:
    """Format previous retrieval queries for the retry prompt."""

    if not previous_queries:
        return "No previous queries."
    return "\n".join(f"- {query}" for query in previous_queries)


def _format_partial_relevance_context(state: AgentState) -> str:
    """Format partial-relevance recovery evidence for retry rewriting."""

    recovery = state.get("partial_relevance_recovery", {})
    if not recovery.get("triggered"):
        return "No partial relevance recovery was triggered."

    documents = state.get("documents", [])
    grades_by_index = {
        grade.get("document_index"): grade for grade in state.get("document_grades", [])
    }
    raw_indices = recovery.get("partial_document_indices", [])
    partial_indices = [
        index
        for index in raw_indices
        if isinstance(index, int) and 1 <= index <= len(documents)
    ]
    lines = [
        f"Recovery action: {recovery.get('action', 'query_refinement')}",
        f"Recovery reason: {recovery.get('reason') or PARTIAL_RELEVANCE_RECOVERY_REASON}",
        (
            "Use these related-but-insufficient chunks to target missing facts, "
            "entities, comparisons, or constraints without broadening off-topic."
        ),
        "Partially related context:",
    ]
    if not partial_indices:
        lines.append("- No valid partially related chunks were available.")
        return "\n".join(lines)

    for index in partial_indices:
        document = documents[index - 1]
        grade = grades_by_index.get(index, {})
        reason = grade.get("reason") or "No partial relevance reason provided."
        confidence = grade.get("confidence", 0.0)
        source = document.get("source") or "unknown source"
        chunk_id = document.get("chunk_id") or "unknown chunk"
        snippet = _make_snippet(document.get("content", ""))
        lines.append(
            f"- [{index}] source={source} chunk_id={chunk_id} "
            f"confidence={confidence} reason={reason}\n  {snippet}"
        )
    return "\n".join(lines)


def _selected_citation_chunk_ids(documents: list[RetrievedDocument]) -> list[str]:
    """Return valid chunk IDs for selected citation documents."""

    chunk_ids: list[str] = []
    for index, document in enumerate(documents, start=1):
        chunk_id = _document_chunk_id(document, index)
        if chunk_id not in chunk_ids:
            chunk_ids.append(chunk_id)
    return chunk_ids


def _document_chunk_id(document: RetrievedDocument, index: int) -> str:
    """Return a citation-verification chunk ID for a selected document."""

    chunk_id = document.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    source = document.get("source") or "unknown-source"
    page = document.get("page")
    if page is not None:
        return f"{source}:p{page}:citation-{index}"
    return f"{source}:citation-{index}"


def _make_snippet(content: str, limit: int = 240) -> str:
    """Create a short citation snippet without exposing full chunks."""

    normalized = " ".join(content.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."
