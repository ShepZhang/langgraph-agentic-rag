"""LangGraph node implementations for Agentic RAG."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import BaseMessage

from agent.prompts import (
    ANSWER_GENERATION_PROMPT,
    CLAIM_VERIFICATION_PROMPT,
    RETRY_QUERY_REWRITE_PROMPT,
    RETRIEVAL_GRADING_PROMPT,
    format_documents,
)
from agent.query_transform import (
    build_query_transform_prompt,
    fallback_query_transform,
    parse_query_transform_response,
)
from agent.state import AgentState, Citation, RetrievedDocument
from agent.tools import create_retriever_tool


FALLBACK_ANSWER = "根据当前已索引文档，无法可靠回答这个问题。请补充相关文档，或换一种更具体的问法。"
logger = logging.getLogger(__name__)


class AgentNodes:
    """State transition nodes for the Agentic RAG graph."""

    def __init__(self, llm: Any, retriever_fn: Any | None = None) -> None:
        self.llm = llm
        self.retriever_tool = create_retriever_tool(retriever_fn)

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
        documents = self.retriever_tool.invoke({"query": query})
        retrieval_attempt = state.get("retrieval_attempt", 0) + 1
        logger.info(
            "Retrieved documents count=%s attempt=%s query=%s",
            len(documents),
            retrieval_attempt,
            query,
        )
        return {
            "documents": documents,
            "retrieval_attempt": retrieval_attempt,
        }

    def grade_documents_node(self, state: AgentState) -> dict[str, Any]:
        """Grade whether retrieved chunks are relevant enough to answer."""

        documents = state.get("documents", [])
        if not documents:
            reason = "No documents retrieved."
            logger.info("Retrieval grading skipped: %s", reason)
            return {
                "is_relevant": False,
                "relevant_documents": [],
                "grading_reason": reason,
                "route": "rewrite_query",
            }

        prompt = RETRIEVAL_GRADING_PROMPT.format(
            question=state["question"],
            current_query=state.get("current_query") or state["question"],
            documents=format_documents(documents),
        )
        raw_result = _coerce_llm_text(self.llm.invoke(prompt))
        grading = _parse_grading_result(raw_result, document_count=len(documents))
        relevant_documents = [
            documents[index - 1] for index in grading["relevant_indices"]
        ]
        is_relevant = bool(relevant_documents)
        logger.info(
            "Retrieval grading relevant=%s relevant_count=%s reason=%s",
            is_relevant,
            len(relevant_documents),
            grading["reason"],
        )
        return {
            "is_relevant": is_relevant,
            "relevant_documents": relevant_documents,
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
        if not citations and not is_unable_to_answer(answer):
            logger.warning("Answer generation returned a normal answer without citations.")
            return _fallback_update(
                "Answer generation did not return valid supporting citations."
            )
        if citations:
            selected_documents = _select_documents_by_indices(
                documents,
                parsed_answer["used_citation_indices"],
            )
            verification = self._verify_answer_claims(
                question=state["question"],
                answer=answer,
                documents=selected_documents,
            )
            if verification is None:
                logger.warning("Claim verification returned invalid JSON.")
                return _fallback_update("Claim verification returned invalid JSON.")
            if not verification["verified"]:
                logger.warning(
                    "Claim verification failed: %s",
                    verification["reason"],
                )
                return _fallback_update(
                    f"Claim verification failed: {verification['reason']}"
                )
        else:
            verification = {
                "verified": False,
                "claims": [],
                "reason": "Unable-to-answer response; claim verification skipped.",
            }
        logger.info(
            "Generated answer citation_count=%s used_indices=%s",
            len(citations),
            parsed_answer["used_citation_indices"],
        )

        return {
            "answer": answer,
            "citations": citations,
            "claims": verification["claims"],
            "claim_verification": verification,
            "claim_verification_reason": verification["reason"],
            "is_verified": verification["verified"],
            "route": "end",
        }

    def fallback_node(self, state: AgentState) -> dict[str, Any]:
        """Return a safe fallback answer when retrieval is insufficient."""

        reason = state.get("grading_reason") or "No reliable supporting evidence found."
        logger.info("Fallback answer returned: %s", reason)
        return _fallback_update(reason)

    def _verify_answer_claims(
        self,
        question: str,
        answer: str,
        documents: list[RetrievedDocument],
    ) -> dict[str, Any] | None:
        """Verify generated answer claims against selected citation chunks."""

        prompt = CLAIM_VERIFICATION_PROMPT.format(
            question=question,
            answer=answer,
            documents=format_documents(documents),
        )
        raw_result = _coerce_llm_text(self.llm.invoke(prompt))
        return _parse_claim_verification_result(
            raw_result,
            document_count=len(documents),
        )


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


def _parse_claim_verification_result(
    raw_result: str,
    document_count: int,
) -> dict[str, Any] | None:
    """Parse and validate claim-level verification JSON."""

    parsed = _extract_first_json_object(raw_result)
    if parsed is None:
        return None

    raw_claims = parsed.get("claims")
    if parsed.get("verified") is not True and parsed.get("verified") is not False:
        return None
    if not isinstance(raw_claims, list):
        return None

    claims: list[dict[str, object]] = []
    for raw_claim in raw_claims:
        if not isinstance(raw_claim, dict):
            continue
        claim_text = raw_claim.get("claim")
        supported = raw_claim.get("supported")
        raw_indices = raw_claim.get("citation_indices", [])
        if not isinstance(claim_text, str) or not isinstance(supported, bool):
            continue
        if not isinstance(raw_indices, list):
            raw_indices = []
        citation_indices: list[int] = []
        for raw_index in raw_indices:
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                continue
            if raw_index < 1 or raw_index > document_count:
                logger.warning(
                    "Ignoring out-of-range claim citation index: %s",
                    raw_index,
                )
                continue
            if raw_index not in citation_indices:
                citation_indices.append(raw_index)
        claims.append(
            {
                "claim": claim_text,
                "supported": supported,
                "citation_indices": citation_indices,
            }
        )

    verified = parsed["verified"] is True and bool(claims) and all(
        claim["supported"] is True and bool(claim["citation_indices"])
        for claim in claims
    )
    reason = str(parsed.get("reason") or "").strip()
    if not reason:
        reason = "All claims verified." if verified else "One or more claims are unsupported."

    return {
        "verified": verified,
        "claims": claims,
        "reason": reason,
    }


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


def _format_previous_queries(previous_queries: list[str]) -> str:
    """Format previous retrieval queries for the retry prompt."""

    if not previous_queries:
        return "No previous queries."
    return "\n".join(f"- {query}" for query in previous_queries)


def _make_snippet(content: str, limit: int = 240) -> str:
    """Create a short citation snippet without exposing full chunks."""

    normalized = " ".join(content.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."
