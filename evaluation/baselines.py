"""Naive RAG baseline used for evaluation comparisons."""

from __future__ import annotations

from typing import Any, Callable

from langchain_openai import ChatOpenAI

from agent.nodes import (
    FALLBACK_ANSWER,
    _coerce_llm_text,
    _parse_answer_result,
    build_citations,
    is_unable_to_answer,
)
from agent.prompts import ANSWER_GENERATION_PROMPT, format_documents
from config import Settings, get_settings
from rag.retriever import retrieve


RetrieverFn = Callable[[str], list[dict[str, Any]]]


def run_naive_rag(
    question: str,
    retriever_fn: RetrieverFn | None = None,
    llm: Any | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run a minimal retrieve-once RAG baseline without agentic control flow."""

    resolved_settings = settings or get_settings()
    resolved_llm = llm or _create_chat_model(resolved_settings)
    resolved_retriever = retriever_fn or retrieve

    documents = resolved_retriever(question)
    if not documents:
        return _fallback_payload(
            question=question,
            retrieved_documents=[],
            reason="Naive RAG retrieved no documents.",
        )

    prompt = ANSWER_GENERATION_PROMPT.format(
        question=question,
        current_query=question,
        documents=format_documents(documents),
    )
    raw_result = _coerce_llm_text(resolved_llm.invoke(prompt))
    parsed_answer = _parse_answer_result(raw_result, document_count=len(documents))
    if parsed_answer is None:
        return _fallback_payload(
            question=question,
            retrieved_documents=documents,
            reason="Naive RAG answer generation returned invalid JSON.",
        )

    answer = parsed_answer["answer"].strip()
    if not answer:
        return _fallback_payload(
            question=question,
            retrieved_documents=documents,
            reason="Naive RAG answer generation returned an empty answer.",
        )

    citations = build_citations(
        documents,
        used_citation_indices=parsed_answer["used_citation_indices"],
    )
    if not citations and not is_unable_to_answer(answer):
        return _fallback_payload(
            question=question,
            retrieved_documents=documents,
            reason="Naive RAG answer generation did not return valid supporting citations.",
        )

    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "claims": [],
        "claim_verification": {},
        "claim_verification_reason": "",
        "is_verified": False,
        "retrieved_documents": documents,
        "relevant_documents": documents,
        "retry_count": 0,
        "fallback_reason": "",
    }


def _fallback_payload(
    question: str,
    retrieved_documents: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    """Return a payload aligned with agentic evaluation results."""

    return {
        "question": question,
        "answer": FALLBACK_ANSWER,
        "citations": [],
        "claims": [],
        "claim_verification": {},
        "claim_verification_reason": reason,
        "is_verified": False,
        "retrieved_documents": retrieved_documents,
        "relevant_documents": [],
        "retry_count": 0,
        "fallback_reason": reason,
    }


def _create_chat_model(settings: Settings) -> ChatOpenAI:
    """Create the chat model used by the naive baseline."""

    settings.require_llm_config()
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=settings.temperature,
    )
