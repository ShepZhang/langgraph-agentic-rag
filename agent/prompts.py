"""Compatibility prompt exports and formatting helpers for Agentic RAG."""

from __future__ import annotations

from agent.state import ChatMessage, RetrievedDocument
from prompting import get_prompt_template

QUERY_REWRITE_PROMPT = get_prompt_template("agent.query_rewrite", version="v1")
RETRY_QUERY_REWRITE_PROMPT = get_prompt_template(
    "agent.retry_query_rewrite",
    version="v1",
)
RETRIEVAL_GRADING_PROMPT = get_prompt_template(
    "agent.retrieval_grading",
    version="v1",
)
ANSWER_GENERATION_PROMPT = get_prompt_template(
    "agent.answer_generation",
    version="v1",
)
CLAIM_EXTRACTION_PROMPT = get_prompt_template(
    "agent.claim_extraction",
    version="v1",
)
CITATION_VERIFICATION_PROMPT = get_prompt_template(
    "agent.citation_verification",
    version="v1",
)
ANSWER_REVISION_PROMPT = get_prompt_template(
    "agent.answer_revision",
    version="v1",
)
CLAIM_VERIFICATION_PROMPT = get_prompt_template(
    "agent.claim_verification",
    version="v1",
)


def format_chat_history(chat_history: list[ChatMessage]) -> str:
    """Format chat history for prompts."""

    if not chat_history:
        return "No prior chat history."

    lines = []
    for message in chat_history:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def format_documents(documents: list[RetrievedDocument]) -> str:
    """Format retrieved documents for prompts."""

    if not documents:
        return "No retrieved chunks."

    blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        source = document.get("source")
        page = document.get("page")
        chunk_id = document.get("chunk_id")
        score = document.get("score")
        rerank_score = document.get("rerank_score")
        content = document.get("content", "")
        rerank_part = (
            f" rerank_score={rerank_score}" if rerank_score is not None else ""
        )
        blocks.append(
            f"[{index}] source={source} page={page} chunk_id={chunk_id} "
            f"score={score}{rerank_part}\n{content}"
        )
    return "\n\n".join(blocks)
