"""Prompt templates and formatting helpers for Agentic RAG."""

from __future__ import annotations

from agent.state import ChatMessage, RetrievedDocument


QUERY_REWRITE_PROMPT = """You are rewriting a user question for private knowledge-base retrieval.

Use the chat history only to resolve references or missing context.
Return one standalone retrieval question.
If the original question is already clear, return it unchanged.

Chat history:
{chat_history}

Original question:
{question}

Standalone retrieval question:"""


RETRIEVAL_GRADING_PROMPT = """You are grading whether retrieved chunks can answer a user's question.

Do not mark chunks relevant just because they share keywords.
Mark them relevant only if they contain enough factual information to answer the question.
Return JSON only in this shape:
{{"relevant": true, "reason": "short reason"}}

Question:
{question}

Retrieved chunks:
{documents}

JSON:"""


ANSWER_GENERATION_PROMPT = """You answer questions using only the retrieved chunks.

Rules:
- Use only facts from the retrieved chunks.
- Do not invent facts that are not present in the retrieved chunks.
- For key facts, include source references in the answer when possible.
- If the retrieved chunks do not contain the answer, say you cannot answer from the current documents.
- Keep the answer concise and useful.

Question:
{question}

Retrieved chunks:
{documents}

Answer:"""


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
        content = document.get("content", "")
        blocks.append(
            f"[{index}] source={source} page={page} chunk_id={chunk_id} "
            f"score={score}\n{content}"
        )
    return "\n\n".join(blocks)
