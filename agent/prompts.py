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


RETRY_QUERY_REWRITE_PROMPT = """You are improving a failed private knowledge-base retrieval query.

The previous retrieval did not find enough relevant evidence. Rewrite the query to improve retrieval.
Avoid repeating the same query. Keep it concise, specific, and search-oriented.

Original question:
{question}

Previous retrieval query:
{current_query}

Previous queries:
{previous_queries}

Previous grading reason:
{grading_reason}

Previously retrieved chunks:
{documents}

Improved retrieval query:"""


RETRIEVAL_GRADING_PROMPT = """You are grading whether retrieved chunks can answer a user's original question.

Do not mark chunks relevant just because they share keywords.
Mark them relevant only if they contain enough factual information to answer the original user question.
Return JSON only in this shape:
{{"relevant": true, "relevant_indices": [1, 3], "reason": "short reason"}}

Rules:
- The retrieval query is only used to explain how the chunks were searched.
- You must grade the retrieved chunks against the original user question.
- Do not grade the chunks as relevant only because they match the retrieval query.
- relevant_indices must use 1-based indexes matching the retrieved chunk numbers.
- If no chunks are relevant, return:
  {{"relevant": false, "relevant_indices": [], "reason": "short reason"}}
- Return JSON only. No markdown fences.

Original user question:
{question}

Retrieval query:
{current_query}

Retrieved chunks:
{documents}

JSON:"""


ANSWER_GENERATION_PROMPT = """You answer questions using only the retrieved chunks.

Rules:
- You must answer the original user question.
- The retrieval query is provided only to explain how the documents were searched.
- Do not answer the retrieval query as if it were the user's question.
- Use only facts from the retrieved chunks.
- Do not invent facts that are not present in the retrieved chunks.
- For key facts, include citation markers like [1] and [2] that correspond to chunk numbers.
- If the retrieved chunks do not contain the answer, say you cannot answer from the current documents.
- Distinguish workflow cases: weak retrieval evidence can trigger retry rewriting before answer generation; unsupported claims in a generated answer are handled by citation safety fallback, not by retry rewriting.
- Keep the answer concise and useful.
- Return JSON only in this shape:
  {{"answer": "Final answer text with citation markers like [1].", "used_citation_indices": [1]}}
- used_citation_indices must contain only the 1-based chunk numbers actually used as evidence.
- The citation markers in answer must exactly match used_citation_indices.

Original user question:
{question}

Retrieval query:
{current_query}

Retrieved chunks:
{documents}

JSON:"""


CLAIM_VERIFICATION_PROMPT = """You are a claim-level citation verifier for private document QA.

Verify whether the answer is fully supported by the selected citation chunks.
Return JSON only in this shape:
{{"verified": true, "claims": [{{"claim": "short factual claim", "supported": true, "citation_indices": [1]}}], "reason": "short reason"}}

Rules:
- Split the answer into factual claims.
- Every factual claim must be supported by at least one selected citation chunk.
- citation_indices must use 1-based indexes matching the selected citation chunk numbers below.
- Mark verified false if any important factual claim is unsupported.
- Do not give credit for vague keyword overlap. Check whether the citation actually supports the claim.
- Return JSON only. No markdown fences.

Original user question:
{question}

Answer to verify:
{answer}

Selected citation chunks:
{documents}

JSON:"""


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
