from __future__ import annotations

from prompting.registry import PromptDefinition, PromptRegistry


QUERY_TRANSFORM_V1 = """You transform user questions for private knowledge-base retrieval.

Use chat history only to resolve references or missing context.
Choose exactly one strategy:
- rewrite: for simple factual questions that only need a standalone retrieval query.
- multi_query: for ambiguous or context-dependent questions that benefit from equivalent or complementary retrieval queries.
- decomposition: for complex comparison or multi-hop questions that benefit from sub-questions.

Return JSON only in this shape:
{{"strategy": "rewrite", "rewritten_query": "standalone retrieval query", "expanded_queries": [], "sub_questions": [], "reason": "short reason"}}

Rules:
- rewritten_query must be a standalone question suitable for retrieval.
- Do not use decomposition for simple factual questions.
- expanded_queries should be empty unless strategy is multi_query.
- sub_questions should be empty unless strategy is decomposition.
- Return JSON only. No markdown fences.

Chat history:
{chat_history}

Original question:
{question}

JSON:"""


QUERY_REWRITE_V1 = """You are rewriting a user question for private knowledge-base retrieval.

Use the chat history only to resolve references or missing context.
Return one standalone retrieval question.
If the original question is already clear, return it unchanged.

Chat history:
{chat_history}

Original question:
{question}

Standalone retrieval question:"""


RETRY_QUERY_REWRITE_V1 = """You are improving a failed private knowledge-base retrieval query.

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

Partial relevance recovery:
{partial_relevance_context}

Previously retrieved chunks:
{documents}

Improved retrieval query:"""


RETRIEVAL_GRADING_V1 = """You are grading whether retrieved chunks can answer a user's original question.

Do not mark chunks relevant just because they share keywords.
Mark them relevant only if they contain enough factual information to answer the original user question.
Return JSON only in this shape:
{{"grades": [{{"document_index": 1, "relevance": "relevant", "confidence": 0.91, "reason": "short reason"}}], "reason": "short overall reason"}}

Rules:
- The retrieval query is only used to explain how the chunks were searched.
- You must grade the retrieved chunks against the original user question.
- Do not grade the chunks as relevant only because they match the retrieval query.
- document_index must use 1-based indexes matching the retrieved chunk numbers.
- relevance must be exactly one of: relevant, partially_relevant, irrelevant.
- Use relevant only when the chunk directly contains enough evidence to answer the original question.
- Use partially_relevant when the chunk is related but does not contain enough evidence to answer.
- Use irrelevant when the chunk does not help answer the original question.
- confidence must be a number between 0 and 1.
- Return JSON only. No markdown fences.

Original user question:
{question}

Retrieval query:
{current_query}

Retrieved chunks:
{documents}

JSON:"""


ANSWER_GENERATION_V1 = """You answer questions using only the retrieved chunks.

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


CLAIM_EXTRACTION_V1 = """You extract atomic factual claims from a draft answer.

Return JSON only in this shape:
{{"claims": [{{"claim_id": "c001", "claim": "short factual claim", "cited_chunk_ids": ["chunk_id"]}}], "reason": "short reason"}}

Rules:
- Extract only factual claims that need citation support.
- Do not create claims for connective text, hedging, or citation markers by themselves.
- claim_id values must be stable IDs like c001, c002, c003.
- cited_chunk_ids must come only from the selected citation chunks below.
- If the answer contains no factual claims, return an empty claims list.
- Return JSON only. No markdown fences.

Original user question:
{question}

Draft answer:
{answer}

Selected citation chunks:
{documents}

JSON:"""


CITATION_VERIFICATION_V1 = """You verify each extracted claim against its cited chunks.

Return JSON only in this shape:
{{"results": [{{"claim_id": "c001", "claim": "short factual claim", "cited_chunk_ids": ["chunk_id"], "verification_label": "supported", "confidence": 0.91, "reason": "short reason"}}], "reason": "short overall reason"}}

Rules:
- verification_label must be exactly one of: supported, partially_supported, unsupported.
- Use supported only when the cited chunks directly support the claim.
- Use partially_supported when the cited chunks support part of the claim but the claim is too broad or too strong.
- Use unsupported when the cited chunks do not support the claim.
- Do not give credit for vague keyword overlap.
- confidence must be a number between 0 and 1.
- Return JSON only. No markdown fences.

Original user question:
{question}

Draft answer:
{answer}

Extracted claims:
{claims}

Selected citation chunks:
{documents}

JSON:"""


ANSWER_REVISION_V1 = """You revise an answer after claim-level citation verification found unsupported content.

Return JSON only in this shape:
{{"answer": "Revised answer with citation markers like [1].", "used_citation_indices": [1]}}

Rules:
- Remove unsupported claims.
- Narrow partially supported claims to exactly what the cited chunks support.
- Preserve valid citation markers like [1] that refer to selected citation chunks.
- Do not introduce new facts or new citations.
- If no supported answer remains, say you cannot answer from the current documents and return an empty used_citation_indices list.
- used_citation_indices must contain only the 1-based chunk numbers actually used as evidence.
- The citation markers in answer must exactly match used_citation_indices.
- Return JSON only. No markdown fences.

Original user question:
{question}

Current draft answer:
{answer}

Unsupported or partially supported claims:
{unsupported_claims}

Selected citation chunks:
{documents}

JSON:"""


CLAIM_VERIFICATION_V1 = """You are a claim-level citation verifier for private document QA.

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


DOCUMENT_SUMMARY_V1 = """Summarize the document using only the supplied text. Return at most {max_points} concise bullet points. Do not add unsupported facts.

Title: {title}

Document:
{content}"""


_PROJECT_PROMPT_DEFINITIONS = (
    PromptDefinition(
        prompt_id="agent.query_transform",
        version="v1",
        template=QUERY_TRANSFORM_V1,
        description="Transform a user question into a retrieval strategy.",
    ),
    PromptDefinition(
        prompt_id="agent.retry_query_rewrite",
        version="v1",
        template=RETRY_QUERY_REWRITE_V1,
        description="Improve a failed retrieval query.",
    ),
    PromptDefinition(
        prompt_id="agent.retrieval_grading",
        version="v1",
        template=RETRIEVAL_GRADING_V1,
        description="Grade retrieved chunks against the user question.",
    ),
    PromptDefinition(
        prompt_id="agent.answer_generation",
        version="v1",
        template=ANSWER_GENERATION_V1,
        description="Generate a grounded answer from retrieved chunks.",
    ),
    PromptDefinition(
        prompt_id="agent.claim_extraction",
        version="v1",
        template=CLAIM_EXTRACTION_V1,
        description="Extract citation-requiring claims from an answer.",
    ),
    PromptDefinition(
        prompt_id="agent.citation_verification",
        version="v1",
        template=CITATION_VERIFICATION_V1,
        description="Verify extracted claims against cited chunks.",
    ),
    PromptDefinition(
        prompt_id="agent.answer_revision",
        version="v1",
        template=ANSWER_REVISION_V1,
        description="Revise an answer after citation verification.",
    ),
    PromptDefinition(
        prompt_id="tool.document_summary",
        version="v1",
        template=DOCUMENT_SUMMARY_V1,
        description="Summarize supplied document text.",
    ),
    PromptDefinition(
        prompt_id="agent.query_rewrite",
        version="v1",
        template=QUERY_REWRITE_V1,
        description="Compatibility prompt for standalone query rewriting.",
    ),
    PromptDefinition(
        prompt_id="agent.claim_verification",
        version="v1",
        template=CLAIM_VERIFICATION_V1,
        description="Compatibility prompt for answer-level claim verification.",
    ),
)


PROJECT_PROMPT_REGISTRY = PromptRegistry(
    _PROJECT_PROMPT_DEFINITIONS,
    active_versions={
        "agent.query_transform": "v1",
        "agent.retry_query_rewrite": "v1",
        "agent.retrieval_grading": "v1",
        "agent.answer_generation": "v1",
        "agent.claim_extraction": "v1",
        "agent.citation_verification": "v1",
        "agent.answer_revision": "v1",
        "tool.document_summary": "v1",
    },
)
