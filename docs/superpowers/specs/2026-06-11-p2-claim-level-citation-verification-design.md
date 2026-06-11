# P2 Claim-Level Citation Verification Design

## Goal

Upgrade citation-aware answer generation from a single lightweight verification call into an explicit LangGraph sub-workflow for claim-level citation verification. Normal factual answers should become final answers only after claim extraction and per-claim citation verification pass, or after one successful answer revision.

## Motivation

The current workflow verifies claims inside `generate_answer_node()`. That is useful, but it hides the verification lifecycle inside one node and cannot model revision. P2 makes citation verification a first-class reliability workflow:

```text
generate_answer
-> extract_claims
-> verify_citations
-> finalize_answer
```

When verification fails:

```text
verify_citations
-> revise_answer
-> extract_claims
-> verify_citations
```

If verification still fails after the configured revision budget, the graph returns fallback instead of an unsupported answer.

## Scope

Implemented behavior:

- Split answer generation from finalization.
- Add claim extraction, citation verification, answer revision, and finalization nodes.
- Store claim-level verification diagnostics in Agent state and `run_agent()` output.
- Route unsupported or partially supported claims to one revision attempt by default.
- Preserve safe fallback behavior for invalid JSON, missing citations, invalid citation markers, failed extraction, failed verification, and repeated unsupported claims.
- Preserve existing public result keys used by UI and evaluation: `answer`, `citations`, `claims`, `claim_verification`, `claim_verification_reason`, and `is_verified`.

Deferred behavior:

- Do not introduce Tool Registry in P2; that remains P3.
- Do not add human-reviewed claim labels in P2.
- Do not add independent ablation toggles in this milestone.
- Do not run P0b evaluation artifacts until after the P2 implementation is stable.

## Workflow

The existing front half remains:

```text
START
-> rewrite_query
-> retrieve
-> grade_documents
-> generate_answer
```

P2 replaces the current direct `generate_answer -> END` edge with:

```text
generate_answer
-> extract_claims
-> verify_citations
-> finalize_answer
-> END
```

Conditional verification routing:

```text
verify_citations
-> finalize_answer  when citation_verification_passed is true
-> revise_answer    when unsupported claims exist and citation_revision_count < max_citation_revision_count
-> fallback         when unsupported claims remain after revision budget

revise_answer
-> extract_claims   when revision produced a non-empty revised draft
-> fallback         when revision failed
```

`fallback` still routes to `END`.

## Node Responsibilities

### `generate_answer_node`

Generates a draft answer from `relevant_documents`.

Responsibilities:

- Parse answer-generation JSON.
- Validate answer citation markers against `used_citation_indices`.
- Build `citations` from selected evidence.
- Build `cited_documents` from selected evidence.
- Store `draft_answer`, `used_citation_indices`, `citations`, and `cited_documents`.
- Do not call claim verification.
- Do not set final `answer` for normal cited answers.

Unable-to-answer behavior:

- If the model explicitly says it cannot answer and returns no citations, mark verification as skipped and route to `finalize_answer`.
- The final answer is allowed because it refuses unsupported content.

### `extract_claims_node`

Extracts atomic claims from `draft_answer`.

Preferred LLM output:

```json
{
  "claims": [
    {
      "claim_id": "c001",
      "claim": "Agentic RAG uses retrieval grading to filter weak chunks.",
      "cited_chunk_ids": ["paper.pdf:p2:c1"]
    }
  ],
  "reason": "The answer contains one factual claim."
}
```

Rules:

- Each claim must be short and factual.
- Non-factual connective text should not become a claim.
- `cited_chunk_ids` must reference selected citation chunks, not arbitrary retrieved chunks.
- If the answer is an unable-to-answer response, claim extraction is skipped and finalization is allowed.

### `verify_citations_node`

Verifies each extracted claim against the selected citation chunks.

Preferred LLM output:

```json
{
  "results": [
    {
      "claim_id": "c001",
      "claim": "Agentic RAG uses retrieval grading to filter weak chunks.",
      "cited_chunk_ids": ["paper.pdf:p2:c1"],
      "verification_label": "supported",
      "confidence": 0.91,
      "reason": "The cited chunk states that retrieval grading filters weak chunks."
    }
  ],
  "reason": "All claims are supported by their cited chunks."
}
```

Allowed labels:

- `supported`
- `partially_supported`
- `unsupported`

Pass condition:

- Every claim must have `verification_label == "supported"`.
- Every supported claim must cite at least one valid selected chunk id.
- Empty claims do not count as a citation verification pass. Unable-to-answer responses can still finalize when `citation_verification_skipped == true`.

### `revise_answer_node`

Revises the draft answer using verifier feedback.

Responsibilities:

- Remove unsupported claims.
- Narrow partially supported claims to what the cited chunks actually support.
- Preserve valid citation markers.
- Return answer-generation-compatible JSON:

```json
{
  "answer": "Revised grounded answer [1].",
  "used_citation_indices": [1]
}
```

The revision node increments `citation_revision_count`, rebuilds `citations` and `cited_documents`, and then routes back to `extract_claims`.

### `finalize_answer_node`

Promotes `draft_answer` to public `answer` only after verification pass or explicit unable-to-answer skip.

Responsibilities:

- Set `answer`.
- Keep `citations`, `claims`, and verification diagnostics.
- Set `is_verified` from `citation_verification_passed`.
- Set route to `end`.

## State Additions

Add these fields to `AgentState`:

```python
draft_answer: str
used_citation_indices: list[int]
cited_documents: list[RetrievedDocument]
claim_verification_results: list[dict[str, object]]
unsupported_claims: list[dict[str, object]]
citation_verification_passed: bool
citation_revision_count: int
max_citation_revision_count: int
citation_verification_skipped: bool
```

`max_citation_revision_count` defaults to `1`, so the workflow gets one chance to repair unsupported claims before fallback.

Upgrade `claims` records to:

```json
{
  "claim_id": "c001",
  "claim": "...",
  "cited_chunk_ids": ["chunk_001"]
}
```

Keep `claim_verification` as a summary payload:

```json
{
  "verified": true,
  "results": [],
  "reason": "All claims supported."
}
```

`is_verified` remains a compatibility alias for `citation_verification_passed`. Unable-to-answer responses that skip verification are finalizable, but keep `citation_verification_passed == false` and `is_verified == false` so evaluation does not count skipped verification as a pass.

## Prompt Additions

Add prompts:

- `CLAIM_EXTRACTION_PROMPT`
- `CITATION_VERIFICATION_PROMPT`
- `ANSWER_REVISION_PROMPT`

The existing `CLAIM_VERIFICATION_PROMPT` can remain in the file for older tests or downstream imports, but the graph should use the new prompts after P2 is complete.

## Parser Modules

Create a focused module:

```text
agent/citation_verification.py
```

Responsibilities:

- Parse claim extraction JSON.
- Parse verification JSON.
- Normalize invalid verification labels to `unsupported`.
- Clamp confidence to `[0, 1]`.
- Validate `cited_chunk_ids` against selected citation chunk ids.
- Build summary verification payloads.

This keeps `agent/nodes.py` from growing further and makes parser behavior independently testable.

## Error Handling

Fallback immediately when:

- answer-generation JSON is invalid.
- a normal answer has missing or invalid citation markers.
- a normal answer returns no valid citations.
- claim extraction JSON is invalid.
- verification JSON is invalid.
- answer revision JSON is invalid.
- answer revision produces an empty answer.
- unsupported claims remain after revision budget.

Do not fallback when:

- the answer is an explicit unable-to-answer response with no citations. In that case, skip claim extraction and finalize the refusal answer.

## Compatibility

Existing UI and evaluation code should keep working:

- `answer` remains the final answer string.
- `citations` remains the final selected citation list.
- `claims` remains available, now with structured claim records.
- `claim_verification` remains available as a summary dictionary.
- `claim_verification_reason` remains available.
- `is_verified` remains available.
- unable-to-answer results may finalize with `is_verified == false` and `citation_verification_skipped == true`.

Existing tests that assume verification occurs inside `generate_answer_node()` will be migrated to new node-level tests and graph-level integration tests.

## Testing

Tests should cover:

- `generate_answer_node` produces `draft_answer`, `used_citation_indices`, `citations`, and `cited_documents` without finalizing normal answers.
- citation marker safety still rejects missing, mismatched, or out-of-range markers.
- claim extraction parser handles valid claims, invalid JSON, missing claim ids, and invalid cited chunk ids.
- `extract_claims_node` writes structured claims.
- `verify_citations_node` writes verification results, unsupported claims, summary payload, and pass/fail flags.
- unsupported or partially supported claims route to `revise_answer`.
- `revise_answer_node` increments revision count and loops back to claim extraction.
- second verification pass finalizes the revised answer.
- repeated unsupported claims route to fallback after revision budget.
- explicit unable-to-answer responses with no citations skip verification and finalize safely.
- `run_agent()` returns the new diagnostics.

## Risks

The largest risk is graph complexity. P2 controls that risk by adding a narrow verification sub-workflow while leaving retrieval, grading, retry, hybrid retrieval, and reranker behavior unchanged.

The second risk is verifier over-strictness. P2 treats `partially_supported` as unsafe for final answers, but permits one revision to salvage grounded content before fallback.

The third risk is breaking existing evaluation and UI consumers. P2 preserves the public result keys and adds new diagnostics instead of removing old fields.
