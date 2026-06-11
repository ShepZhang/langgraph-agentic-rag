# P1e Structured Retrieval Grading Design

## Goal

Upgrade retrieval grading from coarse `relevant_indices` output to chunk-level structured grading with relevance labels, confidence, and reasons, while preserving compatibility with existing simple grading responses.

## Scope

Implemented behavior:

- Parse new structured grading schema with per-document records.
- Preserve legacy `{"relevant": true, "relevant_indices": [...], "reason": "..."}` support.
- Store grading diagnostics in Agent state and final result payload.
- Route to answer generation only when at least one chunk is graded `relevant`.
- Continue retry/fallback behavior when chunks are only `partially_relevant` or `irrelevant`.

Deferred behavior:

- Partial relevance does not yet trigger expanded retrieval or reranker adjustments. That is future P1f work.
- No new LangGraph nodes or conditional edges are added.
- Independent ablation toggles remain future P0b/P1 work.

## Structured Schema

Preferred LLM output:

```json
{
  "grades": [
    {
      "document_index": 1,
      "relevance": "relevant",
      "confidence": 0.91,
      "reason": "Directly answers the question."
    }
  ],
  "reason": "At least one chunk directly supports the answer."
}
```

Allowed labels:

- `relevant`
- `partially_relevant`
- `irrelevant`

Invalid labels become `irrelevant`. Confidence is clamped into `[0, 1]`. Out-of-range document indexes are ignored.

## State Additions

`AgentState` gains:

- `document_grades`
- `relevant_document_count`
- `partial_document_count`
- `max_relevance_confidence`

`relevant_documents` continues to contain only chunks with `relevance == "relevant"`.

## Module Boundary

Create `agent/retrieval_grading.py` with:

- `DocumentGrade` TypedDict.
- `GradingResult` TypedDict.
- `parse_retrieval_grading_response(raw_result, document_count)`.
- `build_legacy_document_grades(relevant_indices, document_count, reason)`.

`agent.nodes.grade_documents_node()` calls this parser and writes the structured state fields.

## Compatibility

Legacy grading output remains valid. Existing fake LLM tests that return `relevant_indices` should keep passing, with additional structured state fields populated.

## Testing

Tests cover:

- Structured grading parsing.
- Legacy grading parsing.
- invalid JSON fallback.
- invalid labels and out-of-range indexes.
- grade node state updates.
- graph result payload diagnostics.

## Risks

The main risk is over-triggering answer generation from weak evidence. P1e avoids this by only answering when at least one chunk is explicitly `relevant`; `partially_relevant` contributes diagnostics but still routes to retry/fallback.
