# Atlas Document QA Product Specification

Atlas is a private document QA app for asking grounded questions about a
user-provided document collection. This specification describes a fictional
demonstration product and contains no customer data.

## Supported Documents

Atlas supports PDF, Markdown, and TXT files. The maximum upload size is 25 MB
per file. Uploaded content is processed locally by the demo workflow and is
used to build the active retrieval collection.

## Indexing Behavior

The demo build replaces the active collection whenever the user builds an
index. The lower-level vectorstore API supports incremental indexing with
deterministic chunk IDs, allowing callers to avoid duplicate chunks across
repeated indexing operations.

## Retrieval And Reranking

Atlas uses Chroma to retrieve vector candidates. An optional cross-encoder can
rerank those candidates before LangGraph performs retrieval grading and
selects the evidence used for answer generation.

## Answer Safety

Normal answers must contain citation markers that match the returned citation
indices. Atlas also performs lightweight claim verification against the
selected evidence. When evidence or citation alignment is insufficient, the
application returns a fallback response instead of an unsupported normal
answer.

## Service Scope

Atlas has no formal production SLA. It is intended for local demo and
evaluation use, not as a production service commitment.
