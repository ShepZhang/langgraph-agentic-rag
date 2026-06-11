# P3c Workspace Isolation Design

## Goal

Make `workspace_id` a real retrieval-isolation boundary instead of only an API
field. A chat request scoped to one workspace must not retrieve chunks indexed
for another workspace.

## Chosen Approach

Use one Chroma collection with metadata filtering:

- API indexing already writes `workspace_id` and `document_id` into chunk
  metadata.
- Dense retrieval passes `filter={"workspace_id": workspace_id}` to Chroma when
  a workspace is supplied.
- BM25 sparse retrieval builds its corpus from the same workspace-filtered
  stored chunks.
- Hybrid retrieval passes the same workspace id to dense and sparse branches.
- `run_agent(workspace_id=...)` creates a workspace-scoped default retriever
  unless a custom `retriever_fn` is injected.

This preserves existing Gradio, CLI, and evaluation behavior when no
`workspace_id` is provided.

## Non-goals

- Do not create one Chroma collection per workspace in P3c.
- Do not add authentication or tenant authorization.
- Do not migrate historical unscoped documents.
- Do not change the P0b evaluation dataset or artifacts.

## Validation

P3c is complete when:

- Dense vector search receives a workspace metadata filter.
- BM25 corpus loading is workspace-filtered.
- Hybrid retrieval applies the same workspace to dense and sparse retrieval.
- Retriever output exposes `workspace_id` and `document_id`.
- `run_agent(workspace_id=...)` scopes the default retriever.
- Full tests pass.
