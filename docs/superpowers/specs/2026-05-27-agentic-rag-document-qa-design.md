# Agentic RAG Document QA System Design

Date: 2026-05-27

## Goal

Build a lightweight, resume-ready Agentic RAG document QA system for private knowledge bases. Users can upload PDF, Markdown, and TXT documents, build a local vector index, ask questions, and receive grounded answers with citations.

The project should be easy to run, easy to explain, and focused on the Agentic RAG workflow rather than backend infrastructure. It should not copy an existing open-source project, though it may follow common engineering patterns from LangGraph and RAG systems.

## Chosen Approach

Use the lightweight MVP approach:

- Gradio for the user interface.
- LangGraph for explicit agent workflow orchestration.
- ChromaDB for local persistent vector storage.
- Local sentence-transformers embeddings by default.
- OpenAI-compatible chat LLM for query rewriting, retrieval grading, and answer generation.
- Simple evaluation script for answer/citation/source-hit statistics.

FastAPI, Docker, authentication, PostgreSQL, Ollama, reranking, and observability are intentionally deferred to later versions.

## MVP Scope

The MVP implements the complete Agentic RAG loop:

```text
Upload PDF / Markdown / TXT
-> parse documents
-> split into chunks
-> embed chunks locally
-> persist in Chroma
-> user asks question
-> rewrite query
-> retrieve context through a retriever tool
-> grade retrieved chunks for relevance
-> if relevant, generate grounded answer with citations
-> if not relevant, rewrite and retrieve again
-> after max retries, return a fallback answer
```

The MVP includes:

- PDF, Markdown, and TXT loading.
- Metadata preservation: source filename, page number where available, and chunk id.
- Recursive text splitting.
- Local `sentence-transformers/all-MiniLM-L6-v2` embeddings by default.
- ChromaDB persistent storage.
- Retriever wrapped as an agent tool named `retrieve_context`.
- LangGraph nodes for query rewrite, retrieve, document grading, answer generation, and fallback.
- Conditional retry with `max_rewrite_attempts=2`.
- Citation-aware answer generation.
- Gradio UI for upload, indexing, question answering, citations, retrieved chunks, rewritten question, and rewrite count.
- Lightweight evaluation script and sample questions.
- README designed for GitHub and resume presentation.

The MVP does not include:

- FastAPI.
- Authentication or permissions.
- PostgreSQL or pgvector.
- Docker.
- Multi-tenant document isolation.
- Local Ollama LLM.
- Advanced observability.
- Production deployment automation.

## Future Enhancements

The roadmap can include:

- FastAPI API layer with `/upload`, `/ask`, and `/health`.
- Ollama local LLM support.
- Naive RAG vs Agentic RAG comparison.
- Reranking.
- Document collection management.
- Document deletion and re-indexing.
- Persistent conversation memory.
- Docker packaging.
- LangSmith or Langfuse tracing.
- Larger evaluation set and optional LLM judge.

## Architecture

The system is split into four layers:

```text
UI Layer
  Gradio app for document upload, indexing, asking questions, and displaying results.

RAG Layer
  Document loading, chunking, embeddings, vector store access, and retrieval.

Agent Layer
  LangGraph state, prompts, tools, nodes, edges, and graph execution.

Evaluation Layer
  JSON question set and a lightweight script for metric reporting.
```

The retriever is exposed as a tool rather than buried inside answer generation. The LangGraph workflow controls when to call it and what to do when retrieved chunks are not relevant.

## Agent Workflow

```text
START
-> rewrite_query
-> retrieve
-> grade_documents
-> if relevant: generate_answer -> END
-> if not relevant and rewrite_count < max_rewrite_attempts: rewrite_query
-> if not relevant and rewrite_count >= max_rewrite_attempts: fallback -> END
```

`rewrite_count` starts at the first rewrite. With the default value `max_rewrite_attempts=2`, the agent can run two rewrite/retrieve attempts before falling back.

## Project Structure

```text
agentic-rag-document-qa/
├── app.py
├── main.py
├── config.py
├── requirements.txt
├── .env.example
├── README.md
├── rag/
│   ├── __init__.py
│   ├── loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   └── retriever.py
├── agent/
│   ├── __init__.py
│   ├── graph.py
│   ├── state.py
│   ├── nodes.py
│   ├── edges.py
│   ├── tools.py
│   └── prompts.py
├── evaluation/
│   ├── __init__.py
│   ├── eval_questions.json
│   └── evaluate.py
├── ui/
│   ├── __init__.py
│   └── gradio_app.py
├── assets/
│   └── architecture.png
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-05-27-agentic-rag-document-qa-design.md
```

## Configuration

`config.py` owns all environment-driven settings:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `EMBEDDING_PROVIDER`
- `EMBEDDING_MODEL`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `TOP_K`
- `MAX_REWRITE_ATTEMPTS`
- `CHROMA_PERSIST_DIR`
- `CHROMA_COLLECTION_NAME`

MVP defaults:

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: local Chroma persistence under `./data/chroma`
- Rewrite attempts: `2`
- Retrieval top k: a small configurable value such as `4`

The OpenAI-compatible LLM is required for agent execution. If LLM settings are missing, the system should fail clearly instead of generating fake offline answers.

## Dependencies

Primary dependencies:

```text
langgraph
langchain
langchain-core
langchain-openai
langchain-community
langchain-text-splitters
langchain-chroma
chromadb
sentence-transformers
pypdf
python-dotenv
pydantic
gradio
```

Optional development dependency:

```text
pytest
```

## RAG Module Design

### `rag/loader.py`

Loads PDF, Markdown, and TXT files into LangChain `Document` objects.

Interface:

```python
load_documents(file_paths: list[str]) -> list[Document]
```

Metadata:

```python
{
    "source": "filename.pdf",
    "page": 3
}
```

PDF files preserve page numbers. Markdown and TXT files use `page=None`.

### `rag/chunker.py`

Splits loaded documents with `RecursiveCharacterTextSplitter`.

Interface:

```python
split_documents(documents: list[Document]) -> list[Document]
```

Each chunk preserves original metadata and adds `chunk_id`, for example:

```text
paper.pdf:p3:c7
```

### `rag/embeddings.py`

Initializes the embedding model.

Interface:

```python
get_embedding_model()
```

The default provider is local Hugging Face sentence-transformers. The design leaves room for OpenAI embeddings later.

### `rag/vectorstore.py`

Owns Chroma creation, loading, indexing, and search.

Interfaces:

```python
create_vectorstore(docs)
load_vectorstore()
add_documents(docs)
similarity_search(query, top_k=None)
```

### `rag/retriever.py`

Wraps vector store retrieval behind a stable project-level interface.

Interface:

```python
retrieve(query: str, top_k: int | None = None) -> list[dict]
```

Returned chunk shape:

```python
{
    "content": "...",
    "source": "paper.pdf",
    "page": 2,
    "chunk_id": "paper.pdf:p2:c3",
    "score": 0.32
}
```

## Agent Module Design

### `agent/state.py`

Use a LangGraph-friendly `TypedDict`:

```python
class AgentState(TypedDict):
    question: str
    rewritten_question: str
    chat_history: list[dict[str, str]]
    documents: list[dict]
    answer: str
    citations: list[dict]
    rewrite_count: int
    is_relevant: bool
    route: str
```

### `agent/tools.py`

Expose the retriever as the tool `retrieve_context`.

Tool description:

```text
Retrieve relevant document chunks from the indexed private knowledge base according to the user's question.
```

Tool input:

```python
query: str
```

Tool output:

```python
list[dict]
```

### `agent/prompts.py`

Centralize prompts:

- Query rewrite prompt.
- Retrieval grading prompt.
- Answer generation prompt.

Prompt requirements:

- Query rewrite turns ambiguous or contextual questions into standalone retrieval questions. If the question is already clear, it returns the original question.
- Retrieval grading checks whether chunks are truly sufficient to answer the question, not just keyword-overlapping.
- Answer generation must use only retrieved chunks and avoid unsupported claims.
- If the answer is not in the retrieved context, the model must say it cannot answer from the current documents.

### `agent/nodes.py`

Nodes:

- `rewrite_query_node`
- `retrieve_node`
- `grade_documents_node`
- `generate_answer_node`
- `fallback_node`

`rewrite_query_node`:

- Reads `question` and `chat_history`.
- Calls the LLM.
- Writes `rewritten_question`.
- Increments `rewrite_count`.

`retrieve_node`:

- Uses `rewritten_question` when available, otherwise `question`.
- Calls `retrieve_context`.
- Writes retrieved chunks to `documents`.

`grade_documents_node`:

- Calls the LLM with question and retrieved chunks.
- Sets `is_relevant`.
- Treats invalid JSON or empty retrieval conservatively as not relevant.

`generate_answer_node`:

- Calls the LLM with question and retrieved chunks.
- Writes `answer`.
- Builds `citations` from retrieved chunk metadata so citations are grounded in actual retrieval results.

`fallback_node`:

- Returns a clear inability message:

```text
根据当前已索引文档，无法可靠回答这个问题。请补充相关文档，或换一种更具体的问法。
```

### `agent/edges.py`

Conditional routing:

```python
def route_after_grading(state: AgentState) -> str:
    if state["is_relevant"]:
        return "generate_answer"
    if state["rewrite_count"] < settings.max_rewrite_attempts:
        return "rewrite_query"
    return "fallback"
```

### `agent/graph.py`

Build and run the LangGraph.

Interfaces:

```python
build_graph(retriever=None)
run_agent(question: str, chat_history: list | None = None) -> dict
```

Return shape:

```python
{
    "answer": "...",
    "citations": [...],
    "retrieved_documents": [...],
    "rewritten_question": "...",
    "rewrite_count": 1,
    "is_relevant": True
}
```

## UI Design

The Gradio UI has two main areas.

Document indexing area:

- Upload multiple PDF, Markdown, or TXT files.
- Build index.
- Show indexing status and chunk count.

Question answering area:

- Ask a question.
- Show final answer.
- Show citations.
- Show rewritten question.
- Show rewrite count.
- Show retrieved chunks.

The UI calls local Python functions directly. It does not call FastAPI in the MVP.

## Evaluation Design

`evaluation/eval_questions.json` format:

```json
[
  {
    "question": "What is Agentic RAG?",
    "expected_keywords": ["agent", "retrieval", "query rewrite"],
    "expected_source": "sample.md"
  }
]
```

`evaluation/evaluate.py`:

- Loads questions.
- Calls `run_agent` for each question.
- Measures latency.
- Checks whether an answer was returned.
- Checks whether citations were returned.
- Checks whether retrieved chunks include `expected_source`.
- Counts rewrite-triggered cases.
- Prints a lightweight report.

Metrics:

- `total_questions`
- `answer_rate`
- `citation_rate`
- `source_hit_rate`
- `average_latency`
- `rewrite_triggered_count`

The MVP does not include an LLM judge.

## Error Handling

The MVP handles:

- Missing LLM configuration with a clear error.
- Empty vector store with a clear prompt to upload and index documents first.
- Unsupported file extensions.
- PDF parsing failures with file-specific messages.
- Empty retrieval results.
- Invalid grading JSON by treating the result as not relevant.

## README Plan

README sections:

```text
# Agentic RAG Document QA System

## Overview
## Why This Is Not a Naive RAG Demo
## Architecture
## Agent Workflow
## Tech Stack
## Features
## Project Structure
## Quick Start
## Environment Variables
## Usage
## Evaluation
## Example Output
## Resume Highlights
## Roadmap
```

The README should emphasize:

- This is Agentic RAG, not naive RAG.
- LangGraph explicitly models agent state transitions.
- The retriever is a tool.
- Query rewriting improves retrieval.
- Retrieval grading checks whether chunks can actually answer the question.
- Conditional retry implements simple self-correction.
- Answers are grounded in retrieved chunks and include citations.
- Evaluation is included.
- The code is modular and easy to extend.

## Implementation Phases

### Phase 1: Project Foundation

Create:

- `.env.example`
- `requirements.txt`
- README draft.
- `app.py`
- `main.py`
- `config.py`
- Package directories and `__init__.py` files.
- Placeholder `evaluation/eval_questions.json`.
- `assets/.gitkeep`.

### Phase 2: RAG Core

Implement:

- Document loader.
- Chunker.
- Embedding initialization.
- Chroma vector store.
- Retriever wrapper.

### Phase 3: Agentic Workflow

Implement:

- LangGraph state.
- Prompts.
- Retriever tool.
- Nodes.
- Conditional edges.
- Graph build/run entrypoint.

### Phase 4: Gradio UI

Implement:

- Upload and indexing controls.
- Question input.
- Answer display.
- Citations display.
- Retrieved chunks display.
- Rewrite diagnostics.

### Phase 5: Evaluation

Implement:

- Sample evaluation questions.
- Evaluation runner.
- Summary metrics.

### Phase 6: README Polish

Finalize:

- Architecture explanation.
- Agent workflow explanation.
- Run commands.
- Example usage.
- Evaluation example.
- Resume bullet points.
- Roadmap.

## Acceptance Criteria

The MVP is complete when:

- `python app.py` starts the Gradio UI.
- Users can upload supported documents and build an index.
- Users can ask questions over the indexed documents.
- The agent performs query rewrite, retrieval, grading, conditional retry, and answer generation.
- Answers include citations when grounded documents are found.
- The system returns a clear fallback when documents do not support the answer.
- `python -m evaluation.evaluate` runs a lightweight evaluation report.
- README explains the project clearly enough for GitHub and resume review.

