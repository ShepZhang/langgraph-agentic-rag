# Agentic RAG Document QA System

基于 LangGraph 的 Agentic RAG 智能文档问答系统，用于面向私有知识库的 PDF / Markdown / TXT 文档问答。

This project is a lightweight, resume-ready Agentic RAG system. Users can upload private documents, build a local vector index, ask questions, and receive grounded answers with source citations. The focus is the Agentic RAG workflow, not backend authentication or production deployment infrastructure.

## Why This Is Not a Naive RAG Demo

A naive RAG pipeline usually follows one fixed path:

```text
question -> retrieve -> generate answer
```

This project makes the retrieval process agentic:

```text
question
-> query rewrite
-> retriever tool
-> retrieval grading
-> conditional retry
-> grounded answer with citations
```

The agent checks whether retrieved chunks can actually answer the question. If they are not relevant enough, it rewrites the query and retries retrieval before falling back with a clear unable-to-answer response.

## Architecture

![Architecture](assets/architecture.png)

```text
UI Layer
  Gradio document upload, indexing, QA, citations, retrieved chunks, diagnostics

RAG Layer
  loader -> chunker -> embeddings -> Chroma vector store -> retriever

Agent Layer
  LangGraph state -> nodes -> conditional edges -> answer or fallback

Evaluation Layer
  eval_questions.json -> evaluation runner -> summary metrics
```

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

Implemented LangGraph nodes:

- `rewrite_query_node`: rewrites contextual or vague questions into standalone retrieval queries.
- `retrieve_node`: calls the `retrieve_context` tool over the private Chroma index.
- `grade_documents_node`: asks the LLM whether retrieved chunks are sufficient.
- `generate_answer_node`: generates citation-aware answers from retrieved chunks only.
- `fallback_node`: returns a clear message when the indexed documents do not support an answer.

## Features

- PDF, Markdown, and TXT document loading.
- Recursive chunking with source, page, and chunk id metadata.
- Local sentence-transformers embeddings by default.
- Persistent Chroma vector store.
- Retriever exposed as an Agent tool named `retrieve_context`.
- Query rewriting for vague or context-dependent questions.
- Retrieval grading with conservative handling for invalid grading output.
- Conditional retry with configurable max rewrite attempts.
- Citation-aware grounded answer generation.
- Gradio UI for upload, indexing, question answering, citations, retrieved chunks, and rewrite diagnostics.
- Lightweight evaluation runner for answer, citation, source-hit, keyword-hit, latency, and rewrite metrics.

## Tech Stack

- Python 3.11+
- LangGraph
- LangChain
- ChromaDB
- sentence-transformers
- OpenAI-compatible chat LLM
- Gradio
- python-dotenv
- pytest

## Quick Start

Create a virtual environment:

```bash
python3 -m venv .venv
```

Install dependencies:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Create an environment file:

```bash
cp .env.example .env
```

Set your OpenAI-compatible LLM config in `.env`:

```bash
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

Start the Gradio UI:

```bash
.venv/bin/python app.py
```

If you activate the virtual environment first, `python app.py` also works:

```bash
source .venv/bin/activate
python app.py
```

## Usage

1. Open the Gradio URL printed by `app.py`.
2. Upload one or more `.pdf`, `.md`, `.markdown`, or `.txt` files.
3. Click `Build Index`.
4. Ask a question about the indexed documents.
5. Inspect:
   - answer
   - citations
   - retrieved chunks
   - rewritten question
   - rewrite count
   - retrieval diagnostics

The chat LLM is required for query rewriting, retrieval grading, and answer generation. If `OPENAI_API_KEY` or `OPENAI_MODEL` is missing, the app returns a clear configuration error instead of producing offline fake answers.

## Evaluation

Run the lightweight evaluation script:

```bash
.venv/bin/python -m evaluation.evaluate --questions evaluation/eval_questions.json
```

Reported metrics:

- `answer_rate`
- `citation_rate`
- `source_hit_rate`
- `keyword_hit_rate`
- `average_latency`
- `rewrite_triggered_count`
- `error_count`

If the LLM config or vector index is missing, evaluation records errors per question and still prints a report.

## Example Output

Example answer payload:

```json
{
  "answer": "Agentic RAG uses query rewriting and retrieval grading to improve document QA.",
  "citations": [
    {
      "source": "notes.md",
      "page": null,
      "chunk_id": "notes.md:pNA:c1",
      "score": 0.91
    }
  ],
  "rewritten_question": "What is Agentic RAG?",
  "rewrite_count": 1,
  "is_relevant": true
}
```

Example evaluation summary:

```text
Evaluation Report

Summary
total_questions: 2
answer_rate: 1.0
citation_rate: 1.0
source_hit_rate: 1.0
average_latency: 1.2345
rewrite_triggered_count: 1
keyword_hit_rate: 0.5
error_count: 0
```

## Environment Variables

- `OPENAI_API_KEY`: API key for the OpenAI-compatible LLM.
- `OPENAI_BASE_URL`: Base URL for the OpenAI-compatible API.
- `OPENAI_MODEL`: Chat model used by the agent.
- `EMBEDDING_PROVIDER`: Embedding backend. MVP default is `sentence_transformers`.
- `EMBEDDING_MODEL`: Local embedding model. Default is `sentence-transformers/all-MiniLM-L6-v2`.
- `CHUNK_SIZE`: Text chunk size.
- `CHUNK_OVERLAP`: Text chunk overlap.
- `TOP_K`: Number of chunks retrieved per query.
- `MAX_REWRITE_ATTEMPTS`: Maximum rewrite and retrieve attempts.
- `CHROMA_PERSIST_DIR`: Local Chroma persistence path.
- `CHROMA_COLLECTION_NAME`: Chroma collection name.
- `GRADIO_SERVER_NAME`: Gradio host.
- `GRADIO_SERVER_PORT`: Gradio port.

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
│   ├── loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   └── retriever.py
├── agent/
│   ├── graph.py
│   ├── state.py
│   ├── nodes.py
│   ├── edges.py
│   ├── tools.py
│   └── prompts.py
├── evaluation/
│   ├── eval_questions.json
│   └── evaluate.py
├── ui/
│   └── gradio_app.py
├── assets/
│   └── architecture.png
└── tests/
```

## Resume Highlights

- Built an Agentic RAG workflow with LangGraph, decomposing document QA into query rewriting, retrieval, retrieval grading, answer generation, and fallback nodes.
- Wrapped vector retrieval as an Agent tool so the workflow can explicitly call a private knowledge base instead of relying on model parameters alone.
- Implemented retrieval grading and conditional retry to improve reliability on vague or poorly matched questions.
- Designed citation-aware answer generation that requires answers to be grounded in retrieved chunks.
- Supported PDF, Markdown, and TXT ingestion with chunk metadata, local embeddings, Chroma indexing, and Gradio-based document QA.
- Added lightweight evaluation for answer rate, citation rate, source hit rate, latency, keyword hit rate, and rewrite behavior.

## Roadmap

- RAG core implemented: loading, chunking, embeddings, Chroma indexing, and retrieval.
- LangGraph agent workflow implemented: query rewriting, retriever tool, retrieval grading, retry routing, answer generation, and fallback.
- Gradio upload and QA flow implemented: document indexing, Agentic QA, citations, retrieved chunks, and rewrite diagnostics.
- Evaluation runner implemented: answer rate, citation rate, source hit rate, latency, keyword hit rate, and rewrite-trigger metrics.
- Add FastAPI API layer.
- Add Ollama local LLM support.
- Add reranking and richer evaluation.
