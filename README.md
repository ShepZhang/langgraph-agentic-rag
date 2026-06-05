# Agentic RAG Document QA System

基于 LangGraph 的 Agentic RAG 智能文档问答系统，用于面向私有知识库的 PDF / Markdown / TXT 文档问答。

This project is a lightweight Agentic RAG prototype for private document QA, focusing on explicit LangGraph-based retrieval control, query rewriting, retrieval grading, retry routing, and citation-aware answer generation. The focus is the Agentic RAG workflow, not backend authentication or production deployment infrastructure.

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

The system keeps a strict distinction between the original user question and the retrieval query. `current_query` is optimized for search; grading and answer generation still target the original user question.

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
-> if no relevant chunks and retry_count < max_retry_count: rewrite_query
-> if no relevant chunks and retry_count >= max_retry_count: fallback -> END
```

Implemented LangGraph nodes:

- `rewrite_query_node`: normalizes the first query, then uses failed retrieval context for retry rewrites.
- `retrieve_node`: calls the `retrieve_context` tool over the private Chroma index.
- `grade_documents_node`: asks the LLM for chunk-level `relevant_indices`, then filters `relevant_documents`.
- `generate_answer_node`: generates JSON answers from relevant chunks only and returns citations selected by `used_citation_indices`.
- `fallback_node`: returns a clear message when the indexed documents do not support an answer.

Key state fields:

- `current_query`: query currently used for retrieval.
- `question`: original user question. Grading and answer generation use this as the target.
- `previous_queries`: retrieval queries already attempted.
- `retrieval_attempt`: number of actual retriever calls.
- `retry_count`: number of failed-retrieval rewrites. Initial query normalization does not count as retry.
- `documents`: raw retrieved chunks.
- `relevant_documents`: chunks accepted by retrieval grading.
- `grading_reason`: LLM reason for accepting or rejecting retrieved evidence.
- `citations`: final answer evidence chunks selected by `used_citation_indices`.

## Features

- PDF, Markdown, and TXT document loading.
- Recursive chunking with source, source path, file hash, page, and chunk id metadata.
- Local sentence-transformers embeddings by default.
- Persistent Chroma vector store with rebuild-on-index strategy to avoid duplicate chunks.
- Retriever exposed as an Agent tool named `retrieve_context`.
- Query rewriting for vague or context-dependent questions.
- Chunk-level retrieval grading with conservative handling for invalid grading output.
- Conditional retry with configurable max retry count.
- Citation-aware grounded answer generation using only selected evidence chunks.
- Citation safety: normal answers without valid supporting citation indices fall back instead of returning unsupported answers.
- Gradio UI for upload, indexing, question answering, citations, retrieved chunks, and retry diagnostics.
- Lightweight evaluation runner comparing naive RAG and Agentic RAG on answer, fallback, citation, source-hit, keyword-hit, retry, relevant filtering, latency, and error metrics.

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
   - retry count
   - retrieval diagnostics

The chat LLM is required for query rewriting, retrieval grading, and answer generation. If `OPENAI_API_KEY` or `OPENAI_MODEL` is missing, the app returns a clear configuration error instead of producing offline fake answers.

## Tests

Run the test suite:

```bash
.venv/bin/python -m pytest -q
```

Tests use fake LLMs and mocked vector stores, so they do not require real OpenAI-compatible API calls.

## Evaluation

Run the lightweight evaluation script:

```bash
.venv/bin/python -m evaluation.evaluate --questions evaluation/eval_questions.json
```

By default, evaluation compares:

- `Naive RAG`: question -> retrieve -> generate.
- `Agentic RAG`: question -> rewrite -> retrieve -> grade -> retry or answer.

Reported comparison metrics:

- `answer_rate`
- `fallback_rate`
- `citation_rate`
- `source_hit_rate`
- `keyword_hit_rate`
- `fallback_correctness_rate`
- `naive_source_hit_rate`
- `agentic_source_hit_rate`
- `naive_keyword_hit_rate`
- `agentic_keyword_hit_rate`
- `naive_citation_rate`
- `agentic_citation_rate`
- `naive_fallback_correctness_rate`
- `agentic_fallback_correctness_rate`
- `naive_average_latency`
- `agentic_average_latency`

Agentic-specific metrics:

- `average_retry_count`
- `average_retrieved_docs`
- `average_relevant_docs`
- `relevant_filtering_rate`
- `rewrite_triggered_count`
- `error_count`

If the LLM config or vector index is missing, evaluation records errors per question and still prints a report.
The current evaluation set is a lightweight local QA set for demonstration. It is useful for comparing behavior, but it is not a rigorous benchmark.

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
  "current_query": "What is Agentic RAG?",
  "retry_count": 0,
  "retrieval_attempt": 1,
  "is_relevant": true
}
```

Example evaluation summary:

```text
Evaluation Report

Comparison Summary

| Metric | Naive RAG | Agentic RAG |
|---|---:|---:|
| Source Hit Rate | 0.6 | 0.8 |
| Keyword Hit Rate | 0.5 | 0.7 |
| Citation Rate | 0.55 | 0.75 |
| Fallback Correctness | 0.7 | 0.85 |
| Avg Latency | 2.1 | 4.8 |
```

## Environment Variables

- `OPENAI_API_KEY`: API key for the OpenAI-compatible LLM.
- `OPENAI_BASE_URL`: Base URL for the OpenAI-compatible API.
- `OPENAI_MODEL`: Chat model used by the agent.
- `OPENAI_TEMPERATURE`: Chat model temperature. Default is `0`.
- `EMBEDDING_PROVIDER`: Embedding backend. MVP default is `sentence_transformers`.
- `EMBEDDING_MODEL`: Local embedding model. Default is `sentence-transformers/all-MiniLM-L6-v2`.
- `CHUNK_SIZE`: Text chunk size.
- `CHUNK_OVERLAP`: Text chunk overlap.
- `TOP_K`: Number of chunks retrieved per query.
- `MAX_RETRY_COUNT`: Maximum failed-retrieval retry rewrites.
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
│   ├── baselines.py
│   ├── eval_questions.json
│   └── evaluate.py
├── docs/
│   └── design.md
├── ui/
│   └── gradio_app.py
├── assets/
│   └── architecture.png
├── sample_docs/
│   └── agentic_rag_notes.md
└── tests/
```

## Resume Highlights

- Built an Agentic RAG workflow with LangGraph, decomposing document QA into query rewriting, retrieval, retrieval grading, answer generation, and fallback nodes.
- Wrapped vector retrieval as an Agent tool so the workflow can explicitly call a private knowledge base instead of relying on model parameters alone.
- Implemented chunk-level retrieval grading and conditional retry to improve reliability on vague or poorly matched questions.
- Designed citation-aware answer generation where the model returns `used_citation_indices`, so final citations map only to evidence chunks used in the answer.
- Supported PDF, Markdown, and TXT ingestion with chunk metadata, local embeddings, Chroma indexing, and Gradio-based document QA.
- Added lightweight evaluation comparing naive RAG and Agentic RAG across source hit rate, keyword hit rate, citation rate, fallback correctness, retry behavior, and relevant chunk filtering.

## Current Limitations

- Citation currently maps generated answers to selected evidence chunks and falls back when normal answers lack valid citations, but it does not yet perform full claim-level verification.
- Retrieval grading depends on LLM JSON output. The parser is defensive, but malformed grading output is treated conservatively.
- Evaluation uses a lightweight local QA set and should be expanded with larger datasets for more rigorous benchmarking.
- The Chroma index currently uses a rebuild-on-index strategy. This avoids duplicate chunks for the MVP, but deterministic chunk IDs would be better for incremental indexing.
- The project is a prototype and is not intended for production deployment without further hardening.

## Roadmap

- RAG core implemented: loading, chunking, embeddings, Chroma indexing, and retrieval.
- LangGraph agent workflow implemented: query rewriting, retriever tool, retrieval grading, retry routing, answer generation, and fallback.
- Gradio upload and QA flow implemented: document indexing, Agentic QA, citations, retrieved chunks, and retry diagnostics.
- Evaluation runner implemented: naive-vs-agentic comparison, answer/fallback/citation/source/keyword metrics, retry metrics, and relevant filtering metrics.
- Add FastAPI API layer.
- Add Ollama local LLM support.
- Add claim-level citation verification.
- Add reranking and richer evaluation.
