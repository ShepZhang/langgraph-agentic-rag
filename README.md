# Agentic RAG Document QA System

基于 LangGraph 的 Agentic RAG 智能文档问答系统，用于面向私有知识库的 PDF / Markdown / TXT 文档问答。

## Overview

This project builds a lightweight Agentic RAG system for private document QA. Users upload documents, build a local vector index, ask questions, and receive grounded answers with source citations.

The MVP focuses on the agent workflow rather than backend infrastructure. It uses LangGraph to model query rewriting, retrieval, document grading, conditional retry, and answer generation as explicit workflow nodes.

## Why This Is Not a Naive RAG Demo

Naive RAG usually follows a fixed path:

```text
question -> retrieve -> generate answer
```

This project follows an Agentic RAG path:

```text
question
-> query rewrite
-> retriever tool
-> retrieval grading
-> conditional retry
-> grounded answer with citations
```

The Agent can inspect whether retrieved chunks are actually useful. If not, it rewrites the query and retrieves again before falling back.

## Architecture

```text
UI Layer
  Gradio document upload and QA interface

RAG Layer
  loader -> chunker -> embeddings -> vectorstore -> retriever

Agent Layer
  LangGraph state -> nodes -> conditional edges -> final answer

Evaluation Layer
  JSON questions -> evaluation runner -> summary metrics
```

## Tech Stack

- Python 3.11+
- LangGraph
- LangChain
- ChromaDB
- sentence-transformers
- OpenAI-compatible chat LLM
- Gradio
- python-dotenv
- pydantic

## Quick Start

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Create your environment file:

```bash
cp .env.example .env
```

Edit `.env` and set:

```bash
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

Start the UI:

```bash
python app.py
```

## Environment Variables

- `OPENAI_API_KEY`: API key for the OpenAI-compatible LLM.
- `OPENAI_BASE_URL`: Base URL for the OpenAI-compatible API.
- `OPENAI_MODEL`: Chat model used by the agent.
- `EMBEDDING_PROVIDER`: Embedding backend. MVP default is `sentence_transformers`.
- `EMBEDDING_MODEL`: Local embedding model.
- `CHUNK_SIZE`: Text chunk size.
- `CHUNK_OVERLAP`: Text chunk overlap.
- `TOP_K`: Number of chunks retrieved per query.
- `MAX_REWRITE_ATTEMPTS`: Maximum rewrite/retrieve attempts.
- `CHROMA_PERSIST_DIR`: Local Chroma persistence path.
- `CHROMA_COLLECTION_NAME`: Chroma collection name.

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
├── agent/
├── evaluation/
├── ui/
├── assets/
└── docs/
```

## Resume Highlights

- Built an Agentic RAG workflow with LangGraph, decomposing document QA into query rewriting, retrieval, retrieval grading, answer generation, and fallback nodes.
- Wrapped vector retrieval as an Agent tool so the workflow can explicitly call a private knowledge base.
- Implemented retrieval grading and conditional retry to improve reliability on vague or poorly matched questions.
- Designed citation-aware answer generation to ground answers in retrieved document chunks.
- Added lightweight evaluation to measure answer rate, citation rate, source hit rate, latency, and rewrite behavior.

## Roadmap

- Implement RAG loading, chunking, embeddings, Chroma indexing, and retrieval.
- Implement LangGraph agent workflow.
- Implement full Gradio upload and QA flow.
- Implement evaluation runner.
- Add FastAPI API layer.
- Add Ollama local LLM support.
- Add reranking and richer evaluation.
