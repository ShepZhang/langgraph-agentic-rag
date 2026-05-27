# Agentic RAG Document QA Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the project foundation for the Agentic RAG Document QA System: package layout, environment defaults, dependency list, README draft, and runnable entrypoints.

**Architecture:** This phase creates an independent Python project with clear package boundaries for `rag`, `agent`, `ui`, and `evaluation`. The code is intentionally lightweight but executable: configuration can be imported, the CLI can print configuration guidance, and `python app.py` can launch a minimal Gradio shell that explains the MVP flow.

**Tech Stack:** Python 3.11+, Gradio, python-dotenv, pydantic, LangGraph/LangChain/Chroma dependencies declared for subsequent phases.

---

## File Structure Map

- `.gitignore`: Keeps virtual environments, caches, local vector stores, uploads, and secrets out of git.
- `.env.example`: Documents all environment variables required by the MVP and future extension points.
- `requirements.txt`: Declares the MVP runtime dependencies.
- `README.md`: Provides the initial GitHub-facing project overview, architecture, setup, and roadmap.
- `config.py`: Reads environment variables and exposes a typed settings object.
- `app.py`: Starts the Gradio app with `python app.py`.
- `main.py`: Provides a lightweight CLI entrypoint for configuration checks.
- `rag/__init__.py`: Marks the RAG package and documents its responsibility.
- `agent/__init__.py`: Marks the Agent package and documents its responsibility.
- `ui/__init__.py`: Marks the UI package.
- `ui/gradio_app.py`: Creates the initial Gradio shell.
- `evaluation/__init__.py`: Marks the evaluation package.
- `evaluation/eval_questions.json`: Adds a small sample evaluation set shape.
- `assets/.gitkeep`: Keeps the assets directory in git until the architecture image is generated.

## Task 1: Add Project Ignore Rules

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Create `.gitignore`**

Create `.gitignore` with exactly this content:

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Virtual environments
.venv/
venv/
env/

# Environment and secrets
.env
.env.*
!.env.example

# Local runtime data
data/
uploads/
*.sqlite3
*.db

# OS/editor files
.DS_Store
.idea/
.vscode/
```

- [ ] **Step 2: Verify ignore file exists**

Run:

```bash
test -f .gitignore
```

Expected: command exits with status `0`.

- [ ] **Step 3: Commit ignore rules**

Run:

```bash
git add .gitignore
git commit -m "chore: add project gitignore"
```

Expected: git creates a commit containing `.gitignore`.

## Task 2: Add Runtime Configuration Files

**Files:**
- Create: `.env.example`
- Create: `requirements.txt`
- Create: `config.py`

- [ ] **Step 1: Create `.env.example`**

Create `.env.example` with exactly this content:

```bash
# OpenAI-compatible chat LLM.
# Required for query rewriting, retrieval grading, and answer generation.
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

# Embeddings.
# MVP default runs sentence-transformers locally.
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Chunking.
CHUNK_SIZE=800
CHUNK_OVERLAP=120

# Retrieval.
TOP_K=4
MAX_REWRITE_ATTEMPTS=2

# Vector store.
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION_NAME=agentic_rag_documents

# Gradio.
GRADIO_SERVER_NAME=127.0.0.1
GRADIO_SERVER_PORT=7860
```

- [ ] **Step 2: Create `requirements.txt`**

Create `requirements.txt` with exactly this content:

```text
langgraph>=0.2.60
langchain>=0.3.14
langchain-core>=0.3.29
langchain-openai>=0.2.14
langchain-community>=0.3.14
langchain-text-splitters>=0.3.4
langchain-chroma>=0.2.0
chromadb>=0.5.23
sentence-transformers>=3.3.1
pypdf>=5.1.0
python-dotenv>=1.0.1
pydantic>=2.10.4
gradio>=5.9.1
pytest>=8.3.4
```

- [ ] **Step 3: Create `config.py`**

Create `config.py` with exactly this content:

```python
"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Typed runtime settings for the Agentic RAG MVP."""

    openai_api_key: str
    openai_base_url: str
    openai_model: str
    embedding_provider: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_rewrite_attempts: int
    chroma_persist_dir: Path
    chroma_collection_name: str
    gradio_server_name: str
    gradio_server_port: int

    @property
    def has_llm_config(self) -> bool:
        """Return True when the required chat LLM settings are present."""

        return bool(self.openai_api_key and self.openai_model)

    def require_llm_config(self) -> None:
        """Raise a clear error if the chat LLM is not configured."""

        if not self.has_llm_config:
            raise RuntimeError(
                "Missing LLM configuration. Set OPENAI_API_KEY and OPENAI_MODEL "
                "in your environment or .env file before running Agentic RAG."
            )


def _get_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc


def get_settings() -> Settings:
    """Load settings from environment variables."""

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "sentence_transformers").strip(),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ).strip(),
        chunk_size=_get_int("CHUNK_SIZE", 800),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 120),
        top_k=_get_int("TOP_K", 4),
        max_rewrite_attempts=_get_int("MAX_REWRITE_ATTEMPTS", 2),
        chroma_persist_dir=Path(os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")),
        chroma_collection_name=os.getenv(
            "CHROMA_COLLECTION_NAME",
            "agentic_rag_documents",
        ).strip(),
        gradio_server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1").strip(),
        gradio_server_port=_get_int("GRADIO_SERVER_PORT", 7860),
    )
```

- [ ] **Step 4: Verify configuration imports**

Run:

```bash
python -c "from config import get_settings; s = get_settings(); print(s.embedding_model); print(s.top_k)"
```

Expected output includes:

```text
sentence-transformers/all-MiniLM-L6-v2
4
```

- [ ] **Step 5: Commit runtime configuration**

Run:

```bash
git add .env.example requirements.txt config.py
git commit -m "chore: add runtime configuration"
```

Expected: git creates a commit containing `.env.example`, `requirements.txt`, and `config.py`.

## Task 3: Add Package Directories

**Files:**
- Create: `rag/__init__.py`
- Create: `agent/__init__.py`
- Create: `ui/__init__.py`
- Create: `evaluation/__init__.py`
- Create: `assets/.gitkeep`

- [ ] **Step 1: Create package directories and files**

Create the package files with exactly this content.

`rag/__init__.py`:

```python
"""RAG components for loading, chunking, embedding, indexing, and retrieval."""
```

`agent/__init__.py`:

```python
"""LangGraph agent workflow for query rewriting, retrieval grading, and answering."""
```

`ui/__init__.py`:

```python
"""User interface components for the Agentic RAG document QA system."""
```

`evaluation/__init__.py`:

```python
"""Evaluation utilities for measuring Agentic RAG behavior."""
```

`assets/.gitkeep`:

```text
```

- [ ] **Step 2: Verify package files exist**

Run:

```bash
test -f rag/__init__.py
test -f agent/__init__.py
test -f ui/__init__.py
test -f evaluation/__init__.py
test -f assets/.gitkeep
```

Expected: all commands exit with status `0`.

- [ ] **Step 3: Commit package directories**

Run:

```bash
git add rag agent ui evaluation assets
git commit -m "chore: add project package structure"
```

Expected: git creates a commit containing package markers and `assets/.gitkeep`.

## Task 4: Add Entrypoints and Initial Gradio Shell

**Files:**
- Create: `app.py`
- Create: `main.py`
- Create: `ui/gradio_app.py`

- [ ] **Step 1: Create `ui/gradio_app.py`**

Create `ui/gradio_app.py` with exactly this content:

```python
"""Gradio UI entrypoint for the Agentic RAG document QA system."""

from __future__ import annotations

import gradio as gr

from config import get_settings


def create_app() -> gr.Blocks:
    """Create the Gradio interface."""

    settings = get_settings()

    with gr.Blocks(title="Agentic RAG Document QA System") as demo:
        gr.Markdown(
            """
            # Agentic RAG Document QA System

            This project will support PDF, Markdown, and TXT upload, local
            vector indexing, LangGraph-based query rewriting, retrieval grading,
            conditional retries, and citation-aware answers.
            """
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Document Indexing")
                gr.File(
                    label="Upload documents",
                    file_count="multiple",
                    file_types=[".pdf", ".md", ".markdown", ".txt"],
                )
                gr.Button("Build Index", interactive=False)
                gr.Textbox(
                    label="Index status",
                    value="RAG indexing will be enabled in the RAG core phase.",
                    interactive=False,
                )

            with gr.Column():
                gr.Markdown("## Question Answering")
                gr.Textbox(label="Question", lines=3)
                gr.Button("Ask", interactive=False)
                gr.Textbox(
                    label="Answer",
                    value=(
                        "Agentic answering will be enabled after the LangGraph "
                        "workflow is implemented."
                    ),
                    lines=6,
                    interactive=False,
                )

        gr.Markdown(
            f"""
            **Current configuration**

            - LLM model: `{settings.openai_model}`
            - Embedding model: `{settings.embedding_model}`
            - Chroma path: `{settings.chroma_persist_dir}`
            - Top K: `{settings.top_k}`
            - Max rewrite attempts: `{settings.max_rewrite_attempts}`
            """
        )

    return demo
```

- [ ] **Step 2: Create `app.py`**

Create `app.py` with exactly this content:

```python
"""Run the Gradio application."""

from __future__ import annotations

from config import get_settings
from ui.gradio_app import create_app


def main() -> None:
    """Launch the Gradio UI."""

    settings = get_settings()
    app = create_app()
    app.launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create `main.py`**

Create `main.py` with exactly this content:

```python
"""Command-line helper for the Agentic RAG document QA system."""

from __future__ import annotations

from config import get_settings


def main() -> None:
    """Print a concise configuration summary."""

    settings = get_settings()
    print("Agentic RAG Document QA System")
    print(f"LLM model: {settings.openai_model}")
    print(f"LLM configured: {settings.has_llm_config}")
    print(f"Embedding model: {settings.embedding_model}")
    print(f"Chroma path: {settings.chroma_persist_dir}")
    print(f"Top K: {settings.top_k}")
    print(f"Max rewrite attempts: {settings.max_rewrite_attempts}")
    print("Run `python app.py` to start the Gradio UI.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify Python files compile**

Run:

```bash
python -m compileall app.py main.py config.py rag agent ui evaluation
```

Expected output includes successful compilation and no syntax errors.

- [ ] **Step 5: Verify CLI output**

Run:

```bash
python main.py
```

Expected output includes:

```text
Agentic RAG Document QA System
Embedding model: sentence-transformers/all-MiniLM-L6-v2
Run `python app.py` to start the Gradio UI.
```

- [ ] **Step 6: Commit entrypoints**

Run:

```bash
git add app.py main.py ui/gradio_app.py
git commit -m "feat: add app entrypoints"
```

Expected: git creates a commit containing `app.py`, `main.py`, and `ui/gradio_app.py`.

## Task 5: Add Evaluation Dataset Shape

**Files:**
- Create: `evaluation/eval_questions.json`

- [ ] **Step 1: Create `evaluation/eval_questions.json`**

Create `evaluation/eval_questions.json` with exactly this content:

```json
[
  {
    "question": "What is Agentic RAG?",
    "expected_keywords": ["agent", "retrieval", "query rewrite"],
    "expected_source": "sample.md"
  },
  {
    "question": "How does retrieval grading improve RAG reliability?",
    "expected_keywords": ["relevant", "grading", "retrieved chunks"],
    "expected_source": "sample.md"
  }
]
```

- [ ] **Step 2: Verify JSON parses**

Run:

```bash
python -m json.tool evaluation/eval_questions.json
```

Expected: formatted JSON is printed and the command exits with status `0`.

- [ ] **Step 3: Commit evaluation dataset shape**

Run:

```bash
git add evaluation/eval_questions.json
git commit -m "chore: add evaluation question examples"
```

Expected: git creates a commit containing `evaluation/eval_questions.json`.

## Task 6: Add README Draft

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create `README.md`**

Create `README.md` with exactly this content:

```markdown
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
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
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
```

- [ ] **Step 2: Verify README exists**

Run:

```bash
test -f README.md
```

Expected: command exits with status `0`.

- [ ] **Step 3: Commit README draft**

Run:

```bash
git add README.md
git commit -m "docs: add project readme draft"
```

Expected: git creates a commit containing `README.md`.

## Task 7: Final Foundation Verification

**Files:**
- Read: all files created in Tasks 1-6

- [ ] **Step 1: Inspect git status**

Run:

```bash
git status --short
```

Expected output:

```text
```

- [ ] **Step 2: Run syntax and JSON verification**

Run:

```bash
python -m compileall app.py main.py config.py rag agent ui evaluation
python -m json.tool evaluation/eval_questions.json >/tmp/agentic-rag-eval-json-check.txt
```

Expected: both commands exit with status `0`.

- [ ] **Step 3: Run configuration smoke test**

Run:

```bash
python main.py
```

Expected output includes:

```text
Agentic RAG Document QA System
LLM model: gpt-4o-mini
Embedding model: sentence-transformers/all-MiniLM-L6-v2
Run `python app.py` to start the Gradio UI.
```

- [ ] **Step 4: Record foundation completion**

Run:

```bash
git log --oneline -6
```

Expected: recent commits include:

```text
docs: add project readme draft
chore: add evaluation question examples
feat: add app entrypoints
chore: add project package structure
chore: add runtime configuration
chore: add project gitignore
```

