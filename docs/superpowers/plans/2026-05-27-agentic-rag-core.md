# Agentic RAG Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the RAG core for loading documents, splitting chunks, initializing embeddings, managing Chroma, and returning normalized retrieval results.

**Architecture:** The RAG layer is split into focused modules: `loader.py` converts files into LangChain `Document` objects, `chunker.py` creates chunked `Document` objects with stable metadata, `embeddings.py` owns embedding initialization, `vectorstore.py` wraps Chroma behind a small manager, and `retriever.py` normalizes search results for the Agent/UI layers. Tests are written first for each module and use dependency injection where external libraries would otherwise make tests slow or brittle.

**Tech Stack:** Python 3.11+, pytest, LangChain `Document`, `RecursiveCharacterTextSplitter`, HuggingFace sentence-transformers embeddings, ChromaDB.

---

## File Structure Map

- `tests/test_loader.py`: Unit tests for TXT, Markdown, PDF extraction with injected reader, and unsupported extensions.
- `tests/test_chunker.py`: Unit tests for recursive splitting and stable `chunk_id` metadata.
- `tests/test_embeddings.py`: Unit tests for embedding provider selection without loading a real model.
- `tests/test_vectorstore.py`: Unit tests for Chroma manager behavior using an injected fake Chroma class.
- `tests/test_retriever.py`: Unit tests for normalized retrieval output using a fake vector store manager.
- `rag/loader.py`: Loads PDF, Markdown, and TXT files into LangChain `Document` objects.
- `rag/chunker.py`: Splits documents and assigns `chunk_id`.
- `rag/embeddings.py`: Initializes local sentence-transformers embeddings by default.
- `rag/vectorstore.py`: Creates, loads, updates, and searches a persistent Chroma store.
- `rag/retriever.py`: Converts vector store search results into dictionaries with `content`, `source`, `page`, `chunk_id`, and `score`.

## Task 0: Prepare RAG Core Execution Environment

**Files:**
- Read: `requirements.txt`

- [ ] **Step 1: Verify Python version**

Run:

```bash
/Users/shep/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 --version
```

Expected: Python version is `3.11` or newer.

- [ ] **Step 2: Create virtual environment when missing**

Run:

```bash
test -d .venv || /Users/shep/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m venv .venv
```

Expected: `.venv` exists.

- [ ] **Step 3: Install dependencies**

Run:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Expected: dependencies install successfully. If the command fails because network access is blocked, rerun the same command with escalated network permission.

- [ ] **Step 4: Verify pytest imports**

Run:

```bash
.venv/bin/python -c "import pytest, langchain_core; print('rag test dependencies ready')"
```

Expected output:

```text
rag test dependencies ready
```

## Task 1: Document Loader

**Files:**
- Create: `tests/test_loader.py`
- Create: `rag/loader.py`

- [ ] **Step 1: Write failing loader tests**

Create `tests/test_loader.py` with exactly this content:

```python
"""Tests for document loading."""

from __future__ import annotations

import pytest

from rag.loader import UnsupportedFileTypeError, load_documents, load_pdf_document


def test_load_text_and_markdown_documents_preserves_metadata(tmp_path):
    txt_path = tmp_path / "notes.txt"
    md_path = tmp_path / "guide.md"
    txt_path.write_text("plain text content", encoding="utf-8")
    md_path.write_text("# Guide\n\nmarkdown content", encoding="utf-8")

    docs = load_documents([txt_path, md_path])

    assert [doc.page_content for doc in docs] == [
        "plain text content",
        "# Guide\n\nmarkdown content",
    ]
    assert docs[0].metadata == {"source": "notes.txt", "page": None}
    assert docs[1].metadata == {"source": "guide.md", "page": None}


def test_load_pdf_document_with_injected_reader_preserves_page_metadata(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")

    class FakePage:
        def __init__(self, text: str):
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class FakeReader:
        def __init__(self, _path):
            self.pages = [FakePage("page one"), FakePage(""), FakePage("page three")]

    docs = load_pdf_document(pdf_path, reader_cls=FakeReader)

    assert [doc.page_content for doc in docs] == ["page one", "page three"]
    assert docs[0].metadata == {"source": "paper.pdf", "page": 1}
    assert docs[1].metadata == {"source": "paper.pdf", "page": 3}


def test_load_documents_rejects_unsupported_file_type(tmp_path):
    csv_path = tmp_path / "table.csv"
    csv_path.write_text("a,b\n1,2", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError, match="Unsupported file type"):
        load_documents([csv_path])
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_loader.py -v
```

Expected: tests fail because `rag.loader` does not exist or loader functions are missing.

- [ ] **Step 3: Implement document loader**

Create `rag/loader.py` with exactly this content:

```python
"""Document loading utilities for PDF, Markdown, and TXT files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol

from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}


class UnsupportedFileTypeError(ValueError):
    """Raised when a document extension is not supported."""


class PdfReaderProtocol(Protocol):
    """Minimal PDF reader protocol used for dependency injection in tests."""

    pages: list[object]


def load_documents(file_paths: Iterable[str | Path]) -> list[Document]:
    """Load supported files into LangChain Documents."""

    documents: list[Document] = []
    for file_path in file_paths:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            documents.extend(load_pdf_document(path))
        elif suffix in {".md", ".markdown", ".txt"}:
            documents.append(load_text_document(path))
        else:
            raise UnsupportedFileTypeError(
                f"Unsupported file type {suffix!r}. Supported types: "
                f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

    return documents


def load_text_document(file_path: str | Path) -> Document:
    """Load a Markdown or TXT file as a single document."""

    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return Document(
        page_content=text,
        metadata={"source": path.name, "page": None},
    )


def load_pdf_document(
    file_path: str | Path,
    reader_cls: type[PdfReaderProtocol] | None = None,
) -> list[Document]:
    """Load a PDF file as one document per non-empty page."""

    path = Path(file_path)
    if reader_cls is None:
        from pypdf import PdfReader

        reader_cls = PdfReader

    reader = reader_cls(path)
    documents: list[Document] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "page": page_number},
            )
        )

    return documents
```

- [ ] **Step 4: Run loader tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_loader.py -v
```

Expected: all loader tests pass.

- [ ] **Step 5: Commit loader**

Run:

```bash
git add tests/test_loader.py rag/loader.py
git commit -m "feat: add document loader"
```

Expected: git creates a commit containing loader tests and implementation.

## Task 2: Document Chunker

**Files:**
- Create: `tests/test_chunker.py`
- Create: `rag/chunker.py`

- [ ] **Step 1: Write failing chunker tests**

Create `tests/test_chunker.py` with exactly this content:

```python
"""Tests for document chunking."""

from __future__ import annotations

from langchain_core.documents import Document

from rag.chunker import split_documents


def test_split_documents_preserves_metadata_and_adds_chunk_ids():
    docs = [
        Document(
            page_content="Alpha beta gamma. " * 20,
            metadata={"source": "notes.md", "page": None},
        )
    ]

    chunks = split_documents(docs, chunk_size=80, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "notes.md" for chunk in chunks)
    assert all(chunk.metadata["page"] is None for chunk in chunks)
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == [
        f"notes.md:pNA:c{index}" for index in range(1, len(chunks) + 1)
    ]


def test_split_documents_uses_page_number_in_chunk_id():
    docs = [
        Document(
            page_content="One two three. " * 12,
            metadata={"source": "paper.pdf", "page": 4},
        )
    ]

    chunks = split_documents(docs, chunk_size=60, chunk_overlap=5)

    assert chunks[0].metadata["chunk_id"] == "paper.pdf:p4:c1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_chunker.py -v
```

Expected: tests fail because `rag.chunker` does not exist or `split_documents` is missing.

- [ ] **Step 3: Implement chunker**

Create `rag/chunker.py` with exactly this content:

```python
"""Text splitting utilities for RAG indexing."""

from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split documents into chunks while preserving source metadata."""

    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
    )

    chunks = splitter.split_documents(documents)
    counters: defaultdict[tuple[str, str], int] = defaultdict(int)

    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        page = chunk.metadata.get("page")
        page_label = f"p{page}" if page is not None else "pNA"
        key = (source, page_label)
        counters[key] += 1
        chunk.metadata["chunk_id"] = f"{source}:{page_label}:c{counters[key]}"

    return chunks
```

- [ ] **Step 4: Run chunker tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_chunker.py -v
```

Expected: all chunker tests pass.

- [ ] **Step 5: Commit chunker**

Run:

```bash
git add tests/test_chunker.py rag/chunker.py
git commit -m "feat: add document chunker"
```

Expected: git creates a commit containing chunker tests and implementation.

## Task 3: Embedding Model Factory

**Files:**
- Create: `tests/test_embeddings.py`
- Create: `rag/embeddings.py`

- [ ] **Step 1: Write failing embedding tests**

Create `tests/test_embeddings.py` with exactly this content:

```python
"""Tests for embedding model initialization."""

from __future__ import annotations

from dataclasses import replace

import pytest

from config import get_settings
from rag import embeddings
from rag.embeddings import UnsupportedEmbeddingProviderError, get_embedding_model


def test_get_embedding_model_uses_sentence_transformers_provider(monkeypatch):
    settings = replace(
        get_settings(),
        embedding_provider="sentence_transformers",
        embedding_model="sentence-transformers/test-model",
    )
    calls = {}

    class FakeEmbeddings:
        def __init__(self, model_name: str):
            calls["model_name"] = model_name

    monkeypatch.setattr(
        embeddings,
        "_load_huggingface_embeddings_class",
        lambda: FakeEmbeddings,
    )

    model = get_embedding_model(settings)

    assert isinstance(model, FakeEmbeddings)
    assert calls["model_name"] == "sentence-transformers/test-model"


def test_get_embedding_model_rejects_unsupported_provider():
    settings = replace(get_settings(), embedding_provider="unsupported")

    with pytest.raises(UnsupportedEmbeddingProviderError, match="Unsupported embedding provider"):
        get_embedding_model(settings)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_embeddings.py -v
```

Expected: tests fail because `rag.embeddings` does not exist or functions are missing.

- [ ] **Step 3: Implement embedding factory**

Create `rag/embeddings.py` with exactly this content:

```python
"""Embedding model initialization."""

from __future__ import annotations

from typing import Any

from config import Settings, get_settings


class UnsupportedEmbeddingProviderError(ValueError):
    """Raised when the configured embedding provider is unsupported."""


def get_embedding_model(settings: Settings | None = None) -> Any:
    """Create the configured embedding model."""

    resolved_settings = settings or get_settings()
    provider = resolved_settings.embedding_provider.lower()

    if provider in {"sentence_transformers", "huggingface", "local"}:
        embeddings_cls = _load_huggingface_embeddings_class()
        return embeddings_cls(model_name=resolved_settings.embedding_model)

    raise UnsupportedEmbeddingProviderError(
        f"Unsupported embedding provider {resolved_settings.embedding_provider!r}. "
        "Supported provider: sentence_transformers"
    )


def _load_huggingface_embeddings_class() -> type[Any]:
    """Load the HuggingFace embeddings class from available LangChain packages."""

    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings
```

- [ ] **Step 4: Run embedding tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_embeddings.py -v
```

Expected: all embedding tests pass.

- [ ] **Step 5: Commit embeddings**

Run:

```bash
git add tests/test_embeddings.py rag/embeddings.py
git commit -m "feat: add embedding model factory"
```

Expected: git creates a commit containing embedding tests and implementation.

## Task 4: Chroma Vector Store Manager

**Files:**
- Create: `tests/test_vectorstore.py`
- Create: `rag/vectorstore.py`

- [ ] **Step 1: Write failing vector store tests**

Create `tests/test_vectorstore.py` with exactly this content:

```python
"""Tests for vector store management."""

from __future__ import annotations

from dataclasses import replace

from langchain_core.documents import Document

from config import get_settings
from rag import vectorstore
from rag.vectorstore import VectorStoreManager


class FakeChroma:
    created_with = None

    def __init__(self, collection_name, embedding_function, persist_directory):
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        self.documents = []
        self.queries = []

    @classmethod
    def from_documents(cls, documents, embedding, collection_name, persist_directory):
        instance = cls(collection_name, embedding, persist_directory)
        instance.documents.extend(documents)
        cls.created_with = {
            "documents": documents,
            "embedding": embedding,
            "collection_name": collection_name,
            "persist_directory": persist_directory,
        }
        return instance

    def add_documents(self, documents):
        self.documents.extend(documents)
        return ["id-1"]

    def similarity_search_with_relevance_scores(self, query, k):
        self.queries.append((query, k))
        return [(Document(page_content="context", metadata={"source": "a.md"}), 0.75)]


def test_create_vectorstore_uses_configured_chroma_settings(tmp_path, monkeypatch):
    settings = replace(
        get_settings(),
        chroma_persist_dir=tmp_path / "chroma",
        chroma_collection_name="test_collection",
    )
    embedding_model = object()
    monkeypatch.setattr(vectorstore, "_load_chroma_class", lambda: FakeChroma)
    manager = VectorStoreManager(settings=settings, embedding_model=embedding_model)
    docs = [Document(page_content="hello", metadata={"source": "a.md"})]

    store = manager.create_vectorstore(docs)

    assert store is manager.store
    assert FakeChroma.created_with["documents"] == docs
    assert FakeChroma.created_with["embedding"] is embedding_model
    assert FakeChroma.created_with["collection_name"] == "test_collection"
    assert FakeChroma.created_with["persist_directory"] == str(tmp_path / "chroma")


def test_similarity_search_loads_store_and_returns_scored_documents(tmp_path, monkeypatch):
    settings = replace(get_settings(), chroma_persist_dir=tmp_path / "chroma")
    monkeypatch.setattr(vectorstore, "_load_chroma_class", lambda: FakeChroma)
    manager = VectorStoreManager(settings=settings, embedding_model=object())

    results = manager.similarity_search("question", top_k=3)

    assert len(results) == 1
    assert results[0][0].page_content == "context"
    assert results[0][1] == 0.75
    assert manager.store.queries == [("question", 3)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_vectorstore.py -v
```

Expected: tests fail because `rag.vectorstore` does not exist or `VectorStoreManager` is missing.

- [ ] **Step 3: Implement vector store manager**

Create `rag/vectorstore.py` with exactly this content:

```python
"""Chroma vector store management."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from config import Settings, get_settings
from rag.embeddings import get_embedding_model


class EmptyVectorStoreError(RuntimeError):
    """Raised when a search is requested before a vector store is available."""


class VectorStoreManager:
    """Small wrapper around a persistent Chroma vector store."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedding_model: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedding_model = embedding_model or get_embedding_model(self.settings)
        self.store: Any | None = None

    def create_vectorstore(self, docs: list[Document]) -> Any:
        """Create a persistent Chroma store from documents."""

        self.settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        chroma_cls = _load_chroma_class()
        self.store = chroma_cls.from_documents(
            documents=docs,
            embedding=self.embedding_model,
            collection_name=self.settings.chroma_collection_name,
            persist_directory=str(self.settings.chroma_persist_dir),
        )
        return self.store

    def load_vectorstore(self) -> Any:
        """Load the configured persistent Chroma store."""

        if self.store is not None:
            return self.store

        chroma_cls = _load_chroma_class()
        self.store = chroma_cls(
            collection_name=self.settings.chroma_collection_name,
            embedding_function=self.embedding_model,
            persist_directory=str(self.settings.chroma_persist_dir),
        )
        return self.store

    def add_documents(self, docs: list[Document]) -> Any:
        """Add documents to the configured vector store."""

        store = self.load_vectorstore()
        return store.add_documents(docs)

    def similarity_search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[tuple[Document, float | None]]:
        """Run similarity search and return documents with optional scores."""

        store = self.load_vectorstore()
        k = top_k or self.settings.top_k

        if hasattr(store, "similarity_search_with_relevance_scores"):
            return store.similarity_search_with_relevance_scores(query, k=k)

        docs = store.similarity_search(query, k=k)
        return [(doc, None) for doc in docs]


_default_manager: VectorStoreManager | None = None


def get_vectorstore_manager() -> VectorStoreManager:
    """Return the process-level vector store manager."""

    global _default_manager
    if _default_manager is None:
        _default_manager = VectorStoreManager()
    return _default_manager


def create_vectorstore(docs: list[Document]) -> Any:
    """Create the default vector store from documents."""

    return get_vectorstore_manager().create_vectorstore(docs)


def load_vectorstore() -> Any:
    """Load the default vector store."""

    return get_vectorstore_manager().load_vectorstore()


def add_documents(docs: list[Document]) -> Any:
    """Add documents to the default vector store."""

    return get_vectorstore_manager().add_documents(docs)


def similarity_search(
    query: str,
    top_k: int | None = None,
) -> list[tuple[Document, float | None]]:
    """Search the default vector store."""

    return get_vectorstore_manager().similarity_search(query, top_k=top_k)


def _load_chroma_class() -> type[Any]:
    """Load the Chroma vector store class."""

    from langchain_chroma import Chroma

    return Chroma
```

- [ ] **Step 4: Run vector store tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_vectorstore.py -v
```

Expected: all vector store tests pass.

- [ ] **Step 5: Commit vector store**

Run:

```bash
git add tests/test_vectorstore.py rag/vectorstore.py
git commit -m "feat: add chroma vector store manager"
```

Expected: git creates a commit containing vector store tests and implementation.

## Task 5: Retriever Wrapper

**Files:**
- Create: `tests/test_retriever.py`
- Create: `rag/retriever.py`

- [ ] **Step 1: Write failing retriever tests**

Create `tests/test_retriever.py` with exactly this content:

```python
"""Tests for retriever result normalization."""

from __future__ import annotations

from langchain_core.documents import Document

from rag.retriever import Retriever, retrieve


class FakeVectorStoreManager:
    def __init__(self):
        self.calls = []

    def similarity_search(self, query, top_k=None):
        self.calls.append((query, top_k))
        return [
            (
                Document(
                    page_content="Relevant context",
                    metadata={"source": "notes.md", "page": None, "chunk_id": "notes.md:pNA:c1"},
                ),
                0.91,
            ),
            (
                Document(
                    page_content="More context",
                    metadata={"source": "paper.pdf", "page": 2, "chunk_id": "paper.pdf:p2:c3"},
                ),
                None,
            ),
        ]


def test_retriever_returns_normalized_chunks():
    manager = FakeVectorStoreManager()
    retriever = Retriever(vectorstore_manager=manager)

    chunks = retriever.retrieve("What is RAG?", top_k=2)

    assert manager.calls == [("What is RAG?", 2)]
    assert chunks == [
        {
            "content": "Relevant context",
            "source": "notes.md",
            "page": None,
            "chunk_id": "notes.md:pNA:c1",
            "score": 0.91,
        },
        {
            "content": "More context",
            "source": "paper.pdf",
            "page": 2,
            "chunk_id": "paper.pdf:p2:c3",
            "score": None,
        },
    ]


def test_module_retrieve_uses_injected_manager(monkeypatch):
    manager = FakeVectorStoreManager()
    monkeypatch.setattr("rag.retriever.get_vectorstore_manager", lambda: manager)

    chunks = retrieve("question", top_k=1)

    assert manager.calls == [("question", 1)]
    assert chunks[0]["source"] == "notes.md"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_retriever.py -v
```

Expected: tests fail because `rag.retriever` does not exist or retriever functions are missing.

- [ ] **Step 3: Implement retriever**

Create `rag/retriever.py` with exactly this content:

```python
"""Retriever wrapper that returns normalized chunks for agents and UI."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.documents import Document

from rag.vectorstore import get_vectorstore_manager


class RetrievedChunk(TypedDict):
    """Normalized retrieved chunk shape."""

    content: str
    source: str | None
    page: int | None
    chunk_id: str | None
    score: float | None


class Retriever:
    """Project-level retriever over the configured vector store."""

    def __init__(self, vectorstore_manager: Any | None = None) -> None:
        self.vectorstore_manager = vectorstore_manager or get_vectorstore_manager()

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Retrieve and normalize relevant chunks."""

        results = self.vectorstore_manager.similarity_search(query, top_k=top_k)
        return [_normalize_result(document, score) for document, score in results]


def retrieve(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """Retrieve chunks using the default vector store manager."""

    return Retriever().retrieve(query, top_k=top_k)


def _normalize_result(document: Document, score: float | None) -> RetrievedChunk:
    metadata = document.metadata or {}
    return {
        "content": document.page_content,
        "source": metadata.get("source"),
        "page": metadata.get("page"),
        "chunk_id": metadata.get("chunk_id"),
        "score": score,
    }
```

- [ ] **Step 4: Run retriever tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_retriever.py -v
```

Expected: all retriever tests pass.

- [ ] **Step 5: Commit retriever**

Run:

```bash
git add tests/test_retriever.py rag/retriever.py
git commit -m "feat: add retriever wrapper"
```

Expected: git creates a commit containing retriever tests and implementation.

## Task 6: RAG Core Verification and README Update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run all RAG core tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_loader.py tests/test_chunker.py tests/test_embeddings.py tests/test_vectorstore.py tests/test_retriever.py -v
```

Expected: all tests pass.

- [ ] **Step 2: Run syntax verification**

Run:

```bash
.venv/bin/python -m compileall rag tests
```

Expected: no syntax errors.

- [ ] **Step 3: Update README roadmap**

Modify the `## Roadmap` section in `README.md` so the first roadmap item reads:

```markdown
- RAG core implemented: loading, chunking, embeddings, Chroma indexing, and retrieval.
```

Keep the remaining roadmap bullets unchanged.

- [ ] **Step 4: Verify README mentions RAG core implemented**

Run:

```bash
rg "RAG core implemented" README.md
```

Expected output includes:

```text
RAG core implemented: loading, chunking, embeddings, Chroma indexing, and retrieval.
```

- [ ] **Step 5: Commit RAG core verification docs**

Run:

```bash
git add README.md
git commit -m "docs: update rag core status"
```

Expected: git creates a commit containing the README update.

## Task 7: Final RAG Core Verification

**Files:**
- Read: all files created or modified in Tasks 1-6

- [ ] **Step 1: Run full test suite**

Run:

```bash
.venv/bin/python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Run foundation smoke test**

Run:

```bash
.venv/bin/python main.py
```

Expected output includes:

```text
Agentic RAG Document QA System
Embedding model: sentence-transformers/all-MiniLM-L6-v2
Run `python app.py` to start the Gradio UI.
```

- [ ] **Step 3: Confirm clean git status**

Run:

```bash
git status --short
```

Expected output:

```text
```

- [ ] **Step 4: Inspect recent commits**

Run:

```bash
git log --oneline -8
```

Expected: recent commits include:

```text
docs: update rag core status
feat: add retriever wrapper
feat: add chroma vector store manager
feat: add embedding model factory
feat: add document chunker
feat: add document loader
```
