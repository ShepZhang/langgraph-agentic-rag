"""Tests for workspace-aware retrieval isolation."""

from __future__ import annotations

from dataclasses import replace

from langchain_core.documents import Document

from agent.features import AgentFeatureFlags
from config import get_settings
from rag.hybrid_retriever import HybridRetriever
from rag.retriever import Retriever
from rag.vectorstore import VectorStoreManager


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def invoke(self, prompt):
        return self.responses.pop(0)


class WorkspaceStore:
    def __init__(self):
        self.search_calls = []
        self.get_calls = []
        self.documents = [
            Document(
                page_content="Workspace A RAG evidence",
                metadata={
                    "source": "a.md",
                    "chunk_id": "a:c1",
                    "workspace_id": "workspace_a",
                    "document_id": "doc_a",
                },
            ),
            Document(
                page_content="Workspace B payroll evidence",
                metadata={
                    "source": "b.md",
                    "chunk_id": "b:c1",
                    "workspace_id": "workspace_b",
                    "document_id": "doc_b",
                },
            ),
        ]

    def similarity_search_with_score(self, query, k, filter=None):
        self.search_calls.append((query, k, filter))
        documents = self._filter_documents(filter)
        return [(document, 0.9 - index * 0.1) for index, document in enumerate(documents[:k])]

    def get(self, include=None, where=None):
        self.get_calls.append((include, where))
        documents = self._filter_documents(where)
        return {
            "documents": [document.page_content for document in documents],
            "metadatas": [document.metadata for document in documents],
        }

    def _filter_documents(self, metadata_filter):
        if not metadata_filter:
            return self.documents
        workspace_id = metadata_filter.get("workspace_id")
        return [
            document
            for document in self.documents
            if document.metadata.get("workspace_id") == workspace_id
        ]


def test_vectorstore_similarity_search_uses_workspace_metadata_filter(tmp_path):
    settings = replace(get_settings(), chroma_persist_dir=tmp_path / "chroma")
    manager = VectorStoreManager(settings=settings, embedding_model=object())
    store = WorkspaceStore()
    manager.store = store

    results = manager.similarity_search(
        "RAG evidence",
        top_k=4,
        workspace_id="workspace_a",
    )

    assert store.search_calls == [
        ("RAG evidence", 4, {"workspace_id": "workspace_a"})
    ]
    assert [document.metadata["workspace_id"] for document, _score in results] == [
        "workspace_a"
    ]


def test_vectorstore_get_all_documents_filters_sparse_corpus_by_workspace(tmp_path):
    settings = replace(get_settings(), chroma_persist_dir=tmp_path / "chroma")
    manager = VectorStoreManager(settings=settings, embedding_model=object())
    store = WorkspaceStore()
    manager.store = store

    documents = manager.get_all_documents(workspace_id="workspace_b")

    assert store.get_calls == [
        (["documents", "metadatas"], {"workspace_id": "workspace_b"})
    ]
    assert [document.metadata["workspace_id"] for document in documents] == [
        "workspace_b"
    ]


def test_hybrid_retriever_passes_workspace_to_dense_and_bm25_corpus(tmp_path):
    settings = replace(
        get_settings(),
        chroma_persist_dir=tmp_path / "chroma",
        dense_top_k=5,
        bm25_top_k=5,
        fusion_top_k=5,
    )
    manager = VectorStoreManager(settings=settings, embedding_model=object())
    store = WorkspaceStore()
    manager.store = store

    results = HybridRetriever(manager, settings=settings).retrieve(
        "payroll evidence",
        top_k=5,
        workspace_id="workspace_b",
    )

    assert store.search_calls == [
        ("payroll evidence", 5, {"workspace_id": "workspace_b"})
    ]
    assert store.get_calls == [
        (["documents", "metadatas"], {"workspace_id": "workspace_b"})
    ]
    assert {document.metadata["workspace_id"] for document, _score in results} == {
        "workspace_b"
    }


def test_retriever_scopes_dense_results_and_exposes_workspace_metadata(tmp_path):
    settings = replace(get_settings(), chroma_persist_dir=tmp_path / "chroma")
    manager = VectorStoreManager(settings=settings, embedding_model=object())
    store = WorkspaceStore()
    manager.store = store
    retriever = Retriever(vectorstore_manager=manager, settings=settings)

    chunks = retriever.retrieve(
        "RAG evidence",
        top_k=2,
        workspace_id="workspace_a",
    )

    assert store.search_calls == [
        ("RAG evidence", 2, {"workspace_id": "workspace_a"})
    ]
    assert chunks == [
        {
            "content": "Workspace A RAG evidence",
            "source": "a.md",
            "page": None,
            "chunk_id": "a:c1",
            "document_id": "doc_a",
            "workspace_id": "workspace_a",
            "score": 0.9,
        }
    ]


def test_run_agent_scopes_default_retriever_by_workspace(monkeypatch):
    from agent.graph import run_agent

    calls = []

    def fake_retrieve(query, top_k=None, workspace_id=None):
        calls.append((query, top_k, workspace_id))
        return [
            {
                "content": "Workspace A RAG evidence",
                "source": "a.md",
                "chunk_id": "a:c1",
                "document_id": "doc_a",
                "workspace_id": "workspace_a",
                "score": 0.9,
            }
        ]

    monkeypatch.setattr("agent.tools.retrieve", fake_retrieve)
    flags = AgentFeatureFlags(
        query_transformation_enabled=False,
        retrieval_grading_enabled=False,
        conditional_retry_enabled=False,
        citation_verification_enabled=False,
    )
    result = run_agent(
        "What is RAG?",
        workspace_id="workspace_a",
        llm=FakeLLM(
            [
                (
                    '{"answer": "Workspace A RAG evidence [1].", '
                    '"used_citation_indices": [1]}'
                )
            ]
        ),
        settings=get_settings(),
        features=flags,
    )

    assert calls == [("What is RAG?", None, "workspace_a")]
    assert result["retrieved_documents"][0]["workspace_id"] == "workspace_a"
