"""Tests for vector store management."""

from __future__ import annotations

from dataclasses import replace

import pytest
from langchain_core.documents import Document

from config import get_settings
from rag import vectorstore
from rag.vectorstore import EmptyVectorStoreError, VectorStoreManager


class FakeChroma:
    created_with = None
    deleted_collections = []

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

    def delete_collection(self):
        self.__class__.deleted_collections.append(
            {
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
            }
        )

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
    FakeChroma.deleted_collections = []
    monkeypatch.setattr(vectorstore, "_load_chroma_class", lambda: FakeChroma)
    manager = VectorStoreManager(settings=settings, embedding_model=embedding_model)
    docs = [Document(page_content="hello", metadata={"source": "a.md"})]

    store = manager.create_vectorstore(docs)

    assert store is manager.store
    assert FakeChroma.deleted_collections == [
        {
            "collection_name": "test_collection",
            "persist_directory": str(tmp_path / "chroma"),
        }
    ]
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


def test_similarity_search_rejects_zero_top_k(tmp_path, monkeypatch):
    settings = replace(get_settings(), chroma_persist_dir=tmp_path / "chroma")
    monkeypatch.setattr(vectorstore, "_load_chroma_class", lambda: FakeChroma)
    manager = VectorStoreManager(settings=settings, embedding_model=object())

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        manager.similarity_search("question", top_k=0)


def test_create_vectorstore_rejects_empty_documents(tmp_path, monkeypatch):
    settings = replace(get_settings(), chroma_persist_dir=tmp_path / "chroma")
    FakeChroma.created_with = None
    monkeypatch.setattr(vectorstore, "_load_chroma_class", lambda: FakeChroma)
    manager = VectorStoreManager(settings=settings, embedding_model=object())

    with pytest.raises(EmptyVectorStoreError, match="Cannot create vector store"):
        manager.create_vectorstore([])

    assert FakeChroma.created_with is None


def test_similarity_search_with_none_top_k_uses_settings(tmp_path, monkeypatch):
    settings = replace(
        get_settings(),
        chroma_persist_dir=tmp_path / "chroma",
        top_k=7,
    )
    monkeypatch.setattr(vectorstore, "_load_chroma_class", lambda: FakeChroma)
    manager = VectorStoreManager(settings=settings, embedding_model=object())

    manager.similarity_search("question", top_k=None)

    assert manager.store.queries == [("question", 7)]


def test_add_documents_delegates_to_loaded_store(tmp_path, monkeypatch):
    settings = replace(get_settings(), chroma_persist_dir=tmp_path / "chroma")
    monkeypatch.setattr(vectorstore, "_load_chroma_class", lambda: FakeChroma)
    manager = VectorStoreManager(settings=settings, embedding_model=object())
    docs = [Document(page_content="new", metadata={"source": "b.md"})]

    result = manager.add_documents(docs)

    assert result == ["id-1"]
    assert manager.store.documents == docs
