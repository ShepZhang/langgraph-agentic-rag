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

        if not docs:
            raise EmptyVectorStoreError("Cannot create vector store without documents.")

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
        k = top_k if top_k is not None else self.settings.top_k
        if k <= 0:
            raise ValueError("top_k must be a positive integer.")

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
