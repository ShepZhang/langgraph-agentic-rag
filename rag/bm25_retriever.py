"""Dependency-free BM25 sparse retriever for local document chunks."""

from __future__ import annotations

import math
import re
from collections import Counter

from langchain_core.documents import Document


ScoredDocument = tuple[Document, float]


class BM25Retriever:
    """Rank documents with BM25 keyword matching."""

    def __init__(
        self,
        documents: list[Document],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self._tokenized_documents = [_tokenize(document.page_content) for document in documents]
        self._term_frequencies = [
            Counter(tokens) for tokens in self._tokenized_documents
        ]
        self._document_lengths = [len(tokens) for tokens in self._tokenized_documents]
        self._average_document_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )
        self._document_frequencies = self._build_document_frequencies()

    def retrieve(self, query: str, top_k: int) -> list[ScoredDocument]:
        """Return top BM25 matches for a query."""

        if not self.documents or top_k <= 0:
            return []

        query_terms = _tokenize(query)
        if not query_terms:
            return []

        scored = [
            (document, self._score_document(query_terms, index))
            for index, document in enumerate(self.documents)
        ]
        matches = [(document, score) for document, score in scored if score > 0]
        return sorted(matches, key=lambda item: item[1], reverse=True)[:top_k]

    def _build_document_frequencies(self) -> dict[str, int]:
        frequencies: dict[str, int] = {}
        for tokens in self._tokenized_documents:
            for token in set(tokens):
                frequencies[token] = frequencies.get(token, 0) + 1
        return frequencies

    def _score_document(self, query_terms: list[str], index: int) -> float:
        score = 0.0
        term_frequencies = self._term_frequencies[index]
        document_length = self._document_lengths[index]
        if document_length == 0 or self._average_document_length == 0:
            return 0.0

        for term in query_terms:
            term_frequency = term_frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            document_frequency = self._document_frequencies.get(term, 0)
            idf = math.log(
                1
                + (len(self.documents) - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * document_length / self._average_document_length
            )
            score += idf * (term_frequency * (self.k1 + 1)) / denominator
        return score


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())
