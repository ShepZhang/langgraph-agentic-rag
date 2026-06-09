# Retrieval Pipeline Notes

The retrieval pipeline is designed to make evidence selection explicit before
generation. Dense retrieval uses vector similarity to find chunks that are
semantically close to the rewritten question. It is useful when the user uses
different words than the source document, but it can miss exact identifiers,
policy names, command names, or file names.

BM25 sparse retrieval complements dense retrieval by matching exact terms. It
is better for literal strings such as `retrieve_context`, `relevant_indices`,
metric names, and source filenames. Sparse retrieval can over-rank repeated
terms, so it should not be the only retrieval signal for conceptual questions.

Hybrid retrieval combines dense and sparse candidate lists. The planned
combination strategy is Reciprocal Rank Fusion (RRF), which rewards chunks that
appear high in either list without requiring comparable raw scores. Hybrid
retrieval is expected to improve recall for both semantic questions and exact
keyword questions.

A cross-encoder reranker can score each query-chunk pair after the hybrid
candidate set is collected. The reranker usually improves precision and helps
move truly supporting chunks above distractors. Its trade-off is latency and
cost: every candidate pair adds model work, so the candidate pool must stay
bounded.

The retriever should return enough metadata for evaluation: source filename,
chunk id, similarity or rank score, and content. Downstream grading can then
separate retrieved documents from relevant documents and measure whether the
filtering step removed noisy chunks.

