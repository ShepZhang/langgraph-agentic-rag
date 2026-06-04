# Agentic RAG Notes

Agentic RAG is a retrieval-augmented generation approach where an agent controls the retrieval workflow instead of following a single fixed retrieve-and-answer path.

In this project, the agent first rewrites unclear or context-dependent questions into standalone search queries. Then it calls a retriever tool named `retrieve_context` to search the indexed private knowledge base.

After retrieval, the agent performs retrieval grading. Retrieval grading checks whether the retrieved chunks are truly relevant and sufficient to answer the user's question. A chunk should not be considered relevant just because it contains a matching keyword.

If the retrieved chunks are not relevant enough, the agent rewrites the question and retrieves again. The system limits this retry loop with a maximum rewrite attempt count. If the documents still do not provide reliable evidence, the agent returns a fallback message saying that the current documents cannot answer the question.

When the retrieved chunks are relevant, the answer generation step must use only the retrieved context. The answer should include citations such as source filename, page number when available, chunk id, and retrieval score. This citation-aware answer generation helps reduce hallucination.

The main difference between naive RAG and Agentic RAG is control flow. Naive RAG usually retrieves once and immediately generates an answer. Agentic RAG can rewrite the query, evaluate retrieval quality, retry retrieval, and fall back when the evidence is insufficient.
