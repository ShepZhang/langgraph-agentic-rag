# Agentic RAG Notes

## What RAG Means

Retrieval-Augmented Generation, or RAG, is a pattern where a system retrieves relevant external context before asking a language model to answer. The goal is to ground the answer in private or domain-specific documents instead of relying only on the model's parameters.

In a basic RAG pipeline, the system usually follows one fixed path: receive a question, retrieve similar chunks, and generate an answer from those chunks. This approach is simple, but it can fail when the question is vague, when retrieval returns weak evidence, or when the model answers from irrelevant context.

## What Agentic RAG Means

Agentic RAG is a retrieval-augmented generation approach where an agent controls the retrieval workflow instead of following a single fixed retrieve-and-answer path. The agent can rewrite a query, call a retriever tool, grade retrieved chunks, retry retrieval, and fall back when evidence is insufficient.

The main difference between naive RAG and Agentic RAG is control flow. Naive RAG usually retrieves once and immediately generates an answer. Agentic RAG can evaluate retrieval quality, improve the query, retry retrieval, and avoid answering when the evidence does not support the question.

## Query Rewriting

Query rewriting turns unclear, conversational, or context-dependent user questions into standalone retrieval queries. For example, if the user asks, "How does it improve reliability?" after discussing Agentic RAG, the system can rewrite the retrieval query to "How does Agentic RAG improve reliability with retrieval grading and fallback?"

The rewritten query is only used for search. The system must still answer the original user question, not the rewritten retrieval query. This distinction matters because a retrieval query may include extra keywords that help search but should not change the user's intent.

## Retriever Tool

In this project, the retriever is exposed as an agent tool named `retrieve_context`. The tool searches the indexed private knowledge base and returns document chunks with metadata such as source filename, page number when available, chunk id, and similarity score.

Treating retrieval as a tool makes the workflow easier to explain: the agent decides when to call the knowledge base, receives structured evidence, and then decides whether that evidence is good enough.

## Vectorstore Indexing

The Chroma vectorstore uses deterministic chunk IDs. Each ID is derived from source filename, file hash, page number, chunk id, and chunk content. This lets incremental indexing skip chunks that already exist instead of writing duplicates.

The Gradio Build Index action uses an explicit rebuild path for a clean uploaded knowledge base. The lower-level vectorstore API also supports incremental add with deterministic ID de-duplication.

## Retrieval Grading

After retrieval, the agent performs retrieval grading. Retrieval grading checks whether the retrieved chunks are truly relevant and sufficient to answer the original user question. A chunk should not be considered relevant just because it contains a matching keyword.

This project uses chunk-level grading. The model returns `relevant_indices`, which identify the retrieved chunks that contain useful evidence. The system then filters the raw retrieved documents into `relevant_documents`, so answer generation does not use unrelated chunks by default.

If the retrieved chunks are not relevant enough, the agent rewrites the question and retrieves again. The system limits this retry loop with a maximum retry count. If the documents still do not provide reliable evidence, the agent returns a fallback message saying that the current documents cannot answer the question.

## Fallback Handling

Fallback handling is a reliability control. If no relevant evidence is found after the allowed retry attempts, the system should say that the current documents cannot answer the question. It should not invent facts about topics that are not present in the indexed knowledge base.

Fallback is especially important for private knowledge-base QA because users may ask questions about policies, people, or systems that were never uploaded. A grounded system should prefer an explicit unable-to-answer response over a confident unsupported answer.

There are two different weak-evidence cases. Weak retrieval evidence can trigger query rewriting and another retrieval attempt before answer generation. Unsupported claims in a generated answer do not trigger another rewrite; citation safety and claim verification route the system to fallback instead.

## Citation-Aware Generation

When the retrieved chunks are relevant, the answer generation step must use only the selected relevant context. The answer should include citation markers such as `[1]` and `[2]`, and the model must return `used_citation_indices` so the program can map those indices back to source chunks.

This project performs selected evidence citation. The program returns citations only for chunks that the model says it used. It also checks that citation markers in the answer text, such as `[1]`, match `used_citation_indices`. Each citation can include source filename, page number when available, chunk id, similarity score, and a short snippet.

If the model returns a normal answer without valid supporting citation indices or matching citation markers, the system falls back instead of returning an unsupported answer. If the model explicitly says it cannot answer from the current documents, citations may be empty.

## Claim-Level Verification

The project includes lightweight claim-level citation verification. After answer generation selects citation chunks, the verifier checks whether the factual claims in the answer are supported by those selected chunks.

If a normal answer contains a claim that is not supported by the selected evidence, the system falls back instead of returning the answer. This makes citation handling stronger than simply attaching retrieved chunks to an answer.

## Current Limitations

Claim-level verification is still LLM-based. It is not the same as complete citation verification, formal proof, or a deterministic guarantee that every claim is correct.

Retrieval grading also depends on language-model judgment. The parser treats malformed JSON conservatively, but future work would add more deterministic checks, reranking, and human-reviewed claim labels.

## Evaluation Metrics

The evaluation runner tracks answer rate, fallback rate, citation rate, claim verification rate, source hit rate, keyword hit rate, fallback correctness, retry count, retrieved document count, and relevant document count.

The project can compare naive RAG and Agentic RAG. Naive RAG retrieves once and generates an answer. Agentic RAG uses query rewriting, retrieval grading, relevant chunk filtering, retry routing, and fallback handling.
