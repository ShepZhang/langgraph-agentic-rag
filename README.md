# Agentic RAG Document QA System

基于 LangGraph 的 Agentic RAG 智能文档问答系统，用于面向私有知识库的 PDF / Markdown / TXT 文档问答。

This project is a lightweight Agentic RAG prototype for private document QA, focusing on explicit LangGraph-based retrieval control, query rewriting, retrieval grading, retry routing, citation-aware answer generation, and lightweight claim-level citation verification. The focus is the Agentic RAG workflow, not backend authentication or production deployment infrastructure.

## Why This Is Not a Naive RAG Demo

A naive RAG pipeline usually follows one fixed path:

```text
question -> retrieve -> generate answer
```

This project makes the retrieval process agentic:

```text
question
-> query rewrite
-> retriever tool
-> retrieval grading
-> conditional retry
-> grounded answer with citations
```

The agent checks whether retrieved chunks can actually answer the question. If they are not relevant enough, it rewrites the query and retries retrieval before falling back with a clear unable-to-answer response.

The system keeps a strict distinction between the original user question and the retrieval query. `current_query` is optimized for search; grading and answer generation still target the original user question.

## Portfolio Materials

- [Evaluation report](docs/evaluation.md)
- [Reproducible demo guide](docs/demo.md)
- [Design notes](docs/design.md)

## Architecture

![Architecture](assets/architecture.png)

```text
UI Layer
  Gradio document upload, indexing, QA, citations, retrieved chunks, diagnostics

RAG Layer
  loader -> chunker -> embeddings -> Chroma vector store -> retriever

Agent Layer
  LangGraph state -> nodes -> conditional edges -> answer or fallback

Evaluation Layer
  eval_questions.json -> evaluation runner -> summary metrics
```

## Agent Workflow

```text
START
-> rewrite_query
-> retrieve
-> grade_documents
-> if relevant: generate_answer -> END
-> if no relevant chunks and retry_count < max_retry_count: rewrite_query
-> if no relevant chunks and retry_count >= max_retry_count: fallback -> END
```

Implemented LangGraph nodes:

- `rewrite_query_node`: normalizes the first query, then uses failed retrieval context for retry rewrites.
- `retrieve_node`: calls the `retrieve_context` tool over the private Chroma index.
- `grade_documents_node`: asks the LLM for chunk-level `relevant_indices`, then filters `relevant_documents`.
- `generate_answer_node`: generates JSON answers from relevant chunks, checks citation marker consistency, maps `used_citation_indices` to evidence, and verifies cited claims before returning normal answers.
- `fallback_node`: returns a clear message when the indexed documents do not support an answer.

Key state fields:

- `current_query`: query currently used for retrieval.
- `question`: original user question. Grading and answer generation use this as the target.
- `previous_queries`: retrieval queries already attempted.
- `retrieval_attempt`: number of actual retriever calls.
- `retry_count`: number of failed-retrieval rewrites. Initial query normalization does not count as retry.
- `documents`: raw retrieved chunks.
- `relevant_documents`: chunks accepted by retrieval grading.
- `grading_reason`: LLM reason for accepting or rejecting retrieved evidence.
- `citations`: final answer evidence chunks selected by `used_citation_indices`.
- `claims`: claim-level verification records extracted from the final answer.
- `is_verified`: whether normal answer claims were verified against selected citation chunks.

## Features

- PDF, Markdown, and TXT document loading.
- Recursive chunking with source, source path, file hash, page, and chunk id metadata.
- Local sentence-transformers embeddings by default.
- Persistent Chroma vector store with deterministic chunk IDs, explicit rebuild, and incremental add support.
- Retriever exposed as an Agent tool named `retrieve_context`.
- Optional cross-encoder reranker: retrieve candidate chunks, rerank them, then pass the strongest chunks to grading.
- Query rewriting for vague or context-dependent questions.
- Chunk-level retrieval grading with conservative handling for invalid grading output.
- Conditional retry with configurable max retry count.
- Citation-aware grounded answer generation using only selected evidence chunks.
- Citation safety: normal answers without valid supporting citation indices or matching answer citation markers fall back instead of returning unsupported answers.
- Lightweight claim-level verification: normal cited answers are split into claims and checked against selected citation chunks.
- Gradio UI for upload, indexing, question answering, citations, retrieved chunks, and retry diagnostics.
- Lightweight evaluation runner comparing Naive RAG, Agentic RAG, and Agentic + Reranker on answer, fallback, citation, source-hit, keyword-hit, retry, relevant filtering, latency, and error metrics.

## Tech Stack

- Python 3.11+
- LangGraph
- LangChain
- ChromaDB
- sentence-transformers
- OpenAI-compatible chat LLM
- Ollama local LLM via OpenAI-compatible endpoint
- Gradio
- python-dotenv
- pytest

## Quick Start

Create a virtual environment:

```bash
python3.11 --version
python3.11 -m venv .venv
```

Use any Python 3.11+ executable available on your machine. On systems where `python3` points to an older interpreter, call the 3.11+ binary explicitly when creating the virtual environment.

Install dependencies:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Create an environment file:

```bash
cp .env.example .env
```

Set your chat LLM config in `.env`.

For OpenAI, DeepSeek, or another OpenAI-compatible remote API:

```bash
LLM_PROVIDER=openai_compatible
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

For local Ollama:

```bash
ollama pull qwen2.5:7b
ollama serve
```

Then set:

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
```

Ollama mode uses the local OpenAI-compatible endpoint at `/v1` internally and does not require `OPENAI_API_KEY`.

Start the Gradio UI:

```bash
.venv/bin/python app.py
```

If you activate the virtual environment first, `python app.py` also works:

```bash
source .venv/bin/activate
python app.py
```

## Usage

1. Open the Gradio URL printed by `app.py`.
2. Upload one or more `.pdf`, `.md`, `.markdown`, or `.txt` files.
3. Click `Build Index`.
4. Ask a question about the indexed documents.
5. Inspect:
   - answer
   - citations
   - retrieved chunks
   - rewritten question
   - retry count
   - retrieval diagnostics

The chat LLM is required for query rewriting, retrieval grading, answer generation, and claim verification. If the selected provider is missing required configuration, the app returns a clear configuration error instead of producing offline fake answers.

## Tests

Run the test suite:

```bash
.venv/bin/python -m pytest -q
```

Tests use fake LLMs and mocked vector stores, so they do not require real OpenAI-compatible API calls.

## Evaluation

Run the lightweight two-variant evaluation script:

```bash
.venv/bin/python -m evaluation.evaluate --questions evaluation/eval_questions.json
```

For the portfolio benchmark, run the three-variant matrix:

```bash
.venv/bin/python -m evaluation.matrix --questions evaluation/eval_questions.json --json-output evaluation/results/deepseek_matrix_2026-06-07.json
```

`evaluation.matrix` compares:

- `Naive RAG`: question -> retrieve -> generate.
- `Agentic RAG`: question -> rewrite -> retrieve -> grade -> retry or answer.
- `Agentic + Reranker`: vector retrieval -> cross-encoder reranking -> agentic grading and generation.

The fixed DeepSeek run is documented in [DeepSeek Evaluation Benchmark](docs/evaluation.md), with raw results in `evaluation/results/deepseek_matrix_2026-06-07.json`.

Current DeepSeek summary from that fixed run:

| Metric | Naive RAG | Agentic RAG | Agentic + Reranker |
|---|---:|---:|---:|
| Retrieval Source Hit Rate | 1.0 | 1.0 | 1.0 |
| Keyword Hit Rate | 0.7143 | 0.7143 | 0.75 |
| Citation Rate | 0.8235 | 0.7647 | 0.7941 |
| Claim Verification Rate | 0.0 | 0.7647 | 0.7941 |
| Fallback Correctness | 0.9706 | 0.9412 | 0.9706 |
| Average Latency | 2.3173 | 13.1514 | 12.3881 |

The matrix validates LLM and reranker configuration before evaluation and exits with a concise configuration error if runner construction fails. After runners are created, per-question retrieval or generation failures are recorded in the report instead of aborting the remaining questions.
The current evaluation set is a lightweight local QA set for demonstration. It is useful for comparing behavior, but it is not a rigorous benchmark.

## Example Output

Example answer payload:

```json
{
  "answer": "Agentic RAG uses query rewriting and retrieval grading to improve document QA.",
  "citations": [
    {
      "source": "notes.md",
      "page": null,
      "chunk_id": "notes.md:pNA:c1",
      "score": 0.91
    }
  ],
  "rewritten_question": "What is Agentic RAG?",
  "current_query": "What is Agentic RAG?",
  "retry_count": 0,
  "retrieval_attempt": 1,
  "is_relevant": true
}
```

For the current evaluation output, see [DeepSeek Evaluation Benchmark](docs/evaluation.md) and `evaluation/results/deepseek_matrix_2026-06-07.json`.

## Environment Variables

- `LLM_PROVIDER`: `openai_compatible` for remote OpenAI-compatible APIs, or `ollama` for local Ollama.
- `LLM_TEMPERATURE`: Chat model temperature. Default is `0`.
- `OPENAI_API_KEY`: API key for the OpenAI-compatible remote LLM.
- `OPENAI_BASE_URL`: Base URL for the OpenAI-compatible API.
- `OPENAI_MODEL`: Remote chat model used by the agent.
- `OLLAMA_BASE_URL`: Local Ollama server URL. Default is `http://localhost:11434`.
- `OLLAMA_MODEL`: Local Ollama model name, such as `qwen2.5:7b`.
- `EMBEDDING_PROVIDER`: Embedding backend. MVP default is `sentence_transformers`.
- `EMBEDDING_MODEL`: Local embedding model. Default is `sentence-transformers/all-MiniLM-L6-v2`.
- `CHUNK_SIZE`: Text chunk size.
- `CHUNK_OVERLAP`: Text chunk overlap.
- `TOP_K`: Number of chunks retrieved per query.
- `RERANKER_ENABLED`: Enable optional cross-encoder reranking. Default is `false`.
- `RERANKER_MODEL`: Cross-encoder model used when reranking is enabled.
- `RERANKER_CANDIDATE_TOP_K`: Number of initial vector candidates to retrieve before reranking.
- `MAX_RETRY_COUNT`: Maximum failed-retrieval retry rewrites.
- `CHROMA_PERSIST_DIR`: Local Chroma persistence path.
- `CHROMA_COLLECTION_NAME`: Chroma collection name.
- `GRADIO_SERVER_NAME`: Gradio host.
- `GRADIO_SERVER_PORT`: Gradio port.

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
│   ├── loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   └── retriever.py
├── agent/
│   ├── graph.py
│   ├── state.py
│   ├── nodes.py
│   ├── edges.py
│   ├── tools.py
│   └── prompts.py
├── evaluation/
│   ├── baselines.py
│   ├── eval_questions.json
│   ├── evaluate.py
│   ├── matrix.py
│   └── results/
│       └── deepseek_matrix_2026-06-07.json
├── docs/
│   ├── demo.md
│   ├── design.md
│   └── evaluation.md
├── ui/
│   └── gradio_app.py
├── assets/
│   └── architecture.png
├── sample_docs/
│   ├── agentic_rag_notes.md
│   ├── employee_handbook.md
│   ├── product_specs.md
│   └── security_policy.md
└── tests/
```

## Resume Highlights

- Built an Agentic RAG workflow with LangGraph, decomposing document QA into query rewriting, retrieval, retrieval grading, answer generation, and fallback nodes.
- Wrapped vector retrieval as an Agent tool so the workflow can explicitly call a private knowledge base instead of relying on model parameters alone.
- Implemented chunk-level retrieval grading and conditional retry to improve reliability on vague or poorly matched questions.
- Designed citation-aware answer generation where the model returns `used_citation_indices`, so final citations map only to evidence chunks used in the answer.
- Added deterministic citation marker consistency checks so answer markers like `[1]` must match `used_citation_indices` before claim verification runs.
- Added lightweight claim-level citation verification that checks whether generated answer claims are supported by selected evidence chunks before returning a normal answer.
- Added provider-aware LLM configuration for remote OpenAI-compatible APIs and local Ollama models without changing the LangGraph workflow.
- Implemented deterministic vectorstore IDs from source metadata and chunk content, allowing incremental add while skipping chunks already indexed.
- Supported PDF, Markdown, and TXT ingestion with chunk metadata, local embeddings, Chroma indexing, and Gradio-based document QA.
- Added lightweight evaluation comparing Naive RAG, Agentic RAG, and Agentic + Reranker across source hit rate, keyword hit rate, citation rate, fallback correctness, retry behavior, relevant chunk filtering, and latency.

## Interview Talking Points

- **Why this is not naive RAG**: The graph can rewrite, grade retrieved evidence, retry retrieval, or fall back instead of always retrieving once and answering.
- **Original question vs retrieval query**: `current_query` can be optimized for search while `question` remains the target for grading and final answer generation.
- **Retriever vs reranker**: The retriever finds candidate chunks with vector similarity; the reranker reorders those candidates before the agent grades evidence.
- **Reranker vs retrieval grading**: Reranking scores candidate-query fit, while retrieval grading decides whether chunks are sufficient for the original question.
- **Citation-aware generation vs claim verification**: Generation asks for used evidence indices and answer markers; verification checks returned claims against selected citation chunks.
- **Reliability tradeoff**: Conservative citation and grading checks can increase fallback rate or latency when evidence is ambiguous.
- **Evaluation limitation**: The DeepSeek matrix is a 34-question local benchmark over fictional sample documents, useful for comparison but not a broad benchmark.

## Current Limitations

- Claim-level citation verification is lightweight and LLM-based. It checks claims against selected evidence chunks, but it is not a formal proof system.
- Citation marker consistency is deterministic, but it only checks marker/index alignment. It does not prove that every cited claim is true.
- Retrieval grading depends on LLM JSON output. The parser is defensive, but malformed grading output is treated conservatively.
- Evaluation uses a lightweight local QA set and should be expanded with larger datasets for more rigorous benchmarking.
- The Gradio `Build Index` workflow intentionally rebuilds the active collection for a clean uploaded knowledge base. The lower-level vectorstore API also supports incremental `add_documents` with deterministic IDs.
- The project is a prototype and is not intended for production deployment without further hardening.

## Roadmap

- RAG core implemented: loading, chunking, embeddings, Chroma indexing, and retrieval.
- LangGraph agent workflow implemented: query rewriting, retriever tool, retrieval grading, retry routing, answer generation, and fallback.
- Gradio upload and QA flow implemented: document indexing, Agentic QA, citations, retrieved chunks, and retry diagnostics.
- Evaluation matrix implemented: Naive RAG, Agentic RAG, and Agentic + Reranker comparison with answer/fallback/citation/source/keyword metrics, retry metrics, and relevant filtering metrics.
- Claim-level verification implemented: cited normal answers are checked against selected evidence before being returned.
- Deterministic citation marker consistency implemented: answer markers must match selected citation indices.
- Deterministic vectorstore IDs implemented: chunk identity is derived from source metadata and content for incremental add de-duplication.
- Optional reranker implemented: vector retrieval can over-retrieve candidates, apply a local cross-encoder reranker, and pass reranked chunks into grading.
- Ollama local LLM support implemented through `LLM_PROVIDER=ollama`.
- Add FastAPI API layer.
- Add model-specific prompt tuning and cost/latency evaluation for local Ollama models.
- Add human-reviewed claim labels for stricter citation validation.
- Add richer reranker evaluation and model/latency comparisons.
