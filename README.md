# Reliability-oriented Agentic RAG Document QA System

基于 LangGraph 的 Agentic RAG 智能文档问答系统，用于面向私有知识库的 PDF / Markdown / TXT 文档问答。

Reliability-oriented Agentic RAG Document QA System is a LangGraph-based document question answering project that upgrades naive retrieve-generate RAG into a stateful Agent workflow. It integrates structured query transformation, optional hybrid retrieval, reranking, structured retrieval grading, partial-relevance recovery, conditional retry, fallback handling, citation-aware answer generation, claim-level citation verification, typed internal tools, answer revision, baseline comparison, an Evaluation Dashboard, and executable V0-V6 ablation artifacts to improve reliability, explainability, debuggability, and evaluability in complex document QA scenarios.

The project is production-oriented as an architecture and evaluation exercise, but it is not a complete production deployment. Authentication, authorization, deployment hardening, and full observability are intentionally left for later milestones.

## Why This Is Not a Naive RAG Demo

A naive RAG pipeline usually follows one fixed path:

```text
question -> retrieve -> generate answer
```

This project makes the retrieval process agentic:

```text
question
-> query transformation
-> retriever tool
-> retrieval grading
-> conditional retry
-> grounded answer with citations
```

The agent checks whether retrieved chunks can actually answer the question. If they are not relevant enough, it rewrites the retrieval query and retries before falling back with a clear unable-to-answer response.

The system keeps a strict distinction between the original user question and the retrieval query. `current_query` is optimized for search; grading and answer generation still target the original user question.

## Architecture

![Architecture](assets/architecture.png)

```text
UI Layer
  Gradio Document QA and Evaluation tabs with QA, comparison, and diagnostics

RAG Layer
  loader -> chunker -> embeddings -> Chroma vector store
  dense retrieval + optional BM25 retrieval + RRF fusion + optional reranker

Agent Layer
  LangGraph state -> nodes -> conditional edges -> answer or fallback

Tool Registry
  typed internal tools -> validation -> dependency injection -> diagnostics

Evaluation Layer
  eval_questions.json -> baseline/agentic runners -> JSON artifacts -> report
```

## Agent Workflow

```text
START
-> rewrite_query
-> retrieve
-> grade_documents
-> if relevant: generate_answer
-> extract_claims
-> verify_citations
-> if verified: finalize_answer -> END
-> if unsupported claims and revision budget remains: revise_answer -> extract_claims
-> if unsupported claims remain after revision budget: fallback -> END
-> if no relevant chunks and retry_count < max_retry_count: rewrite_query
-> if no relevant chunks and retry_count >= max_retry_count: fallback -> END
```

Implemented LangGraph nodes:

- `rewrite_query_node`: performs structured query transformation on the first attempt, then uses failed retrieval context for retry rewrites.
- `retrieve_node`: calls the `retrieve_context` tool over the private Chroma index.
- `grade_documents_node`: asks the LLM for chunk-level relevance labels, confidence scores, and reasons, then filters `relevant_documents` from chunks graded `relevant`.
- `generate_answer_node`: generates draft JSON answers from relevant chunks, checks citation marker consistency, and maps `used_citation_indices` to selected evidence.
- `extract_claims_node`: extracts atomic factual claims from the draft answer.
- `verify_citations_node`: verifies each claim against its cited chunks with `supported`, `partially_supported`, and `unsupported` labels.
- `revise_answer_node`: removes or narrows unsupported claims once before re-running claim extraction and verification.
- `finalize_answer_node`: promotes verified draft answers, or explicit unable-to-answer refusals, to final output.
- `fallback_node`: returns a clear message when the indexed documents do not support an answer.

## Internal Tool Registry

The project uses a typed internal Tool Registry as the execution boundary for
Agent capabilities:

- `retrieve_context`: workspace-scoped dense or hybrid retrieval with optional
  reranking.
- `verify_citations`: claim-level verification against selected citation
  chunks.
- `summarize_document`: grounded summarization for supplied document text.
- `calculator`: bounded arithmetic evaluation through an AST whitelist.

`ToolContext` injects runtime dependencies such as the active LLM, retriever,
and `workspace_id`. Pydantic schemas validate arguments, while `ToolResult`
normalizes success data, structured errors, metadata, and latency. Compact
tool-call events are written into Agent traces without storing prompts,
secrets, or full document bodies.

The primary LangGraph workflow uses `retrieve_context` and
`verify_citations`. Summary and calculator are registered extension points and
independently callable capabilities; P3d does not add autonomous tool
selection, planning, or a ReAct loop.

Key state fields:

- `current_query`: query currently used for retrieval.
- `standalone_question`: standalone retrieval-ready version of the user question.
- `query_transform`: structured query transformation record.
- `query_transform_strategy`: one of `rewrite`, `multi_query`, or `decomposition`.
- `expanded_queries`: complementary retrieval queries used by multi-query retrieval.
- `sub_questions`: decomposition sub-questions recorded for future multi-hop retrieval.
- `retrieval_queries`: actual queries executed by the retriever node.
- `multi_query_used`: whether the retriever node executed more than one query.
- `multi_query_result_count`: number of unique chunks after multi-query merge.
- `question`: original user question. Grading and answer generation use this as the target.
- `previous_queries`: retrieval queries already attempted.
- `retrieval_attempt`: number of retriever-node executions.
- `retry_count`: number of failed-retrieval rewrites. Initial query normalization does not count as retry.
- `documents`: raw retrieved chunks.
- `relevant_documents`: chunks accepted by retrieval grading.
- `document_grades`: chunk-level relevance labels, confidence scores, and reasons.
- `relevant_document_count`: number of chunks graded `relevant`.
- `partial_document_count`: number of chunks graded `partially_relevant`.
- `max_relevance_confidence`: highest confidence score across graded chunks.
- `partial_relevance_recovery`: query-refinement recovery decision when chunks are related but insufficient.
- `grading_reason`: LLM reason for accepting or rejecting retrieved evidence.
- `draft_answer`: generated answer candidate before claim-level verification finalizes it.
- `used_citation_indices`: citation indices selected by answer generation or revision.
- `cited_documents`: selected chunks used for claim extraction and verification.
- `citations`: final answer evidence chunks selected by `used_citation_indices`.
- `claims`: structured claim records with `claim_id`, `claim`, and `cited_chunk_ids`.
- `claim_verification_results`: per-claim verification labels, confidence scores, and reasons.
- `unsupported_claims`: claims that are unsupported or only partially supported.
- `citation_verification_passed`: whether every extracted claim is supported by its cited chunks.
- `citation_revision_count`: number of answer revision attempts used.
- `citation_verification_skipped`: whether verification was skipped for an explicit unable-to-answer response.
- `is_verified`: compatibility alias for `citation_verification_passed`.

## Features

- PDF, Markdown, and TXT document loading.
- Recursive chunking with source, source path, file hash, page, and chunk id metadata.
- Local sentence-transformers embeddings by default.
- Persistent Chroma vector store with deterministic chunk IDs, explicit rebuild, and incremental add support.
- Optional hybrid retrieval: dense vector search and BM25 sparse search are fused with Reciprocal Rank Fusion before grading.
- Typed internal Tool Registry with Pydantic input validation, runtime dependency injection, normalized success/error results, and compact call diagnostics.
- Retriever exposed through the registry as `retrieve_context`; claim verification uses the registry tool `verify_citations`.
- Registered extension tools for grounded document summarization and bounded arithmetic calculation.
- Optional cross-encoder reranker: retrieve candidate chunks, rerank them, then pass the strongest chunks to grading.
- Reranker diagnostics: the reranker can emit structured records with document id, chunk id, original score, rerank score, rank, content, and metadata.
- Structured query transformation with direct rewrite, multi-query retrieval execution, and decomposition metadata.
- Structured chunk-level retrieval grading with `relevant`, `partially_relevant`, and `irrelevant` labels, confidence scores, reasons, and conservative handling for invalid grading output.
- Partial-relevance recovery: when no chunk is directly relevant but some chunks are related, retry rewriting receives partial evidence and targets missing facts instead of blindly rephrasing.
- Conditional retry with configurable max retry count.
- Citation-aware grounded answer generation using only selected evidence chunks.
- Citation safety: normal answers without valid supporting citation indices or matching answer citation markers fall back instead of returning unsupported answers.
- Claim-level citation verification: normal cited answers are split into atomic claims, verified against cited chunks, revised once if unsupported, and otherwise sent to fallback.
- Gradio UI with Document QA and Evaluation tabs for upload, indexing, question answering, quick comparison, saved ablation inspection, citations, retrieved chunks, and failure diagnostics.
- Naive RAG baseline package and CLI for retrieve-once comparison.
- Reliability evaluation runner comparing naive RAG and Agentic RAG on shared documents and a shared structured question set.
- JSON evaluation artifacts for baseline, agentic, comparison, and ablation runs.
- Real cumulative V0-V6 ablation using independent graph feature flags and per-variant retrieval settings.
- Local JSONL trace logging for Agent node events, route decisions, compact tool calls, final answers, citations, latency, retry counts, and errors.
- FastAPI service layer for chat, trace lookup, document upload/index/delete, and evaluation run retrieval.
- Workspace-aware retrieval isolation using Chroma metadata filters for dense retrieval and workspace-filtered BM25 corpora.

## Tech Stack

- Python 3.11+
- LangGraph
- LangChain
- ChromaDB
- sentence-transformers
- OpenAI-compatible chat LLM
- Ollama local LLM via OpenAI-compatible endpoint
- Gradio
- FastAPI
- Uvicorn
- python-dotenv
- pytest

## Quick Start

Create a virtual environment:

```bash
python3 -m venv .venv
```

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

Start the FastAPI backend:

```bash
.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open the interactive API docs at `http://127.0.0.1:8000/docs`.

## Usage

The Gradio app has two main tabs:

- `Document QA`: upload and index documents, ask questions, and inspect answers, citations, retrieved chunks, rewritten queries, retries, and retrieval diagnostics.
- `Evaluation`: run a selected-question Quick Compare or inspect the saved V0-V6 Ablation Snapshot.

For document QA:

1. Open the Gradio URL printed by `app.py` and select `Document QA`.
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

## Hybrid Retrieval Pipeline

P1a adds a configurable retrieval path for term-sensitive document QA:

```text
query
+-- dense retriever top-k from Chroma
+-- BM25 sparse retriever top-k over indexed chunks
+-- RRF fusion
    -> optional reranker
    -> retrieval grading
    -> answer generation or fallback
```

This path is disabled by default to preserve the original dense retrieval behavior. Enable it with:

```bash
HYBRID_RETRIEVAL_ENABLED=true
DENSE_TOP_K=20
BM25_TOP_K=20
FUSION_TOP_K=20
```

Dense retrieval is useful for semantic similarity. BM25 improves recall for exact terms such as filenames, abbreviations, identifiers, and domain-specific keywords. RRF deduplicates overlapping chunks by `chunk_id` and combines rank signals without requiring dense and sparse scores to be on the same scale.

The current BM25 implementation is dependency-free and intentionally lightweight. It uses token-level exact matching without stemming or learned sparse expansion.

## Trace Logging

P3a adds local trace logging for Agentic RAG runs. Enable it with:

```bash
TRACE_LOGGING_ENABLED=true
TRACE_LOG_DIR=./data/traces
```

When enabled, `run_agent()` returns `trace_id`, `trace_path`, and `latency_ms`.
Each trace record is appended to `traces.jsonl` and includes node events,
conditional route decisions, compact tool-call diagnostics, retrieved and
relevant document summaries, document grades, final answer, citations, claim
verification results, retry count, latency, and error metadata.

Trace records intentionally store compact document snippets and metadata rather
than full document bodies or local source paths. Tool-call events store only
allowlisted metadata such as workspace id, result count, claim count, latency,
and sanitized errors. Database-backed trace retention and a Gradio trace
dashboard remain later milestones.

## FastAPI Backend

P3b adds a synchronous FastAPI service layer for integration-oriented use:

```text
POST /documents/upload
POST /documents/index
GET  /documents
DELETE /documents/{document_id}
POST /chat
GET  /chat/{session_id}/trace
POST /evaluation/run
GET  /evaluation/{run_id}
```

Example chat request:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "workspace_id": "default",
    "session_id": "demo-session",
    "question": "How does retrieval grading improve RAG reliability?"
  }'
```

Example chat response:

```json
{
  "answer": "...",
  "citations": [],
  "trace_id": "trace_...",
  "retry_count": 0,
  "latency_ms": 2812.4,
  "fallback_triggered": false
}
```

The API layer reuses the same local vector store and Agent workflow as the
Gradio demo. Document deletion is scoped to documents registered through the
API registry; it does not retroactively manage documents indexed through manual
CLI or Gradio rebuilds.

## Workspace Isolation

P3c makes `workspace_id` a real retrieval boundary for API and programmatic
Agent calls. Documents indexed through the FastAPI document service receive
`workspace_id` and `document_id` metadata. When `run_agent()` receives a
`workspace_id`, the default retriever applies the same metadata filter to:

- dense Chroma retrieval
- BM25 sparse corpus loading
- hybrid dense + BM25 retrieval
- reranker candidate pools after retrieval

The Gradio demo and evaluation runner do not pass a workspace id by default, so
they preserve the previous global knowledge-base behavior. This milestone uses
one Chroma collection with metadata filters; per-workspace collections and
authorization checks remain future hardening work.

## Tests

Run the test suite:

```bash
.venv/bin/python -m pytest -q
```

Tests use fake LLMs and mocked vector stores, so they do not require real OpenAI-compatible API calls.

## Evaluation

P0b evaluates naive RAG and six cumulative Agentic RAG variants on the same 36-question structured dataset and the same indexed documents. Each row resolves to a distinct graph or retriever configuration; the runner rejects duplicate effective configurations.

Run comparison evaluation:

```bash
.venv/bin/python -m evaluation.evaluate \
  --questions evaluation/eval_questions.json \
  --output-dir experiments/results
```

Run the naive baseline only:

```bash
.venv/bin/python -m baseline.run_baseline \
  --questions evaluation/eval_questions.json \
  --output experiments/results/baseline_result.json
```

Run a representative five-question smoke matrix:

```bash
.venv/bin/python -m experiments.run_ablation \
  --questions evaluation/eval_questions.json \
  --config-dir experiments/configs \
  --output-dir experiments/results/smoke \
  --question-ids q001,q016,q027,q030,q033
```

Run the complete matrix:

```bash
.venv/bin/python -m experiments.run_ablation \
  --questions evaluation/eval_questions.json \
  --config-dir experiments/configs \
  --output-dir experiments/results \
  --report experiments/results/ablation_report.md
```

The cumulative matrix is:

| Version | Added capability |
|---|---|
| V0 | Naive dense retrieve-generate RAG |
| V1 | Query transformation |
| V2 | Structured retrieval grading |
| V3 | Conditional retry and fallback |
| V4 | BM25 + dense retrieval + RRF fusion |
| V5 | Reranking |
| V6 | Claim-level citation verification and answer revision |

Generated artifacts:

- `experiments/results/variants/v0_naive.json` through `v6_citation_verification.json`
- `experiments/results/baseline_result.json`
- `experiments/results/agentic_result.json`
- `experiments/results/comparison_result.json`
- `experiments/results/ablation_result.json`
- `experiments/results/ablation_report.md`
- `experiments/report.md`

Every variant is checkpointed before execution and finalized as `completed`, `completed_with_errors`, or `incomplete`. V0 and V6 canonical comparison artifacts are derived from those completed runs without repeating model calls.

Evaluation artifacts include a sanitized `runtime_config` snapshot covering Agent feature flags, model name, temperature, retriever settings, hybrid retrieval settings, reranker settings, and vector collection name. API keys, base URLs, local persistence paths, and other secrets are intentionally excluded.

Metric fields include:

- `answer_rate`
- `correctness_score`
- `context_relevance_score`
- `source_hit_rate`
- `citation_hit_rate`
- `fallback_accuracy`
- `unsupported_claim_count`
- `supported_claim_ratio`
- `citation_verification_pass_rate`
- `average_retry_count`
- `average_latency`
- `average_token_usage`
- `estimated_cost`
- `error_count`
- `failure_type_counts`

### Evaluation Dashboard

P4b adds an `Evaluation` tab with two views:

- `Quick Compare` runs the same selected question records in `Naive RAG`, `Agentic RAG`, or `Compare Both` mode. It reports correctness, context relevance, citation accuracy, fallback accuracy, unsupported claims, latency, retry metrics, failure counts, filterable failed cases, and selected-case diagnostic details.
- `Ablation Snapshot` reads `experiments/results/ablation_result.json` and displays the saved V0-V6 metrics, failed cases, diagnostics source, and runtime configuration. It is read-only and does not run V0-V6 from the browser.

The default smoke selection is `q001`, `q016`, `q027`, `q030`, and `q033`. The dashboard can also select all 36 questions. Quick evaluation runs synchronously, so a full selection can be slow and may incur model cost.

For pre-P4a ablation artifacts that do not contain stored failure analysis, the dashboard enriches complete records with deterministic diagnostics in memory. The UI labels diagnostics as `stored`, `derived`, or unavailable, and never writes the derived data back to the source JSON artifact.

### Failed Case Analysis

Each question result includes a `failure_analysis` object with `failure_type`, `reason`, and `suggestion` fields. The taxonomy covers `retrieval_failure`, `reranking_failure`, `query_rewrite_failure`, `generation_failure`, `citation_failure`, `fallback_failure`, `tool_failure`, and `no_failure`.

Attribution uses available source and citation signals, fallback behavior, unsupported claims, retrieved and relevant document diagnostics, and recorded errors. The analyzer is rule-based and deterministic: it does not use an LLM judge and does not automatically repair failures.

Ablation reports now include failure counts and representative failed cases for debugging and regression triage. This is a framework and reporting capability upgrade; P4a does not claim a new full DeepSeek ablation run, and the historical P0b result numbers below remain unchanged.

If an individual question fails, evaluation records the exception and marks the variant `completed_with_errors`. Configuration, index, or runner-construction failures leave an `incomplete` checkpoint instead of publishing misleading aggregate results.

Latest P0b single-run snapshot using `deepseek-v4-flash`:

- V0 Naive RAG: correctness `0.6111`, fallback accuracy `0.8611`, average latency `4.46s`.
- V5 with hybrid retrieval and reranking: highest heuristic correctness at `0.6389`, average latency `28.70s`.
- V6 with claim-level verification: zero unsupported final claims and supported-claim ratio `1.0000`, but correctness `0.5833` and average latency `41.25s`.

These numbers do not support a blanket claim that the complete Agentic workflow always outperforms naive RAG. They show module-specific trade-offs: reranking helped heuristic correctness in this run, while strict citation verification improved claim support diagnostics at the cost of answer acceptance and latency. See `experiments/report.md` and the generated JSON artifacts for the complete table and limitations.

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
  "is_relevant": true,
  "trace_id": "trace_...",
  "latency_ms": 2812.4
}
```

Example evaluation summary:

```text
Evaluation Report

Comparison Summary

| Metric | Naive RAG | Agentic RAG |
|---|---:|---:|
| Source Hit Rate | 0.6 | 0.8 |
| Keyword Hit Rate | 0.5 | 0.7 |
| Citation Rate | 0.55 | 0.75 |
| Fallback Correctness | 0.7 | 0.85 |
| Avg Latency | 2.1 | 4.8 |
```

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
- `HYBRID_RETRIEVAL_ENABLED`: Enable dense + BM25 retrieval with RRF fusion. Default is `false`.
- `DENSE_TOP_K`: Number of dense vector candidates used by hybrid retrieval.
- `BM25_TOP_K`: Number of sparse keyword candidates used by hybrid retrieval.
- `FUSION_TOP_K`: Number of fused candidates kept before optional reranking.
- `RERANKER_ENABLED`: Enable optional cross-encoder reranking. Default is `false`.
- `RERANKER_MODEL`: Cross-encoder model used when reranking is enabled.
- `RERANKER_TOP_N`: Number of chunks kept after reranking when no explicit `top_k` is passed.
- `RERANKER_CANDIDATE_TOP_K`: Number of dense or fused candidates retrieved before reranking.
- `MAX_RETRY_COUNT`: Maximum failed-retrieval retry rewrites.
- `CHROMA_PERSIST_DIR`: Local Chroma persistence path.
- `CHROMA_COLLECTION_NAME`: Chroma collection name.
- `GRADIO_SERVER_NAME`: Gradio host.
- `GRADIO_SERVER_PORT`: Gradio port.
- `TRACE_LOGGING_ENABLED`: Enable local JSONL trace logging. Default is `false` when unset.
- `TRACE_LOG_DIR`: Directory for local trace JSONL files. Default is `./data/traces`.
- `API_UPLOAD_DIR`: Directory for FastAPI-uploaded files. Default is `./uploads/api`.
- `API_DOCUMENT_REGISTRY_PATH`: JSON registry for API-managed documents. Default is `./data/api_documents/registry.json`.
- `EVALUATION_RUN_DIR`: Directory for FastAPI-triggered evaluation run artifacts. Default is `./data/evaluation_runs`.

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
│   ├── bm25_retriever.py
│   ├── fusion.py
│   ├── hybrid_retriever.py
│   ├── retriever.py
│   └── reranker.py
├── tools/
│   ├── base.py
│   ├── registry.py
│   ├── factory.py
│   ├── retriever_tool.py
│   ├── citation_verifier_tool.py
│   ├── document_summary_tool.py
│   └── calculator_tool.py
├── agent/
│   ├── graph.py
│   ├── state.py
│   ├── nodes.py
│   ├── edges.py
│   ├── multi_query.py
│   ├── query_transform.py
│   ├── tools.py
│   └── prompts.py
├── api/
│   ├── main.py
│   ├── schemas.py
│   ├── dependencies.py
│   ├── routes/
│   └── services/
├── baseline/
│   ├── naive_rag.py
│   └── run_baseline.py
├── evaluation/
│   ├── baselines.py
│   ├── dashboard_models.py
│   ├── dashboard_formatters.py
│   ├── dashboard_service.py
│   ├── eval_questions.json
│   ├── evaluate.py
│   ├── failure_analyzer.py
│   └── runtime_config.py
├── experiments/
│   ├── run_ablation.py
│   ├── configs/
│   └── report.md
├── observability/
│   ├── trace.py
│   ├── storage.py
│   └── logger.py
├── docs/
│   ├── design.md
│   └── resume_bullets.md
├── ui/
│   └── gradio_app.py
├── assets/
│   └── architecture.png
├── sample_docs/
│   ├── agentic_rag_notes.md
│   ├── retrieval_pipeline_notes.md
│   ├── citation_verification_notes.md
│   ├── evaluation_notes.md
│   └── distractor_company_policy.md
└── tests/
```

## Resume Highlights

- Built a LangGraph-based Agentic RAG workflow that upgrades naive retrieve-generate RAG into a state-machine pipeline with structured query transformation, multi-query retrieval, hybrid retrieval, reranking, structured retrieval grading, partial-relevance recovery, conditional retry, citation-aware generation, claim-level citation verification, answer revision, and fallback.
- Implemented a configurable dense retrieval + BM25 sparse retrieval + RRF fusion pipeline so the system can combine semantic recall with exact keyword, filename, and identifier matching.
- Added reranker evaluation readiness with explicit candidate top-k vs final top-n settings, structured reranker records, and sanitized runtime config snapshots in evaluation artifacts.
- Added a standalone naive RAG baseline and comparison runner so Agentic RAG can be evaluated against retrieve-once RAG on the same documents and same questions.
- Designed a reliability evaluation foundation covering correctness, context relevance, source hit rate, citation hit rate, fallback accuracy, unsupported claims, retry count, latency, token usage, and cost fields.
- Expanded the default evaluation dataset to 36 structured questions across single-doc, multi-chunk, ambiguous, unanswerable, distractor, comparison, follow-up, citation-sensitive, cross-file, and false-premise cases.
- Added executable V0-V6 cumulative ablation artifacts with distinct Agent feature flags or retrieval settings, preventing repeated full-workflow runs from being misrepresented as module-level evidence.
- Designed a typed internal Tool Registry and dependency-injection boundary that unifies retriever, claim citation verifier, document summary, and safe calculator tools with validated arguments, normalized results, stable error semantics, and compact trace diagnostics.
- Added local JSONL trace logging so each Agent run can expose node events, route decisions, compact tool calls, evidence summaries, final answer metadata, latency, and errors.
- Added a FastAPI backend for chat, trace lookup, document upload/index/delete, and evaluation run retrieval.
- Added workspace-aware retrieval isolation for dense, BM25, hybrid, retriever, and Agent default retrieval paths.
- Added deterministic failed-case analysis with rule-based failure attribution, per-question reasons and suggestions, summary counts, and representative ablation cases.
- Added a Gradio Evaluation Dashboard for synchronous selected-record comparison, filterable failure diagnostics, and read-only inspection of saved V0-V6 artifacts and runtime configuration.
- Preserved a modular roadmap toward background execution, shared run IDs, trace drill-down, prompt regression tracking, and historical evaluation trends.

## Current Limitations

- Claim-level citation verification is LLM-based. It checks extracted claims against selected evidence chunks and can revise once, but it is not a formal proof system.
- Citation marker consistency is deterministic, but it only checks marker/index alignment. It does not prove that every cited claim is true.
- Retrieval grading depends on LLM JSON output. The parser is defensive, but malformed grading output is treated conservatively.
- `partially_relevant` grading triggers query-refinement recovery and still refuses to answer without directly relevant evidence; it does not yet dynamically increase top-k or rerun reranking with adjusted thresholds.
- Query transformation executes `expanded_queries` for `multi_query` strategy, but decomposition `sub_questions` are still recorded as metadata rather than executed as separate retrieval hops.
- Hybrid BM25 retrieval is lightweight exact-token matching. It supports workspace-filtered corpora, but it does not currently include stemming or learned sparse expansion.
- Evaluation uses a local 36-question reliability dataset. It is useful for reproducible project-level comparison, but it is not a benchmark-grade public dataset.
- P0b ablation variants are executable and distinct, but they are cumulative. They show incremental system trade-offs, not fully isolated causal effects for every module interaction.
- Trace logging currently writes local JSONL records and exposes API lookup through the FastAPI layer. It does not yet provide database-backed retention or a visual trace explorer.
- FastAPI endpoints are integration-oriented but not production-hardened. They do not yet include authentication, authorization, async job queues, rate limiting, or tenant-level access control.
- Workspace isolation currently uses metadata filtering inside a shared Chroma collection. It does not yet create separate collections per workspace or migrate historical unscoped chunks.
- Token usage and estimated cost are recorded only when the active model client exposes usage metadata.
- The Gradio `Build Index` workflow intentionally rebuilds the active collection for a clean uploaded knowledge base. The lower-level vectorstore API also supports incremental `add_documents` with deterministic IDs.
- Quick evaluation in the Gradio Evaluation Dashboard is synchronous. Selecting all 36 questions may be slow and may incur model cost.
- The Ablation Snapshot is read-only and only loads the existing artifact; it cannot launch V0-V6 runs from the browser.
- Dashboard failure diagnostics use deterministic heuristics for debugging and regression triage, not benchmark-grade causal attribution.
- The project is not a complete production deployment without authentication, authorization, deployment hardening, durable trace storage, and operational monitoring.

## Roadmap

### Completed Work

- RAG core implemented: loading, chunking, embeddings, Chroma indexing, and retrieval.
- LangGraph agent workflow implemented: query transformation, retriever tool, retrieval grading, retry routing, answer generation, and fallback.
- Gradio upload and QA flow implemented: document indexing, Agentic QA, citations, retrieved chunks, and retry diagnostics.
- P0a evaluation infrastructure implemented: naive baseline, richer schema, 36-question dataset, reliability metrics, JSON artifacts, and ablation scaffolding.
- P0b real V0-V6 ablation matrix implemented and run with observed DeepSeek metrics and documented trade-offs.
- P2 claim-level citation verification implemented: draft answers are split into claims, verified against cited chunks, revised once when unsupported, and otherwise routed to fallback.
- Deterministic citation marker consistency implemented: answer markers must match selected citation indices.
- Deterministic vectorstore IDs implemented: chunk identity is derived from source metadata and content for incremental add de-duplication.
- Optional reranker implemented: vector retrieval can over-retrieve candidates, apply a local cross-encoder reranker, and pass reranked chunks into grading.
- P1a hybrid retrieval implemented: dense retrieval, BM25 sparse retrieval, RRF fusion, and configurable dense/BM25/fusion top-k values.
- P1b reranker evaluation readiness implemented: explicit `RERANKER_TOP_N`, structured reranker records, and sanitized runtime config snapshots in evaluation/ablation artifacts.
- P1c structured query transformation implemented: direct rewrite, multi-query expansion metadata, decomposition metadata, and result payload fields.
- P1d multi-query retrieval execution implemented: `expanded_queries` are retrieved, de-duplicated, and merged with matched-query diagnostics before grading.
- P1e structured retrieval grading implemented: chunk-level relevance labels, confidence scores, reasons, and result payload diagnostics.
- P1f partial-relevance recovery implemented: related-but-insufficient chunks trigger query-refinement retry context while preserving fallback safety.
- Ollama local LLM support implemented through `LLM_PROVIDER=ollama`.
- P3a local trace logging implemented: node events, route decisions, final answers, citations, compact evidence summaries, latency, retry counts, and errors can be saved to JSONL.
- P3b FastAPI service layer implemented: chat, trace lookup, API-managed document upload/index/delete, and evaluation run retrieval.
- P3c workspace-aware retrieval implemented: API-indexed documents carry workspace metadata, and Agent retrieval can filter dense, BM25, and hybrid candidates by workspace.
- P3d typed internal Tool Registry implemented: retriever, citation verifier, document summary, and safe calculator tools share validated inputs, normalized results, stable errors, and compact trace diagnostics.
- P4a deterministic failed-case analysis implemented: evaluation results include primary failure types, reasons, and suggestions, while summaries and ablation reports include failure counts and representative cases.
- P4b Evaluation Dashboard implemented: Gradio supports synchronous smoke or manually selected quick comparisons, filterable failed-case inspection, and a read-only V0-V6 ablation snapshot with runtime configuration and transparent stored or derived diagnostics.

### Next Milestones

- Upgrade the current Approach A evaluator to Approach B: split dataset loading, schemas, metrics, runners, reporting, and result IO into dedicated modules with typed records, pluggable runners, optional judges, storage backends, and prompt/model config snapshots.
- Add dynamic partial-relevance recovery, such as increasing top-k or reranking again when chunks are only partially relevant.
- Add decomposition sub-question retrieval for multi-hop workflows.
- Harden workspace isolation with optional per-workspace Chroma collections and authorization checks.
- Implement `BackgroundAblationRunner` behind the dashboard runner protocol.
- Add background run status, progress, cancellation, and checkpoint recovery.
- Share evaluation run IDs across Gradio and FastAPI.
- Link failed cases to `trace_id` and a node-level trace viewer.
- Add prompt version tracking and prompt regression checks.
- Persist historical runs and expose trend views.
- Add model-specific prompt tuning and cost/latency evaluation for local and remote models.
- Add human-reviewed claim labels for stricter citation validation.
