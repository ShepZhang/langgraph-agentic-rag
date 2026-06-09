# Reproducible Demo Guide

This guide is a compact, reproducible script for interviews and project demos.
It is not a marketing page. The goal is to show the same local corpus, index,
UI workflow, reranker comparison, fallback behavior, and evaluation command that
support the portfolio benchmark.

Live model outputs can vary across providers and repeated runs. The evaluation
report in `docs/evaluation.md` provides one fixed DeepSeek run for comparison,
with raw output saved under `evaluation/results/`.

## Environment Setup

Create and activate a virtual environment before running the commands below:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then configure the OpenAI-compatible provider
settings for DeepSeek or another compatible endpoint. The matrix runner fails
early when the chat LLM is not configured. Fill in the provider credential only
in your local `.env`; do not commit it and do not paste it into demo notes.

Local Ollama can also be used by setting `LLM_PROVIDER=ollama` and configuring
the Ollama model fields. The published benchmark used DeepSeek through the
OpenAI-compatible path, so use the same provider family when reproducing that
report.

The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`. The
reranker demo uses `cross-encoder/ms-marco-MiniLM-L-6-v2`. First-time users may
need network access to download these Hugging Face models; later runs can use
the local cache. In restricted environments, set `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` only after the models are already cached.

## Build The Sample Index

Run this from the repository root to index every Markdown file in `sample_docs`.
The command follows the same loader -> chunker -> create_vectorstore path used
by the app and prints the number of files, documents, and chunks.

```bash
python -c "from pathlib import Path; from rag.loader import load_documents; from rag.chunker import split_documents; from rag.vectorstore import create_vectorstore; files=sorted(Path('sample_docs').glob('*.md')); docs=load_documents(files); chunks=split_documents(docs); create_vectorstore(chunks); print(f'files={len(files)} docs={len(docs)} chunks={len(chunks)}')"
```

## Start The UI

If the virtual environment is not activated, use the repository-local Python
interpreter:

```bash
.venv/bin/python app.py
```

If the virtual environment is activated, this shorter command is equivalent:

```bash
python app.py
```

If the system only exposes Python 3 as `python3`, use this after activating the
virtual environment:

```bash
python3 app.py
```

Open the Gradio URL printed by the process. The default host and port are
configured in `.env.example`.

## Reranker Toggle

Use the off setting for the baseline agentic run:

```dotenv
RERANKER_ENABLED=false
TOP_K=4
RERANKER_CANDIDATE_TOP_K=12
```

Use the on setting to retrieve a larger candidate pool, rerank it with the
cross-encoder, and return the final top K chunks:

```dotenv
RERANKER_ENABLED=true
TOP_K=4
RERANKER_CANDIDATE_TOP_K=12
```

`RERANKER_CANDIDATE_TOP_K` controls the candidate top_k gathered before
reranking. `TOP_K` controls how many chunks remain after reranking and are shown
to the agent and UI.

## Evaluation Command

Run the fixed question matrix after building the sample index and configuring a
chat LLM:

```bash
python -m evaluation.matrix --questions evaluation/eval_questions.json --json-output evaluation/results/deepseek_matrix_YYYY-MM-DD.json
```

## Demo Sequence

### Direct Answer

Question(s): `What is the maximum Atlas upload size?`

Expected source: `product_specs.md`

Expected route/behavior: The agent should return a direct answer, route to a
normal answer, avoid fallback, and cite the Atlas upload-size evidence.

UI fields to inspect: `Answer`, `Citations`, `Retrieved chunks`. Confirm the
retrieved chunks include `product_specs.md` and the answer cites the selected
evidence.

### Contextual Follow-Up

Question(s): First ask `What is Agentic RAG?`, then ask `How does it improve reliability?`

Expected source: `agentic_rag_notes.md`

Expected route/behavior: The second question should be rewritten using chat
history so the current query is about Agentic RAG reliability instead of a vague
"it". The agent should answer from the Agentic RAG notes when the retrieved
chunks support the follow-up.

UI fields to inspect: `Rewritten question`, `Retry count`, `Answer`,
`Citations`, `Retrieved chunks`. The visible `Rewritten question` field is the
UI display for the agent's current retrieval query; verify the retry count stays
reasonable for a supported follow-up.

### Query Rewrite

Question(s): `How much time off do people get?`

Expected source: `employee_handbook.md`

Expected route/behavior: The agent should normalize the casual wording into a
paid-time-off query, retrieve handbook evidence, and answer with the annual PTO
fact.

UI fields to inspect: `Rewritten question`, `Answer`, `Citations`,
`Retrieved chunks`. Confirm the visible rewritten question reflects the
normalized PTO intent and the retrieved chunks include `employee_handbook.md`.

### Reranker

Question(s): `How does Atlas select the strongest retrieval candidates before grading?`

Expected source: `product_specs.md`

Expected route/behavior: Run once with `RERANKER_ENABLED=false`, then restart
with `RERANKER_ENABLED=true` and run again. With reranking enabled, the retriever
collects the candidate top_k, reorders the candidates, and returns the final
top K chunks before retrieval grading.

UI fields to inspect: `Retrieved chunks`, `Answer`, `Citations`,
`Diagnostics`. Compare source order and scores between the off and on runs. If
reranking is enabled, retrieved chunks should include `rerank_score` values.

### Correct Fallback

Question(s): `What are the company's salary bands?`

Expected source: None. The sample corpus does not include salary-band data.

Expected route/behavior: The agent should retry within the configured limit,
find no relevant supporting documents, and return a fallback instead of
inventing salary bands. The final result should have no citations.

UI fields to inspect: `Answer`, `Citations`, `Retrieved chunks`,
`Retry count`, `Diagnostics`. Confirm the answer says the current documents
cannot answer, citations are empty, and diagnostics explain the fallback.

### Citation Safety

Question(s): No live model failure is required. Use the offline tests for
malformed citation output, invalid citation indices, empty citation indices, and
marker mismatch.

Expected source: The behavior is validated by unit tests, not a source document.

Expected route/behavior: A normal answer with missing citations, out-of-range
citation markers, all-invalid citation indices, or mismatched citation markers
must fall back instead of returning an unsupported answer. An explicit
unable-to-answer response may have no citations.

UI fields to inspect: For a normal successful answer, inspect `Answer` and
`Citations` to verify marker/index alignment. For malformed marker/index cases,
inspect `tests/test_agent_nodes.py`; the live demo should not depend on forcing
a model to fail in a specific way.
