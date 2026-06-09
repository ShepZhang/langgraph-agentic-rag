# DeepSeek Evaluation Benchmark

## Scope

This benchmark was run on June 7, 2026 against the three evaluation variants in `evaluation.matrix`:

- LLM: `deepseek-v4-flash`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Reranker model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Corpus: 4 fictional Markdown documents, indexed as 20 chunks
  - `agentic_rag_notes.md`
  - `employee_handbook.md`
  - `product_specs.md`
  - `security_policy.md`
- Evaluation set: 34 project-specific questions

The corpus describes fictional systems and organizations. It does not contain production data, live infrastructure details, or customer information.

## Results

| Metric | Naive RAG | Agentic RAG | Agentic + Reranker |
|---|---:|---:|---:|
| Retrieval Source Hit Rate | 1.0 | 1.0 | 1.0 |
| Keyword Hit Rate | 0.7143 | 0.7143 | 0.75 |
| Citation Rate | 0.8235 | 0.7647 | 0.7941 |
| Claim Verification Rate | 0.0 | 0.7647 | 0.7941 |
| Fallback Correctness | 0.9706 | 0.9412 | 0.9706 |
| Average Retry Count | 0.0 | 0.4118 | 0.3824 |
| Average Retrieved Docs | 4.0 | 4.0 | 4.0 |
| Average Relevant Docs | 3.8824 | 1.0882 | 1.0588 |
| Average Latency | 2.3173 | 13.1514 | 12.3881 |
| Error Count | 0 | 0 | 0 |

## Metric Definitions

- `source_hit_rate` is retrieval source hit rate. The denominator is the 28 questions with one or more `expected_sources`, and a hit only requires retrieval to include the expected source document or documents. It does not require the final answer to succeed.
- `citation_rate` only means the result payload contained a non-empty `citations` list. A fallback or partial answer can still carry a citation.
- `keyword_hit_rate` requires a normal non-fallback answer that contains all expected keywords for that question.
- `verification_rate` is the rate of the system's LLM verifier setting `is_verified=true`. It is not a human correctness rate.
- `average_relevant_docs` is not identical across variants. In Naive RAG, `relevant_documents` is the baseline payload's returned context and approximates retrieved context size. In the agentic variants, `relevant_documents` is produced after LLM retrieval grading, so it is not fully comparable to the naive count as an independent relevance label.

`source_hit` and `citation_rate` measure different stages. A source hit means retrieval included the expected source document or documents. Citation rate means the final system result returned citations in its payload. A run can retrieve the expected source and still fall back, return a partial answer, or return no final citation.

## Interpretation

Compared with Naive RAG, Agentic RAG was flat on retrieval source hit rate (`1.0`) and keyword hit rate (`0.7143`). It regressed on answer rate (`0.7941` to `0.7647`), citation rate (`0.8235` to `0.7647`), and fallback correctness (`0.9706` to `0.9412`). It added claim verification (`0.0` to `0.7647`). The agentic variants reduced the context passed after retrieval grading; the naive `average_relevant_docs` value is better read as baseline context size, not as an independently graded relevance label. The zero Naive RAG verification rate reflects that the baseline does not run the claim-verification stage, so this comparison is a workflow difference as well as a measured outcome.

Adding the reranker to Agentic RAG improved keyword hit rate from `0.7143` to `0.75`, citation rate from `0.7647` to `0.7941`, verification rate from `0.7647` to `0.7941`, fallback correctness from `0.9412` to `0.9706`, and answer rate from `0.7647` to `0.7941`. Retrieval source hit rate remained flat at `1.0`, so this run does not show a source-hit improvement from reranking. The reranked agentic run passed a slightly smaller graded context on average than Agentic RAG (`1.0588` versus `1.0882`).

The agentic control flow had a substantial latency cost. Average latency increased from `2.3173` seconds for Naive RAG to `13.1514` seconds for Agentic RAG. Agentic + Reranker averaged `12.3881` seconds, which was `0.7633` seconds lower than Agentic RAG in this run but still `10.0708` seconds higher than Naive RAG. This single matrix cannot isolate cross-encoder cost from different retry, grading, generation, and verification paths, so the lower reranked average should not be interpreted as evidence that reranking itself is free or faster.

Fallback and verification behavior was mixed rather than uniformly better. Both agentic variants triggered seven rewrites. Agentic + Reranker recovered Naive RAG's `0.9706` fallback correctness and raised verification to `0.7941`, but individual answerable questions still produced incorrect fallbacks or partial answers.

## Case Studies

### 1. Direct factual success

Question: "How much annual paid time off do full-time employees receive?"

All three systems returned the exact supported fact, "20 days of paid time off per calendar year," with `source_hit=true`, `keyword_hit=true`, `citation_returned=true`, `fallback_triggered=false`, and `retry_count=0`. Agentic RAG and Agentic + Reranker also reported `is_verified=true`; Naive RAG does not run claim verification.

### 2. Evidence-selection and verification anomaly

Question: "How do Atlas citation checks and Northstar secret management reduce answer and credential risk?"

Naive RAG answered without retry and recorded `source_hit=true`, `keyword_hit=true`, and `citation_returned=true`. Agentic RAG selected three relevant documents but returned a fallback with `retry_count=0`, `citation_returned=false`, and `is_verified=false`. Agentic + Reranker used `retry_count=1`, selected security-policy context, returned an answer with a citation, and recorded `is_verified=true`. However, its final answer still said it could not address the Northstar secret-management portion, and `keyword_hit=false`. This is an unstable partial failure: `is_verified=true` only means the verifier accepted the claims made in that answer, not that the system answered the full original question.

### 3. Reranker ordering changed a cross-document outcome

Question: "Which file formats does Atlas support, and how long does temporary production access last?"

The reranked retrieval order began with `product_specs.md`, followed by two `security_policy.md` chunks, then another `product_specs.md` chunk. Agentic + Reranker selected two relevant documents and answered both parts: "PDF, Markdown, and TXT" and "four hours." It recorded `source_hit=true`, `keyword_hit=true`, `citation_returned=true`, `is_verified=true`, `fallback_triggered=false`, and `retry_count=0`.

Naive RAG retrieved both expected sources but answered only the Atlas portion before falling back. Agentic RAG retried twice and still fell back. Their `fallback_correct` values were false because this was an answerable question. This case shows a concrete ordering and selection benefit, not a universal reranker advantage.

### 4. Correct fallback for absent information

Question: "What are Northstar Labs' salary bands?"

The corpus contains no salary-band information. All three systems returned a fallback with `fallback_correct=true`, `answer_returned=false`, `citation_returned=false`, `source_hit=false`, and `is_verified=false`. Naive RAG used `retry_count=0`; both agentic variants used `retry_count=2` and ended with zero relevant documents. The agentic retries increased latency without changing the correct final decision.

### 5. Reranker regression on an answerable question

Question: "How does it improve reliability compared with a one-pass pipeline?"

Naive RAG and Agentic RAG both answered with citations and `source_hit=true`; Agentic RAG also reported `is_verified=true`. Agentic + Reranker retrieved the expected source but returned an incorrect fallback with `fallback_correct=false`, `citation_returned=false`, `is_verified=false`, and `retry_count=0`. Its relevant-document count was one, versus two for Agentic RAG. This result demonstrates that a perfect retrieval source hit does not guarantee a complete evidence selection or a cited final answer.

## Limitations

- The evaluation set is small and tailored to this project, its prompts, and four fictional Markdown documents.
- Keyword matching, LLM retrieval grading, and LLM claim verification are automated proxies, not human judgments of full answer quality.
- The benchmark uses one configured LLM, `deepseek-v4-flash`; it does not establish behavior across providers, model versions, temperatures, or repeated runs.
- There are no independent human relevance labels or human answer-quality labels.
- Latency reflects this environment and the control-flow paths taken during one run; it is not a general performance guarantee.
- These results are not a universal RAG benchmark and do not prove that Agentic RAG or reranking is always better.

The committed raw JSON includes retrieved snippets only from fictional sample documents. Future benchmarks over real or private corpora should publish a redacted summary artifact instead of raw retrieved text.

## Reproduction Notes

The run used `openai_compatible` provider settings with a DeepSeek-compatible base URL configured in `.env`; the API key is not included in this artifact. `LLM_TEMPERATURE=0` was the default and setting used for this run. Chroma persisted to `./data/chroma`.

Rebuild the four-document sample index:

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890 ALL_PROXY=http://127.0.0.1:7890 \
  .venv/bin/python -c "from pathlib import Path; from rag.loader import load_documents; from rag.chunker import split_documents; from rag.vectorstore import create_vectorstore; files=sorted(Path('sample_docs').glob('*.md')); docs=load_documents(files); chunks=split_documents(docs); create_vectorstore(chunks); print(f'files={len(files)} documents={len(docs)} chunks={len(chunks)}')"
```

Run the matrix:

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890 ALL_PROXY=http://127.0.0.1:7890 \
  .venv/bin/python -m evaluation.matrix --questions evaluation/eval_questions.json --json-output evaluation/results/deepseek_matrix_2026-06-07.json
```

During this run, the embedding and reranker models were loaded from the local Hugging Face cache with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. First-time users may need network access to download `sentence-transformers/all-MiniLM-L6-v2` and `cross-encoder/ms-marco-MiniLM-L-6-v2`. DeepSeek API calls require network access and may require a proxy depending on the environment.
