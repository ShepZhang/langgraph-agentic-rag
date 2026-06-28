# GitHub Release Checklist

Use this checklist before publishing the repository or cutting a public version.

## Current Release Candidate

- Repository: `ShepZhang/langgraph-agentic-rag`
- URL: `https://github.com/ShepZhang/langgraph-agentic-rag`
- Branch: `codex/p5b-sqlite-eval-history`
- Target integration branch: `main`
- Version label: `v0.5.1-p5b`
- Tag target: `v0.5.1-p5b`
- Evaluation artifact schema: `4`
- Evaluator version: `p5b`
- Prompt registry: 11 registered `v1` prompts, 9 active
- Positioning: reliability-oriented Agentic RAG document QA system
- Main entry points:
  - Gradio demo: `python app.py`
  - FastAPI backend: `uvicorn api.main:app --reload`
  - Evaluation CLI: `python -m evaluation.evaluate --questions evaluation/eval_questions.json --output-dir evaluation/results`
  - Ablation CLI: `python -m experiments.run_ablation --questions evaluation/eval_questions.json --output-dir experiments/results`

## P5b SQLite Historical Evaluation Release Notes

- SQLite history database: `data/evaluation_history.sqlite3`
- Runtime controls:
  - `EVALUATION_HISTORY_ENABLED=true`
  - `EVALUATION_HISTORY_DB=./data/evaluation_history.sqlite3`
- `data/evaluation_history.sqlite3` is ignored runtime data and must not be
  committed.
- JSON artifacts remain the complete compatibility payload. SQLite stores only
  normalized summaries, failure counts, sanitized runtime config, and prompt
  manifests.
- Prompt manifests store prompt IDs, versions, and fingerprints only. No
  secrets, full prompt templates, or rendered prompt payloads are stored.
- `HistoryStore.save_record()` re-applies sanitization to runtime config,
  prompt manifests, summaries, metrics, and failure counts before SQLite writes.
- Background evaluation and trace drill-down remain future milestones.

## Verification Commands

Run these before publishing:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m compileall prompting agent rag api evaluation experiments baseline tools observability
.venv/bin/python -m pytest \
  tests/test_evaluation_history_store.py \
  tests/test_evaluation_storage.py \
  tests/test_evaluate.py \
  tests/test_fastapi_routes.py \
  tests/test_dashboard_service.py \
  tests/test_gradio_app.py \
  tests/test_ablation.py \
  tests/test_evaluation_matrix.py \
  -q
rg -n \
  'OPENAI_API_KEY|EVALUATION_JUDGE_API_KEY|Bearer |rendered prompt|full prompt template' \
  evaluation api ui README.md CHANGELOG.md docs/github_release_checklist.md \
  .env.example
git diff --check
```

Observed verification for `v0.5.1-p5b`:

- Full test suite: `.venv/bin/python -m pytest -q` → `672 passed in 4.31s`
- Focused compatibility suite: `.venv/bin/python -m pytest tests/test_evaluation_history_store.py tests/test_evaluation_storage.py tests/test_evaluate.py tests/test_fastapi_routes.py tests/test_dashboard_service.py tests/test_gradio_app.py tests/test_ablation.py tests/test_evaluation_matrix.py -q` → `194 passed in 4.50s`
- Focused history tests: `.venv/bin/python -m pytest tests/test_evaluation_history_store.py tests/test_evaluation_storage.py tests/test_evaluate.py -q` → `74 passed in 1.64s`
- API/Dashboard compatibility tests: `.venv/bin/python -m pytest tests/test_fastapi_routes.py tests/test_dashboard_service.py tests/test_gradio_app.py -q` → `83 passed in 4.45s`
- Ruff: `.venv/bin/python -m ruff check .` → `All checks passed!`
- Python `compileall`: `Listing 'prompting'...` through `Listing 'observability'...`, exit code `0`
- Whitespace check: `git diff --check` → no output, exit code `0`
- Forbidden persistence scan: `rg -n 'OPENAI_API_KEY|EVALUATION_JUDGE_API_KEY|Bearer |rendered prompt|full prompt template' evaluation api ui README.md CHANGELOG.md docs/github_release_checklist.md .env.example` returned only environment variable names/placeholders, documentation safety statements, and sanitizer code references; no literal secrets were found.

## GitHub Project Narrative

Use this short description in the GitHub repository summary:

> Reliability-oriented Agentic RAG document QA system built with LangGraph,
> hybrid retrieval, reranking, retrieval grading, retry/fallback routing,
> claim-level citation verification, optional DeepSeek semantic judging,
> versioned prompt fingerprints, trace logging, FastAPI, Gradio, and
> evaluation/ablation tooling with SQLite-backed history trends.

Recommended topics:

- `agentic-rag`
- `langgraph`
- `rag`
- `hybrid-retrieval`
- `reranking`
- `citation-verification`
- `fastapi`
- `gradio`
- `evaluation`
- `llm-agents`

## Suggested Publish Flow

```bash
git switch main
git status --short
git log --oneline --decorate --max-count=5
# Run every command in "Verification Commands" above and confirm expected counts.
git tag v0.5.1-p5b
git push origin main
git push origin v0.5.1-p5b
```

The tag `v0.5.1-p5b` is created only after user-approved integration into an
updated `main` and a successful full-suite run on merged `main`. Also confirm
the worktree is clean and the final local verification matches the observed
baseline above.

## Honest Scope Notes

- The system is production-oriented and reliability-oriented, not a complete
  production deployment.
- FastAPI endpoints do not yet include authentication, authorization, async
  evaluation jobs, or tenant-level access control.
- Deterministic evaluation metrics remain independent and unchanged when the
  optional semantic Judge is enabled.
- The Judge is disabled by default, uses independent configuration, and adds
  one model call per successful system result. Comparison adds two calls per
  question, and ablation runs can increase latency and cost substantially.
- Judge scores can inherit model bias and are model-based signals, not human
  ground truth.
- P5a adds Judge fields to raw reports and existing consumers but adds no
  Evaluation Dashboard UI.
- P5b adds a read-only Dashboard history tab and FastAPI history routes. It does
  not add background evaluation jobs, progress, cancellation, retries, or
  per-question trace drill-down.
- Prompt versioning provides deterministic template fingerprints and safe
  manifests for 11 registered `v1` templates: 9 active runtime prompts and 2
  inactive compatibility-only templates. It does not provide dynamic prompt
  selection, online editing, or behavioral LLM regression testing.
- Claim-level citation verification reduces unsupported claims but is not a
  formal proof system.
