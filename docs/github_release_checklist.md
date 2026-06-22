# GitHub Release Checklist

Use this checklist before publishing the repository or cutting a public version.

## Current Release Candidate

- Repository: `ShepZhang/langgraph-agentic-rag`
- URL: `https://github.com/ShepZhang/langgraph-agentic-rag`
- Branch: `main`
- Version label: `v0.5.0-p5a`
- Evaluation artifact schema: `3`
- Evaluator version: `p5a`
- Prompt registry: 11 registered `v1` prompts, 9 active
- Positioning: reliability-oriented Agentic RAG document QA system
- Main entry points:
  - Gradio demo: `python app.py`
  - FastAPI backend: `uvicorn api.main:app --reload`
  - Evaluation CLI: `python -m evaluation.evaluate --questions evaluation/eval_questions.json --output-dir evaluation/results`
  - Ablation CLI: `python -m experiments.run_ablation --questions evaluation/eval_questions.json --output-dir experiments/results`

## Verification Commands

Run these before publishing:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall prompting agent rag api evaluation experiments baseline tools observability
.venv/bin/python -m pytest \
  tests/test_evaluation_judge_config.py \
  tests/test_evaluation_judge_evidence.py \
  tests/test_evaluation_judge_parsing.py \
  tests/test_evaluation_judges.py \
  tests/test_evaluation_schemas.py \
  tests/test_evaluation_runners.py \
  tests/test_evaluation_metrics.py \
  tests/test_evaluation_comparison.py \
  tests/test_evaluation_reporting.py \
  tests/test_prompt_registry.py \
  tests/test_prompt_catalog.py \
  -q
.venv/bin/python -m pytest \
  tests/test_evaluate.py::test_main_prints_report_with_injected_runner \
  tests/test_evaluate.py::test_main_writes_comparison_artifacts \
  tests/test_evaluate.py::test_main_writes_single_system_agentic_artifact_schema \
  -q
.venv/bin/python -m pytest \
  tests/test_ablation.py \
  tests/test_evaluation_matrix.py \
  tests/test_dashboard_service.py \
  tests/test_fastapi_routes.py \
  -q
rg -n \
  'evaluation\.semantic_judge|EVALUATION_JUDGE_' \
  README.md .env.example prompting evaluation tests
git diff --check
```

Observed verification for `v0.5.0-p5a`:

- Full test suite: `636 passed`
- Focused Judge and evaluation tests: `185 passed`
- CLI compatibility smoke tests: `3 passed`
- Ablation, matrix, Dashboard, and FastAPI compatibility tests: `91 passed`
- Python `compileall`: success

## GitHub Project Narrative

Use this short description in the GitHub repository summary:

> Reliability-oriented Agentic RAG document QA system built with LangGraph,
> hybrid retrieval, reranking, retrieval grading, retry/fallback routing,
> claim-level citation verification, optional DeepSeek semantic judging,
> versioned prompt fingerprints, trace logging, FastAPI, Gradio, and
> evaluation/ablation tooling.

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
git tag v0.5.0-p5a
git push origin main
git push origin v0.5.0-p5a
```

The tag `v0.5.0-p5a` is created only after user-approved integration into an
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
- Prompt versioning provides deterministic template fingerprints and safe
  manifests for 11 registered `v1` templates: 9 active runtime prompts and 2
  inactive compatibility-only templates. It does not provide dynamic prompt
  selection, online editing, or behavioral LLM regression testing.
- Claim-level citation verification reduces unsupported claims but is not a
  formal proof system.
