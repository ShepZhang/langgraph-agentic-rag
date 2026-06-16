# GitHub Release Checklist

Use this checklist before publishing the repository or cutting a public version.

## Current Release Candidate

- Repository: `ShepZhang/langgraph-agentic-rag`
- URL: `https://github.com/ShepZhang/langgraph-agentic-rag`
- Branch: `main`
- Version label: `v0.4.2-p4c`
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
.venv/bin/python -m compileall agent rag api evaluation experiments baseline tools observability
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
```

Expected baseline as of `v0.4.2-p4c`:

- Full test suite: `469 passed`
- CLI compatibility smoke tests: `3 passed`
- Ablation, matrix, dashboard, and FastAPI compatibility tests: `82 passed`

## GitHub Project Narrative

Use this short description in the GitHub repository summary:

> Reliability-oriented Agentic RAG document QA system built with LangGraph,
> hybrid retrieval, reranking, retrieval grading, retry/fallback routing,
> claim-level citation verification, trace logging, FastAPI, Gradio, and
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
git status --short
git log --oneline --decorate --max-count=5
git tag v0.4.2-p4c
git remote add origin git@github.com:ShepZhang/langgraph-agentic-rag.git
git push -u origin main
git push origin v0.4.2-p4c
```

Only create the tag once the final local verification matches the expected
baseline above.

## Honest Scope Notes

- The system is production-oriented and reliability-oriented, not a complete
  production deployment.
- FastAPI endpoints do not yet include authentication, authorization, async
  evaluation jobs, or tenant-level access control.
- Evaluation metrics are deterministic heuristics unless a future semantic judge
  is configured.
- Claim-level citation verification reduces unsupported claims but is not a
  formal proof system.
