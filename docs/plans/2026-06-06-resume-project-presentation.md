# Resume Project Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing Agentic RAG repository into a reproducible, evidence-backed portfolio project with a realistic corpus, three-variant DeepSeek benchmark, demo guide, accurate architecture diagram, interview notes, and offline CI.

**Architecture:** Preserve the current LangGraph and retrieval implementation. Add a separate evaluation matrix layer that composes existing `run_naive_rag`, `run_agent`, and `Retriever` instances with explicit reranker settings. Keep benchmark artifacts and presentation documents outside runtime modules, and generate the architecture PNG from a checked-in script.

**Tech Stack:** Python 3.11+, LangGraph, LangChain, Chroma, sentence-transformers CrossEncoder, DeepSeek through the OpenAI-compatible client, pytest, Ruff, GitHub Actions, Pillow.

---

## File Structure

New files:

- `sample_docs/employee_handbook.md`: fictional HR and workplace policies.
- `sample_docs/product_specs.md`: fictional Atlas Document QA product specification.
- `sample_docs/security_policy.md`: fictional security and production-access policy.
- `evaluation/matrix.py`: reusable three-variant evaluation orchestration.
- `evaluation/results/deepseek_matrix_2026-06-06.json`: raw successful benchmark result.
- `docs/evaluation.md`: human-readable benchmark analysis.
- `docs/demo.md`: reproducible demonstration script.
- `scripts/generate_architecture.py`: deterministic architecture image generator.
- `tests/test_evaluation_matrix.py`: matrix orchestration and formatting tests.
- `tests/test_project_docs.py`: sample corpus and documentation-link checks.
- `pyproject.toml`: pytest and Ruff configuration.
- `.github/workflows/tests.yml`: offline CI workflow.

Modified files:

- `evaluation/eval_questions.json`: 34 questions over the four-document corpus.
- `evaluation/evaluate.py`: validate cross-document source matching and expose public helpers reused by matrix evaluation.
- `README.md`: document links, benchmark summary, interview talking points, updated structure.
- `requirements.txt`: add Pillow as the explicit architecture-generator dependency.
- `assets/architecture.png`: regenerated diagram.

## Task 1: Add A Realistic Sample Corpus

**Files:**
- Create: `sample_docs/employee_handbook.md`
- Create: `sample_docs/product_specs.md`
- Create: `sample_docs/security_policy.md`
- Create: `tests/test_project_docs.py`

- [ ] **Step 1: Write failing sample-corpus tests**

Add:

```python
from pathlib import Path

from rag.loader import load_documents


SAMPLE_DOCS = [
    "sample_docs/agentic_rag_notes.md",
    "sample_docs/employee_handbook.md",
    "sample_docs/product_specs.md",
    "sample_docs/security_policy.md",
]


def test_portfolio_sample_corpus_files_exist_and_load():
    paths = [Path(path) for path in SAMPLE_DOCS]

    assert all(path.exists() for path in paths)
    documents = load_documents(paths)

    assert len(documents) == 4
    assert {doc.metadata["source"] for doc in documents} == {
        "agentic_rag_notes.md",
        "employee_handbook.md",
        "product_specs.md",
        "security_policy.md",
    }
    assert all(len(doc.metadata["file_hash"]) == 64 for doc in documents)


def test_sample_corpus_contains_expected_benchmark_facts():
    combined = "\n".join(
        Path(path).read_text(encoding="utf-8") for path in SAMPLE_DOCS
    )

    for fact in [
        "20 days of paid time off",
        "three remote-work days per week",
        "25 MB",
        "PDF, Markdown, and TXT",
        "four hours",
        "managed secrets vault",
    ]:
        assert fact in combined
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_project_docs.py -q
```

Expected: FAIL because the three new Markdown files do not exist.

- [ ] **Step 3: Create the employee handbook**

Create `sample_docs/employee_handbook.md` with these exact sections and facts:

```markdown
# Northstar Labs Employee Handbook

## Working Model

Full-time employees may work remotely up to three days per week. Teams choose shared office days, while core collaboration hours are 10:00 AM to 3:00 PM Pacific Time.

## Paid Time Off

Full-time employees receive 20 days of paid time off per calendar year. Requests for five or more consecutive days must be submitted in the PeopleOps portal at least 10 business days before the first day of leave.

## Learning Budget

Each full-time employee receives an annual professional-development budget of USD 1,500. Manager approval is required before purchase, and reimbursement requests must include an itemized receipt.

## Travel Expenses

Travel expenses above USD 500 require manager approval before booking. Expense reports must be submitted within 30 days after the trip ends.
```

- [ ] **Step 4: Create the product specification**

Create `sample_docs/product_specs.md` with these exact sections and facts:

```markdown
# Atlas Document QA Product Specification

## Purpose

Atlas is a private document question-answering application for internal teams. It supports PDF, Markdown, and TXT files.

## Upload And Indexing

The maximum file size is 25 MB per uploaded file. Building an index replaces the active demo collection, while the lower-level indexing API supports incremental additions with deterministic chunk IDs.

## Retrieval Pipeline

Atlas retrieves vector-search candidates from Chroma. Optional cross-encoder reranking can reorder a larger candidate set before the LangGraph retrieval-grading node selects relevant evidence.

## Answer Safety

Normal answers must include citation markers that match selected citation indices. Answers then pass lightweight claim verification against the selected evidence chunks.

## Availability Target

The prototype has no formal production service-level agreement. It is intended for local demonstrations and controlled evaluation.
```

- [ ] **Step 5: Create the security policy**

Create `sample_docs/security_policy.md` with these exact sections and facts:

```markdown
# Northstar Labs Security And Access Policy

## Authentication

Multi-factor authentication is required for company email, source control, cloud administration, and the Atlas production environment.

## Production Access

Production access uses just-in-time approval and expires after four hours. The request must include a ticket reference and approval from the on-call engineering lead.

## Secret Management

API keys, database passwords, and service credentials must be stored in the managed secrets vault. Secrets must not be committed to source control or placed in shared documents.

## Access Reviews

System owners review privileged access every quarter. Access that is no longer required must be removed within one business day.

## Incident Reporting

Suspected credential exposure or unauthorized production access must be reported to the security channel within 30 minutes.
```

- [ ] **Step 6: Run sample-corpus tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_project_docs.py tests/test_loader.py tests/test_chunker.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add sample_docs/employee_handbook.md sample_docs/product_specs.md sample_docs/security_policy.md tests/test_project_docs.py
git commit -m "docs: add realistic sample knowledge base"
```

## Task 2: Expand And Validate The Evaluation Dataset

**Files:**
- Modify: `evaluation/eval_questions.json`
- Modify: `evaluation/evaluate.py`
- Modify: `tests/test_evaluate.py`
- Modify: `tests/test_project_docs.py`

- [ ] **Step 1: Write failing dataset-shape tests**

Add:

```python
from evaluation.evaluate import load_eval_questions


def test_portfolio_evaluation_set_has_balanced_question_groups():
    questions = load_eval_questions("evaluation/eval_questions.json")

    assert len(questions) >= 30
    assert sum(item["should_answer"] for item in questions) >= 20
    assert sum(not item["should_answer"] for item in questions) >= 6
    assert sum(item["requires_rewrite"] for item in questions) >= 6
    assert sum(item["source_match_mode"] == "all" for item in questions) >= 2

    expected_sources = {
        source
        for item in questions
        for source in item["expected_sources"]
    }
    assert {
        "agentic_rag_notes.md",
        "employee_handbook.md",
        "product_specs.md",
        "security_policy.md",
    }.issubset(expected_sources)
```

- [ ] **Step 2: Write failing source-match validation tests**

Add to `tests/test_evaluate.py`:

```python
import json

import pytest

from evaluation.evaluate import evaluate_questions, load_eval_questions


def test_load_eval_questions_accepts_all_source_match_mode(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question": "Cross-document question",
                    "expected_keywords": ["PDF", "four hours"],
                    "expected_sources": [
                        "product_specs.md",
                        "security_policy.md",
                    ],
                    "source_match_mode": "all",
                    "should_answer": True,
                    "requires_rewrite": False,
                }
            ]
        ),
        encoding="utf-8",
    )

    questions = load_eval_questions(path)

    assert questions[0]["source_match_mode"] == "all"


def test_load_eval_questions_rejects_unknown_source_match_mode(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question": "Invalid mode",
                    "expected_sources": ["product_specs.md"],
                    "source_match_mode": "some",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source_match_mode"):
        load_eval_questions(path)


def test_cross_document_source_hit_requires_all_expected_sources():
    questions = [
        {
            "question": "Cross-document question",
            "expected_keywords": ["PDF", "four hours"],
            "expected_sources": ["product_specs.md", "security_policy.md"],
            "source_match_mode": "all",
            "should_answer": True,
            "requires_rewrite": False,
        }
    ]

    report = evaluate_questions(
        questions,
        run_agent_fn=lambda question: {
            "answer": "Atlas supports PDF files and access lasts four hours.",
            "citations": [{"source": "product_specs.md"}],
            "retrieved_documents": [{"source": "product_specs.md"}],
        },
    )

    assert report["results"][0]["source_hit"] is False
```

- [ ] **Step 3: Run the tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_project_docs.py::test_portfolio_evaluation_set_has_balanced_question_groups \
  tests/test_evaluate.py::test_load_eval_questions_accepts_all_source_match_mode \
  tests/test_evaluate.py::test_load_eval_questions_rejects_unknown_source_match_mode \
  tests/test_evaluate.py::test_cross_document_source_hit_requires_all_expected_sources \
  -q
```

Expected: FAIL because the current set has only 20 questions, `source_match_mode` is not normalized, and source matching uses `any`.

- [ ] **Step 4: Implement explicit source-match semantics**

In `load_eval_questions()`, normalize and validate:

```python
source_match_mode = record.get("source_match_mode", "any")
if source_match_mode not in {"any", "all"}:
    raise ValueError("source_match_mode must be 'any' or 'all'")
normalized["source_match_mode"] = source_match_mode
```

Pass `item.get("source_match_mode", "any")` into `_has_expected_source()`. Update the helper:

```python
"source_hit": _has_expected_source(
    expected_sources,
    citations,
    retrieved_documents,
    item.get("source_match_mode", "any"),
),
```

Update the helper implementation:

```python
def _has_expected_source(
    expected_sources: Any,
    citations: list[Any],
    retrieved_documents: list[Any],
    source_match_mode: str = "any",
) -> bool:
    if not expected_sources:
        return False

    evidence = citations if citations else retrieved_documents
    observed_sources = {
        document["source"]
        for document in evidence
        if isinstance(document, dict)
        and isinstance(document.get("source"), str)
    }
    expected = set(expected_sources)
    if source_match_mode == "all":
        return expected.issubset(observed_sources)
    return bool(expected & observed_sources)
```

- [ ] **Step 5: Replace the evaluation set with 34 records**

Use these groups:

- 8 Agentic RAG questions from `agentic_rag_notes.md`.
  - Mark these three contextual/search-oriented records with `requires_rewrite: true`:
    - `"How does it improve reliability compared with a one-pass pipeline?"`
    - `"What happens when the first retrieved chunks are not useful?"`
    - `"How does the system keep its answer tied to evidence?"`
- 5 handbook questions:
  - remote-work allowance
  - core collaboration hours
  - annual PTO
  - long-leave notice
  - learning budget
- 5 product questions:
  - supported file types
  - maximum upload size
  - index rebuild versus incremental add
  - optional reranker role
  - prototype SLA limitation
- 5 security questions:
  - MFA scope
  - JIT access duration
  - production approval requirements
  - secrets storage
  - incident-reporting deadline
- 3 vague/search-oriented questions with `requires_rewrite: true`:
  - `"How much time off do people get?"`
  - `"How long does elevated access last?"`
  - `"What file limit does the document tool have?"`
- 2 cross-document questions with `source_match_mode: "all"`:
  - `"Which file formats does Atlas support, and how long does temporary production access last?"`
    - expected sources: `product_specs.md`, `security_policy.md`
    - expected keywords: `"PDF"`, `"four hours"`
  - `"How do Atlas citation checks and Northstar secret management reduce answer and credential risk?"`
    - expected sources: `product_specs.md`, `security_policy.md`
    - expected keywords: `"citation markers"`, `"managed secrets vault"`
- 6 unanswerable questions:
  - payroll salary bands
  - company CEO
  - quarterly revenue
  - production database hostname
  - customer retention rate
  - office parking policy

Every single-document answerable record must contain narrow expected keywords copied from source wording and exactly one expected source. Cross-document records must contain two expected sources and `source_match_mode: "all"`. Unanswerable records must use empty keyword/source arrays and `should_answer: false`. Every record must explicitly include `requires_rewrite`; at least six records must set it to `true`.

- [ ] **Step 6: Validate the dataset**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluate.py tests/test_project_docs.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add evaluation/eval_questions.json evaluation/evaluate.py tests/test_evaluate.py tests/test_project_docs.py
git commit -m "test: expand portfolio evaluation dataset"
```

## Task 3: Implement Three-Variant Evaluation Matrix

**Files:**
- Create: `evaluation/matrix.py`
- Create: `tests/test_evaluation_matrix.py`
- Modify: `evaluation/evaluate.py`

- [ ] **Step 1: Write failing matrix orchestration test**

Add:

```python
from evaluation.matrix import evaluate_matrix


def test_evaluate_matrix_keeps_three_variants_separate():
    questions = [
        {
            "question": "What is Atlas?",
            "expected_keywords": ["private"],
            "expected_sources": ["product_specs.md"],
            "should_answer": True,
            "requires_rewrite": False,
        }
    ]
    runners = {
        "naive": lambda question: {
            "answer": "Atlas is private.",
            "citations": [{"source": "product_specs.md"}],
            "retrieved_documents": [{"source": "product_specs.md"}],
            "relevant_documents": [{"source": "product_specs.md"}],
        },
        "agentic": lambda question: {
            "answer": "Atlas is private.",
            "citations": [{"source": "product_specs.md"}],
            "retrieved_documents": [{"source": "product_specs.md"}],
            "relevant_documents": [{"source": "product_specs.md"}],
            "is_verified": True,
            "claims": [{"claim": "Atlas is private.", "supported": True}],
        },
        "agentic_reranker": lambda question: {
            "answer": "Atlas is private.",
            "citations": [{"source": "product_specs.md"}],
            "retrieved_documents": [{"source": "product_specs.md", "rerank_score": 0.9}],
            "relevant_documents": [{"source": "product_specs.md", "rerank_score": 0.9}],
            "is_verified": True,
            "claims": [{"claim": "Atlas is private.", "supported": True}],
        },
    }

    report = evaluate_matrix(questions, runners=runners, timer=lambda: 0.0)

    assert list(report["summary"]["variants"]) == [
        "naive",
        "agentic",
        "agentic_reranker",
    ]
    assert report["results"][0]["systems"]["agentic_reranker"]["source_hit"] is True
```

- [ ] **Step 2: Write failing Markdown formatter test**

Add:

```python
from evaluation.matrix import format_matrix_report


def test_format_matrix_report_includes_all_variants():
    report = {
        "summary": {
            "total_questions": 1,
            "variants": {
                "naive": {"source_hit_rate": 0.5, "average_latency": 1.0},
                "agentic": {"source_hit_rate": 0.75, "average_latency": 2.0},
                "agentic_reranker": {
                    "source_hit_rate": 1.0,
                    "average_latency": 2.5,
                },
            },
        },
        "results": [],
    }

    text = format_matrix_report(report)

    assert "| Metric | Naive RAG | Agentic RAG | Agentic + Reranker |" in text
    assert "| Source Hit Rate | 0.5 | 0.75 | 1.0 |" in text
```

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_matrix.py -q
```

Expected: collection error because `evaluation.matrix` does not exist.

- [ ] **Step 4: Expose reusable evaluation helpers**

Rename or wrap private helpers in `evaluation/evaluate.py`:

```python
def evaluate_single_system(
    item: dict[str, Any],
    runner: Callable[[str], dict[str, Any]],
    timer: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    return _evaluate_single_system(item, runner, timer)


def summarize_results(
    results: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    return _summarize(results, questions)
```

Do not change the existing `evaluate_questions()` behavior.

- [ ] **Step 5: Implement generic matrix evaluation**

Create `evaluation/matrix.py` with:

```python
VARIANT_LABELS = {
    "naive": "Naive RAG",
    "agentic": "Agentic RAG",
    "agentic_reranker": "Agentic + Reranker",
}


def evaluate_matrix(questions, runners, timer=time.perf_counter):
    variant_results = {name: [] for name in runners}
    rows = []
    for item in questions:
        systems = {}
        for name, runner in runners.items():
            result = evaluate_single_system(item, runner, timer)
            systems[name] = result
            variant_results[name].append(result)
        rows.append(
            {
                "question": item["question"],
                "requires_rewrite": item.get("requires_rewrite", False),
                "systems": systems,
            }
        )

    return {
        "summary": {
            "mode": "matrix",
            "total_questions": len(questions),
            "variants": {
                name: summarize_results(results, questions)
                for name, results in variant_results.items()
            },
        },
        "results": rows,
    }
```

Implement `format_matrix_report()` with rows for:

- Source Hit Rate
- Keyword Hit Rate
- Citation Rate
- Claim Verification Rate
- Fallback Correctness
- Average Retry Count
- Average Retrieved Docs
- Average Relevant Docs
- Average Latency
- Error Count

Use `"N/A"` when a metric is absent.

- [ ] **Step 6: Implement real benchmark runner construction**

Add:

```python
from dataclasses import replace

from agent.graph import run_agent
from config import Settings, get_settings
from evaluation.baselines import run_naive_rag
from rag.retriever import Retriever


def build_benchmark_runners(settings: Settings | None = None):
    base = settings or get_settings()
    base.require_llm_config()
    without_reranker = replace(base, reranker_enabled=False)
    with_reranker = replace(base, reranker_enabled=True)

    plain_retriever = Retriever(settings=without_reranker).retrieve
    reranked_retriever = Retriever(settings=with_reranker).retrieve

    return {
        "naive": lambda question: run_naive_rag(
            question,
            retriever_fn=plain_retriever,
            settings=without_reranker,
        ),
        "agentic": lambda question: run_agent(
            question,
            retriever_fn=plain_retriever,
            settings=without_reranker,
        ),
        "agentic_reranker": lambda question: run_agent(
            question,
            retriever_fn=reranked_retriever,
            settings=with_reranker,
        ),
    }
```

Add a CLI:

```bash
.venv/bin/python -m evaluation.matrix \
  --questions evaluation/eval_questions.json \
  --json-output evaluation/results/deepseek_matrix_2026-06-06.json
```

The CLI must print Markdown and optionally write the full JSON result.
If the LLM configuration is missing, it must exit with a concise configuration error before running any question. It must create the JSON output parent directory before writing.

- [ ] **Step 7: Run matrix tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_evaluation_matrix.py tests/test_evaluate.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add evaluation/matrix.py evaluation/evaluate.py tests/test_evaluation_matrix.py
git commit -m "feat: add three-variant evaluation matrix"
```

## Task 4: Run The DeepSeek Benchmark And Publish Evidence

**Files:**
- Create: `evaluation/results/deepseek_matrix_2026-06-06.json`
- Create: `docs/evaluation.md`

- [ ] **Step 1: Confirm DeepSeek configuration without exposing secrets**

Run:

```bash
.venv/bin/python main.py
```

Expected:

```text
LLM provider: openai_compatible
LLM configured: True
```

Do not print or inspect the API key value.

- [ ] **Step 2: Rebuild the complete sample index**

Run:

```bash
.venv/bin/python -c "from pathlib import Path; from rag.loader import load_documents; from rag.chunker import split_documents; from rag.vectorstore import create_vectorstore; paths=sorted(Path('sample_docs').glob('*.md')); docs=load_documents(paths); chunks=split_documents(docs); create_vectorstore(chunks); print(f'files={len(paths)} documents={len(docs)} chunks={len(chunks)}')"
```

Expected: four files indexed and a non-zero chunk count.

- [ ] **Step 3: Run the complete matrix benchmark**

Run:

```bash
.venv/bin/python -m evaluation.matrix \
  --questions evaluation/eval_questions.json \
  --json-output evaluation/results/deepseek_matrix_2026-06-06.json
```

Expected: a complete three-column report with `error_count: 0` for every variant. If errors occur, preserve them, diagnose them, and rerun only after correcting the actual issue.

- [ ] **Step 4: Write the evaluation report from actual JSON**

Create `docs/evaluation.md` with:

```markdown
# Evaluation Report

## Scope

- Date: June 6, 2026
- LLM: configured DeepSeek OpenAI-compatible model
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
- Corpus: four fictional Markdown documents
- Questions: 34

## Results

Use the exact Markdown table printed by `evaluation.matrix`; do not transcribe
or round values independently.

## Interpretation

- Describe which metrics improved, stayed flat, or regressed.
- Explain the latency cost of Agentic control and reranking.
- Identify whether reranking improved source/keyword hit rates.
- Explain fallback and verification behavior.

## Case Studies

Include at least:

1. one direct factual success
2. one rewrite/retry example
3. one reranker ordering example
4. one correct fallback
5. one failure or unstable case, if present

## Limitations

- small project-specific dataset
- LLM-based grading and claim verification
- single benchmark model
- no human relevance labels
- results are not a universal RAG benchmark
```

All numeric values must come from the generated JSON.

- [ ] **Step 5: Validate the published artifact**

Run:

```bash
.venv/bin/python -m json.tool evaluation/results/deepseek_matrix_2026-06-06.json >/dev/null
rg "\\[PLACEHOLDER\\]|<actual|<paste" docs/evaluation.md
```

Expected: JSON validation succeeds and the placeholder scan returns no matches. `"N/A"` is acceptable only when explained by the report.

- [ ] **Step 6: Commit**

```bash
git add evaluation/results/deepseek_matrix_2026-06-06.json docs/evaluation.md
git commit -m "docs: publish DeepSeek evaluation benchmark"
```

## Task 5: Add A Reproducible Demo Guide

**Files:**
- Create: `docs/demo.md`
- Modify: `tests/test_project_docs.py`

- [ ] **Step 1: Write failing documentation-link tests**

Add:

```python
def test_portfolio_documentation_files_exist():
    for path in [
        Path("docs/evaluation.md"),
        Path("docs/demo.md"),
        Path("assets/architecture.png"),
    ]:
        assert path.exists()


def test_demo_contains_required_scenarios():
    demo = Path("docs/demo.md").read_text(encoding="utf-8")

    for heading in [
        "Direct Answer",
        "Contextual Follow-Up",
        "Query Rewrite",
        "Correct Fallback",
        "Reranker",
        "Citation Safety",
    ]:
        assert heading in demo
```

- [ ] **Step 2: Run test and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_project_docs.py::test_portfolio_documentation_files_exist tests/test_project_docs.py::test_demo_contains_required_scenarios -q
```

Expected: FAIL because `docs/demo.md` does not exist.

- [ ] **Step 3: Create the demo guide**

Create `docs/demo.md` with:

- environment setup commands
- complete sample index command
- Gradio start command
- reranker off/on `.env` examples
- the following demonstration sequence:

```text
Direct Answer:
What is the maximum Atlas upload size?

Contextual Follow-Up:
What is Agentic RAG?
How does it improve reliability?

Query Rewrite:
How much time off do people get?

Reranker:
How does Atlas select the strongest retrieval candidates before grading?

Correct Fallback:
What are the company's salary bands?

Citation Safety:
Explain that malformed marker/index outputs are covered by offline tests rather
than relying on the live model to fail on demand.
```

For each scenario, document expected source, expected route, and UI fields to inspect.

- [ ] **Step 4: Run documentation tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_project_docs.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/demo.md tests/test_project_docs.py
git commit -m "docs: add reproducible project demo"
```

## Task 6: Update README For Interview Presentation

**Files:**
- Modify: `README.md`
- Modify: `tests/test_project_docs.py`

- [ ] **Step 1: Write failing README presentation test**

Add:

```python
def test_readme_links_portfolio_materials_and_interview_topics():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "docs/evaluation.md" in readme
    assert "docs/demo.md" in readme
    assert "## Interview Talking Points" in readme
    assert "Reranker vs retrieval grading" in readme
    assert "Original question vs retrieval query" in readme
    assert "Citation-aware generation vs claim verification" in readme
```

- [ ] **Step 2: Run test and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_project_docs.py::test_readme_links_portfolio_materials_and_interview_topics -q
```

Expected: FAIL because the links and section are absent.

- [ ] **Step 3: Update README**

Add near the architecture section:

```markdown
## Portfolio Materials

- [Evaluation report](docs/evaluation.md)
- [Reproducible demo guide](docs/demo.md)
- [Design notes](docs/design.md)
```

Add:

```markdown
## Interview Talking Points

- **Why this is not naive RAG:** LangGraph explicitly controls rewrite, retrieve, grade, retry, answer, and fallback.
- **Original question vs retrieval query:** rewriting improves search without changing the user's requested answer.
- **Retriever vs reranker:** vector search maximizes candidate recall; the cross-encoder reranker improves candidate ordering.
- **Reranker vs retrieval grading:** reranking ranks candidates, while grading decides whether evidence can answer the original question.
- **Citation-aware generation vs claim verification:** selected evidence indices are checked deterministically, then answer claims are checked by an LLM verifier.
- **Reliability tradeoff:** retries and verification reduce unsupported answers but increase latency and may produce conservative fallbacks.
- **Evaluation limitation:** the included benchmark is small, local, and project-specific.
```

Replace the fabricated example evaluation numbers with either a link to `docs/evaluation.md` or the actual published matrix values.

- [ ] **Step 4: Run README tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_project_docs.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_project_docs.py
git commit -m "docs: improve interview presentation"
```

## Task 7: Replace The Architecture Diagram

**Files:**
- Create: `scripts/generate_architecture.py`
- Modify: `assets/architecture.png`
- Modify: `requirements.txt`
- Modify: `tests/test_project_docs.py`

- [ ] **Step 1: Write failing architecture checks**

Add:

```python
from PIL import Image


def test_architecture_diagram_has_portfolio_dimensions():
    with Image.open("assets/architecture.png") as image:
        assert image.width >= 1600
        assert image.height >= 1000
```

- [ ] **Step 2: Run test and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_project_docs.py::test_architecture_diagram_has_portfolio_dimensions -q
```

Expected: FAIL because the current image height is 950.

- [ ] **Step 3: Add the explicit image dependency**

Append to `requirements.txt`:

```text
Pillow>=10.4.0
```

- [ ] **Step 4: Implement the diagram generator**

Create `scripts/generate_architecture.py` using Pillow. Generate a 1800x1200 PNG with four horizontal bands:

1. Ingestion:
   - Gradio Upload
   - PDF / Markdown / TXT Loader
   - Recursive Chunker + Metadata
   - Local Embeddings
   - Deterministic Chroma Index
2. Retrieval:
   - Query Rewrite
   - Vector Candidate Retrieval
   - Optional Cross-Encoder Reranker
   - Retriever Tool
3. LangGraph:
   - Grade Chunks
   - Retry Rewrite loop
   - Generate Answer
   - Citation Marker Check
   - Claim Verification
   - Fallback
4. Providers and evaluation:
   - DeepSeek / OpenAI-compatible
   - Local Ollama
   - Naive vs Agentic vs Agentic + Reranker

Use dark text on a light neutral background, no overlapping labels, minimum 24 px body text, and arrows with explicit direction.

- [ ] **Step 5: Generate and inspect**

Run:

```bash
.venv/bin/python scripts/generate_architecture.py
```

Open `assets/architecture.png` with the image viewer and verify:

- no overlaps
- no clipped labels
- readable at full size
- all current system stages present

- [ ] **Step 6: Run architecture tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_project_docs.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/generate_architecture.py assets/architecture.png requirements.txt tests/test_project_docs.py
git commit -m "docs: refresh architecture diagram"
```

## Task 8: Add Pyproject, Ruff, And Offline GitHub Actions

**Files:**
- Create: `pyproject.toml`
- Create: `.github/workflows/tests.yml`

- [ ] **Step 1: Create pytest and Ruff configuration**

Create:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"

[tool.ruff]
target-version = "py311"
line-length = 88
exclude = [
  ".git",
  ".venv",
  "data",
]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
ignore = ["E501"]
```

- [ ] **Step 2: Run Ruff and record real failures**

Run:

```bash
.venv/bin/python -m pip install ruff
.venv/bin/python -m ruff check .
```

Expected: either PASS or a concrete list of lint findings. Fix findings without changing behavior, then rerun until PASS.

- [ ] **Step 3: Create offline CI workflow**

Create:

```yaml
name: tests

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          python -m pip install -r requirements.txt
          python -m pip install ruff
      - name: Lint
        run: python -m ruff check .
      - name: Test
        env:
          OPENAI_API_KEY: ""
          OLLAMA_MODEL: ""
          RERANKER_ENABLED: "false"
        run: python -m pytest -q
```

- [ ] **Step 4: Run final local verification**

Run:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -q
git diff --check
```

Expected: Ruff exits 0, all tests pass, diff check exits 0.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .github/workflows/tests.yml
git commit -m "ci: add offline lint and test workflow"
```

## Task 9: Final Portfolio Verification

**Files:**
- Modify only files required to fix discovered issues.

- [ ] **Step 1: Verify repository cleanliness**

Run:

```bash
git status --short
find . -name '.DS_Store' -o -name '__pycache__' -o -name '.pytest_cache'
```

Expected: no tracked or unignored garbage files.

- [ ] **Step 2: Verify all documentation links**

Run:

```bash
.venv/bin/python -m pytest tests/test_project_docs.py -q
```

Expected: PASS.

- [ ] **Step 3: Verify complete test and lint suite**

Run:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -q
```

Expected: both commands exit 0.

- [ ] **Step 4: Verify benchmark artifact**

Run:

```bash
.venv/bin/python -m json.tool evaluation/results/deepseek_matrix_2026-06-06.json >/dev/null
```

Expected: exits 0.

- [ ] **Step 5: Review the rendered architecture image**

Inspect `assets/architecture.png` one final time for readability and current terminology.

- [ ] **Step 6: Confirm the final tree is clean**

```bash
git status --short
```

Expected: no output. If a verification step discovers a defect, return to the task that owns that file, add a focused regression test, fix the defect, rerun that task's checks, and commit it under that task before repeating this final verification.
