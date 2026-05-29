# README Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finalize the GitHub/resume presentation layer with a complete README and an architecture diagram asset.

**Architecture:** This phase changes documentation and static assets only. It documents the completed MVP flow, references `assets/architecture.png`, and updates commands to use `.venv/bin/python` so macOS shells without `python` still work.

**Tech Stack:** Markdown, deterministic PNG generation with Pillow, existing Python test/compile commands.

---

## Task 1: Architecture Diagram Asset

**Files:**
- Create: `assets/architecture.png`

- [ ] **Step 1: Generate `assets/architecture.png`**

Create a 1600x950 PNG showing:

- UI Layer: Gradio upload, build index, ask question
- RAG Layer: loader, chunker, embeddings, Chroma, retriever
- Agent Layer: query rewrite, retrieve tool, grade docs, conditional retry, answer/fallback
- Evaluation Layer: eval questions, metrics report

- [ ] **Step 2: Verify asset exists**

Run:

```bash
file assets/architecture.png
.venv/bin/python -c "from PIL import Image; im = Image.open('assets/architecture.png'); print(im.size)"
```

Expected output includes `PNG image data` and `(1600, 950)`.

- [ ] **Step 3: Commit architecture asset**

Run:

```bash
git add assets/architecture.png
git commit -m "docs: add architecture diagram asset"
```

## Task 2: README Final Polish

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite README for GitHub/resume presentation**

Add these sections:

- Overview
- Why This Is Not a Naive RAG Demo
- Architecture, with `![Architecture](assets/architecture.png)`
- Agent Workflow
- Features
- Tech Stack
- Quick Start
- Usage
- Evaluation
- Example Output
- Project Structure
- Resume Highlights
- Roadmap

Use `.venv/bin/python` in run commands.

- [ ] **Step 2: Verify README references implemented features**

Run:

```bash
rg "assets/architecture.png|Agent Workflow|Evaluation|\\.venv/bin/python app.py|Resume Highlights" README.md
```

Expected: all patterns are present.

- [ ] **Step 3: Commit README polish**

Run:

```bash
git add README.md
git commit -m "docs: polish project readme"
```

## Task 3: Final Documentation Verification

**Files:**
- Read: `README.md`, `assets/architecture.png`

- [ ] **Step 1: Run full test suite**

Run:

```bash
.venv/bin/python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Run smoke commands**

Run:

```bash
.venv/bin/python main.py
.venv/bin/python -m evaluation.evaluate --questions evaluation/eval_questions.json
```

Expected: `main.py` prints config summary and evaluation prints `Evaluation Report`.

- [ ] **Step 3: Confirm clean git status**

Run:

```bash
git status --short
```

Expected output is empty.

## Self-Review

- Spec coverage: The plan adds the required architecture image, README sections, Mac-safe commands, and verification.
- Placeholder scan: No unfinished placeholder markers remain in the plan.
- Type consistency: File references match the repository structure.
