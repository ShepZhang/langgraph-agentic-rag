# Agent Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the LangGraph Agentic RAG workflow for query rewriting, retriever tool execution, retrieval grading, citation-aware answer generation, retry routing, and fallback behavior.

**Architecture:** The Agent layer will stay modular: `state.py` defines the LangGraph state contract, `prompts.py` stores prompts and formatting helpers, `tools.py` wraps the RAG retriever as a LangChain tool, `nodes.py` contains testable node methods with injected LLM/retriever dependencies, `edges.py` owns conditional routing, and `graph.py` compiles/runs the LangGraph. Tests use fake LLMs and fake retrievers so Agent behavior is verified without real API calls.

**Tech Stack:** Python 3.12, pytest, LangGraph `StateGraph`, LangChain Core tools/messages, OpenAI-compatible chat model via `langchain-openai`.

---

## File Structure Map

- `agent/state.py`: TypedDict state and result/citation types used by graph nodes.
- `agent/prompts.py`: Query rewrite, retrieval grading, and answer generation prompt templates plus document/history formatting helpers.
- `agent/tools.py`: `create_retriever_tool()` and module-level `retrieve_context` tool.
- `agent/nodes.py`: `AgentNodes` class and node methods for rewrite/retrieve/grade/generate/fallback.
- `agent/edges.py`: `route_after_grading()` conditional edge function.
- `agent/graph.py`: `build_graph()` and `run_agent()` entrypoints.
- `tests/test_agent_state_prompts.py`: Tests state defaults and prompt formatting.
- `tests/test_agent_tools.py`: Tests retriever tool wrapping.
- `tests/test_agent_nodes.py`: Tests node behavior using fake LLM and fake retriever.
- `tests/test_agent_edges.py`: Tests conditional routing.
- `tests/test_agent_graph.py`: Tests compiled graph happy path and retry fallback path.

## Task 0: Baseline Verification

**Files:**
- Read: `requirements.txt`

- [ ] **Step 1: Verify Agent dependencies import**

Run:

```bash
.venv/bin/python -c "import langgraph, langchain_core, langchain_openai; print('agent dependencies ready')"
```

Expected output:

```text
agent dependencies ready
```

- [ ] **Step 2: Run current test suite**

Run:

```bash
.venv/bin/python -m pytest -v
```

Expected: existing tests pass.

## Task 1: State and Prompt Contracts

**Files:**
- Create: `tests/test_agent_state_prompts.py`
- Create: `agent/state.py`
- Create: `agent/prompts.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_agent_state_prompts.py` with exactly this content:

```python
"""Tests for Agent state and prompt formatting."""

from __future__ import annotations

from agent.prompts import (
    ANSWER_GENERATION_PROMPT,
    QUERY_REWRITE_PROMPT,
    RETRIEVAL_GRADING_PROMPT,
    format_chat_history,
    format_documents,
)
from agent.state import create_initial_state


def test_create_initial_state_sets_defaults():
    state = create_initial_state("What is RAG?")

    assert state["question"] == "What is RAG?"
    assert state["rewritten_question"] == ""
    assert state["chat_history"] == []
    assert state["documents"] == []
    assert state["answer"] == ""
    assert state["citations"] == []
    assert state["rewrite_count"] == 0
    assert state["is_relevant"] is False
    assert state["route"] == ""


def test_create_initial_state_preserves_chat_history():
    history = [{"role": "user", "content": "Tell me about LangGraph"}]

    state = create_initial_state("How does it help?", chat_history=history)

    assert state["chat_history"] == history


def test_format_chat_history_handles_empty_and_nonempty_history():
    assert format_chat_history([]) == "No prior chat history."

    formatted = format_chat_history(
        [
            {"role": "user", "content": "What is RAG?"},
            {"role": "assistant", "content": "Retrieval augmented generation."},
        ]
    )

    assert "user: What is RAG?" in formatted
    assert "assistant: Retrieval augmented generation." in formatted


def test_format_documents_includes_metadata_and_content():
    docs = [
        {
            "content": "Chunk text",
            "source": "paper.pdf",
            "page": 2,
            "chunk_id": "paper.pdf:p2:c1",
            "score": 0.87,
        }
    ]

    formatted = format_documents(docs)

    assert "[1]" in formatted
    assert "source=paper.pdf" in formatted
    assert "page=2" in formatted
    assert "chunk_id=paper.pdf:p2:c1" in formatted
    assert "score=0.87" in formatted
    assert "Chunk text" in formatted


def test_prompts_contain_required_guardrails():
    assert "standalone" in QUERY_REWRITE_PROMPT.lower()
    assert "retrieved chunks" in ANSWER_GENERATION_PROMPT.lower()
    assert "json" in RETRIEVAL_GRADING_PROMPT.lower()
    assert "keyword" in RETRIEVAL_GRADING_PROMPT.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_state_prompts.py -v
```

Expected: tests fail because `agent.state` and `agent.prompts` are missing.

- [ ] **Step 3: Create `agent/state.py`**

Create `agent/state.py` with exactly this content:

```python
"""LangGraph state definitions for the Agentic RAG workflow."""

from __future__ import annotations

from typing import TypedDict


class ChatMessage(TypedDict):
    """A minimal chat history message."""

    role: str
    content: str


class RetrievedDocument(TypedDict, total=False):
    """Retrieved document chunk passed through the agent state."""

    content: str
    source: str | None
    page: int | None
    chunk_id: str | None
    score: float | None


class Citation(TypedDict, total=False):
    """Citation returned with the final answer."""

    source: str | None
    page: int | None
    chunk_id: str | None
    score: float | None


class AgentState(TypedDict):
    """State passed between LangGraph nodes."""

    question: str
    rewritten_question: str
    chat_history: list[ChatMessage]
    documents: list[RetrievedDocument]
    answer: str
    citations: list[Citation]
    rewrite_count: int
    is_relevant: bool
    route: str


def create_initial_state(
    question: str,
    chat_history: list[ChatMessage] | None = None,
) -> AgentState:
    """Create the initial state for an Agentic RAG run."""

    return {
        "question": question,
        "rewritten_question": "",
        "chat_history": chat_history or [],
        "documents": [],
        "answer": "",
        "citations": [],
        "rewrite_count": 0,
        "is_relevant": False,
        "route": "",
    }
```

- [ ] **Step 4: Create `agent/prompts.py`**

Create `agent/prompts.py` with exactly this content:

```python
"""Prompt templates and formatting helpers for Agentic RAG."""

from __future__ import annotations

from agent.state import ChatMessage, RetrievedDocument


QUERY_REWRITE_PROMPT = """You are rewriting a user question for private knowledge-base retrieval.

Use the chat history only to resolve references or missing context.
Return one standalone retrieval question.
If the original question is already clear, return it unchanged.

Chat history:
{chat_history}

Original question:
{question}

Standalone retrieval question:"""


RETRIEVAL_GRADING_PROMPT = """You are grading whether retrieved chunks can answer a user's question.

Do not mark chunks relevant just because they share keywords.
Mark them relevant only if they contain enough factual information to answer the question.
Return JSON only in this shape:
{{"relevant": true, "reason": "short reason"}}

Question:
{question}

Retrieved chunks:
{documents}

JSON:"""


ANSWER_GENERATION_PROMPT = """You answer questions using only the retrieved chunks.

Rules:
- Use only facts from the retrieved chunks.
- Do not invent facts that are not present in the retrieved chunks.
- For key facts, include source references in the answer when possible.
- If the retrieved chunks do not contain the answer, say you cannot answer from the current documents.
- Keep the answer concise and useful.

Question:
{question}

Retrieved chunks:
{documents}

Answer:"""


def format_chat_history(chat_history: list[ChatMessage]) -> str:
    """Format chat history for prompts."""

    if not chat_history:
        return "No prior chat history."

    lines = []
    for message in chat_history:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def format_documents(documents: list[RetrievedDocument]) -> str:
    """Format retrieved documents for prompts."""

    if not documents:
        return "No retrieved chunks."

    blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        source = document.get("source")
        page = document.get("page")
        chunk_id = document.get("chunk_id")
        score = document.get("score")
        content = document.get("content", "")
        blocks.append(
            f"[{index}] source={source} page={page} chunk_id={chunk_id} "
            f"score={score}\n{content}"
        )
    return "\n\n".join(blocks)
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_state_prompts.py -v
```

Expected: all state/prompt tests pass.

- [ ] **Step 6: Commit state and prompts**

Run:

```bash
git add tests/test_agent_state_prompts.py agent/state.py agent/prompts.py
git commit -m "feat: add agent state and prompts"
```

Expected: git creates a commit containing state, prompts, and tests.

## Task 2: Retriever Tool

**Files:**
- Create: `tests/test_agent_tools.py`
- Create: `agent/tools.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_agent_tools.py` with exactly this content:

```python
"""Tests for Agent retriever tools."""

from __future__ import annotations

from agent.tools import create_retriever_tool


def test_create_retriever_tool_invokes_retriever_function():
    calls = []

    def fake_retriever(query: str):
        calls.append(query)
        return [{"content": "context", "source": "notes.md"}]

    tool = create_retriever_tool(fake_retriever)
    result = tool.invoke({"query": "What is RAG?"})

    assert calls == ["What is RAG?"]
    assert result == [{"content": "context", "source": "notes.md"}]
    assert tool.name == "retrieve_context"
    assert "private knowledge base" in tool.description
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_tools.py -v
```

Expected: tests fail because `agent.tools` is missing.

- [ ] **Step 3: Create `agent/tools.py`**

Create `agent/tools.py` with exactly this content:

```python
"""Agent tools for accessing the private knowledge base."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import StructuredTool

from rag.retriever import retrieve


RetrieverFn = Callable[[str], list[dict[str, Any]]]


def create_retriever_tool(retriever_fn: RetrieverFn | None = None) -> StructuredTool:
    """Create a retriever tool for Agent use."""

    resolved_retriever = retriever_fn or retrieve

    def _retrieve_context(query: str) -> list[dict[str, Any]]:
        """Retrieve relevant document chunks from the indexed private knowledge base."""

        return resolved_retriever(query)

    return StructuredTool.from_function(
        func=_retrieve_context,
        name="retrieve_context",
        description=(
            "Retrieve relevant document chunks from the indexed private knowledge base "
            "according to the user's question."
        ),
    )


retrieve_context = create_retriever_tool()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_tools.py -v
```

Expected: all tool tests pass.

- [ ] **Step 5: Commit retriever tool**

Run:

```bash
git add tests/test_agent_tools.py agent/tools.py
git commit -m "feat: add retriever tool"
```

Expected: git creates a commit containing tool tests and implementation.

## Task 3: Agent Nodes

**Files:**
- Create: `tests/test_agent_nodes.py`
- Create: `agent/nodes.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_agent_nodes.py` with exactly this content:

```python
"""Tests for Agent workflow nodes."""

from __future__ import annotations

from agent.nodes import AgentNodes
from agent.state import create_initial_state


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_rewrite_query_node_rewrites_and_increments_count():
    llm = FakeLLM(["What is Agentic RAG?"])
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state(
        "What is it?",
        chat_history=[{"role": "user", "content": "Tell me about Agentic RAG"}],
    )

    update = nodes.rewrite_query_node(state)

    assert update["rewritten_question"] == "What is Agentic RAG?"
    assert update["rewrite_count"] == 1
    assert "Tell me about Agentic RAG" in llm.prompts[0]


def test_retrieve_node_uses_rewritten_question_when_present():
    calls = []

    def fake_retriever(query):
        calls.append(query)
        return [{"content": "context", "source": "notes.md"}]

    nodes = AgentNodes(llm=FakeLLM([]), retriever_fn=fake_retriever)
    state = create_initial_state("original")
    state["rewritten_question"] = "rewritten"

    update = nodes.retrieve_node(state)

    assert calls == ["rewritten"]
    assert update["documents"] == [{"content": "context", "source": "notes.md"}]


def test_grade_documents_node_marks_empty_docs_irrelevant_without_llm_call():
    llm = FakeLLM(['{"relevant": true}'])
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")

    update = nodes.grade_documents_node(state)

    assert update["is_relevant"] is False
    assert update["route"] == "rewrite_query"
    assert llm.prompts == []


def test_grade_documents_node_parses_relevance_json():
    llm = FakeLLM(['{"relevant": true, "reason": "enough context"}'])
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["documents"] = [{"content": "answer context", "source": "a.md"}]

    update = nodes.grade_documents_node(state)

    assert update["is_relevant"] is True
    assert update["route"] == "generate_answer"


def test_grade_documents_node_treats_invalid_json_as_irrelevant():
    llm = FakeLLM(["not json"])
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["documents"] = [{"content": "context", "source": "a.md"}]

    update = nodes.grade_documents_node(state)

    assert update["is_relevant"] is False
    assert update["route"] == "rewrite_query"


def test_generate_answer_node_returns_answer_and_grounded_citations():
    llm = FakeLLM(["Grounded answer."])
    nodes = AgentNodes(llm=llm, retriever_fn=lambda query: [])
    state = create_initial_state("question")
    state["documents"] = [
        {
            "content": "context",
            "source": "paper.pdf",
            "page": 2,
            "chunk_id": "paper.pdf:p2:c1",
            "score": 0.8,
        }
    ]

    update = nodes.generate_answer_node(state)

    assert update["answer"] == "Grounded answer."
    assert update["citations"] == [
        {
            "source": "paper.pdf",
            "page": 2,
            "chunk_id": "paper.pdf:p2:c1",
            "score": 0.8,
        }
    ]


def test_fallback_node_returns_clear_message():
    nodes = AgentNodes(llm=FakeLLM([]), retriever_fn=lambda query: [])

    update = nodes.fallback_node(create_initial_state("question"))

    assert "无法可靠回答" in update["answer"]
    assert update["citations"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_nodes.py -v
```

Expected: tests fail because `agent.nodes` is missing.

- [ ] **Step 3: Create `agent/nodes.py`**

Create `agent/nodes.py` with exactly this content:

```python
"""LangGraph node implementations for Agentic RAG."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import BaseMessage

from agent.prompts import (
    ANSWER_GENERATION_PROMPT,
    QUERY_REWRITE_PROMPT,
    RETRIEVAL_GRADING_PROMPT,
    format_chat_history,
    format_documents,
)
from agent.state import AgentState, Citation, RetrievedDocument
from agent.tools import create_retriever_tool


FALLBACK_ANSWER = "根据当前已索引文档，无法可靠回答这个问题。请补充相关文档，或换一种更具体的问法。"


class AgentNodes:
    """State transition nodes for the Agentic RAG graph."""

    def __init__(self, llm: Any, retriever_fn: Any | None = None) -> None:
        self.llm = llm
        self.retriever_tool = create_retriever_tool(retriever_fn)

    def rewrite_query_node(self, state: AgentState) -> dict[str, Any]:
        """Rewrite the user question for retrieval."""

        prompt = QUERY_REWRITE_PROMPT.format(
            chat_history=format_chat_history(state.get("chat_history", [])),
            question=state["question"],
        )
        rewritten_question = _coerce_llm_text(self.llm.invoke(prompt)).strip()
        if not rewritten_question:
            rewritten_question = state["question"]

        return {
            "rewritten_question": rewritten_question,
            "rewrite_count": state.get("rewrite_count", 0) + 1,
        }

    def retrieve_node(self, state: AgentState) -> dict[str, Any]:
        """Retrieve relevant chunks with the retriever tool."""

        query = state.get("rewritten_question") or state["question"]
        documents = self.retriever_tool.invoke({"query": query})
        return {"documents": documents}

    def grade_documents_node(self, state: AgentState) -> dict[str, Any]:
        """Grade whether retrieved chunks are relevant enough to answer."""

        documents = state.get("documents", [])
        if not documents:
            return {"is_relevant": False, "route": "rewrite_query"}

        prompt = RETRIEVAL_GRADING_PROMPT.format(
            question=state.get("rewritten_question") or state["question"],
            documents=format_documents(documents),
        )
        raw_result = _coerce_llm_text(self.llm.invoke(prompt))
        is_relevant = _parse_relevance(raw_result)
        return {
            "is_relevant": is_relevant,
            "route": "generate_answer" if is_relevant else "rewrite_query",
        }

    def generate_answer_node(self, state: AgentState) -> dict[str, Any]:
        """Generate a grounded answer and citations."""

        documents = state.get("documents", [])
        prompt = ANSWER_GENERATION_PROMPT.format(
            question=state.get("rewritten_question") or state["question"],
            documents=format_documents(documents),
        )
        answer = _coerce_llm_text(self.llm.invoke(prompt)).strip()
        if not answer:
            answer = FALLBACK_ANSWER

        return {
            "answer": answer,
            "citations": build_citations(documents),
        }

    def fallback_node(self, state: AgentState) -> dict[str, Any]:
        """Return a safe fallback answer when retrieval is insufficient."""

        return {
            "answer": FALLBACK_ANSWER,
            "citations": [],
            "is_relevant": False,
            "route": "fallback",
        }


def build_citations(documents: list[RetrievedDocument]) -> list[Citation]:
    """Build citations from retrieved document metadata."""

    citations: list[Citation] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for document in documents:
        citation: Citation = {
            "source": document.get("source"),
            "page": document.get("page"),
            "chunk_id": document.get("chunk_id"),
            "score": document.get("score"),
        }
        key = (
            citation.get("source"),
            citation.get("page"),
            citation.get("chunk_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        citations.append(citation)
    return citations


def _parse_relevance(raw_result: str) -> bool:
    """Parse a relevance grading JSON response."""

    try:
        parsed = json.loads(raw_result)
    except json.JSONDecodeError:
        return False

    return bool(parsed.get("relevant") is True)


def _coerce_llm_text(response: Any) -> str:
    """Convert LangChain or fake LLM responses into text."""

    if isinstance(response, str):
        return response
    if isinstance(response, BaseMessage):
        content = response.content
        if isinstance(content, str):
            return content
        return str(content)
    content = getattr(response, "content", None)
    if content is not None:
        return str(content)
    return str(response)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_nodes.py -v
```

Expected: all node tests pass.

- [ ] **Step 5: Commit nodes**

Run:

```bash
git add tests/test_agent_nodes.py agent/nodes.py
git commit -m "feat: add agent workflow nodes"
```

Expected: git creates a commit containing node tests and implementation.

## Task 4: Conditional Edges

**Files:**
- Create: `tests/test_agent_edges.py`
- Create: `agent/edges.py`

- [ ] **Step 1: Write failing edge tests**

Create `tests/test_agent_edges.py` with exactly this content:

```python
"""Tests for Agent graph routing."""

from __future__ import annotations

from dataclasses import replace

from config import get_settings
from agent.edges import route_after_grading
from agent.state import create_initial_state


def test_route_after_grading_generates_when_relevant():
    state = create_initial_state("question")
    state["is_relevant"] = True

    route = route_after_grading(state)

    assert route == "generate_answer"


def test_route_after_grading_rewrites_when_under_attempt_limit():
    settings = replace(get_settings(), max_rewrite_attempts=2)
    state = create_initial_state("question")
    state["is_relevant"] = False
    state["rewrite_count"] = 1

    route = route_after_grading(state, settings=settings)

    assert route == "rewrite_query"


def test_route_after_grading_falls_back_at_attempt_limit():
    settings = replace(get_settings(), max_rewrite_attempts=2)
    state = create_initial_state("question")
    state["is_relevant"] = False
    state["rewrite_count"] = 2

    route = route_after_grading(state, settings=settings)

    assert route == "fallback"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_edges.py -v
```

Expected: tests fail because `agent.edges` is missing.

- [ ] **Step 3: Create `agent/edges.py`**

Create `agent/edges.py` with exactly this content:

```python
"""Conditional routing for the Agentic RAG graph."""

from __future__ import annotations

from config import Settings, get_settings
from agent.state import AgentState


def route_after_grading(
    state: AgentState,
    settings: Settings | None = None,
) -> str:
    """Route after retrieval grading."""

    resolved_settings = settings or get_settings()

    if state.get("is_relevant"):
        return "generate_answer"
    if state.get("rewrite_count", 0) < resolved_settings.max_rewrite_attempts:
        return "rewrite_query"
    return "fallback"
```

- [ ] **Step 4: Run edge tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_edges.py -v
```

Expected: all edge tests pass.

- [ ] **Step 5: Commit edges**

Run:

```bash
git add tests/test_agent_edges.py agent/edges.py
git commit -m "feat: add agent routing edges"
```

Expected: git creates a commit containing edge tests and implementation.

## Task 5: LangGraph Build and Run Entrypoint

**Files:**
- Create: `tests/test_agent_graph.py`
- Create: `agent/graph.py`

- [ ] **Step 1: Write failing graph tests**

Create `tests/test_agent_graph.py` with exactly this content:

```python
"""Tests for compiled Agentic RAG graph execution."""

from __future__ import annotations

from dataclasses import replace

from config import get_settings
from agent.graph import build_graph, run_agent


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_run_agent_generates_answer_when_retrieval_is_relevant():
    llm = FakeLLM(
        [
            "What is Agentic RAG?",
            '{"relevant": true, "reason": "context answers it"}',
            "Agentic RAG uses an agent workflow around retrieval.",
        ]
    )

    def fake_retriever(query):
        return [
            {
                "content": "Agentic RAG uses query rewriting and retrieval grading.",
                "source": "notes.md",
                "page": None,
                "chunk_id": "notes.md:pNA:c1",
                "score": 0.9,
            }
        ]

    result = run_agent(
        "What is it?",
        llm=llm,
        retriever_fn=fake_retriever,
    )

    assert result["answer"] == "Agentic RAG uses an agent workflow around retrieval."
    assert result["rewritten_question"] == "What is Agentic RAG?"
    assert result["rewrite_count"] == 1
    assert result["is_relevant"] is True
    assert result["citations"] == [
        {
            "source": "notes.md",
            "page": None,
            "chunk_id": "notes.md:pNA:c1",
            "score": 0.9,
        }
    ]
    assert result["retrieved_documents"][0]["source"] == "notes.md"


def test_run_agent_retries_then_falls_back_when_documents_are_irrelevant():
    settings = replace(get_settings(), max_rewrite_attempts=2)
    llm = FakeLLM(
        [
            "first rewritten question",
            '{"relevant": false, "reason": "not enough"}',
            "second rewritten question",
            '{"relevant": false, "reason": "still not enough"}',
        ]
    )
    queries = []

    def fake_retriever(query):
        queries.append(query)
        return [{"content": "unrelated", "source": "notes.md"}]

    graph = build_graph(llm=llm, retriever_fn=fake_retriever, settings=settings)
    result = run_agent(
        "unclear question",
        graph=graph,
        settings=settings,
    )

    assert queries == ["first rewritten question", "second rewritten question"]
    assert result["rewrite_count"] == 2
    assert result["is_relevant"] is False
    assert "无法可靠回答" in result["answer"]
    assert result["citations"] == []


def test_build_graph_requires_llm_config_when_no_llm_is_injected(monkeypatch):
    settings = replace(get_settings(), openai_api_key="", openai_model="")

    try:
        build_graph(settings=settings)
    except RuntimeError as exc:
        assert "Missing LLM configuration" in str(exc)
    else:
        raise AssertionError("build_graph should require LLM configuration")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_graph.py -v
```

Expected: tests fail because `agent.graph` is missing.

- [ ] **Step 3: Create `agent/graph.py`**

Create `agent/graph.py` with exactly this content:

```python
"""LangGraph construction and execution for Agentic RAG."""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent.edges import route_after_grading
from agent.nodes import AgentNodes
from agent.state import AgentState, ChatMessage, create_initial_state
from config import Settings, get_settings


def build_graph(
    llm: Any | None = None,
    retriever_fn: Any | None = None,
    settings: Settings | None = None,
) -> Any:
    """Build and compile the Agentic RAG graph."""

    resolved_settings = settings or get_settings()
    resolved_llm = llm or _create_chat_model(resolved_settings)
    nodes = AgentNodes(llm=resolved_llm, retriever_fn=retriever_fn)

    graph = StateGraph(AgentState)
    graph.add_node("rewrite_query", nodes.rewrite_query_node)
    graph.add_node("retrieve", nodes.retrieve_node)
    graph.add_node("grade_documents", nodes.grade_documents_node)
    graph.add_node("generate_answer", nodes.generate_answer_node)
    graph.add_node("fallback", nodes.fallback_node)

    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        lambda state: route_after_grading(state, settings=resolved_settings),
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query",
            "fallback": "fallback",
        },
    )
    graph.add_edge("generate_answer", END)
    graph.add_edge("fallback", END)

    return graph.compile()


def run_agent(
    question: str,
    chat_history: list[ChatMessage] | None = None,
    graph: Any | None = None,
    llm: Any | None = None,
    retriever_fn: Any | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run the Agentic RAG graph and return a UI-friendly result."""

    resolved_settings = settings or get_settings()
    compiled_graph = graph or build_graph(
        llm=llm,
        retriever_fn=retriever_fn,
        settings=resolved_settings,
    )
    final_state = compiled_graph.invoke(create_initial_state(question, chat_history))

    return {
        "answer": final_state.get("answer", ""),
        "citations": final_state.get("citations", []),
        "retrieved_documents": final_state.get("documents", []),
        "rewritten_question": final_state.get("rewritten_question", ""),
        "rewrite_count": final_state.get("rewrite_count", 0),
        "is_relevant": final_state.get("is_relevant", False),
    }


def _create_chat_model(settings: Settings) -> ChatOpenAI:
    """Create the configured OpenAI-compatible chat model."""

    settings.require_llm_config()
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0,
    )
```

- [ ] **Step 4: Run graph tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_graph.py -v
```

Expected: all graph tests pass.

- [ ] **Step 5: Commit graph**

Run:

```bash
git add tests/test_agent_graph.py agent/graph.py
git commit -m "feat: add langgraph agent workflow"
```

Expected: git creates a commit containing graph tests and implementation.

## Task 6: Agent Workflow Verification and README Update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run Agent tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_state_prompts.py tests/test_agent_tools.py tests/test_agent_nodes.py tests/test_agent_edges.py tests/test_agent_graph.py -v
```

Expected: all Agent workflow tests pass.

- [ ] **Step 2: Run full suite**

Run:

```bash
.venv/bin/python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 3: Run syntax verification**

Run:

```bash
.venv/bin/python -m compileall agent tests
```

Expected: no syntax errors.

- [ ] **Step 4: Update README roadmap**

Modify the `## Roadmap` section in `README.md` so the second roadmap item reads:

```markdown
- LangGraph agent workflow implemented: query rewriting, retriever tool, retrieval grading, retry routing, answer generation, and fallback.
```

Keep remaining roadmap bullets unchanged.

- [ ] **Step 5: Verify README mentions Agent workflow implemented**

Run:

```bash
rg "LangGraph agent workflow implemented" README.md
```

Expected output includes:

```text
LangGraph agent workflow implemented: query rewriting, retriever tool, retrieval grading, retry routing, answer generation, and fallback.
```

- [ ] **Step 6: Commit README update**

Run:

```bash
git add README.md
git commit -m "docs: update agent workflow status"
```

Expected: git creates a commit containing the README update.

## Task 7: Final Agent Workflow Verification

**Files:**
- Read: all files created or modified in Tasks 1-6

- [ ] **Step 1: Run full test suite**

Run:

```bash
.venv/bin/python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Run CLI smoke test**

Run:

```bash
.venv/bin/python main.py
```

Expected output includes:

```text
Agentic RAG Document QA System
Embedding model: sentence-transformers/all-MiniLM-L6-v2
Run `python app.py` to start the Gradio UI.
```

- [ ] **Step 3: Confirm clean git status**

Run:

```bash
git status --short
```

Expected output:

```text
```

- [ ] **Step 4: Inspect recent commits**

Run:

```bash
git log --oneline -10
```

Expected: recent commits include:

```text
docs: update agent workflow status
feat: add langgraph agent workflow
feat: add agent routing edges
feat: add agent workflow nodes
feat: add retriever tool
feat: add agent state and prompts
```
