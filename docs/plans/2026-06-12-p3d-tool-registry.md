# P3d Typed Tool Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an internal typed Tool Registry and route the existing retrieval and claim-level citation verification workflow through it while providing independently callable summary and calculator tools.

**Architecture:** Introduce a focused `tools/` package containing typed tool contracts, runtime dependency context, a validating registry, four concrete tools, and a default composition factory. Keep LangGraph nodes responsible for state and routing, use an observer callback for compact trace diagnostics, and preserve the current `agent.tools` LangChain `StructuredTool` API as a compatibility adapter.

**Tech Stack:** Python 3.12, Pydantic v2, LangGraph, LangChain `StructuredTool`, pytest.

---

## File Map

### Create

- `tools/__init__.py`: public exports for base types, registry, and default factory.
- `tools/base.py`: `ToolContext`, `ToolErrorInfo`, `ToolResult`, `BaseTool`, exceptions, and LLM text coercion.
- `tools/registry.py`: registration, lookup, argument validation, execution wrapping, timing, and observer callbacks.
- `tools/factory.py`: default composition of all P3d tools.
- `tools/retriever_tool.py`: workspace-scoped knowledge-base retrieval.
- `tools/citation_verifier_tool.py`: claim-level citation verification prompt execution and parsing.
- `tools/document_summary_tool.py`: independently callable grounded document summary.
- `tools/calculator_tool.py`: bounded AST arithmetic evaluator.
- `tests/test_tool_registry.py`: registry contracts and error behavior.
- `tests/test_retriever_tool.py`: retrieval dependency and workspace tests.
- `tests/test_citation_verifier_tool.py`: verifier parsing and failure tests.
- `tests/test_document_summary_tool.py`: summary input and LLM behavior.
- `tests/test_calculator_tool.py`: arithmetic and unsafe expression rejection.

### Modify

- `agent/tools.py`: retain the LangChain compatibility API while delegating to the new registry.
- `agent/nodes.py`: invoke retrieval and citation verification through `ToolRegistry`.
- `agent/graph.py`: compose or accept a registry and attach trace observation.
- `observability/trace.py`: record compact tool-call events and expose `tool_calls`.
- `tests/test_agent_tools.py`: verify compatibility behavior.
- `tests/test_agent_nodes.py`: verify registry-driven node behavior and safe failures.
- `tests/test_agent_graph.py`: verify a supplied registry is used end to end.
- `tests/test_trace_logging.py`: verify persisted tool-call diagnostics.
- `README.md`: document the internal registry and non-autonomous scope.
- `CHANGELOG.md`: add the `v0.3.3-p3d` release entry.
- `docs/resume_bullets.md`: add accurate tool-boundary engineering language.

## Stable Interfaces

The implementation tasks must preserve these signatures:

```python
@dataclass(frozen=True)
class ToolContext:
    llm: Any | None = None
    retriever_fn: Callable[[str], list[dict[str, Any]]] | None = None
    workspace_id: str | None = None


@dataclass(frozen=True)
class ToolErrorInfo:
    code: str
    message: str


@dataclass(frozen=True)
class ToolResult(Generic[ResultT]):
    tool_name: str
    success: bool
    data: ResultT | None
    error: ToolErrorInfo | None
    metadata: dict[str, Any]


class ToolRegistry:
    def register(self, tool: BaseTool[Any, Any]) -> None: ...
    def get(self, name: str) -> BaseTool[Any, Any]: ...
    def list_tools(self) -> list[dict[str, str]]: ...
    def invoke(self, name: str, arguments: Mapping[str, Any]) -> ToolResult[Any]: ...
    def set_call_observer(
        self,
        observer: Callable[[dict[str, Any]], None] | None,
    ) -> None: ...


def create_default_tool_registry(
    *,
    llm: Any,
    retriever_fn: Callable[[str], list[dict[str, Any]]] | None = None,
    workspace_id: str | None = None,
) -> ToolRegistry: ...
```

---

### Task 1: Typed Base Contracts and Registry

**Files:**
- Create: `tools/__init__.py`
- Create: `tools/base.py`
- Create: `tools/registry.py`
- Test: `tests/test_tool_registry.py`

- [ ] **Step 1: Write failing registry contract tests**

Create `tests/test_tool_registry.py`:

```python
from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from tools.base import BaseTool, ToolContext
from tools.registry import (
    ToolNotFoundError,
    ToolRegistrationError,
    ToolRegistry,
)


class EchoArgs(BaseModel):
    text: str = Field(min_length=1)


class EchoTool(BaseTool[EchoArgs, str]):
    name = "echo"
    description = "Return validated text."
    args_schema = EchoArgs

    def run(self, arguments: EchoArgs) -> str:
        return arguments.text

    def build_metadata(
        self,
        arguments: EchoArgs,
        result: str,
    ) -> dict[str, object]:
        return {"length": len(result)}


def test_registry_registers_lists_and_invokes_typed_tool():
    calls = []
    registry = ToolRegistry(call_observer=calls.append)
    registry.register(EchoTool(ToolContext()))

    result = registry.invoke("echo", {"text": "hello"})

    assert result.success is True
    assert result.data == "hello"
    assert result.error is None
    assert result.metadata["length"] == 5
    assert result.metadata["latency_ms"] >= 0
    assert registry.list_tools() == [
        {"name": "echo", "description": "Return validated text."}
    ]
    assert calls[0]["tool_name"] == "echo"
    assert calls[0]["success"] is True
    assert calls[0]["metadata"]["length"] == 5


def test_registry_rejects_duplicate_and_unknown_tools():
    registry = ToolRegistry()
    registry.register(EchoTool(ToolContext()))

    with pytest.raises(ToolRegistrationError):
        registry.register(EchoTool(ToolContext()))

    with pytest.raises(ToolNotFoundError):
        registry.invoke("missing", {})

    class BlankNameTool(EchoTool):
        name = " "

    with pytest.raises(ToolRegistrationError):
        registry.register(BlankNameTool(ToolContext()))


def test_registry_returns_structured_input_and_execution_failures():
    registry = ToolRegistry()
    registry.register(EchoTool(ToolContext()))

    invalid = registry.invoke("echo", {"text": ""})

    assert invalid.success is False
    assert invalid.error is not None
    assert invalid.error.code == "tool_input_error"
    assert invalid.data is None

    class BrokenTool(EchoTool):
        name = "broken"

        def run(self, arguments: EchoArgs) -> str:
            raise RuntimeError("backend unavailable")

    registry.register(BrokenTool(ToolContext()))
    failed = registry.invoke("broken", {"text": "hello"})

    assert failed.success is False
    assert failed.error is not None
    assert failed.error.code == "tool_execution_error"
    assert "backend unavailable" in failed.error.message


def test_registry_redacts_secrets_and_ignores_observer_failure():
    def broken_observer(record):
        raise RuntimeError("observer unavailable")

    class SecretFailureTool(EchoTool):
        name = "secret_failure"

        def run(self, arguments: EchoArgs) -> str:
            raise RuntimeError("provider rejected sk-secretvalue123")

    registry = ToolRegistry(call_observer=broken_observer)
    registry.register(SecretFailureTool(ToolContext()))

    result = registry.invoke("secret_failure", {"text": "hello"})

    assert result.success is False
    assert "sk-secretvalue123" not in result.error.message
    assert "[REDACTED]" in result.error.message
```

- [ ] **Step 2: Run the tests and verify the expected import failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_tool_registry.py -q
```

Expected: FAIL during collection because `tools.base` and `tools.registry` do not exist.

- [ ] **Step 3: Implement base contracts**

Create `tools/base.py`:

```python
"""Typed contracts shared by internal Agent tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

from pydantic import BaseModel


ArgsT = TypeVar("ArgsT", bound=BaseModel)
ResultT = TypeVar("ResultT")


class ToolError(Exception):
    """Base exception for tool composition and execution."""


class ToolRegistrationError(ToolError):
    """Raised when a tool cannot be registered."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not registered."""


class ToolInputError(ToolError):
    """Categorize invalid tool arguments."""


class ToolExecutionError(ToolError):
    """Categorize capability execution failures."""


@dataclass(frozen=True)
class ToolContext:
    """Runtime dependencies injected into tools."""

    llm: Any | None = None
    retriever_fn: Callable[[str], list[dict[str, Any]]] | None = None
    workspace_id: str | None = None


@dataclass(frozen=True)
class ToolErrorInfo:
    """Serializable runtime error summary."""

    code: str
    message: str


@dataclass(frozen=True)
class ToolResult(Generic[ResultT]):
    """Normalized result returned by the registry."""

    tool_name: str
    success: bool
    data: ResultT | None = None
    error: ToolErrorInfo | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC, Generic[ArgsT, ResultT]):
    """Typed internal tool interface."""

    name: str
    description: str
    args_schema: type[ArgsT]

    def __init__(self, context: ToolContext) -> None:
        self.context = context

    @abstractmethod
    def run(self, arguments: ArgsT) -> ResultT:
        """Execute the capability with validated arguments."""

    def build_metadata(
        self,
        arguments: ArgsT,
        result: ResultT,
    ) -> dict[str, Any]:
        """Return compact result metadata for diagnostics."""

        return {}


def coerce_llm_text(result: Any) -> str:
    """Normalize string and LangChain message responses."""

    if isinstance(result, str):
        return result
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)
```

- [ ] **Step 4: Implement the validating registry and public exports**

Create `tools/registry.py`:

```python
"""Registration and execution boundary for internal Agent tools."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import ValidationError

from tools.base import (
    BaseTool,
    ToolErrorInfo,
    ToolInputError,
    ToolNotFoundError,
    ToolRegistrationError,
    ToolResult,
)


logger = logging.getLogger(__name__)
ToolCallObserver = Callable[[dict[str, Any]], None]
_SECRET_PATTERN = re.compile(
    r"(?i)(sk-[A-Za-z0-9_-]{8,}|bearer\s+\S+)"
)


class ToolRegistry:
    """Register, discover, validate, and execute internal tools."""

    def __init__(self, call_observer: ToolCallObserver | None = None) -> None:
        self._tools: dict[str, BaseTool[Any, Any]] = {}
        self._call_observer = call_observer

    def register(self, tool: BaseTool[Any, Any]) -> None:
        name = tool.name.strip()
        if not name:
            raise ToolRegistrationError("Tool name must not be blank.")
        if name in self._tools:
            raise ToolRegistrationError(f"Tool already registered: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> BaseTool[Any, Any]:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"Tool not registered: {name}") from exc

    def list_tools(self) -> list[dict[str, str]]:
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self._tools.values()
        ]

    def set_call_observer(self, observer: ToolCallObserver | None) -> None:
        self._call_observer = observer

    def invoke(
        self,
        name: str,
        arguments: Mapping[str, Any],
    ) -> ToolResult[Any]:
        tool = self.get(name)
        started_at = time.perf_counter()
        try:
            validated = tool.args_schema.model_validate(dict(arguments))
            data = tool.run(validated)
            metadata = tool.build_metadata(validated, data)
            result = ToolResult(
                tool_name=name,
                success=True,
                data=data,
                metadata=metadata,
            )
        except (ValidationError, ToolInputError) as exc:
            result = ToolResult(
                tool_name=name,
                success=False,
                error=ToolErrorInfo(
                    code="tool_input_error",
                    message=_safe_error_message(exc),
                ),
            )
        except Exception as exc:
            result = ToolResult(
                tool_name=name,
                success=False,
                error=ToolErrorInfo(
                    code="tool_execution_error",
                    message=_safe_error_message(exc),
                ),
            )

        elapsed_ms = round(max((time.perf_counter() - started_at) * 1000, 0.0), 4)
        result = ToolResult(
            tool_name=result.tool_name,
            success=result.success,
            data=result.data,
            error=result.error,
            metadata={**result.metadata, "latency_ms": elapsed_ms},
        )
        self._notify(result)
        return result

    def _notify(self, result: ToolResult[Any]) -> None:
        if self._call_observer is None:
            return
        record = {
            "tool_name": result.tool_name,
            "success": result.success,
            "latency_ms": result.metadata["latency_ms"],
            "error": (
                {
                    "code": result.error.code,
                    "message": result.error.message,
                }
                if result.error
                else None
            ),
            "metadata": {
                key: value
                for key, value in result.metadata.items()
                if key != "latency_ms"
            },
        }
        try:
            self._call_observer(record)
        except Exception:
            logger.exception("Tool call observer failed for %s", result.tool_name)


def _safe_error_message(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return _SECRET_PATTERN.sub("[REDACTED]", message)[:500]
```

Create `tools/__init__.py`:

```python
"""Internal tools for the reliability-oriented Agent workflow."""

from tools.base import BaseTool, ToolContext, ToolErrorInfo, ToolResult
from tools.registry import ToolRegistry

__all__ = [
    "BaseTool",
    "ToolContext",
    "ToolErrorInfo",
    "ToolRegistry",
    "ToolResult",
]
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_tool_registry.py -q
```

Expected: `3 passed`.

- [ ] **Step 6: Commit the registry foundation**

```bash
git add tools/__init__.py tools/base.py tools/registry.py tests/test_tool_registry.py
git commit -m "feat: add typed internal tool registry"
```

---

### Task 2: Retriever Tool and LangChain Compatibility Adapter

**Files:**
- Create: `tools/retriever_tool.py`
- Create: `tests/test_retriever_tool.py`
- Modify: `agent/tools.py`
- Modify: `tests/test_agent_tools.py`

- [ ] **Step 1: Write failing retriever and compatibility tests**

Create `tests/test_retriever_tool.py`:

```python
from tools.base import ToolContext
from tools.registry import ToolRegistry
from tools.retriever_tool import RetrieverTool


def test_retriever_tool_uses_injected_retriever_and_reports_metadata():
    calls = []

    def fake_retriever(query: str):
        calls.append(query)
        return [{"content": "context", "chunk_id": "c1"}]

    registry = ToolRegistry()
    registry.register(
        RetrieverTool(
            ToolContext(
                retriever_fn=fake_retriever,
                workspace_id="workspace_1",
            )
        )
    )

    result = registry.invoke("retrieve_context", {"query": "What is RAG?"})

    assert result.success is True
    assert calls == ["What is RAG?"]
    assert result.data == [{"content": "context", "chunk_id": "c1"}]
    assert result.metadata["workspace_id"] == "workspace_1"
    assert result.metadata["result_count"] == 1


def test_retriever_arguments_cannot_override_workspace():
    registry = ToolRegistry()
    registry.register(
        RetrieverTool(
            ToolContext(
                retriever_fn=lambda query: [],
                workspace_id="workspace_1",
            )
        )
    )

    result = registry.invoke(
        "retrieve_context",
        {"query": "question", "workspace_id": "workspace_2"},
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "tool_input_error"
```

Append to `tests/test_agent_tools.py`:

```python
def test_compatibility_retriever_tool_propagates_registry_failure():
    def broken_retriever(query: str):
        raise RuntimeError("retrieval unavailable")

    tool = create_retriever_tool(broken_retriever)

    with pytest.raises(RuntimeError, match="retrieval unavailable"):
        tool.invoke({"query": "question"})
```

Add `import pytest` to `tests/test_agent_tools.py`.

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_retriever_tool.py tests/test_agent_tools.py -q
```

Expected: FAIL because `tools.retriever_tool` does not exist.

- [ ] **Step 3: Implement the retriever tool**

Create `tools/retriever_tool.py`:

```python
"""Workspace-scoped private knowledge-base retrieval tool."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rag.retriever import retrieve
from tools.base import BaseTool


class RetrieverArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)


class RetrieverTool(BaseTool[RetrieverArgs, list[dict[str, Any]]]):
    name = "retrieve_context"
    description = (
        "Retrieve relevant document chunks from the indexed private "
        "knowledge base."
    )
    args_schema = RetrieverArgs

    def run(self, arguments: RetrieverArgs) -> list[dict[str, Any]]:
        if self.context.retriever_fn is not None:
            return self.context.retriever_fn(arguments.query)
        return retrieve(
            arguments.query,
            workspace_id=self.context.workspace_id,
        )

    def build_metadata(
        self,
        arguments: RetrieverArgs,
        result: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "workspace_id": self.context.workspace_id,
            "result_count": len(result),
        }
```

- [ ] **Step 4: Replace `agent/tools.py` with a compatibility adapter**

Keep the same public function and object names:

```python
"""Backward-compatible LangChain adapters for internal Agent tools."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from tools.base import ToolContext
from tools.registry import ToolRegistry
from tools.retriever_tool import RetrieverTool


RetrieverFn = Any


def create_retriever_tool(
    retriever_fn: RetrieverFn | None = None,
    workspace_id: str | None = None,
) -> StructuredTool:
    """Create the legacy LangChain retriever adapter."""

    registry = ToolRegistry()
    registry.register(
        RetrieverTool(
            ToolContext(
                retriever_fn=retriever_fn,
                workspace_id=workspace_id,
            )
        )
    )

    def _retrieve_context(query: str) -> list[dict[str, Any]]:
        result = registry.invoke("retrieve_context", {"query": query})
        if not result.success:
            message = result.error.message if result.error else "Unknown tool failure."
            raise RuntimeError(message)
        return result.data or []

    return StructuredTool.from_function(
        func=_retrieve_context,
        name="retrieve_context",
        description=registry.get("retrieve_context").description,
    )


retrieve_context = create_retriever_tool()
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_retriever_tool.py tests/test_agent_tools.py tests/test_workspace_isolation.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit retrieval migration**

```bash
git add tools/retriever_tool.py agent/tools.py tests/test_retriever_tool.py tests/test_agent_tools.py
git commit -m "feat: register workspace-scoped retriever tool"
```

---

### Task 3: Claim-Level Citation Verifier Tool

**Files:**
- Create: `tools/citation_verifier_tool.py`
- Create: `tests/test_citation_verifier_tool.py`

- [ ] **Step 1: Write failing verifier tests**

Create `tests/test_citation_verifier_tool.py`:

```python
from tools.base import ToolContext
from tools.citation_verifier_tool import CitationVerifierTool
from tools.registry import ToolRegistry


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.prompts = []

    def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_citation_verifier_returns_normalized_claim_results():
    llm = FakeLLM(
        '{"results": [{"claim_id": "c001", "claim": "RAG retrieves evidence", '
        '"cited_chunk_ids": ["c1"], "verification_label": "supported", '
        '"confidence": 0.95, "reason": "Direct support."}], '
        '"reason": "All claims supported."}'
    )
    registry = ToolRegistry()
    registry.register(CitationVerifierTool(ToolContext(llm=llm)))

    result = registry.invoke(
        "verify_citations",
        {
            "question": "What does RAG do?",
            "answer": "RAG retrieves evidence [1].",
            "claims": [
                {
                    "claim_id": "c001",
                    "claim": "RAG retrieves evidence",
                    "cited_chunk_ids": ["c1"],
                }
            ],
            "documents": [
                {
                    "content": "RAG retrieves evidence.",
                    "chunk_id": "c1",
                    "source": "notes.md",
                }
            ],
        },
    )

    assert result.success is True
    assert result.data["results"][0]["verification_label"] == "supported"
    assert result.data["results"][0]["cited_chunk_ids"] == ["c1"]
    assert result.metadata["claim_count"] == 1
    assert result.metadata["unsupported_count"] == 0
    assert "RAG retrieves evidence." in llm.prompts[0]


def test_citation_verifier_rejects_invalid_model_output():
    registry = ToolRegistry()
    registry.register(
        CitationVerifierTool(ToolContext(llm=FakeLLM("not json")))
    )

    result = registry.invoke(
        "verify_citations",
        {
            "question": "question",
            "answer": "answer [1]",
            "claims": [{"claim_id": "c1", "claim": "answer", "cited_chunk_ids": ["d1"]}],
            "documents": [{"content": "context", "chunk_id": "d1"}],
        },
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "tool_execution_error"
    assert "invalid JSON" in result.error.message
```

- [ ] **Step 2: Run tests and verify import failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_citation_verifier_tool.py -q
```

Expected: FAIL because the verifier tool module does not exist.

- [ ] **Step 3: Implement the verifier tool**

Create `tools/citation_verifier_tool.py`:

```python
"""Claim-level citation verification capability."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent.citation_verification import parse_citation_verification_response
from agent.prompts import CITATION_VERIFICATION_PROMPT, format_documents
from tools.base import BaseTool, ToolExecutionError, coerce_llm_text


class CitationVerifierArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    claims: list[dict[str, Any]] = Field(min_length=1)
    documents: list[dict[str, Any]] = Field(min_length=1)


class CitationVerifierTool(
    BaseTool[CitationVerifierArgs, dict[str, Any]]
):
    name = "verify_citations"
    description = "Verify each answer claim against its cited document chunks."
    args_schema = CitationVerifierArgs

    def run(self, arguments: CitationVerifierArgs) -> dict[str, Any]:
        if self.context.llm is None:
            raise ToolExecutionError("Citation verifier requires an LLM.")

        valid_chunk_ids = [
            str(document["chunk_id"])
            for document in arguments.documents
            if document.get("chunk_id")
        ]
        prompt = CITATION_VERIFICATION_PROMPT.format(
            question=arguments.question,
            answer=arguments.answer,
            claims=json.dumps(arguments.claims, ensure_ascii=False),
            documents=format_documents(arguments.documents),
        )
        raw_result = coerce_llm_text(self.context.llm.invoke(prompt))
        verification = parse_citation_verification_response(
            raw_result,
            valid_chunk_ids=valid_chunk_ids,
        )
        if verification is None:
            raise ToolExecutionError(
                "Citation verification returned invalid JSON."
            )
        return verification

    def build_metadata(
        self,
        arguments: CitationVerifierArgs,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        records = result.get("results", [])
        unsupported_count = sum(
            record.get("verification_label") != "supported"
            or not record.get("cited_chunk_ids")
            for record in records
        )
        return {
            "claim_count": len(records),
            "unsupported_count": unsupported_count,
        }
```

- [ ] **Step 4: Run verifier tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_citation_verifier_tool.py tests/test_citation_verification.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit verifier tool**

```bash
git add tools/citation_verifier_tool.py tests/test_citation_verifier_tool.py
git commit -m "feat: add claim citation verifier tool"
```

---

### Task 4: Document Summary, Safe Calculator, and Default Factory

**Files:**
- Create: `tools/document_summary_tool.py`
- Create: `tools/calculator_tool.py`
- Create: `tools/factory.py`
- Create: `tests/test_document_summary_tool.py`
- Create: `tests/test_calculator_tool.py`
- Modify: `tests/test_tool_registry.py`
- Modify: `tools/__init__.py`

- [ ] **Step 1: Write failing summary tests**

Create `tests/test_document_summary_tool.py`:

```python
from tools.base import ToolContext
from tools.document_summary_tool import DocumentSummaryTool
from tools.registry import ToolRegistry


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "- Retrieval grading filters weak chunks."


def test_document_summary_uses_title_content_and_point_limit():
    llm = FakeLLM()
    registry = ToolRegistry()
    registry.register(DocumentSummaryTool(ToolContext(llm=llm)))

    result = registry.invoke(
        "summarize_document",
        {
            "title": "RAG Reliability",
            "content": "Retrieval grading filters weak chunks.",
            "max_points": 3,
        },
    )

    assert result.success is True
    assert result.data == "- Retrieval grading filters weak chunks."
    assert result.metadata["max_points"] == 3
    assert "RAG Reliability" in llm.prompts[0]
    assert "at most 3" in llm.prompts[0]


def test_document_summary_requires_nonempty_content():
    registry = ToolRegistry()
    registry.register(DocumentSummaryTool(ToolContext(llm=FakeLLM())))

    result = registry.invoke("summarize_document", {"content": ""})

    assert result.success is False
    assert result.error.code == "tool_input_error"
```

- [ ] **Step 2: Write failing calculator tests**

Create `tests/test_calculator_tool.py`:

```python
import pytest

from tools.base import ToolContext
from tools.calculator_tool import CalculatorTool
from tools.registry import ToolRegistry


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(12 + 8) / 5", 4.0),
        ("2 ** 4 + 3", 19),
        ("-7 // 2", -4),
    ],
)
def test_calculator_evaluates_whitelisted_arithmetic(expression, expected):
    registry = ToolRegistry()
    registry.register(CalculatorTool(ToolContext()))

    result = registry.invoke("calculator", {"expression": expression})

    assert result.success is True
    assert result.data["value"] == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('id')",
        "value + 1",
        "(1).__class__",
        "[1, 2][0]",
        "2 ** 100",
    ],
)
def test_calculator_rejects_unsafe_or_unbounded_expressions(expression):
    registry = ToolRegistry()
    registry.register(CalculatorTool(ToolContext()))

    result = registry.invoke("calculator", {"expression": expression})

    assert result.success is False
    assert result.error.code == "tool_execution_error"
```

- [ ] **Step 3: Add a failing default registry composition test**

Append to `tests/test_tool_registry.py`:

```python
def test_default_factory_registers_all_p3d_tools():
    from tools.factory import create_default_tool_registry

    registry = create_default_tool_registry(
        llm=object(),
        retriever_fn=lambda query: [],
        workspace_id="workspace_1",
    )

    assert [item["name"] for item in registry.list_tools()] == [
        "retrieve_context",
        "verify_citations",
        "summarize_document",
        "calculator",
    ]
```

- [ ] **Step 4: Run tests and verify failures**

Run:

```bash
.venv/bin/python -m pytest tests/test_document_summary_tool.py tests/test_calculator_tool.py tests/test_tool_registry.py -q
```

Expected: FAIL because the new tool and factory modules do not exist.

- [ ] **Step 5: Implement the document summary tool**

Create `tools/document_summary_tool.py`:

```python
"""Grounded document summary tool."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tools.base import BaseTool, ToolExecutionError, coerce_llm_text


class DocumentSummaryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1)
    title: str | None = None
    max_points: int = Field(default=5, ge=1, le=10)


class DocumentSummaryTool(BaseTool[DocumentSummaryArgs, str]):
    name = "summarize_document"
    description = "Summarize supplied document text without adding new facts."
    args_schema = DocumentSummaryArgs

    def run(self, arguments: DocumentSummaryArgs) -> str:
        if self.context.llm is None:
            raise ToolExecutionError("Document summary requires an LLM.")
        title = arguments.title or "Untitled document"
        prompt = (
            "Summarize the document using only the supplied text. "
            f"Return at most {arguments.max_points} concise bullet points. "
            "Do not add unsupported facts.\n\n"
            f"Title: {title}\n\nDocument:\n{arguments.content}"
        )
        summary = coerce_llm_text(self.context.llm.invoke(prompt)).strip()
        if not summary:
            raise ToolExecutionError("Document summary returned empty text.")
        return summary

    def build_metadata(
        self,
        arguments: DocumentSummaryArgs,
        result: str,
    ) -> dict[str, Any]:
        return {"max_points": arguments.max_points}
```

- [ ] **Step 6: Implement the bounded AST calculator**

Create `tools/calculator_tool.py`:

```python
"""Safe arithmetic calculator based on a bounded AST whitelist."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from tools.base import BaseTool, ToolExecutionError


class CalculatorArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expression: str = Field(min_length=1, max_length=200)


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorTool(BaseTool[CalculatorArgs, dict[str, int | float]]):
    name = "calculator"
    description = "Evaluate a bounded arithmetic expression."
    args_schema = CalculatorArgs

    def run(self, arguments: CalculatorArgs) -> dict[str, int | float]:
        try:
            tree = ast.parse(arguments.expression, mode="eval")
        except SyntaxError as exc:
            raise ToolExecutionError("Invalid arithmetic expression.") from exc
        if sum(1 for _ in ast.walk(tree)) > 64:
            raise ToolExecutionError("Arithmetic expression is too complex.")
        value = self._evaluate(tree.body)
        if isinstance(value, float) and not math.isfinite(value):
            raise ToolExecutionError("Arithmetic result must be finite.")
        return {"value": value}

    def _evaluate(self, node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(
                node.value,
                (int, float),
            ):
                raise ToolExecutionError("Only numeric constants are allowed.")
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](self._evaluate(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = self._evaluate(node.left)
            right = self._evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ToolExecutionError("Exponent magnitude must not exceed 10.")
            return _BINARY_OPERATORS[type(node.op)](left, right)
        raise ToolExecutionError("Expression contains a disallowed operation.")
```

- [ ] **Step 7: Implement the default factory and exports**

Create `tools/factory.py`:

```python
"""Default internal tool composition."""

from __future__ import annotations

from typing import Any, Callable

from tools.base import ToolContext
from tools.calculator_tool import CalculatorTool
from tools.citation_verifier_tool import CitationVerifierTool
from tools.document_summary_tool import DocumentSummaryTool
from tools.registry import ToolRegistry
from tools.retriever_tool import RetrieverTool


def create_default_tool_registry(
    *,
    llm: Any,
    retriever_fn: Callable[[str], list[dict[str, Any]]] | None = None,
    workspace_id: str | None = None,
) -> ToolRegistry:
    context = ToolContext(
        llm=llm,
        retriever_fn=retriever_fn,
        workspace_id=workspace_id,
    )
    registry = ToolRegistry()
    registry.register(RetrieverTool(context))
    registry.register(CitationVerifierTool(context))
    registry.register(DocumentSummaryTool(context))
    registry.register(CalculatorTool(context))
    return registry
```

Update `tools/__init__.py` to export `create_default_tool_registry`.

- [ ] **Step 8: Run all tool tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_tool_registry.py tests/test_retriever_tool.py tests/test_citation_verifier_tool.py tests/test_document_summary_tool.py tests/test_calculator_tool.py -q
```

Expected: all tool tests pass.

- [ ] **Step 9: Commit the complete default toolset**

```bash
git add tools/factory.py tools/document_summary_tool.py tools/calculator_tool.py tools/__init__.py tests/test_document_summary_tool.py tests/test_calculator_tool.py tests/test_tool_registry.py
git commit -m "feat: add summary calculator and default toolset"
```

---

### Task 5: Route LangGraph Retrieval and Verification Through the Registry

**Files:**
- Modify: `agent/nodes.py`
- Modify: `agent/graph.py`
- Modify: `tests/test_agent_nodes.py`
- Modify: `tests/test_agent_graph.py`

- [ ] **Step 1: Add failing node tests using explicit stub tools**

Append helpers and tests to `tests/test_agent_nodes.py`:

```python
from pydantic import BaseModel

from tools.base import BaseTool, ToolContext
from tools.registry import ToolRegistry


class QueryArgs(BaseModel):
    query: str


class StubRetrieverTool(BaseTool[QueryArgs, list[dict]]):
    name = "retrieve_context"
    description = "Stub retrieval."
    args_schema = QueryArgs

    def __init__(self, results):
        super().__init__(ToolContext())
        self.results = results
        self.calls = []

    def run(self, arguments):
        self.calls.append(arguments.query)
        if isinstance(self.results, Exception):
            raise self.results
        return self.results


def test_retrieve_node_uses_supplied_registry():
    tool = StubRetrieverTool(
        [{"content": "context", "source": "notes.md", "chunk_id": "c1"}]
    )
    registry = ToolRegistry()
    registry.register(tool)
    nodes = AgentNodes(llm=FakeLLM([]), tool_registry=registry)
    state = create_initial_state("question")
    state["current_query"] = "rewritten"

    update = nodes.retrieve_node(state)

    assert tool.calls == ["rewritten"]
    assert update["documents"][0]["chunk_id"] == "c1"


def test_retrieve_node_turns_tool_failure_into_retry_context():
    registry = ToolRegistry()
    registry.register(StubRetrieverTool(RuntimeError("retrieval unavailable")))
    nodes = AgentNodes(llm=FakeLLM([]), tool_registry=registry)
    state = create_initial_state("question")
    state["current_query"] = "rewritten"

    update = nodes.retrieve_node(state)

    assert update["documents"] == []
    assert "retrieval unavailable" in update["grading_reason"]
    assert update["retrieval_attempt"] == 1
```

```python
class VerifyArgs(BaseModel):
    question: str
    answer: str
    claims: list[dict]
    documents: list[dict]


class StubVerifierTool(BaseTool[VerifyArgs, dict]):
    name = "verify_citations"
    description = "Stub citation verification."
    args_schema = VerifyArgs

    def __init__(self, result):
        super().__init__(ToolContext())
        self.result = result
        self.calls = []

    def run(self, arguments):
        self.calls.append(arguments.model_dump())
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _build_verification_state():
    state = create_initial_state("What does RAG do?")
    state["draft_answer"] = "RAG retrieves evidence [1]."
    state["claims"] = [
        {
            "claim_id": "c001",
            "claim": "RAG retrieves evidence",
            "cited_chunk_ids": ["c1"],
        }
    ]
    state["cited_documents"] = [
        {
            "content": "RAG retrieves evidence.",
            "source": "notes.md",
            "chunk_id": "c1",
        }
    ]
    return state


def test_verify_citations_node_uses_supplied_registry():
    verifier = StubVerifierTool(
        {
            "results": [
                {
                    "claim_id": "c001",
                    "claim": "RAG retrieves evidence",
                    "cited_chunk_ids": ["c1"],
                    "verification_label": "supported",
                    "confidence": 0.95,
                    "reason": "Direct support.",
                }
            ],
            "reason": "All claims supported.",
        }
    )
    registry = ToolRegistry()
    registry.register(verifier)
    nodes = AgentNodes(llm=FakeLLM([]), tool_registry=registry)
    state = _build_verification_state()

    update = nodes.verify_citations_node(state)

    assert verifier.calls[0]["question"] == state["question"]
    assert update["citation_verification_passed"] is True
    assert update["route"] == "finalize_answer"


def test_verify_citations_node_falls_back_when_tool_fails():
    registry = ToolRegistry()
    registry.register(
        StubVerifierTool(RuntimeError("verifier unavailable"))
    )
    nodes = AgentNodes(llm=FakeLLM([]), tool_registry=registry)
    state = _build_verification_state()

    update = nodes.verify_citations_node(state)

    assert update["route"] == "fallback"
    assert "citation verification tool failed" in update["fallback_reason"].lower()
    assert update["citation_verification_passed"] is False
```

- [ ] **Step 2: Add a failing graph injection test**

Append to `tests/test_agent_graph.py`:

```python
def test_run_agent_uses_supplied_tool_registry():
    from pydantic import BaseModel

    from agent.graph import run_agent
    from tools.base import BaseTool, ToolContext
    from tools.registry import ToolRegistry

    class RetrieveArgs(BaseModel):
        query: str

    class VerifyArgs(BaseModel):
        question: str
        answer: str
        claims: list[dict]
        documents: list[dict]

    class RecordingRetriever(BaseTool[RetrieveArgs, list[dict]]):
        name = "retrieve_context"
        description = "Test retrieval."
        args_schema = RetrieveArgs

        def __init__(self, documents):
            super().__init__(ToolContext())
            self.documents = documents
            self.calls = []

        def run(self, arguments):
            self.calls.append(arguments.query)
            return self.documents

    class RecordingVerifier(BaseTool[VerifyArgs, dict]):
        name = "verify_citations"
        description = "Test verification."
        args_schema = VerifyArgs

        def __init__(self):
            super().__init__(ToolContext())
            self.calls = []

        def run(self, arguments):
            self.calls.append(arguments.model_dump())
            return {
                "results": [
                    {
                        "claim_id": "c001",
                        "claim": "Agentic RAG uses retrieval grading",
                        "cited_chunk_ids": ["c1"],
                        "verification_label": "supported",
                        "confidence": 0.95,
                        "reason": "Direct support.",
                    }
                ],
                "reason": "All claims supported.",
            }

    documents = [
        {
            "content": "Agentic RAG uses retrieval grading.",
            "source": "notes.md",
            "chunk_id": "c1",
        }
    ]
    retriever = RecordingRetriever(documents)
    verifier = RecordingVerifier()
    registry = ToolRegistry()
    registry.register(retriever)
    registry.register(verifier)
    llm = FakeLLM(
        [
            "agentic rag retrieval grading",
            '{"relevant": true, "relevant_indices": [1], "reason": "matches"}',
            (
                '{"answer": "Agentic RAG uses retrieval grading [1].", '
                '"used_citation_indices": [1]}'
            ),
            (
                '{"claims": [{"claim_id": "c001", '
                '"claim": "Agentic RAG uses retrieval grading", '
                '"cited_chunk_ids": ["c1"]}], '
                '"reason": "Extracted one claim."}'
            ),
        ]
    )

    result = run_agent(
        "How does Agentic RAG work?",
        llm=llm,
        settings=get_settings(),
        tool_registry=registry,
    )

    assert retriever.calls == ["agentic rag retrieval grading"]
    assert len(verifier.calls) == 1
    assert result["answer"] == "Agentic RAG uses retrieval grading [1]."
    assert result["citation_verification_passed"] is True
```

- [ ] **Step 3: Run tests and verify constructor/signature failures**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_nodes.py tests/test_agent_graph.py -q
```

Expected: FAIL because `AgentNodes`, `build_graph()`, and `run_agent()` do not
accept `tool_registry`.

- [ ] **Step 4: Inject the registry into `AgentNodes`**

In `agent/nodes.py`:

```python
from tools.factory import create_default_tool_registry
from tools.registry import ToolRegistry
```

Replace the constructor with:

```python
def __init__(
    self,
    llm: Any,
    retriever_fn: Any | None = None,
    features: AgentFeatureFlags | None = None,
    workspace_id: str | None = None,
    tool_registry: ToolRegistry | None = None,
) -> None:
    self.llm = llm
    self.features = features or AgentFeatureFlags()
    self.tool_registry = tool_registry or create_default_tool_registry(
        llm=llm,
        retriever_fn=retriever_fn,
        workspace_id=workspace_id,
    )
```

Remove the import and use of `create_retriever_tool`.

- [ ] **Step 5: Replace retrieval calls with registry invocations**

Change `retrieve_node()` so each query invokes:

```python
query_results = []
tool_errors = []
for retrieval_query in retrieval_queries:
    result = self.tool_registry.invoke(
        "retrieve_context",
        {"query": retrieval_query},
    )
    if result.success:
        query_results.append((retrieval_query, result.data or []))
    else:
        message = result.error.message if result.error else "Unknown retrieval failure."
        tool_errors.append(message)

documents = merge_retrieved_documents(query_results)
update = {
    "documents": documents,
    "retrieval_queries": retrieval_queries,
    "multi_query_used": len(retrieval_queries) > 1,
    "multi_query_result_count": len(documents),
    "retrieval_attempt": state.get("retrieval_attempt", 0) + 1,
}
if not documents and tool_errors:
    update["grading_reason"] = (
        "Retriever tool failed: " + "; ".join(dict.fromkeys(tool_errors))
    )
return update
```

Update the empty-document branch in `grade_documents_node()`:

```python
reason = state.get("grading_reason") or "No documents retrieved."
```

This preserves a tool failure reason for retry and fallback diagnostics.

Also update the empty-document branch in
`accept_retrieved_documents_node()`:

```python
return _fallback_update(
    state.get("grading_reason")
    or "No documents retrieved while retrieval grading was disabled."
)
```

This preserves the same diagnostic when retrieval grading is disabled.

- [ ] **Step 6: Replace citation verification prompt execution with registry invocation**

In `verify_citations_node()`, replace prompt construction, direct LLM
invocation, and response parsing with:

```python
result = self.tool_registry.invoke(
    "verify_citations",
    {
        "question": state["question"],
        "answer": state.get("draft_answer", ""),
        "claims": claims,
        "documents": cited_documents,
    },
)
if not result.success:
    message = (
        result.error.message
        if result.error
        else "Unknown citation verifier failure."
    )
    return _fallback_update(
        f"Citation verification tool failed: {message}"
    )

verification = result.data
if not isinstance(verification, dict):
    return _fallback_update(
        "Citation verification tool returned invalid data."
    )
```

Keep `build_claim_verification_summary()` and all route selection in the node.
Remove no-longer-used imports for `CITATION_VERIFICATION_PROMPT` and
`parse_citation_verification_response`.

- [ ] **Step 7: Add registry parameters to graph entrypoints**

In `agent/graph.py`, import `ToolRegistry` and
`create_default_tool_registry`.

Add `tool_registry: ToolRegistry | None = None` to `build_graph()` and
`run_agent()`.

In `build_graph()`:

```python
resolved_tool_registry = tool_registry or create_default_tool_registry(
    llm=resolved_llm,
    retriever_fn=retriever_fn,
    workspace_id=workspace_id,
)
nodes = AgentNodes(
    llm=resolved_llm,
    features=resolved_features,
    tool_registry=resolved_tool_registry,
)
```

In `run_agent()`, forward `tool_registry` into `build_graph()`.

- [ ] **Step 8: Run focused and regression tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent_nodes.py tests/test_agent_graph.py tests/test_agent_tools.py tests/test_workspace_isolation.py -q
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit graph integration**

```bash
git add agent/nodes.py agent/graph.py tests/test_agent_nodes.py tests/test_agent_graph.py
git commit -m "feat: route agent workflow through tool registry"
```

---

### Task 6: Tool Call Trace Diagnostics

**Files:**
- Modify: `observability/trace.py`
- Modify: `agent/graph.py`
- Modify: `tests/test_trace_logging.py`

- [ ] **Step 1: Write failing trace recorder test**

Append to `tests/test_trace_logging.py`:

```python
def test_trace_recorder_records_compact_tool_calls():
    from observability.trace import TraceRecorder

    recorder = TraceRecorder(
        original_question="question",
        workspace_id="workspace_1",
    )
    recorder.record_tool_call(
        {
            "tool_name": "retrieve_context",
            "success": True,
            "latency_ms": 12.5,
            "error": None,
            "metadata": {
                "workspace_id": "workspace_1",
                "result_count": 2,
            },
        }
    )

    record = recorder.build_record({}, latency_ms=20)

    assert record["tool_calls"] == [
        {
            "tool_name": "retrieve_context",
            "success": True,
            "latency_ms": 12.5,
            "error": None,
            "metadata": {
                "workspace_id": "workspace_1",
                "result_count": 2,
            },
        }
    ]
    assert {
        "event_type": "tool",
        **record["tool_calls"][0],
    } in record["events"]
```

- [ ] **Step 2: Extend the existing Agent trace test**

In `test_run_agent_writes_node_events_and_route_decisions_to_trace`, add:

```python
assert trace["tool_calls"][0]["tool_name"] == "retrieve_context"
assert trace["tool_calls"][0]["success"] is True
assert trace["tool_calls"][0]["metadata"]["workspace_id"] == "workspace_1"
assert "content" not in trace["tool_calls"][0]["metadata"]
```

- [ ] **Step 3: Run trace tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_trace_logging.py -q
```

Expected: FAIL because `TraceRecorder.record_tool_call()` and `tool_calls` do
not exist and graph construction does not install an observer.

- [ ] **Step 4: Add tool events to `TraceRecorder`**

In `observability/trace.py`:

```python
class TraceRecorder:
    def __init__(...):
        # existing fields
        self.tool_calls: list[dict[str, Any]] = []

    def record_tool_call(self, record: dict[str, Any]) -> None:
        """Record a compact internal tool invocation."""

        normalized = _jsonable(record)
        self.tool_calls.append(normalized)
        self.events.append(
            {
                "event_type": "tool",
                **normalized,
            }
        )
```

Add `"tool_calls": self.tool_calls` to `build_record()`.

- [ ] **Step 5: Attach the trace observer during graph composition**

In `build_graph()` after resolving the registry:

```python
if trace_recorder is not None:
    resolved_tool_registry.set_call_observer(
        trace_recorder.record_tool_call
    )
```

When no recorder is active, call:

```python
resolved_tool_registry.set_call_observer(None)
```

This prevents a reused registry from retaining an observer from an earlier
run.

- [ ] **Step 6: Run trace and graph tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_trace_logging.py tests/test_agent_graph.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit observability integration**

```bash
git add observability/trace.py agent/graph.py tests/test_trace_logging.py
git commit -m "feat: trace internal tool calls"
```

---

### Task 7: Documentation, Versioning, and Full Verification

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/resume_bullets.md`
- Modify: `api/main.py`

- [ ] **Step 1: Add the P3d changelog entry**

Insert above the P3c entry:

```markdown
## v0.3.3-p3d - Typed Internal Tool Registry

Date: 2026-06-12

### Added

- Added a typed internal Tool Registry with Pydantic argument validation,
  runtime dependency injection, normalized results, and compact diagnostics.
- Registered retriever, claim citation verifier, document summary, and safe
  calculator tools.
- Routed LangGraph retrieval and citation verification through the registry.
- Added tool-call trace events with success, latency, metadata, and sanitized
  errors.
- Preserved the existing LangChain retriever tool API through a compatibility
  adapter.

### Notes

- P3d does not add autonomous tool selection, planning, a ReAct loop, or a
  generic public tool execution endpoint.
- Document summary and calculator tools are registered for extension and
  independent use but are not part of the primary QA workflow.
```

- [ ] **Step 2: Update README**

Add this section after the Agent workflow description:

```markdown
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
```

Also make these content changes:

- Add `Tool Registry` between the Agent workflow and capability
  implementations in the architecture description.
- Move Tool Registry from `Next Milestones` to `Completed Work`.

- [ ] **Step 3: Update resume bullets**

Add or revise one bullet to say:

```markdown
- 设计 typed Tool Registry 与依赖注入边界，统一 retriever、claim citation verifier、document summary 和 safe calculator 的参数校验、执行结果、错误语义与 trace diagnostics；将检索和引用验证节点接入 Registry，同时保持非自主规划的可扩展 Agent 工具架构。
```

Do not use “autonomous agent” or “production-ready”.

- [ ] **Step 4: Update the API version**

In `api/main.py`, change:

```python
version="0.3.2-p3c"
```

to:

```python
version="0.3.3-p3d"
```

- [ ] **Step 5: Run formatting and static repository checks**

Run:

```bash
git diff --check
.venv/bin/python -m compileall agent tools observability api
```

Expected: no whitespace errors and all modules compile successfully.

- [ ] **Step 6: Run focused P3d tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_tool_registry.py \
  tests/test_retriever_tool.py \
  tests/test_citation_verifier_tool.py \
  tests/test_document_summary_tool.py \
  tests/test_calculator_tool.py \
  tests/test_agent_tools.py \
  tests/test_agent_nodes.py \
  tests/test_agent_graph.py \
  tests/test_trace_logging.py \
  tests/test_workspace_isolation.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Run the complete test suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all project tests pass with no failures.

- [ ] **Step 8: Verify API import and version**

Run:

```bash
.venv/bin/python -c \
  "from api.main import app; print(app.title, app.version, len(app.routes))"
```

Expected:

```text
Reliability-oriented Agentic RAG API 0.3.3-p3d 12
```

- [ ] **Step 9: Commit P3d documentation and version**

```bash
git add README.md CHANGELOG.md docs/resume_bullets.md api/main.py
git commit -m "docs: publish p3d tool registry release"
```

- [ ] **Step 10: Inspect final scope**

Run:

```bash
git status --short
git log --oneline -8
```

Expected: clean worktree and the P3d commits visible above
`de61a18 docs: design p3d typed tool registry`.

Do not create the `v0.3.3-p3d` tag until the user reviews the implementation
and explicitly requests the version commit/tag step.
