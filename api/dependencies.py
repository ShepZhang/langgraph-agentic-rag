"""FastAPI dependency factories."""

from __future__ import annotations

from typing import Any, Callable

from fastapi import Depends

from agent.graph import run_agent
from api.services.documents import DocumentService
from api.services.evaluation import EvaluationService
from api.services.traces import TraceService
from config import Settings, get_settings


def get_settings_dependency() -> Settings:
    """Return runtime settings."""

    return get_settings()


def get_agent_runner() -> Callable[..., dict[str, Any]]:
    """Return the Agentic RAG runner."""

    return run_agent


def get_document_service(
    settings: Settings = Depends(get_settings_dependency),
) -> DocumentService:
    """Return the document service."""

    return DocumentService(settings=settings)


def get_trace_service(
    settings: Settings = Depends(get_settings_dependency),
) -> TraceService:
    """Return the trace lookup service."""

    return TraceService(settings=settings)


def get_evaluation_service(
    settings: Settings = Depends(get_settings_dependency),
) -> EvaluationService:
    """Return the evaluation service."""

    return EvaluationService(settings=settings)
