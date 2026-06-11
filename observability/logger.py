"""Trace-store factory functions."""

from __future__ import annotations

from config import Settings
from observability.storage import JsonlTraceStore


def create_trace_store(settings: Settings) -> JsonlTraceStore:
    """Create the configured local trace store."""

    return JsonlTraceStore(settings.trace_log_dir)
