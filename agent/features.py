"""Feature flags for composing Agentic RAG workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AgentFeatureFlags:
    """Immutable capability switches used by graph construction and evaluation."""

    query_transformation_enabled: bool = True
    retrieval_grading_enabled: bool = True
    conditional_retry_enabled: bool = True
    citation_verification_enabled: bool = True

    def to_dict(self) -> dict[str, bool]:
        """Return a JSON-serializable feature snapshot."""

        return asdict(self)
