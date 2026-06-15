"""Typed evaluation-domain records with compatibility serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from agent.state import ChatMessage


JudgeStatus = Literal["disabled", "completed", "failed"]


@dataclass(frozen=True)
class EvaluationQuestion:
    id: str
    question: str
    question_type: str
    gold_answer: str
    expected_sources: list[str]
    expected_keywords: list[str]
    source_match_mode: Literal["any", "all"]
    answerable: bool
    expected_behavior: str
    chat_history: list[ChatMessage]
    requires_rewrite: bool
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def to_compat_dict(self) -> dict[str, Any]:
        payload = dict(self.extra_fields)
        payload.update(
            {
                "id": self.id,
                "question": self.question,
                "question_type": self.question_type,
                "gold_answer": self.gold_answer,
                "expected_sources": list(self.expected_sources),
                "expected_keywords": list(self.expected_keywords),
                "source_match_mode": self.source_match_mode,
                "answerable": self.answerable,
                "should_answer": self.answerable,
                "expected_behavior": self.expected_behavior,
                "chat_history": [dict(message) for message in self.chat_history],
                "requires_rewrite": self.requires_rewrite,
            }
        )
        return payload


@dataclass
class EvaluationResult:
    question_id: str
    question_type: str
    question: str
    chat_history_supplied: bool = False
    chat_history_used: bool = False
    answer_returned: bool = False
    fallback_triggered: bool = False
    fallback_correct: bool = False
    correct: bool = False
    context_relevant: bool = False
    citation_hit: bool = False
    citation_returned: bool = False
    is_verified: bool = False
    citation_verification_applicable: bool = False
    claim_count: int = 0
    unsupported_claim_count: int | None = None
    supported_claim_count: int | None = None
    total_claim_count: int | None = None
    source_hit: bool = False
    keyword_hit: bool = False
    citation_verification_passed: bool = False
    rewrite_triggered: bool = False
    retry_count: int = 0
    retrieved_doc_count: int = 0
    relevant_doc_count: int = 0
    token_usage: Any = None
    estimated_cost: float | None = None
    latency: float = 0
    error: str | None = None
    answer: str = ""
    citations: list[Any] = field(default_factory=list)
    claims: list[Any] = field(default_factory=list)
    claim_verification_results: list[Any] = field(default_factory=list)
    retrieved_documents: list[Any] = field(default_factory=list)
    relevant_documents: list[Any] = field(default_factory=list)
    failure_analysis: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(
        cls,
        question_id: str,
        question_type: str,
        question: str,
    ) -> EvaluationResult:
        return cls(
            question_id=question_id,
            question_type=question_type,
            question=question,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationSummary:
    total_questions: int = 0
    answer_rate: float = 0
    fallback_rate: float = 0
    citation_rate: float = 0
    source_hit_rate: float = 0
    keyword_hit_rate: float = 0
    fallback_correctness_rate: float = 0
    verification_rate: float = 0
    average_claim_count: float = 0
    correctness_score: float = 0
    context_relevance_score: float = 0
    citation_hit_rate: float = 0
    fallback_accuracy: float = 0
    unsupported_claim_count: int | None = None
    supported_claim_ratio: float | None = None
    citation_verification_pass_rate: float | None = None
    average_token_usage: float = 0
    estimated_cost: float = 0
    average_retry_count: float = 0
    average_retrieved_docs: float = 0
    average_relevant_docs: float = 0
    relevant_filtering_rate: float = 0
    average_latency: float = 0
    rewrite_triggered_count: int = 0
    error_count: int = 0
    failure_type_counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> EvaluationSummary:
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PairedEvaluationResult:
    question: str
    requires_rewrite: bool
    naive: EvaluationResult
    agentic: EvaluationResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "requires_rewrite": self.requires_rewrite,
            "naive": self.naive.to_dict(),
            "agentic": self.agentic.to_dict(),
        }


@dataclass
class ComparisonEvaluationSummary:
    total_questions: int
    naive: EvaluationSummary
    agentic: EvaluationSummary
    comparison: dict[str, Any]
    mode: str = "comparison"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "total_questions": self.total_questions,
            "naive": self.naive.to_dict(),
            "agentic": self.agentic.to_dict(),
            "comparison": dict(self.comparison),
        }


@dataclass
class EvaluationReport:
    summary: EvaluationSummary | ComparisonEvaluationSummary
    results: list[EvaluationResult] | list[PairedEvaluationResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(frozen=True)
class RuntimeMetadata:
    schema_version: int
    evaluator_version: str
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluator_version": self.evaluator_version,
            **self.config,
        }


@dataclass(frozen=True)
class JudgeResult:
    status: JudgeStatus
    scores: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    error: str | None = None

    @classmethod
    def disabled(cls) -> JudgeResult:
        return cls(status="disabled")

    @classmethod
    def completed(
        cls,
        scores: dict[str, float],
        reason: str = "",
    ) -> JudgeResult:
        return cls(
            status="completed",
            scores=dict(scores),
            reason=reason,
        )

    @classmethod
    def failed(cls, error: str) -> JudgeResult:
        return cls(status="failed", error=error)
