"""Typed cumulative variants for P0b ablation experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal

from agent.features import AgentFeatureFlags
from agent.state import ChatMessage
from config import Settings


RunnerKind = Literal["naive", "agentic"]
VariantRunner = Callable[[str, list[ChatMessage]], dict[str, Any]]
_BOOLEAN_KEYS = {
    "query_transformation_enabled",
    "retrieval_grading_enabled",
    "conditional_retry_enabled",
    "hybrid_retrieval_enabled",
    "reranker_enabled",
    "citation_verification_enabled",
}
_ALLOWED_KEYS = {"id", "method", "runner", *_BOOLEAN_KEYS}


@dataclass(frozen=True)
class AblationVariant:
    """One executable cumulative ablation configuration."""

    id: str
    method: str
    runner: RunnerKind
    features: AgentFeatureFlags
    settings_overrides: dict[str, bool]

    def apply_settings(self, base: Settings) -> Settings:
        """Apply retriever capability overrides to base runtime settings."""

        return replace(base, **self.settings_overrides)

    def effective_signature(self) -> tuple[object, ...]:
        """Return the behavior-defining fields used for duplicate detection."""

        flags = self.features
        return (
            self.runner,
            flags.query_transformation_enabled,
            flags.retrieval_grading_enabled,
            flags.conditional_retry_enabled,
            self.settings_overrides["hybrid_retrieval_enabled"],
            self.settings_overrides["reranker_enabled"],
            flags.citation_verification_enabled,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a serializable variant snapshot."""

        return {
            "id": self.id,
            "method": self.method,
            "runner": self.runner,
            "feature_flags": self.features.to_dict(),
            "settings_overrides": dict(self.settings_overrides),
        }


def load_ablation_variants(config_dir: str | Path) -> list[AblationVariant]:
    """Load and validate all ablation variants in filename order."""

    variants = [
        _parse_variant(path)
        for path in sorted(Path(config_dir).glob("*.yaml"))
    ]
    if not variants:
        raise ValueError(f"No ablation configs found in {config_dir}")
    validate_cumulative_variants(variants)
    return variants


def validate_cumulative_variants(variants: list[AblationVariant]) -> None:
    """Ensure variants are unique and add exactly one capability per row."""

    if not variants:
        raise ValueError("At least one ablation variant is required")

    seen_signatures: dict[tuple[object, ...], str] = {}
    for variant in variants:
        signature = variant.effective_signature()
        previous_id = seen_signatures.get(signature)
        if previous_id is not None:
            raise ValueError(
                "Ablation variants have duplicate effective configurations: "
                f"{previous_id} and {variant.id}"
            )
        seen_signatures[signature] = variant.id
        _validate_dependencies(variant)

    for previous, current in zip(variants, variants[1:]):
        previous_capabilities = _capability_vector(previous)
        current_capabilities = _capability_vector(current)
        if any(
            was_enabled and not is_enabled
            for was_enabled, is_enabled in zip(
                previous_capabilities,
                current_capabilities,
            )
        ):
            raise ValueError(
                f"Ablation variant {current.id} disables a previous capability"
            )
        additions = sum(
            not was_enabled and is_enabled
            for was_enabled, is_enabled in zip(
                previous_capabilities,
                current_capabilities,
            )
        )
        if additions != 1:
            raise ValueError(
                f"Ablation variant {current.id} must add exactly one capability"
            )
        if previous.runner == "agentic" and current.runner != "agentic":
            raise ValueError("Ablation runner cannot revert from agentic to naive")


def create_variant_runner(
    variant: AblationVariant,
    base_settings: Settings,
    retriever_factory: Callable[[Settings], Any] | None = None,
    agent_runner: Callable[..., dict[str, Any]] | None = None,
    naive_runner: Callable[..., dict[str, Any]] | None = None,
) -> VariantRunner:
    """Build a history-aware runner with variant-specific retrieval settings."""

    if retriever_factory is None:
        from rag.retriever import Retriever

        retriever_factory = lambda settings: Retriever(settings=settings).retrieve
    if agent_runner is None:
        from agent.graph import run_agent

        agent_runner = run_agent
    if naive_runner is None:
        from baseline.naive_rag import run_naive_rag

        naive_runner = run_naive_rag

    resolved_settings = variant.apply_settings(base_settings)
    retriever = retriever_factory(resolved_settings)
    retriever_fn = retriever.retrieve if hasattr(retriever, "retrieve") else retriever
    if not callable(retriever_fn):
        raise TypeError("retriever_factory must return a callable or Retriever")

    if variant.runner == "naive":
        def run_naive(
            question: str,
            chat_history: list[ChatMessage],
        ) -> dict[str, Any]:
            return naive_runner(
                question,
                chat_history=chat_history,
                settings=resolved_settings,
                retriever_fn=retriever_fn,
            )

        return run_naive

    def run_agentic(
        question: str,
        chat_history: list[ChatMessage],
    ) -> dict[str, Any]:
        return agent_runner(
            question,
            chat_history,
            settings=resolved_settings,
            features=variant.features,
            retriever_fn=retriever_fn,
        )

    return run_agentic


def _parse_variant(path: Path) -> AblationVariant:
    raw = _load_simple_config(path)
    unknown_keys = sorted(set(raw) - _ALLOWED_KEYS)
    if unknown_keys:
        raise ValueError(f"{path} contains unknown keys: {', '.join(unknown_keys)}")

    variant_id = raw.get("id", "").strip()
    if not variant_id:
        raise ValueError(f"{path} requires id")
    if variant_id != path.stem:
        raise ValueError(f"{path} id must match filename stem")

    method = raw.get("method", "").strip()
    if not method:
        raise ValueError(f"{path} requires method")

    runner = raw.get("runner", "").strip()
    if runner not in {"naive", "agentic"}:
        raise ValueError(f"{path} runner must be naive or agentic")

    booleans = {
        key: _parse_bool(path, key, raw.get(key))
        for key in _BOOLEAN_KEYS
    }
    return AblationVariant(
        id=variant_id,
        method=method,
        runner=runner,
        features=AgentFeatureFlags(
            query_transformation_enabled=booleans[
                "query_transformation_enabled"
            ],
            retrieval_grading_enabled=booleans["retrieval_grading_enabled"],
            conditional_retry_enabled=booleans["conditional_retry_enabled"],
            citation_verification_enabled=booleans[
                "citation_verification_enabled"
            ],
        ),
        settings_overrides={
            "hybrid_retrieval_enabled": booleans["hybrid_retrieval_enabled"],
            "reranker_enabled": booleans["reranker_enabled"],
        },
    )


def _load_simple_config(path: Path) -> dict[str, str]:
    config: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"{path}:{line_number} missing ':'")
        key, value = line.split(":", 1)
        normalized_key = key.strip()
        if not normalized_key:
            raise ValueError(f"{path}:{line_number} missing key")
        if normalized_key in config:
            raise ValueError(f"{path}:{line_number} duplicate key {normalized_key}")
        config[normalized_key] = value.strip()
    return config


def _parse_bool(path: Path, key: str, value: str | None) -> bool:
    if value is None:
        raise ValueError(f"{path} requires {key}")
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{path} {key} must be true or false")


def _capability_vector(variant: AblationVariant) -> tuple[bool, ...]:
    flags = variant.features
    return (
        flags.query_transformation_enabled,
        flags.retrieval_grading_enabled,
        flags.conditional_retry_enabled,
        variant.settings_overrides["hybrid_retrieval_enabled"],
        variant.settings_overrides["reranker_enabled"],
        flags.citation_verification_enabled,
    )


def _validate_dependencies(variant: AblationVariant) -> None:
    flags = variant.features
    if flags.retrieval_grading_enabled and not flags.query_transformation_enabled:
        raise ValueError(
            f"{variant.id} retrieval grading requires query transformation"
        )
    if flags.conditional_retry_enabled and (
        not flags.query_transformation_enabled
        or not flags.retrieval_grading_enabled
    ):
        raise ValueError(
            f"{variant.id} conditional retry requires transformation and grading"
        )
    if variant.settings_overrides["reranker_enabled"] and not (
        variant.settings_overrides["hybrid_retrieval_enabled"]
    ):
        raise ValueError(f"{variant.id} reranker requires hybrid retrieval")
    if flags.citation_verification_enabled and variant.runner != "agentic":
        raise ValueError(f"{variant.id} citation verification requires agentic runner")
