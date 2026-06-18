from __future__ import annotations

from prompting.catalog import PROJECT_PROMPT_REGISTRY
from prompting.registry import PromptDefinition


def get_prompt_definition(
    prompt_id: str,
    *,
    version: str | None = None,
) -> PromptDefinition:
    return PROJECT_PROMPT_REGISTRY.get(prompt_id, version=version)


def get_prompt_template(
    prompt_id: str,
    *,
    version: str | None = None,
) -> str:
    return get_prompt_definition(prompt_id, version=version).template


def render_prompt(
    prompt_id: str,
    *,
    version: str | None = None,
    **variables: object,
) -> str:
    return PROJECT_PROMPT_REGISTRY.render(
        prompt_id,
        version=version,
        variables=variables,
    )


def get_active_prompt_manifest() -> dict[str, dict[str, str]]:
    return PROJECT_PROMPT_REGISTRY.active_manifest()


__all__ = [
    "PromptDefinition",
    "get_active_prompt_manifest",
    "get_prompt_definition",
    "get_prompt_template",
    "render_prompt",
]
