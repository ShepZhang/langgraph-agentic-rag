from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from prompting.registry import PromptDefinition, PromptRegistry


def prompt(
    prompt_id: str = "agent.example",
    version: str = "v1",
    template: str = "Hello {name}",
) -> PromptDefinition:
    return PromptDefinition(
        prompt_id=prompt_id,
        version=version,
        template=template,
        description="Test prompt.",
    )


def test_active_lookup_and_strict_render():
    definition = prompt()
    definitions = [definition]
    active_versions = {"agent.example": "v1"}
    registry = PromptRegistry(
        definitions,
        active_versions=active_versions,
    )
    definitions.clear()
    active_versions["agent.example"] = "v999"

    assert definition.variables == ("name",)
    assert registry.get("agent.example") is definition
    assert registry.render("agent.example", variables={"name": "Ada"}) == "Hello Ada"
    with pytest.raises(FrozenInstanceError):
        definition.template = "Changed"


def test_explicit_historical_version_lookup():
    historical = prompt(version="v1", template="Hello from v1")
    active = prompt(version="v2", template="{z} then {a}")
    registry = PromptRegistry(
        [historical, active],
        active_versions={"agent.example": "v2"},
    )

    assert registry.get("agent.example") is active
    assert registry.get("agent.example", version="v1") is historical
    assert active.variables == ("a", "z")


def test_duplicate_prompt_version_is_rejected():
    with pytest.raises(ValueError, match="agent\\.example.*v1"):
        PromptRegistry(
            [prompt(), prompt(template="Goodbye {name}")],
            active_versions={"agent.example": "v1"},
        )


def test_invalid_definition_metadata_is_rejected():
    invalid_definitions = [
        ("Agent.Example", "v1", "Hello"),
        ("agent", "v1", "Hello"),
        ("agent.example", "1", "Hello"),
        ("agent.example", "v0", "Hello"),
        ("agent.example", "v1", ""),
    ]

    for prompt_id, version, template in invalid_definitions:
        with pytest.raises(ValueError):
            prompt(prompt_id=prompt_id, version=version, template=template)


def test_malformed_or_advanced_template_fields_are_rejected():
    invalid_templates = [
        "Hello {name",
        "Hello {user.name}",
        "Hello {items[0]}",
        "Hello {name!r}",
        "Hello {name:>10}",
    ]

    for template in invalid_templates:
        with pytest.raises(ValueError, match="template"):
            prompt(template=template)


def test_invalid_active_mappings_are_rejected():
    invalid_active_mappings = [
        {"agent.unknown": "v1"},
        {"agent.example": "v2"},
    ]

    for active_versions in invalid_active_mappings:
        with pytest.raises(ValueError):
            PromptRegistry([prompt()], active_versions=active_versions)


def test_render_rejects_missing_and_unexpected_variables_without_leaking_values():
    registry = PromptRegistry(
        [prompt()],
        active_versions={"agent.example": "v1"},
    )

    with pytest.raises(ValueError) as exc_info:
        registry.render(
            "agent.example",
            variables={"secret": "must-not-appear-in-error"},
        )

    assert str(exc_info.value) == (
        "Prompt 'agent.example' version 'v1' variables mismatch: "
        "missing=['name'], unexpected=['secret']"
    )
    assert "must-not-appear-in-error" not in str(exc_info.value)


def test_fingerprint_is_deterministic_and_escaped_braces_render_literally():
    definition = prompt(template='Return {{"answer": "ok"}} for {name}.')
    registry = PromptRegistry(
        [definition],
        active_versions={"agent.example": "v1"},
    )

    assert definition.fingerprint == (
        "sha256:028f419a8c14cbe0efcb58e5b0f67cbfdd54cdcd2ee4c4ab3a065c5f103eb72e"
    )
    assert registry.render("agent.example", variables={"name": "Ada"}) == (
        'Return {"answer": "ok"} for Ada.'
    )


def test_active_manifest_is_sorted_minimal_and_defensively_copied():
    alpha = prompt(prompt_id="agent.alpha")
    beta_old = prompt(
        prompt_id="agent.beta",
        version="v1",
        template="Old {name}",
    )
    beta_active = prompt(
        prompt_id="agent.beta",
        version="v2",
        template="New {name}",
    )
    inactive = prompt(prompt_id="agent.inactive")
    registry = PromptRegistry(
        [beta_old, inactive, beta_active, alpha],
        active_versions={"agent.beta": "v2", "agent.alpha": "v1"},
    )

    manifest = registry.active_manifest()

    assert list(manifest) == ["agent.alpha", "agent.beta"]
    assert manifest == {
        "agent.alpha": {
            "version": "v1",
            "fingerprint": (
                "sha256:"
                "833583c574131c1ec81313e982643b9f5fba312df50f7c97db0572ad1ce5a929"
            ),
        },
        "agent.beta": {
            "version": "v2",
            "fingerprint": beta_active.fingerprint,
        },
    }
    assert set(manifest["agent.alpha"]) == {"version", "fingerprint"}

    manifest["agent.alpha"]["version"] = "changed"
    assert registry.active_manifest()["agent.alpha"]["version"] == "v1"
