from __future__ import annotations

import hashlib
import re
import string
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field


_PROMPT_ID_PATTERN = re.compile(
    r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+"
)
_VERSION_PATTERN = re.compile(r"v[1-9][0-9]*")


@dataclass(frozen=True)
class PromptDefinition:
    prompt_id: str
    version: str
    template: str
    description: str
    variables: tuple[str, ...] = field(init=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_id, str) or not _PROMPT_ID_PATTERN.fullmatch(
            self.prompt_id
        ):
            raise ValueError(f"Invalid prompt ID: {self.prompt_id!r}")
        if not isinstance(self.version, str) or not _VERSION_PATTERN.fullmatch(
            self.version
        ):
            raise ValueError(f"Invalid prompt version: {self.version!r}")
        if not isinstance(self.template, str) or not self.template:
            raise ValueError("Prompt template must be non-empty")

        try:
            parsed_template = tuple(string.Formatter().parse(self.template))
        except ValueError as exc:
            raise ValueError(f"Prompt template is invalid: {exc}") from exc

        variables: set[str] = set()
        for _, field_name, format_spec, conversion in parsed_template:
            if field_name is None:
                continue
            if (
                not field_name.isidentifier()
                or conversion is not None
                or format_spec
            ):
                raise ValueError(
                    "Prompt template is invalid: only simple fields are allowed"
                )
            variables.add(field_name)

        digest = hashlib.sha256(self.template.encode("utf-8")).hexdigest()
        object.__setattr__(self, "variables", tuple(sorted(variables)))
        object.__setattr__(self, "fingerprint", f"sha256:{digest}")


class PromptRegistry:
    def __init__(
        self,
        definitions: Iterable[PromptDefinition],
        *,
        active_versions: Mapping[str, str],
    ) -> None:
        self._definitions: dict[tuple[str, str], PromptDefinition] = {}
        for definition in list(definitions):
            key = (definition.prompt_id, definition.version)
            if key in self._definitions:
                raise ValueError(
                    "Duplicate prompt definition: "
                    f"{definition.prompt_id} {definition.version}"
                )
            self._definitions[key] = definition

        self._active_versions = dict(active_versions)
        for prompt_id, version in self._active_versions.items():
            if (prompt_id, version) not in self._definitions:
                raise ValueError(
                    f"Active prompt definition is not registered: "
                    f"{prompt_id} {version}"
                )

    def get(
        self,
        prompt_id: str,
        *,
        version: str | None = None,
    ) -> PromptDefinition:
        resolved_version = version
        if resolved_version is None:
            try:
                resolved_version = self._active_versions[prompt_id]
            except KeyError:
                raise KeyError(f"No active version for prompt '{prompt_id}'") from None

        try:
            return self._definitions[(prompt_id, resolved_version)]
        except KeyError:
            raise KeyError(
                f"Prompt '{prompt_id}' version '{resolved_version}' is not registered"
            ) from None

    def render(
        self,
        prompt_id: str,
        *,
        variables: Mapping[str, object],
        version: str | None = None,
    ) -> str:
        definition = self.get(prompt_id, version=version)
        supplied_variables = dict(variables)
        expected_names = set(definition.variables)
        supplied_names = set(supplied_variables)
        missing = sorted(expected_names - supplied_names)
        unexpected = sorted(supplied_names - expected_names)
        if missing or unexpected:
            raise ValueError(
                f"Prompt '{definition.prompt_id}' version '{definition.version}' "
                f"variables mismatch: missing={missing}, unexpected={unexpected}"
            )
        return definition.template.format(**supplied_variables)

    def active_manifest(self) -> dict[str, dict[str, str]]:
        return {
            prompt_id: {
                "version": version,
                "fingerprint": self._definitions[
                    (prompt_id, version)
                ].fingerprint,
            }
            for prompt_id, version in sorted(self._active_versions.items())
        }
