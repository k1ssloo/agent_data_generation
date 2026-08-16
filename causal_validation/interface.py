"""Validate whether a public tool surface is identifiable before execution."""

from __future__ import annotations

import json
import re
from typing import Any

from task_factory.bundle import TaskBundle


_OPAQUE_NAME = re.compile(r"^(?:operation|tool|function)_\d+_[0-9a-f]+$")


def _text(value: Any) -> str:
    return " ".join(value.strip().casefold().split()) if isinstance(value, str) else ""


def _schema_signature(parameters: dict[str, Any]) -> str:
    """Ignore opaque property names while retaining their public semantics."""
    properties = parameters.get("properties", {})
    required = set(parameters.get("required", []))
    fields = []
    for name, definition in properties.items():
        fields.append(
            {
                "type": definition.get("type"),
                "description": _text(definition.get("description")),
                "enum": definition.get("enum"),
                "const": definition.get("const"),
                "default": definition.get("default"),
                "required": name in required,
            }
        )
    return json.dumps(sorted(fields, key=lambda item: json.dumps(item, sort_keys=True)), sort_keys=True)


def validate_tool_identifiability(
    bundle: TaskBundle, *, require_descriptions: bool | None = None
) -> dict[str, Any]:
    """Reject public APIs whose tools cannot be distinguished from schemas.

    Canonical snake-case names can themselves carry affordance semantics. Opaque
    names cannot, so rendered APIs must provide descriptions for every tool and
    renamed parameter. Exact description/schema duplicates are also rejected.
    """
    renderer = bundle.manifest.get("renderer", {})
    if require_descriptions is None:
        require_descriptions = bool(renderer) or any(
            _OPAQUE_NAME.fullmatch(str(tool.get("name", ""))) for tool in bundle.tools
        )
    errors: list[str] = []
    signatures: dict[tuple[str, str], list[str]] = {}
    described_tools = 0
    described_parameters = 0
    parameter_count = 0
    for index, tool in enumerate(bundle.tools):
        name = str(tool.get("name", ""))
        description = _text(tool.get("description"))
        opaque = bool(_OPAQUE_NAME.fullmatch(name))
        if description:
            described_tools += 1
        elif require_descriptions or opaque:
            errors.append(f"tool {name or index!r} has no public semantic description")
        parameters = tool.get("parameters", {})
        properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        for argument, definition in properties.items():
            parameter_count += 1
            argument_description = _text(
                definition.get("description") if isinstance(definition, dict) else None
            )
            if argument_description:
                described_parameters += 1
            elif require_descriptions or opaque:
                errors.append(
                    f"tool {name!r} parameter {argument!r} has no public semantic description"
                )
        signature = (description, _schema_signature(parameters))
        signatures.setdefault(signature, []).append(name)

    collisions = [
        names
        for (description, _schema), names in signatures.items()
        if len(names) > 1 and require_descriptions
    ]
    for names in collisions:
        errors.append(
            "public tools are indistinguishable before execution: " + ", ".join(names)
        )
    return {
        "task_id": bundle.task_id,
        "valid": not errors,
        "errors": errors,
        "metrics": {
            "tool_count": len(bundle.tools),
            "described_tool_count": described_tools,
            "parameter_count": parameter_count,
            "described_parameter_count": described_parameters,
            "indistinguishable_groups": collisions,
        },
    }
