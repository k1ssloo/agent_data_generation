"""Compile and validate total public tool interfaces for hidden environments."""

from __future__ import annotations

import copy
import re
from dataclasses import replace
from typing import Any

from .bundle import TaskBundle


_ERROR_SLUG = re.compile(r"[^A-Za-z0-9]+")


def _is_unconditional(branch: dict[str, Any]) -> bool:
    return branch.get("when", True) is True


def _fallback_branch(capability_id: str) -> dict[str, Any]:
    code = _ERROR_SLUG.sub("_", capability_id).strip("_").upper()
    return {
        "id": "public_input_not_applicable",
        "when": True,
        "response": {
            "ok": False,
            "error_code": f"{code}_INPUT_NOT_APPLICABLE",
            "message": (
                "The public arguments are well-formed but do not identify an "
                "applicable action in the current observable workflow."
            ),
            "retryable": True,
        },
        "effects": [],
        "reads": [],
        "writes": [],
        "resolves_errors": [],
    }


def totalize_public_capabilities(bundle: TaskBundle) -> TaskBundle:
    """Add a deterministic public result for every schema-valid call."""

    environment = copy.deepcopy(bundle.environment)
    bindings = copy.deepcopy(bundle.bindings)
    for tool in bindings.get("tools", []):
        parameters = tool.get("parameters", {})
        properties = parameters.get("properties", {})
        parameters["required"] = [
            name
            for name in parameters.get("required", [])
            if not (
                isinstance(properties.get(name), dict)
                and "default" in properties[name]
            )
        ]
    public_capabilities = {
        tool.get("capability_id")
        for tool in bundle.tools
        if isinstance(tool, dict) and isinstance(tool.get("capability_id"), str)
    }
    for capability_id in public_capabilities:
        capability = environment.get("capabilities", {}).get(capability_id)
        if not isinstance(capability, dict):
            continue
        branches = capability.get("branches", [])
        conditional = [
            copy.deepcopy(branch)
            for branch in branches
            if isinstance(branch, dict) and not _is_unconditional(branch)
        ]
        unconditional = [
            copy.deepcopy(branch)
            for branch in branches
            if isinstance(branch, dict) and _is_unconditional(branch)
        ]
        capability["branches"] = (
            conditional + unconditional
            if unconditional
            else conditional + [_fallback_branch(capability_id)]
        )
    return replace(bundle, environment=environment, bindings=bindings)


def _schema_errors(schema: Any, path: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(schema, dict):
        return [f"{path} must be an object schema"]
    schema_type = schema.get("type")
    if schema_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict):
            errors.append(f"{path}.properties must be an object")
            properties = {}
        if not isinstance(required, list) or any(
            not isinstance(item, str) for item in required
        ):
            errors.append(f"{path}.required must be a string list")
            required = []
        for name in required:
            if name not in properties:
                errors.append(f"{path}.required names unknown property {name!r}")
        for name, definition in properties.items():
            child_path = f"{path}.properties.{name}"
            errors.extend(_schema_errors(definition, child_path))
            if name in required and isinstance(definition, dict) and "default" in definition:
                errors.append(f"{child_path} cannot be both required and defaulted")
        additional = schema.get("additionalProperties", True)
        if additional not in {True, False} and not isinstance(additional, dict):
            errors.append(f"{path}.additionalProperties must be boolean or schema")
    elif schema_type == "array" and "items" in schema:
        errors.extend(_schema_errors(schema["items"], f"{path}.items"))
    elif schema_type not in {
        "string", "integer", "number", "boolean", "object", "array"
    }:
        errors.append(f"{path}.type is unsupported or missing")
    if "enum" in schema and (
        not isinstance(schema["enum"], list) or not schema["enum"]
    ):
        errors.append(f"{path}.enum must be a non-empty list")
    return errors


def validate_public_executability(bundle: TaskBundle) -> dict[str, Any]:
    """Prove that public calls are schema-described and runtime-total."""

    errors: list[str] = []
    total_capabilities = 0
    structured_terminal_branches = 0
    capabilities = bundle.environment.get("capabilities", {})
    for index, tool in enumerate(bundle.tools):
        name = str(tool.get("name", index))
        errors.extend(
            f"tool {name}: {error}"
            for error in _schema_errors(
                tool.get("parameters"), f"bindings.tools[{index}].parameters"
            )
        )
        capability_id = tool.get("capability_id")
        capability = capabilities.get(capability_id, {})
        branches = capability.get("branches", []) if isinstance(capability, dict) else []
        unconditional_indices = [
            branch_index
            for branch_index, branch in enumerate(branches)
            if isinstance(branch, dict) and _is_unconditional(branch)
        ]
        if not unconditional_indices:
            errors.append(
                f"tool {name}: public capability {capability_id!r} has no total fallback branch"
            )
            continue
        if unconditional_indices != [len(branches) - 1]:
            errors.append(
                f"tool {name}: unconditional public fallback must be the unique final branch"
            )
            continue
        total_capabilities += 1
        fallback = branches[-1]
        response = fallback.get("response", {})
        if fallback.get("id") == "public_input_not_applicable":
            if fallback.get("effects") or fallback.get("writes"):
                errors.append(f"tool {name}: public fallback must be read-only")
            if not isinstance(response.get("error_code"), str):
                errors.append(
                    f"tool {name}: public fallback must expose error_code"
                )
            else:
                structured_terminal_branches += 1
        elif isinstance(response, dict) and response:
            structured_terminal_branches += 1
        else:
            errors.append(f"tool {name}: unconditional success branch has no response")
    return {
        "task_id": bundle.task_id,
        "valid": not errors,
        "errors": errors,
        "metrics": {
            "public_tool_count": len(bundle.tools),
            "total_capability_count": total_capabilities,
            "structured_terminal_branch_count": structured_terminal_branches,
        },
    }
