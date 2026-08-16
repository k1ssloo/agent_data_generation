"""Semantic fingerprints for lineage-collapse and near-duplicate detection."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from runtime.predicates import predicate_paths

from .bundle import TaskBundle


def _shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_shape(item) for item in value]
    if isinstance(value, bool):
        return "<bool>"
    if isinstance(value, (int, float)):
        return "<number>"
    if isinstance(value, str):
        if value.startswith("$"):
            return value
        return "<string>"
    return f"<{type(value).__name__}>"


def semantic_signature(bundle: TaskBundle) -> dict[str, Any]:
    capabilities = bundle.environment.get("capabilities", {})
    capability_shapes = {
        capability_id: {
            "branches": [
                {
                    "when": _shape(branch.get("when", True)),
                    "effects": _shape(branch.get("effects", [])),
                    "reads": sorted(branch.get("reads", [])),
                    "writes": sorted(branch.get("writes", [])),
                    "resolves_error_count": len(branch.get("resolves_errors", [])),
                }
                for branch in capability.get("branches", [])
            ]
        }
        for capability_id, capability in sorted(capabilities.items())
    }
    goal_paths = sorted(
        {
            path
            for goal in bundle.contract.get("goal_predicates", [])
            for path in predicate_paths(goal.get("predicate", goal))
        }
    )
    return {
        "domain": bundle.manifest.get(
            "domain", bundle.manifest.get("lineage", {}).get("domain", "unknown")
        ),
        "capabilities": capability_shapes,
        "goal_paths": goal_paths,
        "tool_schema_shapes": sorted(
            (
                len(tool.get("parameters", {}).get("properties", {})),
                tuple(
                    sorted(
                        str(definition.get("type", "unknown"))
                        for definition in tool.get("parameters", {})
                        .get("properties", {})
                        .values()
                    )
                ),
                len(tool.get("parameters", {}).get("required", [])),
            )
            for tool in bundle.tools
        ),
        "counterfactual_axes": sorted(
            axis.get("state_path", "")
            for axis in bundle.contract.get("counterfactual_axes", [])
        ),
    }


def semantic_fingerprint(bundle: TaskBundle) -> str:
    payload = json.dumps(
        semantic_signature(bundle), sort_keys=True, ensure_ascii=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
