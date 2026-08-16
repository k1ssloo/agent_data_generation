"""Load and validate task-first bundle manifests."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from runtime.spec import validate_runtime_spec
from .contracts import validate_contract


class BundleError(ValueError):
    """Raised when a task bundle is malformed or unsafe to load."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleError(f"{path} must contain a JSON object")
    return value


def _resolve_member(root: Path, relative: Any, field: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise BundleError(f"manifest.{field} must be a non-empty relative path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise BundleError(f"manifest.{field} escapes the bundle directory") from exc
    if not candidate.is_file():
        raise BundleError(f"manifest.{field} does not exist: {candidate}")
    return candidate


@dataclass(frozen=True)
class TaskBundle:
    root: Path
    manifest: dict[str, Any]
    instruction: str
    contract: dict[str, Any]
    environment: dict[str, Any]
    bindings: dict[str, Any]
    reference_plan: dict[str, Any]

    @property
    def task_id(self) -> str:
        return str(self.manifest["task_id"])

    @property
    def tools(self) -> list[dict[str, Any]]:
        tools = self.bindings.get("tools", [])
        return tools if isinstance(tools, list) else []


def validate_bundle(bundle: TaskBundle) -> list[str]:
    errors: list[str] = []
    if bundle.manifest.get("bundle_version") != "task-bundle-v1":
        errors.append("manifest.bundle_version must be 'task-bundle-v1'")
    if not isinstance(bundle.manifest.get("task_id"), str) or not bundle.manifest.get("task_id"):
        errors.append("manifest.task_id must be a non-empty string")
    if not bundle.instruction.strip():
        errors.append("instruction must not be empty")
    if bundle.contract.get("contract_version") != "task-contract-v1":
        errors.append("contract.contract_version must be 'task-contract-v1'")
    if not isinstance(bundle.contract.get("goal_predicates"), list) or not bundle.contract.get("goal_predicates"):
        errors.append("contract.goal_predicates must be a non-empty list")
    if bundle.environment.get("runtime_version") != "causal-runtime-v1":
        errors.append("environment.runtime_version must be 'causal-runtime-v1'")
    if not isinstance(bundle.environment.get("initial_state"), dict):
        errors.append("environment.initial_state must be an object")
    else:
        errors.extend(
            f"contract: {error}"
            for error in validate_contract(bundle.contract, bundle.environment["initial_state"])
        )
    capabilities = bundle.environment.get("capabilities")
    if not isinstance(capabilities, dict) or not capabilities:
        errors.append("environment.capabilities must be a non-empty object")
        capabilities = {}
    if bundle.bindings.get("binding_version") != "tool-binding-v1":
        errors.append("bindings.binding_version must be 'tool-binding-v1'")
    errors.extend(f"environment: {error}" for error in validate_runtime_spec(bundle.environment))
    seen_names: set[str] = set()
    for index, tool in enumerate(bundle.tools):
        if not isinstance(tool, dict):
            errors.append(f"bindings.tools[{index}] must be an object")
            continue
        name = tool.get("name")
        capability = tool.get("capability_id")
        if not isinstance(name, str) or not name:
            errors.append(f"bindings.tools[{index}].name must be a non-empty string")
        elif name in seen_names:
            errors.append(f"duplicate public tool name {name!r}")
        else:
            seen_names.add(name)
        if capability not in capabilities:
            errors.append(f"tool {name!r} references unknown capability {capability!r}")
        parameters = tool.get("parameters")
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            errors.append(f"tool {name!r} parameters must be an object JSON schema")
    actions = bundle.reference_plan.get("actions")
    if not isinstance(actions, list) or not actions:
        errors.append("reference_plan.actions must be a non-empty list")
    elif any(not isinstance(action, dict) for action in actions):
        errors.append("every reference action must be an object")
    else:
        for index, action in enumerate(actions):
            if action.get("tool") not in seen_names:
                errors.append(f"reference_plan.actions[{index}] uses unknown tool {action.get('tool')!r}")
            if not isinstance(action.get("arguments", {}), dict):
                errors.append(f"reference_plan.actions[{index}].arguments must be an object")
    counterfactuals = bundle.reference_plan.get("counterfactuals", [])
    if not isinstance(counterfactuals, list):
        errors.append("reference_plan.counterfactuals must be a list")
    else:
        for variant_index, variant in enumerate(counterfactuals):
            prefix = f"reference_plan.counterfactuals[{variant_index}]"
            if not isinstance(variant, dict):
                errors.append(f"{prefix} must be an object")
                continue
            overrides = variant.get("state_overrides")
            if not isinstance(overrides, dict) or not overrides:
                errors.append(f"{prefix}.state_overrides must be a non-empty object")
            elif any(
                not isinstance(path, str) or not path.startswith("$state.")
                for path in overrides
            ):
                errors.append(f"{prefix}.state_overrides keys must target $state")
            variant_actions = variant.get("actions")
            if not isinstance(variant_actions, list) or not variant_actions:
                errors.append(f"{prefix}.actions must be a non-empty list")
                continue
            for action_index, action in enumerate(variant_actions):
                action_prefix = f"{prefix}.actions[{action_index}]"
                if not isinstance(action, dict):
                    errors.append(f"{action_prefix} must be an object")
                    continue
                if action.get("tool") not in seen_names:
                    errors.append(f"{action_prefix} uses unknown tool {action.get('tool')!r}")
                if not isinstance(action.get("arguments", {}), dict):
                    errors.append(f"{action_prefix}.arguments must be an object")
    return errors


def load_task_bundle(path: Path) -> TaskBundle:
    root = path.resolve()
    manifest_path = root / "manifest.json" if root.is_dir() else root
    if not manifest_path.is_file():
        raise BundleError(f"bundle manifest does not exist: {manifest_path}")
    root = manifest_path.parent.resolve()
    manifest = _load_json(manifest_path)
    instruction_path = _resolve_member(root, manifest.get("instruction_file"), "instruction_file")
    contract_path = _resolve_member(root, manifest.get("contract_file"), "contract_file")
    environment_path = _resolve_member(root, manifest.get("environment_file"), "environment_file")
    bindings_path = _resolve_member(root, manifest.get("bindings_file"), "bindings_file")
    reference_path = _resolve_member(root, manifest.get("reference_plan_file"), "reference_plan_file")
    bundle = TaskBundle(
        root=root,
        manifest=manifest,
        instruction=instruction_path.read_text(encoding="utf-8"),
        contract=_load_json(contract_path),
        environment=_load_json(environment_path),
        bindings=_load_json(bindings_path),
        reference_plan=_load_json(reference_path),
    )
    errors = validate_bundle(bundle)
    if errors:
        raise BundleError("; ".join(errors))
    return bundle
