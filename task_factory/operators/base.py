"""Interfaces and helpers for deterministic task evolution."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any, Protocol

from task_factory.bundle import TaskBundle


@dataclass(frozen=True)
class EvolutionProduct:
    bundle: TaskBundle
    patch: dict[str, Any]


class EvolutionOperator(Protocol):
    operator_id: str

    def apply(self, parent: TaskBundle, *, generation: int) -> EvolutionProduct:
        ...


def clone_bundle(parent: TaskBundle, *, task_id: str, operator_id: str, generation: int) -> TaskBundle:
    manifest = copy.deepcopy(parent.manifest)
    parent_lineage = manifest.get("lineage", {})
    operators = list(parent_lineage.get("operators", []))
    operators.append(operator_id)
    manifest["task_id"] = task_id
    manifest["lineage"] = {
        "parent_task_id": parent.task_id,
        "root_task_id": parent_lineage.get("root_task_id", parent.task_id),
        "generation": generation,
        "operators": operators,
    }
    return replace(
        parent,
        manifest=manifest,
        instruction=str(parent.instruction),
        contract=copy.deepcopy(parent.contract),
        environment=copy.deepcopy(parent.environment),
        bindings=copy.deepcopy(parent.bindings),
        reference_plan=copy.deepcopy(parent.reference_plan),
    )


def append_goal_condition(bundle: TaskBundle, condition: dict[str, Any]) -> None:
    goals = bundle.contract["goal_predicates"]
    predicate = goals[0]["predicate"]
    if "all" in predicate and isinstance(predicate["all"], list):
        predicate["all"].append(condition)
    else:
        goals[0]["predicate"] = {"all": [predicate, condition]}


def tool_binding(bundle: TaskBundle, name: str) -> dict[str, Any]:
    for tool in bundle.bindings["tools"]:
        if tool["name"] == name:
            return tool
    raise ValueError(f"operator requires public tool {name!r}")


def capability(bundle: TaskBundle, capability_id: str) -> dict[str, Any]:
    try:
        return bundle.environment["capabilities"][capability_id]
    except KeyError as exc:
        raise ValueError(f"operator requires capability {capability_id!r}") from exc


def action_index(bundle: TaskBundle, tool_name: str, *, last: bool = False) -> int:
    return action_index_in(bundle.reference_plan["actions"], tool_name, last=last)


def action_index_in(
    actions: list[dict[str, Any]], tool_name: str, *, last: bool = False
) -> int:
    matches = [
        index
        for index, action in enumerate(actions)
        if action["tool"] == tool_name
    ]
    if not matches:
        raise ValueError(f"operator requires reference action {tool_name!r}")
    return matches[-1] if last else matches[0]


def solution_action_lists(bundle: TaskBundle) -> list[list[dict[str, Any]]]:
    """Return baseline and counterfactual solution plans for aligned rewrites."""
    plans = [bundle.reference_plan["actions"]]
    plans.extend(
        variant["actions"]
        for variant in bundle.reference_plan.get("counterfactuals", [])
    )
    return plans


def solution_commit_branches(
    bundle: TaskBundle, *, commit_tool_name: str, last: bool = False
) -> set[str]:
    """Replay every admitted solution and return its selected commit branch."""
    from causal_validation.intervention import bundle_with_state_overrides
    from rollout import run_reference_plan

    executions = [(bundle, bundle.reference_plan["actions"])]
    for variant in bundle.reference_plan.get("counterfactuals", []):
        variant_bundle = bundle_with_state_overrides(
            bundle, variant.get("state_overrides", {})
        )
        executions.append((variant_bundle, variant["actions"]))
    branch_ids: set[str] = set()
    for variant_bundle, actions in executions:
        report = run_reference_plan(variant_bundle, actions=actions)
        matches = [
            step
            for step in report.get("trace", [])
            if step.get("public_tool") == commit_tool_name
        ]
        if not matches:
            raise ValueError(
                f"solution plan does not execute semantic commit {commit_tool_name!r}"
            )
        step = matches[-1] if last else matches[0]
        branch_id = step.get("selected_branch")
        if not isinstance(branch_id, str) or not branch_id:
            raise ValueError("semantic commit execution has no selected branch")
        branch_ids.add(branch_id)
    return branch_ids


def commit_target_argument(
    bundle: TaskBundle,
    *,
    commit_tool_name: str,
    target_value: str,
    last: bool = False,
) -> tuple[str, str]:
    """Infer the public/internal commit argument carrying the semantic target."""
    binding = tool_binding(bundle, commit_tool_name)
    actions = bundle.reference_plan["actions"]
    action = actions[action_index_in(actions, commit_tool_name, last=last)]
    matches = [
        name
        for name, value in action.get("arguments", {}).items()
        if value == target_value
    ]
    if len(matches) != 1:
        raise ValueError(
            "semantic commit hook target must match exactly one public commit argument"
        )
    public_name = matches[0]
    internal_name = binding.get("input_map", {}).get(public_name, public_name)
    return public_name, internal_name


def manifest_metadata(bundle: TaskBundle) -> dict[str, Any]:
    """Return non-structural manifest fields that must survive materialization."""
    structural = {
        "bundle_version",
        "task_id",
        "instruction_file",
        "contract_file",
        "environment_file",
        "bindings_file",
        "reference_plan_file",
        "lineage",
    }
    return {
        key: copy.deepcopy(value)
        for key, value in bundle.manifest.items()
        if key not in structural
    }
