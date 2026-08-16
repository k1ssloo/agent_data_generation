"""Add an observation-dependent execution route before a semantic commit."""

from __future__ import annotations

import copy
from dataclasses import replace
import re
from typing import Any

from task_factory.bundle import TaskBundle, validate_bundle

from .base import (
    EvolutionProduct,
    action_index_in,
    append_goal_condition,
    capability,
    clone_bundle,
    commit_target_argument,
    solution_action_lists,
    solution_commit_branches,
    tool_binding,
)


def _required_string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"execution route hook requires non-empty {key!r}")
    return item


class ExecutionRouteOperator:
    """Require route discovery and route-specific reservation before commit."""

    operator_id = "execution_route_branch_v1"

    def apply(self, parent: TaskBundle, *, generation: int) -> EvolutionProduct:
        if self.operator_id in parent.manifest.get("lineage", {}).get("operators", []):
            raise ValueError(f"{self.operator_id} cannot be applied twice in one lineage")
        hooks = parent.manifest.get("evolution_hooks", {})
        hook = hooks.get("audit_checkpoint") if isinstance(hooks, dict) else None
        if not isinstance(hook, dict):
            raise ValueError("bundle does not declare a semantic commit hook")
        scope = _required_string(hook, "scope")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", scope):
            raise ValueError("execution route scope must be snake_case")
        commit_tool_name = _required_string(hook, "commit_tool")
        commit_capability_id = _required_string(hook, "commit_capability")
        verify_capability_id = _required_string(hook, "verify_capability")
        target_value = _required_string(hook, "target_value")
        target_argument, internal_target_argument = commit_target_argument(
            parent,
            commit_tool_name=commit_tool_name,
            target_value=target_value,
            last=bool(hook.get("commit_last")),
        )
        state_key = f"{scope}_route"
        state_path = f"$state.{state_key}"
        inspect_tool_name = f"inspect_{scope}_routes"
        reserve_fallback_tool_name = f"reserve_{scope}_fallback_route"
        reserve_primary_tool_name = f"reserve_{scope}_primary_route"
        reservation_argument = "route_reservation_ref"

        child = clone_bundle(
            parent,
            task_id=f"{parent.task_id}__g{generation}_{scope}_route",
            operator_id=self.operator_id,
            generation=generation,
        )
        child.environment["initial_state"][state_key] = {
            "mode": "fallback_only",
            "current_version": 1,
            "selected": "",
            "reserved": False,
            "reserved_version": 0,
            "target": "",
            "committed": "",
            "committed_target": "",
        }
        inspect_capability = f"route.inspect.{scope}.v1"
        reserve_fallback_capability = f"route.reserve.{scope}.fallback.v1"
        reserve_primary_capability = f"route.reserve.{scope}.primary.v1"
        child.environment["capabilities"][inspect_capability] = {
            "branches": [
                {
                    "id": "primary_available",
                    "when": {"eq": [f"{state_path}.mode", "primary_available"]},
                    "response": {
                        "route_report_handle": f"route_report_{scope}_primary_v1",
                        "recommended_route": "primary",
                        "route_version": f"{state_path}.current_version",
                    },
                    "reads": [f"{state_path}.mode", f"{state_path}.current_version"],
                    "observes": [f"{state_path}.mode"],
                },
                {
                    "id": "fallback_required",
                    "when": {"eq": [f"{state_path}.mode", "fallback_only"]},
                    "response": {
                        "route_report_handle": f"route_report_{scope}_fallback_v1",
                        "recommended_route": "fallback",
                        "route_version": f"{state_path}.current_version",
                    },
                    "reads": [f"{state_path}.mode", f"{state_path}.current_version"],
                    "observes": [f"{state_path}.mode"],
                },
            ]
        }
        fallback_reservation_branch = {
                    "id": "reserve_primary",
                    "when": {"all": [
                        {"eq": [f"{state_path}.mode", "primary_available"]},
                        {"eq": ["$args.route_report_handle", f"route_report_{scope}_primary_v1"]},
                    ]},
                    "response": {
                        "route_reservation_handle": f"route_reservation_{scope}_primary_v1",
                        "route": "primary",
                        "reserved_version": f"{state_path}.current_version",
                    },
                    "effects": [
                        {"set": f"{state_path}.selected", "value": "primary"},
                        {"set": f"{state_path}.reserved", "value": True},
                        {
                            "set": f"{state_path}.reserved_version",
                            "value": f"{state_path}.current_version",
                        },
                        {"set": f"{state_path}.target", "value": "$args.target_handle"},
                    ],
                    "reads": [f"{state_path}.mode", f"{state_path}.current_version"],
                    "writes": [f"{state_path}.selected", f"{state_path}.reserved", f"{state_path}.reserved_version", f"{state_path}.target"],
                }
        primary_reservation_branch = {
                    "id": "reserve_fallback",
                    "when": {"all": [
                        {"eq": [f"{state_path}.mode", "fallback_only"]},
                        {"eq": ["$args.route_report_handle", f"route_report_{scope}_fallback_v1"]},
                    ]},
                    "response": {
                        "route_reservation_handle": f"route_reservation_{scope}_fallback_v1",
                        "route": "fallback",
                        "reserved_version": f"{state_path}.current_version",
                    },
                    "effects": [
                        {"set": f"{state_path}.selected", "value": "fallback"},
                        {"set": f"{state_path}.reserved", "value": True},
                        {
                            "set": f"{state_path}.reserved_version",
                            "value": f"{state_path}.current_version",
                        },
                        {"set": f"{state_path}.target", "value": "$args.target_handle"},
                    ],
                    "reads": [f"{state_path}.mode", f"{state_path}.current_version"],
                    "writes": [f"{state_path}.selected", f"{state_path}.reserved", f"{state_path}.reserved_version", f"{state_path}.target"],
                }
        # The public tools represent different route mechanisms, not aliases
        # that merely pass a different enum to one generic reservation tool.
        fallback_reservation_branch, primary_reservation_branch = (
            primary_reservation_branch,
            fallback_reservation_branch,
        )
        child.environment["capabilities"][reserve_fallback_capability] = {
            "branches": [fallback_reservation_branch]
        }
        child.environment["capabilities"][reserve_primary_capability] = {
            "branches": [primary_reservation_branch]
        }
        child.bindings["tools"].extend(
            [
                {
                    "name": inspect_tool_name,
                    "description": "Inspect currently available execution routes for the pending action.",
                    "capability_id": inspect_capability,
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
                {
                    "name": reserve_fallback_tool_name,
                    "description": "Reserve the observed fallback execution route for the exact target.",
                    "capability_id": reserve_fallback_capability,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "route_report_ref": {"type": "string"},
                            "target_ref": {"type": "string"},
                        },
                        "required": ["route_report_ref", "target_ref"],
                    },
                    "input_map": {
                        "route_report_ref": "route_report_handle",
                        "target_ref": "target_handle",
                    },
                    "provenance_required": ["route_report_ref", "target_ref"],
                },
                {
                    "name": reserve_primary_tool_name,
                    "description": "Reserve the observed primary execution route for the exact target.",
                    "capability_id": reserve_primary_capability,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "route_report_ref": {"type": "string"},
                            "target_ref": {"type": "string"},
                        },
                        "required": ["route_report_ref", "target_ref"],
                    },
                    "input_map": {
                        "route_report_ref": "route_report_handle",
                        "target_ref": "target_handle",
                    },
                    "provenance_required": ["route_report_ref", "target_ref"],
                },
            ]
        )

        commit_tool = tool_binding(child, commit_tool_name)
        commit_tool["parameters"]["properties"][reservation_argument] = {
            "type": "string",
            "description": "Reservation for the route selected from current environment evidence.",
        }
        commit_tool["parameters"].setdefault("required", []).append(
            reservation_argument
        )
        commit_tool.setdefault("input_map", {})[reservation_argument] = "route_reservation_handle"
        commit_tool.setdefault("provenance_required", []).append(reservation_argument)

        commit = capability(child, commit_capability_id)
        commit_branch_ids = solution_commit_branches(
            parent,
            commit_tool_name=commit_tool_name,
            last=bool(hook.get("commit_last")),
        )
        resolved_ids = {
            branch.get("id")
            for branch in commit.get("branches", [])
            if branch.get("id") in commit_branch_ids
        }
        if resolved_ids != commit_branch_ids:
            raise ValueError("execution route could not resolve every solution commit branch")
        expanded_branches = []
        for commit_branch in commit.get("branches", []):
            if commit_branch.get("id") not in commit_branch_ids:
                expanded_branches.append(commit_branch)
                continue
            original_when = commit_branch.get("when", True)
            branch_base = {
                "response": copy.deepcopy(commit_branch.get("response", {})),
                "effects": copy.deepcopy(commit_branch.get("effects", [])),
                "reads": sorted(
                    set(
                        commit_branch.get("reads", [])
                        + [
                            f"{state_path}.selected",
                            f"{state_path}.reserved",
                            f"{state_path}.reserved_version",
                            f"{state_path}.current_version",
                            f"{state_path}.target",
                        ]
                    )
                ),
                "writes": sorted(
                    set(
                        commit_branch.get("writes", [])
                        + [f"{state_path}.committed", f"{state_path}.committed_target"]
                    )
                ),
                "resolves_errors": copy.deepcopy(
                    commit_branch.get("resolves_errors", [])
                ),
            }
            for route in ("primary", "fallback"):
                branch = {
                    "id": f"{commit_branch.get('id', 'commit')}_{route}_route",
                    "when": {
                        "all": [
                            copy.deepcopy(original_when),
                            {
                                "eq": [
                                    "$args.route_reservation_handle",
                                    f"route_reservation_{scope}_{route}_v1",
                                ]
                            },
                            {"eq": [f"{state_path}.selected", route]},
                            {
                                "eq": [
                                    f"{state_path}.reserved_version",
                                    f"{state_path}.current_version",
                                ]
                            },
                            {
                                "eq": [
                                    f"{state_path}.mode",
                                    "primary_available" if route == "primary" else "fallback_only",
                                ]
                            },
                            {"eq": [f"{state_path}.reserved", True]},
                            {
                                "eq": [
                                    f"$args.{internal_target_argument}",
                                    f"{state_path}.target",
                                ]
                            },
                        ]
                    },
                    **copy.deepcopy(branch_base),
                }
                branch["effects"].append(
                    {"set": f"{state_path}.committed", "value": route}
                )
                branch["effects"].append(
                    {
                        "set": f"{state_path}.committed_target",
                        "value": f"$args.{internal_target_argument}",
                    }
                )
                expanded_branches.append(branch)
        commit["branches"] = expanded_branches

        verify = capability(child, verify_capability_id)
        verify_branch_id = hook.get("verify_branch")
        verify_matching = [
            branch
            for branch in verify.get("branches", [])
            if verify_branch_id is None or branch.get("id") == verify_branch_id
        ]
        if len(verify_matching) != 1:
            raise ValueError("execution route hook must identify exactly one verify branch")
        verify_branch = verify_matching[0]
        verify_branch.setdefault("response", {})["committed_route"] = f"{state_path}.committed"
        verify_branch.setdefault("response", {})["route_committed_target"] = (
            f"{state_path}.committed_target"
        )
        verify_branch.setdefault("response", {})["route_reserved_target"] = (
            f"{state_path}.target"
        )
        verify_branch.setdefault("response", {})["route_current_version"] = (
            f"{state_path}.current_version"
        )
        verify_branch.setdefault("response", {})["route_reserved_version"] = (
            f"{state_path}.reserved_version"
        )
        verify_branch.setdefault("reads", []).extend(
            [
                f"{state_path}.committed",
                f"{state_path}.committed_target",
                f"{state_path}.target",
                f"{state_path}.current_version",
                f"{state_path}.reserved_version",
            ]
        )
        append_goal_condition(
            child,
            {
                "all": [
                    {"in": [f"{state_path}.committed", ["primary", "fallback"]]},
                    {"eq": [f"{state_path}.target", target_value]},
                    {
                        "eq": [
                            f"{state_path}.reserved_version",
                            f"{state_path}.current_version",
                        ]
                    },
                    {"eq": [f"{state_path}.committed_target", target_value]},
                ]
            },
        )
        child.contract.setdefault("counterfactual_axes", []).append(
            {"state_path": f"{state_path}.mode", "variants": ["fallback_only", "primary_available"]}
        )
        child.contract.setdefault("forbidden_shortcuts", []).append(
            "reuse a route reservation after the available route changes"
        )

        for actions in solution_action_lists(child):
            commit_indices = [
                index
                for index, action in enumerate(actions)
                if action.get("tool") == commit_tool_name
            ]
            if not commit_indices:
                raise ValueError("solution plan has no semantic commit action")
            first_commit_index = commit_indices[0]
            plan_target = actions[first_commit_index].get("arguments", {}).get(target_argument)
            if not isinstance(plan_target, str) or not plan_target:
                raise ValueError("every solution plan must provide the semantic commit target")
            actions.insert(first_commit_index, {"tool": inspect_tool_name, "arguments": {}})
            actions.insert(
                first_commit_index + 1,
                {
                    "tool": reserve_fallback_tool_name,
                    "arguments": {
                        "route_report_ref": f"route_report_{scope}_fallback_v1",
                        "target_ref": plan_target,
                    },
                },
            )
            for action in actions:
                if action.get("tool") == commit_tool_name:
                    action.setdefault("arguments", {})[
                        reservation_argument
                    ] = f"route_reservation_{scope}_fallback_v1"
        primary_actions = copy.deepcopy(child.reference_plan["actions"])
        for action in primary_actions:
            if action["tool"] == reserve_fallback_tool_name:
                action["tool"] = reserve_primary_tool_name
                action["arguments"]["route_report_ref"] = f"route_report_{scope}_primary_v1"
            if action["tool"] == commit_tool_name:
                action["arguments"][reservation_argument] = (
                    f"route_reservation_{scope}_primary_v1"
                )
        child.reference_plan.setdefault("counterfactuals", []).append(
            {
                "id": f"{scope}_primary_route",
                "state_overrides": {f"{state_path}.mode": "primary_available"},
                "actions": primary_actions,
            }
        )
        route_instruction = (
            "Any consequential action must use an execution route that is currently "
            "valid for its exact target."
        )
        child = replace(
            child,
            instruction=child.instruction.rstrip() + " " + route_instruction + "\n",
        )
        if isinstance(child.contract.get("instruction_claims"), list) and isinstance(
            child.contract.get("goal_clauses"), list
        ):
            clause_id = f"{scope}_route_committed"
            child.contract["instruction_claims"].append(
                {
                    "evidence_span": route_instruction,
                    "kind": "synthetic_constraint",
                    "clause_ids": [clause_id],
                }
            )
            child.contract["goal_clauses"].append(
                {
                    "id": clause_id,
                    "predicate": {
                        "all": [
                            {"in": [f"{state_path}.committed", ["primary", "fallback"]]},
                            {"eq": [f"{state_path}.target", target_value]},
                            {
                                "eq": [
                                    f"{state_path}.reserved_version",
                                    f"{state_path}.current_version",
                                ]
                            },
                            {"eq": [f"{state_path}.committed_target", target_value]},
                        ]
                    },
                    "transition_paths": [
                        f"{state_path}.committed",
                        f"{state_path}.committed_target",
                        f"{state_path}.target",
                        f"{state_path}.current_version",
                        f"{state_path}.reserved_version",
                    ],
                    "evidence_paths": [
                        f"{state_path}.committed",
                        f"{state_path}.committed_target",
                        f"{state_path}.target",
                        f"{state_path}.current_version",
                        f"{state_path}.reserved_version",
                    ],
                    "witness_tools": [
                        inspect_tool_name,
                        reserve_tool_name,
                        commit_tool_name,
                    ],
                }
            )
        errors = validate_bundle(child)
        if errors:
            raise ValueError("execution route operator produced invalid bundle: " + "; ".join(errors))
        return EvolutionProduct(
            bundle=child,
            patch={
                "operator_id": self.operator_id,
                "semantic_changes": [
                    "the consequential action now depends on an observed execution route",
                    "the selected route produces a target-specific reservation",
                    "a counterfactual environment requires a different reservation strategy",
                ],
                "added_goal_paths": [f"{state_path}.committed"],
            },
        )
