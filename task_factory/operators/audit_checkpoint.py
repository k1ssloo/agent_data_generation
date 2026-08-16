"""Insert a portable evidence checkpoint before a consequential action."""

from __future__ import annotations

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
        raise ValueError(f"audit checkpoint hook requires non-empty {key!r}")
    return item


class AuditCheckpointOperator:
    """Make an existing commit consume newly inspected and derived evidence.

    Domain bundles expose semantic roles through ``manifest.evolution_hooks``.
    The operator never relies on a particular public API name or domain schema.
    """

    operator_id = "audit_checkpoint_v1"

    def apply(self, parent: TaskBundle, *, generation: int) -> EvolutionProduct:
        if self.operator_id in parent.manifest.get("lineage", {}).get("operators", []):
            raise ValueError(f"{self.operator_id} cannot be applied twice in one lineage")
        hooks = parent.manifest.get("evolution_hooks", {})
        hook = hooks.get("audit_checkpoint") if isinstance(hooks, dict) else None
        if not isinstance(hook, dict):
            raise ValueError("bundle does not declare evolution_hooks.audit_checkpoint")

        scope = _required_string(hook, "scope")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", scope):
            raise ValueError("audit checkpoint scope must be snake_case")
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
        inspect_tool_name = str(hook.get("inspect_tool", f"inspect_{scope}_policy"))
        approve_tool_name = str(hook.get("approve_tool", f"approve_{scope}_action"))
        state_key = str(hook.get("state_key", f"{scope}_audit"))
        policy_handle = str(hook.get("policy_handle", f"policy_{scope}_audit_1"))
        approval_handle = str(hook.get("approval_handle", f"approval_{scope}_1"))
        approval_argument = str(hook.get("approval_argument", "approval_ref"))

        child = clone_bundle(
            parent,
            task_id=f"{parent.task_id}__g{generation}_{scope}_audit",
            operator_id=self.operator_id,
            generation=generation,
        )
        state_path = f"$state.{state_key}"
        child.environment["initial_state"][state_key] = {
            "current_policy": policy_handle,
            "reviewed": False,
            "approved_policy": "",
            "target": "",
            "committed": False,
            "committed_policy": "",
            "committed_target": "",
        }

        inspect_capability = f"audit.inspect.{scope}.v1"
        approve_capability = f"audit.approve.{scope}.v1"
        if inspect_capability in child.environment["capabilities"]:
            raise ValueError(f"capability collision: {inspect_capability}")
        child.environment["capabilities"][inspect_capability] = {
            "branches": [
                {
                    "id": "policy_discovered",
                    "when": True,
                    "response": {
                        "policy_handle": policy_handle,
                        "scope": scope,
                        "requires_explicit_review": True,
                    },
                    "reads": [f"{state_path}.current_policy"],
                }
            ]
        }
        child.environment["capabilities"][approve_capability] = {
            "branches": [
                {
                    "id": "target_reviewed",
                    "when": {
                        "all": [
                            {"eq": ["$args.policy_handle", policy_handle]},
                            {
                                "eq": [
                                    "$args.policy_handle",
                                    f"{state_path}.current_policy",
                                ]
                            },
                        ]
                    },
                    "response": {
                        "approval_handle": approval_handle,
                        "approved_target": "$args.target_handle",
                    },
                    "effects": [
                        {"set": f"{state_path}.reviewed", "value": True},
                        {
                            "set": f"{state_path}.approved_policy",
                            "value": "$args.policy_handle",
                        },
                        {"set": f"{state_path}.target", "value": "$args.target_handle"},
                    ],
                    "reads": [f"{state_path}.current_policy"],
                    "writes": [
                        f"{state_path}.reviewed",
                        f"{state_path}.approved_policy",
                        f"{state_path}.target",
                    ],
                }
            ]
        }

        child.bindings["tools"].extend(
            [
                {
                    "name": inspect_tool_name,
                    "description": f"Inspect the current evidence policy for {scope}.",
                    "capability_id": inspect_capability,
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
                {
                    "name": approve_tool_name,
                    "description": f"Review a discovered target under the {scope} policy.",
                    "capability_id": approve_capability,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "policy_ref": {"type": "string"},
                            "target_ref": {"type": "string"},
                        },
                        "required": ["policy_ref", "target_ref"],
                    },
                    "input_map": {
                        "policy_ref": "policy_handle",
                        "target_ref": "target_handle",
                    },
                    "provenance_required": ["policy_ref", "target_ref"],
                },
            ]
        )

        commit_tool = tool_binding(child, commit_tool_name)
        properties = commit_tool["parameters"].setdefault("properties", {})
        required = commit_tool["parameters"].setdefault("required", [])
        if approval_argument in properties:
            raise ValueError(f"commit tool already defines {approval_argument!r}")
        properties[approval_argument] = {
            "type": "string",
            "description": "Approval evidence derived for the exact commit target.",
        }
        required.append(approval_argument)
        commit_tool.setdefault("input_map", {})[approval_argument] = "approval_handle"
        commit_tool.setdefault("provenance_required", []).append(approval_argument)

        commit = capability(child, commit_capability_id)
        branch_ids = solution_commit_branches(
            parent,
            commit_tool_name=commit_tool_name,
            last=bool(hook.get("commit_last")),
        )
        matching = [
            branch
            for branch in commit.get("branches", [])
            if branch.get("id") in branch_ids
        ]
        if len(matching) != len(branch_ids):
            raise ValueError("audit checkpoint could not resolve every solution commit branch")
        for branch in matching:
            branch["when"] = {
                "all": [
                    branch.get("when", True),
                    {"eq": ["$args.approval_handle", approval_handle]},
                    {"eq": [f"{state_path}.reviewed", True]},
                    {
                        "eq": [
                            f"{state_path}.approved_policy",
                            f"{state_path}.current_policy",
                        ]
                    },
                    {
                        "eq": [
                            f"$args.{internal_target_argument}",
                            f"{state_path}.target",
                        ]
                    },
                ]
            }
            branch.setdefault("effects", []).append(
                {"set": f"{state_path}.committed", "value": True}
            )
            branch.setdefault("effects", []).append(
                {
                    "set": f"{state_path}.committed_policy",
                    "value": f"{state_path}.current_policy",
                }
            )
            branch.setdefault("effects", []).append(
                {
                    "set": f"{state_path}.committed_target",
                    "value": f"$args.{internal_target_argument}",
                }
            )
            branch.setdefault("reads", []).extend(
                [
                    f"{state_path}.reviewed",
                    f"{state_path}.current_policy",
                    f"{state_path}.approved_policy",
                    f"{state_path}.target",
                ]
            )
            branch.setdefault("writes", []).extend(
                [
                    f"{state_path}.committed",
                    f"{state_path}.committed_policy",
                    f"{state_path}.committed_target",
                ]
            )

        verify = capability(child, verify_capability_id)
        verify_branch_id = hook.get("verify_branch")
        verify_matching = [
            item
            for item in verify.get("branches", [])
            if verify_branch_id is None or item.get("id") == verify_branch_id
        ]
        if len(verify_matching) != 1:
            raise ValueError("audit checkpoint hook must identify exactly one verify branch")
        verify_branch = verify_matching[0]
        verify_branch.setdefault("response", {})["audit_committed"] = (
            f"{state_path}.committed"
        )
        verify_branch.setdefault("response", {})["audit_committed_target"] = (
            f"{state_path}.committed_target"
        )
        verify_branch.setdefault("response", {})["audit_approved_target"] = (
            f"{state_path}.target"
        )
        verify_branch.setdefault("response", {})["audit_current_policy"] = (
            f"{state_path}.current_policy"
        )
        verify_branch.setdefault("response", {})["audit_approved_policy"] = (
            f"{state_path}.approved_policy"
        )
        verify_branch.setdefault("response", {})["audit_committed_policy"] = (
            f"{state_path}.committed_policy"
        )
        verify_branch.setdefault("reads", []).extend(
            [
                f"{state_path}.committed",
                f"{state_path}.committed_target",
                f"{state_path}.target",
                f"{state_path}.current_policy",
                f"{state_path}.approved_policy",
                f"{state_path}.committed_policy",
            ]
        )
        append_goal_condition(
            child,
            {
                "all": [
                    {"eq": [f"{state_path}.committed", True]},
                    {
                        "eq": [
                            f"{state_path}.approved_policy",
                            f"{state_path}.current_policy",
                        ]
                    },
                    {
                        "eq": [
                            f"{state_path}.committed_policy",
                            f"{state_path}.current_policy",
                        ]
                    },
                    {"eq": [f"{state_path}.target", target_value]},
                    {"eq": [f"{state_path}.committed_target", target_value]},
                ]
            },
        )
        child.contract.setdefault("forbidden_shortcuts", []).append(
            f"perform {scope} commit without target-specific reviewed evidence"
        )

        for actions in solution_action_lists(child):
            commit_index = action_index_in(
                actions, commit_tool_name, last=bool(hook.get("commit_last"))
            )
            plan_target = actions[commit_index].get("arguments", {}).get(target_argument)
            if not isinstance(plan_target, str) or not plan_target:
                raise ValueError("every solution plan must provide the semantic commit target")
            actions.insert(commit_index, {"tool": inspect_tool_name, "arguments": {}})
            actions.insert(
                commit_index + 1,
                {
                    "tool": approve_tool_name,
                    "arguments": {"policy_ref": policy_handle, "target_ref": plan_target},
                },
            )
            actions[commit_index + 2].setdefault("arguments", {})[
                approval_argument
            ] = approval_handle
        child = replace(
            child,
            instruction=child.instruction.rstrip()
            + f" The consequential {scope} action must satisfy the current target-specific authorization policy.\n",
        )

        errors = validate_bundle(child)
        if errors:
            raise ValueError("audit checkpoint operator produced invalid bundle: " + "; ".join(errors))
        return EvolutionProduct(
            bundle=child,
            patch={
                "operator_id": self.operator_id,
                "semantic_changes": [
                    f"{scope} commit now requires a freshly inspected policy handle",
                    "approval is derived for a previously discovered target handle",
                    "the original plan fails because the commit schema and causal precondition changed",
                ],
                "added_goal_paths": [f"{state_path}.committed"],
            },
        )
