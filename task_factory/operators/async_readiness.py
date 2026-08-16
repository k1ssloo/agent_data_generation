"""Add a stateful pending-to-ready lifecycle before a semantic commit."""

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
        raise ValueError(f"async readiness hook requires non-empty {key!r}")
    return item


class AsyncReadinessOperator:
    """Force a poll, pending observation, retry, and evidence-backed commit."""

    operator_id = "async_readiness_retry_v1"

    def apply(self, parent: TaskBundle, *, generation: int) -> EvolutionProduct:
        if self.operator_id in parent.manifest.get("lineage", {}).get("operators", []):
            raise ValueError(f"{self.operator_id} cannot be applied twice in one lineage")
        hooks = parent.manifest.get("evolution_hooks", {})
        hook = hooks.get("audit_checkpoint") if isinstance(hooks, dict) else None
        if not isinstance(hook, dict):
            raise ValueError("bundle does not declare a semantic commit hook")

        scope = _required_string(hook, "scope")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", scope):
            raise ValueError("async readiness scope must be snake_case")
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
        poll_tool_name = f"poll_{scope}_readiness"
        readiness_argument = "readiness_ref"
        readiness_handle = f"readiness_{scope}_ready_1"
        state_key = f"{scope}_lifecycle"
        state_path = f"$state.{state_key}"

        child = clone_bundle(
            parent,
            task_id=f"{parent.task_id}__g{generation}_{scope}_readiness",
            operator_id=self.operator_id,
            generation=generation,
        )
        child.environment["initial_state"][state_key] = {
            "poll_count": 0,
            "ready": False,
            "target": "",
            "committed": False,
        }
        poll_capability = f"lifecycle.poll.{scope}.v1"
        child.environment["capabilities"][poll_capability] = {
            "branches": [
                {
                    "id": "pending_retry",
                    "when": {
                        "all": [
                            {"eq": [f"{state_path}.poll_count", 0]},
                        ]
                    },
                    "response": {"status": "pending", "retryable": True},
                    "effects": [
                        {"increment": f"{state_path}.poll_count", "by": 1},
                        {"set": f"{state_path}.target", "value": "$args.target_handle"},
                    ],
                    "reads": [f"{state_path}.poll_count"],
                    "writes": [f"{state_path}.poll_count", f"{state_path}.target"],
                },
                {
                    "id": "ready_with_evidence",
                    "when": {
                        "all": [
                            {"eq": ["$args.target_handle", f"{state_path}.target"]},
                            {"eq": [f"{state_path}.poll_count", 1]},
                        ]
                    },
                    "response": {
                        "status": "ready",
                        "retryable": False,
                        "readiness_handle": readiness_handle,
                    },
                    "effects": [{"set": f"{state_path}.ready", "value": True}],
                    "reads": [f"{state_path}.poll_count"],
                    "writes": [f"{state_path}.ready"],
                },
            ]
        }
        child.bindings["tools"].append(
            {
                "name": poll_tool_name,
                "description": "Observe whether the pending target is ready; retry only while pending.",
                "capability_id": poll_capability,
                "parameters": {
                    "type": "object",
                    "properties": {"target_ref": {"type": "string"}},
                    "required": ["target_ref"],
                },
                "input_map": {"target_ref": "target_handle"},
                "provenance_required": ["target_ref"],
            }
        )

        commit_tool = tool_binding(child, commit_tool_name)
        properties = commit_tool["parameters"].setdefault("properties", {})
        required = commit_tool["parameters"].setdefault("required", [])
        if readiness_argument in properties:
            raise ValueError(f"commit tool already defines {readiness_argument!r}")
        properties[readiness_argument] = {
            "type": "string",
            "description": "Readiness evidence produced only after the lifecycle becomes ready.",
        }
        required.append(readiness_argument)
        commit_tool.setdefault("input_map", {})[readiness_argument] = "readiness_handle"
        commit_tool.setdefault("provenance_required", []).append(readiness_argument)

        commit = capability(child, commit_capability_id)
        commit_branch_ids = solution_commit_branches(
            parent,
            commit_tool_name=commit_tool_name,
            last=bool(hook.get("commit_last")),
        )
        matching = [
            branch
            for branch in commit.get("branches", [])
            if branch.get("id") in commit_branch_ids
        ]
        if len(matching) != len(commit_branch_ids):
            raise ValueError("async readiness could not resolve every solution commit branch")
        for commit_branch in matching:
            commit_branch["when"] = {
                "all": [
                    commit_branch.get("when", True),
                    {"eq": ["$args.readiness_handle", readiness_handle]},
                    {"eq": [f"{state_path}.ready", True]},
                    {
                        "eq": [
                            f"$args.{internal_target_argument}",
                            f"{state_path}.target",
                        ]
                    },
                ]
            }
            commit_branch.setdefault("effects", []).append(
                {"set": f"{state_path}.committed", "value": True}
            )
            commit_branch.setdefault("reads", []).extend(
                [f"{state_path}.ready", f"{state_path}.target"]
            )
            commit_branch.setdefault("writes", []).append(f"{state_path}.committed")

        verify = capability(child, verify_capability_id)
        verify_branch_id = hook.get("verify_branch")
        verify_matching = [
            branch
            for branch in verify.get("branches", [])
            if verify_branch_id is None or branch.get("id") == verify_branch_id
        ]
        if len(verify_matching) != 1:
            raise ValueError("async readiness hook must identify exactly one verify branch")
        verify_branch = verify_matching[0]
        verify_branch.setdefault("response", {})["lifecycle_committed"] = (
            f"{state_path}.committed"
        )
        verify_branch.setdefault("reads", []).append(f"{state_path}.committed")
        append_goal_condition(child, {"eq": [f"{state_path}.committed", True]})
        child.contract.setdefault("requirements", {})["async_decision"] = True
        child.contract.setdefault("forbidden_shortcuts", []).append(
            f"commit {scope} before observing pending and then ready lifecycle states"
        )

        for actions in solution_action_lists(child):
            commit_index = action_index_in(
                actions, commit_tool_name, last=bool(hook.get("commit_last"))
            )
            plan_target = actions[commit_index].get("arguments", {}).get(target_argument)
            if not isinstance(plan_target, str) or not plan_target:
                raise ValueError("every solution plan must provide the semantic commit target")
            actions.insert(
                commit_index,
                {"tool": poll_tool_name, "arguments": {"target_ref": plan_target}},
            )
            actions.insert(
                commit_index + 1,
                {"tool": poll_tool_name, "arguments": {"target_ref": plan_target}},
            )
            actions[commit_index + 2].setdefault("arguments", {})[
                readiness_argument
            ] = readiness_handle
        child = replace(
            child,
            instruction=child.instruction.rstrip()
            + f" Complete the consequential {scope} action only while its exact target is ready.\n",
        )
        errors = validate_bundle(child)
        if errors:
            raise ValueError("async readiness operator produced invalid bundle: " + "; ".join(errors))
        return EvolutionProduct(
            bundle=child,
            patch={
                "operator_id": self.operator_id,
                "semantic_changes": [
                    "one public poll tool reaches pending and ready branches in the same rollout",
                    "the pending observation causes a retry instead of a premature commit",
                    "the consequential action consumes readiness evidence from the later observation",
                ],
                "added_goal_paths": [f"{state_path}.committed"],
            },
        )
