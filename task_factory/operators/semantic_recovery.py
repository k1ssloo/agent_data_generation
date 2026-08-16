"""Inject a recoverable semantic failure into a consequential action."""

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
        raise ValueError(f"semantic recovery hook requires non-empty {key!r}")
    return item


class SemanticRecoveryOperator:
    """Require failure observation, diagnosis, state repair, and exact retry."""

    operator_id = "semantic_failure_recovery_v1"

    def apply(self, parent: TaskBundle, *, generation: int) -> EvolutionProduct:
        if self.operator_id in parent.manifest.get("lineage", {}).get("operators", []):
            raise ValueError(f"{self.operator_id} cannot be applied twice in one lineage")
        hooks = parent.manifest.get("evolution_hooks", {})
        hook = hooks.get("audit_checkpoint") if isinstance(hooks, dict) else None
        if not isinstance(hook, dict):
            raise ValueError("bundle does not declare a semantic commit hook")

        scope = _required_string(hook, "scope")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", scope):
            raise ValueError("semantic recovery scope must be snake_case")
        commit_tool_name = _required_string(hook, "commit_tool")
        commit_capability_id = _required_string(hook, "commit_capability")
        verify_capability_id = _required_string(hook, "verify_capability")
        target_value = _required_string(hook, "target_value")
        commit_last = bool(hook.get("commit_last"))
        target_argument, internal_target_argument = commit_target_argument(
            parent,
            commit_tool_name=commit_tool_name,
            target_value=target_value,
            last=commit_last,
        )
        commit_branch_ids = solution_commit_branches(
            parent,
            commit_tool_name=commit_tool_name,
            last=commit_last,
        )

        child = clone_bundle(
            parent,
            task_id=f"{parent.task_id}__g{generation}_{scope}_recovery",
            operator_id=self.operator_id,
            generation=generation,
        )
        state_key = f"{scope}_recovery"
        state_path = f"$state.{state_key}"
        error_code = f"{scope.upper()}_TARGET_STATE_CONFLICT"
        failure_report_handle = f"failure_report_{scope}_1"
        diagnosis_handle = f"diagnosis_{scope}_1"
        repair_handle = f"repair_{scope}_1"
        diagnosis_tool_name = f"diagnose_{scope}_failure"
        repair_tool_name = f"repair_{scope}_target_state"
        prepare_tool_name = f"prepare_{scope}_action"
        preparation_argument = "preparation_ref"
        stale_preparation_handle = f"preparation_{scope}_revision_1"
        current_preparation_handle = f"preparation_{scope}_revision_2"
        repair_argument = "repair_evidence_ref"
        target_revision_path = str(
            hook.get("target_revision_path", f"{state_path}.current_revision")
        )
        if not target_revision_path.startswith("$state."):
            raise ValueError("target_revision_path must be under $state")

        child.environment["initial_state"][state_key] = {
            "failure_observed": False,
            "failed_target": "",
            "prepared": False,
            "prepared_target": "",
            "prepared_revision": 0,
            "current_revision": 1,
            "diagnosed": False,
            "repaired": False,
            "repaired_target": "",
            "retried_successfully": False,
        }
        diagnose_capability_id = f"recovery.diagnose.{scope}.v1"
        repair_capability_id = f"recovery.repair.{scope}.v1"
        prepare_capability_id = f"recovery.prepare.{scope}.v1"
        child.environment["capabilities"][prepare_capability_id] = {
            "branches": [
                {
                    "id": "prepare_exact_target_after_repair",
                    "when": {
                        "all": [
                            {"eq": [f"{state_path}.repaired", True]},
                            {"eq": ["$args.target_handle", target_value]},
                        ]
                    },
                    "response": {
                        "preparation_handle": current_preparation_handle,
                        "prepared_target": "$args.target_handle",
                        "prepared_revision": target_revision_path,
                        "stable_after_repair": True,
                    },
                    "effects": [
                        {"set": f"{state_path}.prepared", "value": True},
                        {
                            "set": f"{state_path}.prepared_target",
                            "value": "$args.target_handle",
                        },
                        {
                            "set": f"{state_path}.prepared_revision",
                            "value": target_revision_path,
                        },
                    ],
                    "reads": [
                        f"{state_path}.repaired",
                        target_revision_path,
                    ],
                    "writes": [
                        f"{state_path}.prepared",
                        f"{state_path}.prepared_target",
                        f"{state_path}.prepared_revision",
                    ],
                },
                {
                    "id": "prepare_exact_target_before_concurrent_change",
                    "when": {
                        "all": [
                            {"eq": [f"{state_path}.repaired", False]},
                            {"eq": ["$args.target_handle", target_value]},
                        ]
                    },
                    "response": {
                        "preparation_handle": stale_preparation_handle,
                        "prepared_target": "$args.target_handle",
                        "prepared_revision": target_revision_path,
                    },
                    "effects": [
                        {"set": f"{state_path}.prepared", "value": True},
                        {
                            "set": f"{state_path}.prepared_target",
                            "value": "$args.target_handle",
                        },
                        {
                            "set": f"{state_path}.prepared_revision",
                            "value": target_revision_path,
                        },
                    ],
                    "after_response_effects": [
                        {"increment": target_revision_path, "by": 1}
                    ],
                    "reads": [
                        f"{state_path}.repaired",
                        target_revision_path,
                    ],
                    "writes": [
                        f"{state_path}.prepared",
                        f"{state_path}.prepared_target",
                        f"{state_path}.prepared_revision",
                        target_revision_path,
                    ],
                }
            ]
        }
        child.environment["capabilities"][diagnose_capability_id] = {
            "branches": [
                {
                    "id": "diagnose_observed_failure",
                    "when": {
                        "all": [
                            {"eq": [f"{state_path}.failure_observed", True]},
                            {"eq": ["$args.failure_report_handle", failure_report_handle]},
                            {"eq": ["$args.target_handle", f"{state_path}.failed_target"]},
                        ]
                    },
                    "response": {
                        "diagnosis_handle": diagnosis_handle,
                        "conflict": "target_state_changed_after_preparation",
                        "prepared_revision": f"{state_path}.prepared_revision",
                        "current_revision": target_revision_path,
                        "repairable": True,
                        "affected_target": f"{state_path}.failed_target",
                    },
                    "effects": [{"set": f"{state_path}.diagnosed", "value": True}],
                    "reads": [
                        f"{state_path}.failure_observed",
                        f"{state_path}.failed_target",
                        f"{state_path}.prepared_revision",
                        target_revision_path,
                    ],
                    "writes": [f"{state_path}.diagnosed"],
                }
            ]
        }
        child.environment["capabilities"][repair_capability_id] = {
            "branches": [
                {
                    "id": "repair_diagnosed_target",
                    "when": {
                        "all": [
                            {"eq": [f"{state_path}.diagnosed", True]},
                            {"eq": ["$args.diagnosis_handle", diagnosis_handle]},
                            {"eq": ["$args.target_handle", f"{state_path}.failed_target"]},
                        ]
                    },
                    "response": {
                        "repair_evidence_handle": repair_handle,
                        "status": "repaired",
                        "repaired_target": f"{state_path}.failed_target",
                        "repaired_revision": target_revision_path,
                    },
                    "effects": [
                        {"set": f"{state_path}.repaired", "value": True},
                        {
                            "set": f"{state_path}.repaired_target",
                            "value": "$args.target_handle",
                        },
                    ],
                    "reads": [
                        f"{state_path}.diagnosed",
                        f"{state_path}.failed_target",
                        target_revision_path,
                    ],
                    "writes": [
                        f"{state_path}.repaired",
                        f"{state_path}.repaired_target",
                    ],
                    "resolves_errors": [error_code],
                }
            ]
        }
        child.bindings["tools"].extend(
            [
                {
                    "name": prepare_tool_name,
                    "description": (
                        "Prepare the exact discovered target for the consequential "
                        "action and return the revision snapshot used at commit time."
                    ),
                    "capability_id": prepare_capability_id,
                    "parameters": {
                        "type": "object",
                        "properties": {"target_ref": {"type": "string"}},
                        "required": ["target_ref"],
                    },
                    "input_map": {"target_ref": "target_handle"},
                    "provenance_required": ["target_ref"],
                },
                {
                    "name": diagnosis_tool_name,
                    "description": "Diagnose the recoverable failure returned by the consequential action.",
                    "capability_id": diagnose_capability_id,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "failure_report_ref": {"type": "string"},
                            "target_ref": {"type": "string"},
                        },
                        "required": ["failure_report_ref", "target_ref"],
                    },
                    "input_map": {
                        "failure_report_ref": "failure_report_handle",
                        "target_ref": "target_handle",
                    },
                    "provenance_required": ["failure_report_ref", "target_ref"],
                },
                {
                    "name": repair_tool_name,
                    "description": "Repair the diagnosed state conflict for the exact affected target.",
                    "capability_id": repair_capability_id,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "diagnosis_ref": {"type": "string"},
                            "target_ref": {"type": "string"},
                        },
                        "required": ["diagnosis_ref", "target_ref"],
                    },
                    "input_map": {
                        "diagnosis_ref": "diagnosis_handle",
                        "target_ref": "target_handle",
                    },
                    "provenance_required": ["diagnosis_ref", "target_ref"],
                },
            ]
        )

        commit_tool = tool_binding(child, commit_tool_name)
        properties = commit_tool["parameters"].setdefault("properties", {})
        required = commit_tool["parameters"].setdefault("required", [])
        if repair_argument in properties:
            raise ValueError(f"commit tool already defines {repair_argument!r}")
        properties[preparation_argument] = {
            "type": "string",
            "description": "Revision snapshot prepared for this exact target.",
        }
        properties[repair_argument] = {
            "type": "string",
            "description": "Evidence that a reported target-state conflict was repaired.",
        }
        required.append(preparation_argument)
        commit_tool.setdefault("input_map", {})[preparation_argument] = (
            "preparation_handle"
        )
        commit_tool.setdefault("input_map", {})[repair_argument] = "repair_evidence_handle"
        commit_tool.setdefault("provenance_required", []).append(preparation_argument)
        commit_tool.setdefault("provenance_required", []).append(repair_argument)

        commit = capability(child, commit_capability_id)
        resolved_ids = {
            branch.get("id")
            for branch in commit.get("branches", [])
            if branch.get("id") in commit_branch_ids
        }
        if resolved_ids != commit_branch_ids:
            raise ValueError("semantic recovery could not resolve every solution commit branch")
        failure_branch = {
            "id": f"recoverable_{scope}_conflict",
            "when": {
                "all": [
                    {"eq": [f"{state_path}.prepared", True]},
                    {
                        "eq": [
                            "$args.preparation_handle",
                            stale_preparation_handle,
                        ]
                    },
                    {
                        "eq": [
                            f"$args.{internal_target_argument}",
                            f"{state_path}.prepared_target",
                        ]
                    },
                    {
                        "ne": [
                            f"{state_path}.prepared_revision",
                            target_revision_path,
                        ]
                    },
                ]
            },
            "response": {
                "ok": False,
                "error_code": error_code,
                "failure_report_handle": failure_report_handle,
                "affected_target": f"$args.{internal_target_argument}",
                "retryable": True,
                "prepared_revision": f"{state_path}.prepared_revision",
                "current_revision": target_revision_path,
            },
            "effects": [
                {"set": f"{state_path}.failure_observed", "value": True},
                {
                    "set": f"{state_path}.failed_target",
                    "value": f"$args.{internal_target_argument}",
                },
            ],
            "writes": [
                f"{state_path}.failure_observed",
                f"{state_path}.failed_target",
            ],
            "reads": [
                f"{state_path}.prepared",
                f"{state_path}.prepared_target",
                f"{state_path}.prepared_revision",
                target_revision_path,
            ],
        }
        rewritten_branches = [failure_branch]
        for branch in commit.get("branches", []):
            branch = copy.deepcopy(branch)
            if branch.get("id") in commit_branch_ids:
                branch["when"] = {
                    "all": [
                        branch.get("when", True),
                        {
                            "eq": [
                                "$args.preparation_handle",
                                current_preparation_handle,
                            ]
                        },
                        {
                            "eq": [
                                f"$args.{internal_target_argument}",
                                f"{state_path}.prepared_target",
                            ]
                        },
                        {
                            "eq": [
                                f"{state_path}.prepared_revision",
                                target_revision_path,
                            ]
                        },
                        {"eq": ["$args.repair_evidence_handle", repair_handle]},
                        {"eq": [f"{state_path}.repaired", True]},
                        {
                            "eq": [
                                f"$args.{internal_target_argument}",
                                f"{state_path}.repaired_target",
                            ]
                        },
                    ]
                }
                branch.setdefault("effects", []).append(
                    {"set": f"{state_path}.retried_successfully", "value": True}
                )
                branch.setdefault("reads", []).extend(
                    [
                        f"{state_path}.prepared_target",
                        f"{state_path}.prepared_revision",
                        target_revision_path,
                        f"{state_path}.repaired",
                        f"{state_path}.repaired_target",
                    ]
                )
                branch.setdefault("writes", []).append(
                    f"{state_path}.retried_successfully"
                )
            rewritten_branches.append(branch)
        commit["branches"] = rewritten_branches

        verify = capability(child, verify_capability_id)
        verify_branch_id = hook.get("verify_branch")
        verify_matching = [
            branch
            for branch in verify.get("branches", [])
            if verify_branch_id is None or branch.get("id") == verify_branch_id
        ]
        if len(verify_matching) != 1:
            raise ValueError("semantic recovery hook must identify exactly one verify branch")
        verify_branch = verify_matching[0]
        original_verify_when = copy.deepcopy(verify_branch.get("when", True))
        verify_branch["when"] = {
            "all": [
                original_verify_when,
                {"exists": target_revision_path},
            ]
        }
        verify_branches = verify.get("branches", [])
        verify_index = verify_branches.index(verify_branch)
        verify_branches.insert(
            verify_index + 1,
            {
                "id": f"{scope}_target_not_ready",
                "when": {
                    "all": [
                        copy.deepcopy(original_verify_when),
                        {"not_exists": target_revision_path},
                    ]
                },
                "response": {
                    "ok": False,
                    "error_code": f"{scope.upper()}_TARGET_NOT_READY",
                    "message": (
                        "The workflow context is valid, but its target does not yet "
                        "exist for final outcome inspection."
                    ),
                    "retryable": True,
                },
                "effects": [],
                "reads": sorted(
                    set(verify_branch.get("reads", [])) | {target_revision_path}
                ),
                "writes": [],
                "resolves_errors": [],
            },
        )
        verify_branch.setdefault("response", {})["recovery_completed"] = (
            f"{state_path}.retried_successfully"
        )
        verify_branch.setdefault("response", {})["recovery_failed_target"] = (
            f"{state_path}.failed_target"
        )
        verify_branch.setdefault("response", {})["recovery_repaired_target"] = (
            f"{state_path}.repaired_target"
        )
        verify_branch.setdefault("response", {})["recovery_current_revision"] = (
            target_revision_path
        )
        verify_branch.setdefault("response", {})["recovery_prepared_target"] = (
            f"{state_path}.prepared_target"
        )
        verify_branch.setdefault("response", {})["recovery_prepared_revision"] = (
            f"{state_path}.prepared_revision"
        )
        verify_branch.setdefault("reads", []).extend(
            [
                f"{state_path}.retried_successfully",
                f"{state_path}.failed_target",
                f"{state_path}.repaired_target",
                f"{state_path}.prepared_target",
                f"{state_path}.prepared_revision",
                target_revision_path,
            ]
        )
        append_goal_condition(
            child,
            {
                "all": [
                    {"eq": [f"{state_path}.retried_successfully", True]},
                    {"eq": [f"{state_path}.failed_target", target_value]},
                    {"eq": [f"{state_path}.repaired_target", target_value]},
                    {"eq": [f"{state_path}.prepared_target", target_value]},
                    {
                        "eq": [
                            f"{state_path}.prepared_revision",
                            target_revision_path,
                        ]
                    },
                ]
            },
        )
        child.contract.setdefault("requirements", {})["semantic_recovery"] = True
        child.contract.setdefault("forbidden_shortcuts", []).append(
            f"bypass the recoverable {scope} state conflict or retry a different target"
        )

        for actions in solution_action_lists(child):
            commit_index = action_index_in(actions, commit_tool_name, last=commit_last)
            retry_action = actions[commit_index]
            plan_target = retry_action.get("arguments", {}).get(target_argument)
            if not isinstance(plan_target, str) or not plan_target:
                raise ValueError("every solution plan must provide the semantic commit target")
            actions.insert(
                commit_index,
                {
                    "tool": prepare_tool_name,
                    "arguments": {"target_ref": plan_target},
                },
            )
            commit_index += 1
            retry_action = actions[commit_index]
            retry_action.setdefault("arguments", {})[preparation_argument] = (
                stale_preparation_handle
            )
            failed_attempt = copy.deepcopy(retry_action)
            failed_attempt.setdefault("arguments", {}).pop(repair_argument, None)
            actions.insert(commit_index, failed_attempt)
            actions.insert(
                commit_index + 1,
                {
                    "tool": diagnosis_tool_name,
                    "arguments": {
                        "failure_report_ref": failure_report_handle,
                        "target_ref": plan_target,
                    },
                },
            )
            actions.insert(
                commit_index + 2,
                {
                    "tool": repair_tool_name,
                    "arguments": {
                        "diagnosis_ref": diagnosis_handle,
                        "target_ref": plan_target,
                    },
                },
            )
            actions.insert(
                commit_index + 3,
                {
                    "tool": prepare_tool_name,
                    "arguments": {"target_ref": plan_target},
                },
            )
            actions[commit_index + 4].setdefault("arguments", {})[
                preparation_argument
            ] = current_preparation_handle
            actions[commit_index + 4].setdefault("arguments", {})[
                repair_argument
            ] = repair_handle

        child.manifest["evolution_hooks"]["audit_checkpoint"]["commit_last"] = True

        recovery_instruction = (
            f"If the consequential {scope} action encounters a recoverable consistency "
            "conflict, preserve the original target and complete the goal without "
            "bypassing that conflict."
        )
        child = replace(
            child,
            instruction=child.instruction.rstrip() + " " + recovery_instruction + "\n",
        )
        if isinstance(child.contract.get("instruction_claims"), list) and isinstance(
            child.contract.get("goal_clauses"), list
        ):
            clause_id = f"{scope}_recovery_completed"
            child.contract["instruction_claims"].append(
                {
                    "evidence_span": recovery_instruction,
                    "kind": "synthetic_constraint",
                    "clause_ids": [clause_id],
                }
            )
            child.contract["goal_clauses"].append(
                {
                    "id": clause_id,
                    "predicate": {
                        "all": [
                            {"eq": [f"{state_path}.retried_successfully", True]},
                            {"eq": [f"{state_path}.failed_target", target_value]},
                            {"eq": [f"{state_path}.repaired_target", target_value]},
                            {"eq": [f"{state_path}.prepared_target", target_value]},
                            {
                                "eq": [
                                    f"{state_path}.prepared_revision",
                                    target_revision_path,
                                ]
                            },
                        ]
                    },
                    "transition_paths": [
                        f"{state_path}.retried_successfully",
                        f"{state_path}.failed_target",
                        f"{state_path}.repaired_target",
                        f"{state_path}.prepared_target",
                        f"{state_path}.prepared_revision",
                        target_revision_path,
                    ],
                    "evidence_paths": [
                        f"{state_path}.retried_successfully",
                        f"{state_path}.failed_target",
                        f"{state_path}.repaired_target",
                        f"{state_path}.prepared_target",
                        f"{state_path}.prepared_revision",
                        target_revision_path,
                    ],
                    "witness_tools": [
                        prepare_tool_name,
                        diagnosis_tool_name,
                        repair_tool_name,
                        commit_tool_name,
                    ],
                }
            )
        errors = validate_bundle(child)
        if errors:
            raise ValueError(
                "semantic recovery operator produced invalid bundle: " + "; ".join(errors)
            )
        return EvolutionProduct(
            bundle=child,
            patch={
                "operator_id": self.operator_id,
                "semantic_changes": [
                    "the consequential action now fails once with an observable semantic conflict",
                    "diagnosis binds the failure report to the original target",
                    "state repair produces evidence required by the retried action",
                    "final verification checks that the repaired action actually succeeded",
                ],
                "added_goal_paths": [f"{state_path}.retried_successfully"],
            },
        )
