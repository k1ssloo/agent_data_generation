"""Compile a replayed WikiHow workflow into a causal task bundle candidate."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import re
from typing import Any

from scripts.executable_environment import (
    build_environment_for_row,
    eval_condition,
    execute_tool,
)

class WikiHowCompileError(ValueError):
    """Raised when a legacy workflow cannot be lifted without guessing."""


_PUBLIC_OPTION_ARGUMENTS = {
    "collection",
    "target_collection",
    "view",
    "resource_type",
    "job_type",
    "provider",
    "relationship",
    "mode",
    "format",
    "channel",
    "action",
}


@dataclass(frozen=True)
class WikiHowSeed:
    task_id: str
    domain: str
    instruction: str
    contract: dict[str, Any]
    environment: dict[str, Any]
    bindings: dict[str, Any]
    reference_plan: dict[str, Any]
    evolution_hooks: dict[str, Any]

    @property
    def candidate(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "environment": self.environment,
            "bindings": self.bindings,
            "reference_plan": self.reference_plan,
        }

    @property
    def manifest_metadata(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "seed_family": "wikihow_compiled_v1",
            "evolution_hooks": self.evolution_hooks,
        }


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return result or "workflow"


def _state_paths(value: Any) -> set[str]:
    paths: set[str] = set()
    if isinstance(value, str) and value.startswith("$state"):
        paths.add(value)
    elif isinstance(value, list):
        for item in value:
            paths |= _state_paths(item)
    elif isinstance(value, dict):
        for item in value.values():
            paths |= _state_paths(item)
    return paths


def _response_scalars(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for item in value.values():
            result |= _response_scalars(item)
    elif isinstance(value, list):
        for item in value:
            result |= _response_scalars(item)
    elif isinstance(value, str):
        result.add(value)
    return result


def _top_level_effects(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    effects = []
    for key in sorted(set(before) | set(after)):
        path = f"$state.{key}"
        if key not in after:
            effects.append({"delete": path})
        elif key not in before or before[key] != after[key]:
            effects.append({"set": path, "value": copy.deepcopy(after[key])})
    return effects


def _exact_args_predicate(arguments: dict[str, Any]) -> dict[str, Any] | bool:
    conditions = [
        {"eq": [f"$args.{name}", copy.deepcopy(value)]}
        for name, value in sorted(arguments.items())
    ]
    if not conditions:
        return True
    return conditions[0] if len(conditions) == 1 else {"all": conditions}


def _tool_calls(row: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    messages = row.get("messages", [])
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        call = message.get("tool_call")
        if not isinstance(call, dict):
            continue
        if index + 1 >= len(messages) or messages[index + 1].get("role") != "tool":
            raise WikiHowCompileError(f"tool call at message {index} has no response")
        response = messages[index + 1]
        if response.get("name") != call.get("name"):
            raise WikiHowCompileError(f"tool response mismatch at message {index + 1}")
        arguments = call.get("arguments", {})
        if not isinstance(arguments, dict):
            raise WikiHowCompileError(f"tool arguments at message {index} are not an object")
        calls.append(
            {
                "name": call.get("name"),
                "arguments": copy.deepcopy(arguments),
                "response": copy.deepcopy(response.get("content", {})),
            }
        )
    if len(calls) < 4:
        raise WikiHowCompileError("workflow needs at least four executable calls")
    return calls


def _public_instruction(row: dict[str, Any]) -> str:
    for message in row.get("messages", []):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    summary = row.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    raise WikiHowCompileError("workflow has no public user objective")


def _compile_public_argument_choices(
    parameters: dict[str, Any],
    calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """Expose genuine API choices without leaking task-specific identifiers."""
    result = copy.deepcopy(parameters)
    properties = result.get("properties", {})
    if not isinstance(properties, dict):
        return result
    values_by_name: dict[str, list[Any]] = {}
    for call in calls:
        for name, value in call.get("arguments", {}).items():
            values_by_name.setdefault(name, []).append(value)
    for name, values in values_by_name.items():
        definition = properties.get(name)
        if not isinstance(definition, dict):
            continue
        unique = []
        for value in values:
            if value not in unique:
                unique.append(copy.deepcopy(value))
        if name in _PUBLIC_OPTION_ARGUMENTS and all(
            isinstance(value, (str, int, float)) and not isinstance(value, bool)
            for value in unique
        ):
            definition["enum"] = unique
        elif len(unique) == 1 and unique[0] in ("", [], {}):
            definition["default"] = unique[0]
            required = result.get("required", [])
            if isinstance(required, list):
                result["required"] = [item for item in required if item != name]
    return result


def compile_wikihow_row(row: dict[str, Any]) -> WikiHowSeed:
    """Lift one already replayable WikiHow workflow into causal-runtime-v1.

    The legacy execution is treated as a solvability witness. Branches are
    grounded in observed arguments, while state effects are derived by replay.
    Synthetic recursive requirements are deliberately not introduced here.
    """
    task_id = str(row.get("id", ""))
    if not task_id:
        raise WikiHowCompileError("row has no id")
    legacy_environment = build_environment_for_row(row)
    state = copy.deepcopy(legacy_environment.get("initial_state", {}))
    initial_state = copy.deepcopy(state)
    calls = _tool_calls(row)
    workflow_context_handle = f"workflow_context_{_slug(task_id)}"
    initial_state["workflow_context"] = {
        "handle": workflow_context_handle,
        "source_task_id": task_id,
    }
    capabilities: dict[str, Any] = {}
    branches_by_tool: dict[str, list[dict[str, Any]]] = {}
    actions = []
    observed_values: set[str] = set()
    provenance_by_tool: dict[str, set[str]] = {}
    selected_branch_ids: list[str] = []

    for call_index, call in enumerate(calls, start=1):
        name = call["name"]
        if not isinstance(name, str) or not name:
            raise WikiHowCompileError(f"call {call_index} has no tool name")
        arguments = call["arguments"]
        before = copy.deepcopy(state)
        expected, errors = execute_tool(name, arguments, state, legacy_environment)
        if errors:
            raise WikiHowCompileError(f"{name} replay failed: {'; '.join(errors)}")
        if expected != call["response"]:
            raise WikiHowCompileError(f"{name} stored response does not match replay")

        for argument, value in arguments.items():
            if isinstance(value, str) and value in observed_values:
                provenance_by_tool.setdefault(name, set()).add(argument)
        observed_values |= _response_scalars(expected)
        effects = _top_level_effects(before, state)
        branch_id = f"observed_{call_index:02d}"
        selected_branch_ids.append(branch_id)
        response = copy.deepcopy(expected)
        if isinstance(response, dict) and response.get("status") == "failed":
            response.setdefault("error_code", f"{_slug(name).upper()}_FAILED")
        branches_by_tool.setdefault(name, []).append(
            {
                "id": branch_id,
                "when": _exact_args_predicate(arguments),
                "response": response,
                "effects": effects,
                "reads": sorted(
                    {
                        path
                        for legacy_branch in legacy_environment["tool_rules"][name]["branches"]
                        if eval_condition(
                            legacy_branch.get("if", True),
                            {"state": before, "args": arguments, "response": None},
                        )
                        for path in _state_paths(legacy_branch.get("if", True))
                    }
                ),
                "writes": [effect.get("set") or effect.get("delete") for effect in effects],
            }
        )
        actions.append({"tool": name, "arguments": copy.deepcopy(arguments)})

    changed_roots = sorted(key for key in set(initial_state) | set(state) if initial_state.get(key) != state.get(key))
    if not changed_roots:
        raise WikiHowCompileError("workflow produces no observable state change")
    created_roots = sorted(key for key in changed_roots if key not in initial_state)
    if created_roots:
        raise WikiHowCompileError(
            "workflow creates goal state absent from the initial schema: "
            + ", ".join(created_roots)
        )
    goal_conditions = [
        {"eq": [f"$state.{key}", copy.deepcopy(state[key])]}
        for key in changed_roots
        if key in state
    ]
    if not goal_conditions:
        raise WikiHowCompileError("workflow deletes state but produces no verifiable outcome")

    last_tool = calls[-1]["name"]

    tools = []
    for tool in row.get("tools", []):
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = function.get("name")
        if name not in branches_by_tool:
            continue
        capability_id = f"wikihow.{_slug(name)}.v1"
        capabilities[capability_id] = {"branches": branches_by_tool[name]}
        tool_calls = [call for call in calls if call["name"] == name]
        binding: dict[str, Any] = {
            "name": name,
            "description": str(function.get("description", "")),
            "capability_id": capability_id,
            "parameters": _compile_public_argument_choices(
                function.get("parameters", {}), tool_calls
            ),
        }
        provenance = sorted(provenance_by_tool.get(name, set()))
        if provenance:
            binding["provenance_required"] = provenance
        tools.append(binding)
    if {call["name"] for call in calls} - {tool["name"] for tool in tools}:
        raise WikiHowCompileError("one or more called tools have no public schema")

    context_capability = "wikihow.workflow_context.observe.v1"
    context_tool = "observe_workflow_context"
    capabilities[context_capability] = {
        "branches": [
            {
                "id": "workflow_context_visible",
                "when": True,
                "response": {
                    "workflow_context_handle": "$state.workflow_context.handle"
                },
                "effects": [],
                "reads": ["$state.workflow_context.handle"],
                "writes": [],
                "resolves_errors": [],
            }
        ]
    }
    tools.insert(
        0,
        {
            "name": context_tool,
            "description": (
                "Observe the public context for the current workflow before acting."
            ),
            "capability_id": context_capability,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "provenance_required": [],
        },
    )
    actions.insert(0, {"tool": context_tool, "arguments": {}})

    outcome_capability = "wikihow.workflow_outcome.observe.v1"
    outcome_tool = "observe_workflow_outcome"
    goal_state_paths = [f"$state.{key}" for key in changed_roots]
    capabilities[outcome_capability] = {
        "branches": [
            {
                "id": "current_workflow_outcome_visible",
                "when": {
                    "eq": [
                        "$args.workflow_context_handle",
                        "$state.workflow_context.handle",
                    ]
                },
                "response": {
                    "outcome_state": {
                        key: f"$state.{key}" for key in changed_roots
                    },
                },
                "effects": [],
                "reads": ["$state.workflow_context.handle", *goal_state_paths],
                "writes": [],
                "resolves_errors": [],
            },
            {
                "id": "workflow_context_mismatch",
                "when": True,
                "response": {
                    "error_code": "WORKFLOW_CONTEXT_MISMATCH",
                    "recoverable": False,
                },
                "effects": [],
                "reads": ["$state.workflow_context.handle"],
                "writes": [],
                "resolves_errors": [],
            },
        ]
    }
    tools.append(
        {
            "name": outcome_tool,
            "description": (
                "Observe the resulting state for the same public workflow context."
            ),
            "capability_id": outcome_capability,
            "parameters": {
                "type": "object",
                "properties": {
                    "workflow_context_handle": {
                        "type": "string",
                        "description": (
                            "Opaque context returned by the initial workflow observation."
                        ),
                    }
                },
                "required": ["workflow_context_handle"],
                "additionalProperties": False,
            },
            "provenance_required": ["workflow_context_handle"],
        }
    )
    actions.append(
        {
            "tool": outcome_tool,
            "arguments": {"workflow_context_handle": workflow_context_handle},
        }
    )

    goal_predicate: dict[str, Any] = (
        goal_conditions[0] if len(goal_conditions) == 1 else {"all": goal_conditions}
    )
    contract = {
        "contract_version": "task-contract-v1",
        "goal": str(row.get("summary") or _public_instruction(row)),
        "goal_predicates": [{"id": "wikihow_outcome", "predicate": goal_predicate}],
        "invariants": [
            {
                "id": "workflow_state_remains_structured",
                "predicate": {"all": [{"exists": f"$state.{key}"} for key in changed_roots]},
            }
        ],
        "requirements": {
            "semantic_recovery": False,
            "async_decision": False,
            "goal_grounded_verification": True,
        },
        "forbidden_shortcuts": [
            "claim completion without producing the WikiHow workflow outcome",
            "claim completion without inspecting the resulting state",
        ],
        "expected_reasoning_features": ["derived_object_dependency", "goal_grounded_verification"],
        "counterfactual_axes": [],
    }

    commit_index = len(calls) - 2
    while commit_index >= 0 and not branches_by_tool[calls[commit_index]["name"]][0].get("effects"):
        commit_index -= 1
    if commit_index < 0:
        raise WikiHowCompileError("workflow has no consequential action before verification")
    commit_call = calls[commit_index]
    target_candidates = [
        value
        for name, value in commit_call["arguments"].items()
        if name in provenance_by_tool.get(commit_call["name"], set()) and isinstance(value, str)
    ]
    if not target_candidates:
        raise WikiHowCompileError("commit action has no observed target handle")
    commit_capability = f"wikihow.{_slug(commit_call['name'])}.v1"
    verify_capability = outcome_capability
    title = str(row.get("metadata", {}).get("title") or row.get("summary") or task_id)
    domain = str(row.get("domain") or row.get("platform") or "wikihow")
    return WikiHowSeed(
        task_id=f"{task_id}__task_first",
        domain=f"wikihow_{_slug(domain)}",
        instruction=_public_instruction(row),
        contract=contract,
        environment={
            "runtime_version": "causal-runtime-v1",
            "initial_state": initial_state,
            "capabilities": capabilities,
        },
        bindings={"binding_version": "tool-binding-v1", "tools": tools},
        reference_plan={"actions": actions, "counterfactuals": []},
        evolution_hooks={
            "audit_checkpoint": {
                "scope": "workflow_commit",
                "commit_tool": commit_call["name"],
                "commit_capability": commit_capability,
                "commit_branch": selected_branch_ids[commit_index],
                "target_value": target_candidates[0],
                "verify_capability": verify_capability,
                "verify_branch": "current_workflow_outcome_visible",
            }
        },
    )
