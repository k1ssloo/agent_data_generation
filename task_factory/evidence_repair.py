"""Deterministic repairs for narrowly provable final-evidence omissions."""

from __future__ import annotations

import copy
from typing import Any

from runtime.predicates import predicate_paths


HOOK_ERROR_PREFIX = (
    "recursive evolution hook unavailable: no final domain observation covers "
    "every goal path"
)
FINAL_OBSERVATION_ERROR = "final observations do not cover every goal predicate path"
_CASCADE_ERRORS = {
    FINAL_OBSERVATION_ERROR,
    "alternate API rendering changed task validity",
    "counterfactual strategy adaptation failed",
    (
        "assigned alternative_plan_affordance requires at least one valid state "
        "intervention where the adapted strategy succeeds and the stale strategy fails"
    ),
}
_CAUSAL_COMPANION_ERRORS = {
    "required arguments lack tool-output provenance",
}


def _goal_paths(contract: dict[str, Any]) -> set[str]:
    return {
        path
        for item in contract.get("goal_predicates", [])
        if isinstance(item, dict)
        for path in predicate_paths(item.get("predicate", item))
    }


def _covered(path: str, reads: list[str]) -> bool:
    return any(path.startswith(read) or read.startswith(path) for read in reads)


def _set_evidence_path(response: dict[str, Any], state_path: str) -> bool:
    leaf = state_path.rsplit(".", 1)[-1]
    if leaf in response and response[leaf] != state_path:
        return False
    response[leaf] = state_path
    return True


def _unique_name(existing: set[str], base: str) -> str:
    if base not in existing:
        return base
    index = 2
    while f"{base}_{index}" in existing:
        index += 1
    return f"{base}_{index}"


def _append_read_only_goal_observation(
    contract: dict[str, Any],
    candidate: dict[str, Any],
    report: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Append a domain observation when a successful final action is a mutation."""
    errors = report.get("errors", [])
    causal = report.get("causal_validation", {})
    causal_errors = causal.get("errors", []) if isinstance(causal, dict) else []
    causal_error_strings = list(map(str, causal_errors))
    disallowed_causal_errors = [
        error
        for error in causal_error_strings
        if error != FINAL_OBSERVATION_ERROR
        and error in _CAUSAL_COMPANION_ERRORS
    ]
    if (
        report.get("phase") != "execution"
        or not isinstance(errors, list)
        or not errors
        or FINAL_OBSERVATION_ERROR not in causal_error_strings
        or disallowed_causal_errors
    ):
        return None, {"reason": "not_a_post_mutation_evidence_cascade"}
    episode = report.get("episode", {})
    trace = episode.get("trace", []) if isinstance(episode, dict) else []
    if episode.get("status") != "goal_satisfied" or not trace:
        return None, {"reason": "reference_execution_not_goal_satisfied"}
    final_step = trace[-1]
    if not final_step.get("write_set"):
        return None, {"reason": "final_step_is_not_a_mutation"}

    goal_paths = sorted(_goal_paths(contract))
    if not goal_paths or any(not path.startswith("$state.") for path in goal_paths):
        return None, {"reason": "no_repairable_goal_paths"}
    result = copy.deepcopy(candidate)
    environment = result.get("environment", {})
    capabilities = environment.get("capabilities", {})
    bindings = result.get("bindings", {}).get("tools", [])
    reference_plan = result.get("reference_plan", {})
    if (
        not isinstance(capabilities, dict)
        or not isinstance(bindings, list)
        or not isinstance(reference_plan.get("actions"), list)
    ):
        return None, {"reason": "candidate_structure_is_not_repairable"}

    existing_capabilities = set(map(str, capabilities))
    existing_tools = {
        str(tool.get("name")) for tool in bindings if isinstance(tool, dict)
    }
    capability_id = _unique_name(
        existing_capabilities, "deterministic_final_domain_observation"
    )
    tool_name = _unique_name(existing_tools, "observe_final_domain_state")
    response = {
        f"goal_evidence_{index:02d}": path
        for index, path in enumerate(goal_paths, start=1)
    }
    capabilities[capability_id] = {
        "branches": [
            {
                "id": "current_domain_state_visible",
                "when": True,
                "response": response,
                "reads": goal_paths,
                "writes": [],
                "effects": [],
                "resolves_errors": [],
            }
        ]
    }
    bindings.append(
        {
            "name": tool_name,
            "description": (
                "Read the current domain state after the preceding operation."
            ),
            "capability_id": capability_id,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "provenance_required": [],
        }
    )
    observation_action = {"tool": tool_name, "arguments": {}}
    reference_plan["actions"].append(copy.deepcopy(observation_action))
    counterfactual_count = 0
    for variant in reference_plan.get("counterfactuals", []):
        if isinstance(variant, dict) and isinstance(variant.get("actions"), list):
            variant["actions"].append(copy.deepcopy(observation_action))
            counterfactual_count += 1
    return result, {
        "mode": "append_post_mutation_observation",
        "capability_id": capability_id,
        "tool_name": tool_name,
        "added_goal_paths": goal_paths,
        "counterfactuals_updated": counterfactual_count,
    }


def repair_final_goal_evidence(
    contract: dict[str, Any],
    candidate: dict[str, Any],
    report: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Expose missing goal state on the already successful final observation.

    This deliberately handles one case only: execution and causal validation
    succeeded, and recursive preparation rejected the bundle solely because the
    final successful observation did not cover every goal path.
    """
    appended, append_details = _append_read_only_goal_observation(
        contract, candidate, report
    )
    if appended is not None:
        return appended, append_details

    errors = report.get("errors", [])
    if (
        report.get("phase") != "execution"
        or not isinstance(errors, list)
        or len(errors) != 1
        or not str(errors[0]).startswith(HOOK_ERROR_PREFIX)
        or not report.get("causal_validation", {}).get("valid")
    ):
        return None, {
            "reason": "not_a_final_evidence_only_failure",
            "append_reason": append_details.get("reason"),
        }
    episode = report.get("episode", {})
    trace = episode.get("trace", []) if isinstance(episode, dict) else []
    if episode.get("status") != "goal_satisfied" or not trace:
        return None, {"reason": "reference_execution_not_goal_satisfied"}
    final_step = trace[-1]
    if final_step.get("write_set"):
        return None, {"reason": "final_step_is_a_mutation_not_an_observation"}
    capability_id = final_step.get("capability_id")
    branch_id = final_step.get("selected_branch")
    if not isinstance(capability_id, str) or not isinstance(branch_id, str):
        return None, {"reason": "final_step_has_no_selected_domain_branch"}
    reads = final_step.get("read_set", [])
    missing = sorted(path for path in _goal_paths(contract) if not _covered(path, reads))
    if not missing or any(not path.startswith("$state.") for path in missing):
        return None, {"reason": "no_repairable_missing_goal_paths"}

    result = copy.deepcopy(candidate)
    capabilities = result.get("environment", {}).get("capabilities", {})
    capability = capabilities.get(capability_id) if isinstance(capabilities, dict) else None
    branches = capability.get("branches", []) if isinstance(capability, dict) else []
    branch = next(
        (
            item
            for item in branches
            if isinstance(item, dict) and item.get("id") == branch_id
        ),
        None,
    )
    if branch is None or not isinstance(branch.get("response"), dict):
        return None, {"reason": "selected_branch_not_found_in_candidate"}
    branch_reads = branch.setdefault("reads", [])
    if not isinstance(branch_reads, list):
        return None, {"reason": "selected_branch_reads_is_not_a_list"}
    for path in missing:
        if not _set_evidence_path(branch["response"], path):
            return None, {"reason": "goal_evidence_response_collision", "path": path}
        if path not in branch_reads:
            branch_reads.append(path)
    branch["reads"] = sorted(branch_reads)
    return result, {
        "capability_id": capability_id,
        "branch_id": branch_id,
        "added_goal_paths": missing,
    }


__all__ = [
    "FINAL_OBSERVATION_ERROR",
    "HOOK_ERROR_PREFIX",
    "repair_final_goal_evidence",
]
