"""Deterministic grounding for immediate ordinal control arguments."""

from __future__ import annotations

import copy
import re
from typing import Any


_ORDINAL_ARGUMENT = re.compile(
    r"(?:^|_)(?:page|step|item|part|attempt|retry|sequence|index|count|number)(?:_|$)"
)


def _binding_by_name(candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tools = candidate.get("bindings", {}).get("tools", [])
    return {
        tool["name"]: tool
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("name"), str)
    }


def repair_immediate_ordinal_provenance(
    candidate: dict[str, Any], report: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Expose an ordinal on the successful observation immediately before use."""
    if report.get("phase") != "execution":
        return None, {"reason": "not_an_execution_report"}
    causal = report.get("causal_validation", {})
    metrics = causal.get("metrics", {}) if isinstance(causal, dict) else {}
    unexplained = metrics.get("unexplained_arguments", [])
    trace = report.get("episode", {}).get("trace", [])
    if not isinstance(unexplained, list) or not unexplained or not isinstance(trace, list):
        return None, {"reason": "no_unexplained_arguments"}
    bindings = _binding_by_name(candidate)
    repairs: list[dict[str, Any]] = []
    for item in unexplained:
        if not isinstance(item, dict):
            return None, {"reason": "malformed_unexplained_argument"}
        step = item.get("step")
        tool_name = item.get("tool")
        argument = item.get("argument")
        value = item.get("value")
        if (
            not isinstance(step, int)
            or step < 2
            or not isinstance(tool_name, str)
            or not isinstance(argument, str)
            or not isinstance(value, int)
            or isinstance(value, bool)
            or not _ORDINAL_ARGUMENT.search(argument)
            or step - 2 >= len(trace)
        ):
            return None, {"reason": "argument_is_not_an_immediate_ordinal"}
        binding = bindings.get(tool_name, {})
        definition = (
            binding.get("parameters", {}).get("properties", {}).get(argument, {})
            if isinstance(binding, dict)
            else {}
        )
        if not isinstance(definition, dict) or definition.get("type") != "integer":
            return None, {"reason": "ordinal_schema_is_not_integer"}
        prior = trace[step - 2]
        if (
            not isinstance(prior, dict)
            or prior.get("error_code")
            or not isinstance(prior.get("capability_id"), str)
            or not isinstance(prior.get("selected_branch"), str)
            or not isinstance(prior.get("response"), dict)
        ):
            return None, {"reason": "prior_step_is_not_a_successful_observation"}
        repairs.append(
            {
                "step": step,
                "tool": tool_name,
                "argument": argument,
                "value": value,
                "producer_capability": prior["capability_id"],
                "producer_branch": prior["selected_branch"],
            }
        )

    result = copy.deepcopy(candidate)
    capabilities = result.get("environment", {}).get("capabilities", {})
    for repair in repairs:
        capability = capabilities.get(repair["producer_capability"])
        branches = capability.get("branches", []) if isinstance(capability, dict) else []
        branch = next(
            (
                item
                for item in branches
                if isinstance(item, dict)
                and item.get("id") == repair["producer_branch"]
            ),
            None,
        )
        if branch is None or not isinstance(branch.get("response"), dict):
            return None, {"reason": "producer_branch_not_found"}
        response = branch["response"]
        key = repair["argument"]
        response_text = " ".join(
            str(value).casefold() for value in response.values() if isinstance(value, str)
        )
        if "next" in response_text:
            key = "next_" + key
        elif repair["value"] == 1 and any(
            name in response for name in ("page_count", "item_count", "part_count")
        ):
            key = "first_" + key
        if key in response and response[key] != repair["value"]:
            key = "next_" + key
        if key in response and response[key] != repair["value"]:
            return None, {"reason": "producer_response_collision", "key": key}
        response[key] = repair["value"]
        repair["response_field"] = key
    return result, {"repairs": repairs}


__all__ = ["repair_immediate_ordinal_provenance"]
