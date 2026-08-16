"""Hard quality gates for decision-dense adaptive trajectories."""

from __future__ import annotations

import re
from typing import Any

from task_factory.bundle import TaskBundle

from .ablation import evaluate_action_ablation


_ORACLE_NAME_PATTERNS = (
    re.compile(r"^(?:diagnose|repair|recover|fix)_"),
    re.compile(r"(?:^|_)open_exact(?:_|$)"),
    re.compile(r"(?:^|_)(?:exact|correct|recommended)_solution(?:_|$)"),
    re.compile(r"(?:^|_)solution_for(?:_|$)"),
)
_ORACLE_DESCRIPTION_PATTERNS = (
    re.compile(r"\b(?:exact|correct|recommended) solution\b"),
    re.compile(r"\b(?:diagnose|repair|fix) this (?:failure|error)\b"),
    re.compile(r"\buse (?:this|it) after (?:the )?(?:failure|error)\b"),
    re.compile(r"\bafter\b.{0,80}\b(?:failure|error)\b"),
)
_PROCEDURAL_RECOVERY = re.compile(
    r"\b(?:if|when|after)\b.{0,100}\b(?:error|fail|failure)\b.{0,140}"
    r"\b(?:diagnose|repair|fix|clear|edit|open|navigate|retry)\b",
    re.IGNORECASE | re.DOTALL,
)


def validate_tool_oracle_resistance(bundle: TaskBundle) -> dict[str, Any]:
    """Reject intent-labelled tools that disclose the privileged solution role."""
    violations = []
    for tool in bundle.tools:
        name = str(tool.get("name", ""))
        description = " ".join(str(tool.get("description", "")).casefold().split())
        reasons = []
        if any(pattern.search(name.casefold()) for pattern in _ORACLE_NAME_PATTERNS):
            reasons.append("solution_role_in_name")
        if any(pattern.search(description) for pattern in _ORACLE_DESCRIPTION_PATTERNS):
            reasons.append("solution_role_in_description")
        if reasons:
            violations.append({"tool": name, "reasons": reasons})
    return {
        "valid": not violations,
        "violations": violations,
        "checked_tool_count": len(bundle.tools),
    }


def validate_instruction_route_hiding(
    bundle: TaskBundle, counterfactual: dict[str, Any]
) -> dict[str, Any]:
    """Ensure the user goal does not prescribe a discovered recovery route."""
    instruction = " ".join(bundle.instruction.casefold().split())
    strategies = list(counterfactual.get("baseline_recovery_strategies", []))
    for variant in counterfactual.get("variants", []):
        strategies.extend(variant.get("recovery_strategies", []))
    exposed_codes = sorted(
        {
            str(item.get("error_code"))
            for item in strategies
            if item.get("error_code")
            and str(item["error_code"]).casefold() in instruction
        }
    )
    exposed_tools = sorted(
        {
            str(item.get("recovery_tool"))
            for item in strategies
            if item.get("recovery_tool")
            and str(item["recovery_tool"]).casefold() in instruction
        }
    )
    procedural = bool(_PROCEDURAL_RECOVERY.search(bundle.instruction)) or bool(
        exposed_codes
        and re.search(
            r"\b(?:diagnose|repair|fix|clear|edit|open|navigate|retry)\b",
            bundle.instruction,
            re.IGNORECASE,
        )
    )
    errors = []
    if exposed_codes:
        errors.append("instruction exposes a runtime recovery error code")
    if exposed_tools:
        errors.append("instruction names the recovery action")
    if procedural:
        errors.append("instruction prescribes a conditional recovery procedure")
    return {
        "valid": not errors,
        "errors": errors,
        "exposed_error_codes": exposed_codes,
        "exposed_recovery_tools": exposed_tools,
        "procedural_recovery_language": procedural,
    }


def alternative_recovery_metrics(counterfactual: dict[str, Any]) -> dict[str, Any]:
    """Prove that one failure has two successful public recovery strategies."""
    by_error: dict[str, dict[str, Any]] = {}

    def add(source: str, strategies: list[dict[str, Any]]) -> None:
        for item in strategies:
            code = str(item.get("error_code", ""))
            tool = str(item.get("recovery_tool", ""))
            if not code or not tool:
                continue
            group = by_error.setdefault(
                code,
                {
                    "error_code": code,
                    "tools": set(),
                    "mechanisms": set(),
                    "witnesses": [],
                },
            )
            capability = str(item.get("capability_id", ""))
            branch = str(item.get("selected_branch", ""))
            writes = tuple(sorted(str(path) for path in item.get("write_set", [])))
            mechanism = (capability, branch, writes) if capability else (tool, "", ())
            group["tools"].add(tool)
            group["mechanisms"].add(mechanism)
            group["witnesses"].append(
                {
                    "source": source,
                    "recovery_tool": tool,
                    "capability_id": capability or None,
                    "selected_branch": branch or None,
                    "write_set": list(writes),
                }
            )

    add("baseline", counterfactual.get("baseline_recovery_strategies", []))
    for variant in counterfactual.get("variants", []):
        if variant.get("valid") and variant.get("adapted_valid"):
            add(str(variant.get("id", "variant")), variant.get("recovery_strategies", []))

    failures = []
    for code, group in sorted(by_error.items()):
        tools = sorted(group.pop("tools"))
        mechanisms = sorted(group.pop("mechanisms"), key=repr)
        failures.append(
            {
                **group,
                "distinct_recovery_tools": tools,
                "strategy_count": len(tools),
                "mechanism_count": len(mechanisms),
                "valid": len(tools) >= 2 and len(mechanisms) >= 2,
            }
        )
    return {
        "valid": any(item["valid"] for item in failures),
        "failures": failures,
        "alternative_failure_count": sum(item["valid"] for item in failures),
    }


def validate_vnext_adaptive_profile(
    bundle: TaskBundle,
    episode: dict[str, Any],
    causal: dict[str, Any],
    counterfactual: dict[str, Any],
    *,
    ablation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the concrete vNext target requested for training admission."""
    ablation = ablation or evaluate_action_ablation(bundle)
    metrics = causal.get("metrics", {})
    decisions = counterfactual.get("decision_metrics", {})
    steps = len(episode.get("trace", []))
    necessary = int(ablation.get("necessary_actions", 0))
    decision_nodes = int(decisions.get("meaningful_planning_decision_count", 0))
    recoveries = len(metrics.get("semantic_recoveries", []))
    route_hiding = validate_instruction_route_hiding(bundle, counterfactual)
    oracle_resistance = validate_tool_oracle_resistance(bundle)
    alternative_recovery = alternative_recovery_metrics(counterfactual)
    errors = []
    if not 15 <= steps <= 25:
        errors.append("tool-call horizon must be between 15 and 25")
    if not 15 <= necessary <= 25:
        errors.append("necessary tool-call count must be between 15 and 25")
    if float(ablation.get("necessary_action_ratio", 0.0)) < 0.75:
        errors.append("necessary action ratio must be at least 0.75")
    if not 3 <= decision_nodes <= 5:
        errors.append("grounded decision-node count must be between 3 and 5")
    if not 1 <= recoveries <= 2:
        errors.append("semantic failure/recovery count must be between 1 and 2")
    if not alternative_recovery["valid"]:
        errors.append("no failure has two validated alternative recovery paths")
    if not counterfactual.get("valid"):
        errors.append("all declared counterfactual policies must be valid")
    if metrics.get("missing_provenance"):
        errors.append("strict argument provenance failed")
    if metrics.get("invariant_violations"):
        errors.append("an execution invariant was violated")
    if float(metrics.get("goal_evidence_coverage", 0.0)) < 1.0:
        errors.append("final observation does not cover every goal predicate")
    if metrics.get("final_goal_observation_step") is None:
        errors.append("no read-only final goal observation was found")
    if not route_hiding["valid"]:
        errors.extend(route_hiding["errors"])
    if not oracle_resistance["valid"]:
        errors.append("public tool surface contains solution-role oracle names")
    return {
        "valid": not errors,
        "errors": errors,
        "targets": {
            "tool_calls": [15, 25],
            "necessary_tool_calls": [15, 25],
            "decision_nodes": [3, 5],
            "semantic_failures": [1, 2],
            "necessary_action_ratio_min": 0.75,
        },
        "metrics": {
            "tool_calls": steps,
            "necessary_tool_calls": necessary,
            "necessary_action_ratio": ablation.get("necessary_action_ratio", 0.0),
            "grounded_decision_nodes": decision_nodes,
            "semantic_failures": recoveries,
            "strict_provenance": not bool(metrics.get("missing_provenance")),
            "final_state_verified": metrics.get("final_goal_observation_step") is not None
            and float(metrics.get("goal_evidence_coverage", 0.0)) == 1.0,
        },
        "alternative_recovery": alternative_recovery,
        "route_hiding": route_hiding,
        "tool_oracle_resistance": oracle_resistance,
    }


__all__ = [
    "alternative_recovery_metrics",
    "validate_instruction_route_hiding",
    "validate_tool_oracle_resistance",
    "validate_vnext_adaptive_profile",
]
