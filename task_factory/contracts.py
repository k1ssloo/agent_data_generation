"""Static checks for generated task transformation contracts."""

from __future__ import annotations

from typing import Any

from runtime.predicates import (
    EvaluationError,
    evaluate_predicate,
    predicate_paths,
    validate_predicate_syntax,
)


CONTRACT_FIELDS = frozenset(
    {
        "contract_version",
        "goal",
        "preserved_requirements",
        "new_requirements",
        "discoverable_evidence",
        "goal_predicates",
        "invariants",
        "forbidden_shortcuts",
        "counterfactual_axes",
        "expected_reasoning_features",
        "selected_operator",
        "rationale",
        "requirements",
        "instruction_claims",
        "goal_clauses",
    }
)


def normalize_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Drop executable bundle payloads accidentally copied into a contract."""
    return {key: value for key, value in contract.items() if key in CONTRACT_FIELDS}


ALLOWED_REASONING_FEATURES = {
    "delayed_handle_use",
    "observation_dependent_decision",
    "derived_object_dependency",
    "async_decision",
    "semantic_recovery",
    "goal_grounded_verification",
    "alternative_plan",
    "closed_loop_control",
    "temporal_provenance",
}


def validate_contract(contract: dict[str, Any], initial_state: dict[str, Any] | None = None) -> list[str]:
    errors: list[str] = []
    unexpected = sorted(set(contract) - CONTRACT_FIELDS)
    if unexpected:
        errors.append(f"unexpected contract fields: {unexpected}")
    if contract.get("contract_version") != "task-contract-v1":
        errors.append("contract_version must be 'task-contract-v1'")
    if not isinstance(contract.get("goal"), str) or not contract.get("goal", "").strip():
        errors.append("goal must be a non-empty string")
    goals = contract.get("goal_predicates")
    if not isinstance(goals, list) or not goals:
        errors.append("goal_predicates must be a non-empty list")
        goals = []
    invariants = contract.get("invariants", [])
    if not isinstance(invariants, list):
        errors.append("invariants must be a list")
        invariants = []
    seen_ids: set[str] = set()
    for kind, items in (("goal", goals), ("invariant", invariants)):
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append(f"{kind}[{index}] must be an object")
                continue
            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id:
                errors.append(f"{kind}[{index}].id must be a non-empty string")
            elif item_id in seen_ids:
                errors.append(f"duplicate predicate id {item_id!r}")
            else:
                seen_ids.add(item_id)
            predicate = item.get("predicate")
            errors.extend(
                f"{kind}[{index}] {error}"
                for error in validate_predicate_syntax(predicate)
            )
            paths = predicate_paths(predicate)
            if not paths:
                errors.append(f"{kind}[{index}] does not reference observable state")
            if initial_state is not None and not validate_predicate_syntax(predicate):
                try:
                    initial_value = evaluate_predicate(
                        predicate, {"state": initial_state, "args": {}, "response": {}}
                    )
                    if kind == "invariant" and not initial_value:
                        errors.append(
                            f"{kind}[{index}] must hold on the initial state"
                        )
                except EvaluationError as exc:
                    # A goal may be initially false, but every referenced path must be resolvable.
                    errors.append(f"{kind}[{index}] cannot be evaluated on initial state: {exc}")
    shortcuts = contract.get("forbidden_shortcuts")
    if not isinstance(shortcuts, list) or not shortcuts or any(not isinstance(item, str) for item in shortcuts):
        errors.append("forbidden_shortcuts must be a non-empty string list")
    features = contract.get("expected_reasoning_features", [])
    if not isinstance(features, list):
        errors.append("expected_reasoning_features must be a list")
    else:
        unknown = sorted(set(features) - ALLOWED_REASONING_FEATURES)
        if unknown:
            errors.append(f"unknown expected_reasoning_features: {unknown}")
        if not ({"delayed_handle_use", "observation_dependent_decision", "derived_object_dependency"} & set(features)):
            errors.append("contract must request at least one causal dependency feature")
    axes = contract.get("counterfactual_axes", [])
    if not isinstance(axes, list):
        errors.append("counterfactual_axes must be a list")
    else:
        for index, axis in enumerate(axes):
            if not isinstance(axis, dict) or not isinstance(axis.get("state_path"), str) or not axis["state_path"].startswith("$state"):
                errors.append(f"counterfactual_axes[{index}] must define a $state path")
            if not isinstance(axis, dict) or not isinstance(axis.get("variants"), list) or len(axis.get("variants", [])) < 2:
                errors.append(f"counterfactual_axes[{index}] must define at least two variants")
    claims = contract.get("instruction_claims")
    clauses = contract.get("goal_clauses")
    if (claims is None) != (clauses is None):
        errors.append("instruction_claims and goal_clauses must be declared together")
    if claims is not None and not isinstance(claims, list):
        errors.append("instruction_claims must be a list")
    if clauses is not None and (not isinstance(clauses, list) or not clauses):
        errors.append("goal_clauses must be a non-empty list")
    return errors


__all__ = ["CONTRACT_FIELDS", "normalize_contract", "validate_contract"]
