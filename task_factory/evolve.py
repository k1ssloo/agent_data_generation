"""Apply semantic operators and reject children without proven difficulty gain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from causal_validation import evaluate_action_ablation, validate_episode
from causal_validation.intervention import evaluate_counterfactuals
from rollout import run_reference_plan
from runtime.predicates import predicate_paths

from .bundle import TaskBundle
from .operators import EvolutionProduct, get_operator
from .patches import normalize_contract_patch


def _leaf_count(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_leaf_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(_leaf_count(item) for item in value)
    return 1


def _goal_paths(bundle: TaskBundle) -> set[str]:
    paths: set[str] = set()
    for goal in bundle.contract.get("goal_predicates", []):
        paths |= predicate_paths(goal.get("predicate", goal))
    return paths


def complexity_profile(bundle: TaskBundle, report: dict[str, Any]) -> dict[str, int]:
    validation = validate_episode(
        bundle,
        report,
        min_delayed_handle_distance=0,
        min_handle_chain_depth=0,
        require_semantic_recovery=False,
    )
    metrics = validation["metrics"]
    return {
        "steps": len(report.get("trace", [])),
        "state_leaves": _leaf_count(bundle.environment.get("initial_state", {})),
        "capabilities": len(bundle.environment.get("capabilities", {})),
        "public_tools": len(bundle.tools),
        "goal_paths": len(_goal_paths(bundle)),
        "handle_chain_depth": int(metrics["handle_chain_depth"]),
        "observation_dependent_branches": int(
            metrics["observation_dependent_branch_count"]
        ),
        "counterfactual_variants": len(bundle.reference_plan.get("counterfactuals", [])),
    }


@dataclass(frozen=True)
class EvolutionEvaluation:
    product: EvolutionProduct
    report: dict[str, Any]


def evolve_once(
    parent: TaskBundle,
    operator_id: str,
    *,
    objective: str = "semantic",
) -> EvolutionEvaluation:
    if objective not in {"semantic", "decision_nodes"}:
        raise ValueError(f"unknown evolution objective: {objective!r}")
    generation = int(parent.manifest.get("lineage", {}).get("generation", 0)) + 1
    product = get_operator(operator_id).apply(parent, generation=generation)
    child = product.bundle
    normalized_patch = normalize_contract_patch(
        product.patch,
        parent_task_id=parent.task_id,
        child_task_id=child.task_id,
        generation=generation,
    )
    product = EvolutionProduct(bundle=child, patch=normalized_patch)

    parent_report = run_reference_plan(parent)
    child_report = run_reference_plan(child)
    child_validation = validate_episode(child, child_report)
    stale_report = run_reference_plan(child, actions=parent.reference_plan["actions"])
    stale_validation = validate_episode(
        child,
        stale_report,
        min_delayed_handle_distance=0,
        min_handle_chain_depth=0,
        require_semantic_recovery=False,
    )
    parent_profile = complexity_profile(parent, parent_report)
    child_profile = complexity_profile(child, child_report)
    deltas = {
        key: child_profile[key] - parent_profile[key]
        for key in parent_profile
    }
    semantic_keys = {
        "state_leaves",
        "capabilities",
        "goal_paths",
        "observation_dependent_branches",
        "counterfactual_variants",
    }
    semantic_gain = any(deltas[key] > 0 for key in semantic_keys)
    no_semantic_regression = all(deltas[key] >= 0 for key in semantic_keys)
    parent_counterfactual = evaluate_counterfactuals(parent)
    counterfactual = evaluate_counterfactuals(child)
    parent_decisions = int(
        parent_counterfactual["decision_metrics"][
            "meaningful_planning_decision_count"
        ]
    )
    child_decisions = int(
        counterfactual["decision_metrics"][
            "meaningful_planning_decision_count"
        ]
    )
    decision_node_delta = child_decisions - parent_decisions
    requires_counterfactual = bool(child.reference_plan.get("counterfactuals"))
    counterfactual_valid = counterfactual["valid"] if requires_counterfactual else True
    ablation = evaluate_action_ablation(child)
    necessary_action_gate = ablation["necessary_action_ratio"] >= 0.6
    errors = []
    if not child_validation["valid"]:
        errors.append("child oracle failed causal validation")
    if stale_validation["valid"]:
        errors.append("parent plan still solves the child task unchanged")
    if objective == "semantic" and not semantic_gain:
        errors.append("child did not increase a semantic complexity dimension")
    if objective == "decision_nodes" and decision_node_delta <= 0:
        errors.append("child did not add a grounded decision node")
    if not no_semantic_regression:
        errors.append("child regressed a semantic complexity dimension")
    if not counterfactual_valid:
        errors.append("counterfactual strategy adaptation failed")
    if not necessary_action_gate:
        errors.append("necessary action ratio below 0.6")
    report = {
        "parent_task_id": parent.task_id,
        "child_task_id": child.task_id,
        "operator_id": operator_id,
        "evolution_objective": objective,
        "accepted": not errors,
        "errors": errors,
        "patch": product.patch,
        "parent_profile": parent_profile,
        "child_profile": child_profile,
        "complexity_delta": deltas,
        "parent_decision_node_count": parent_decisions,
        "child_decision_node_count": child_decisions,
        "decision_node_delta": decision_node_delta,
        "child_validation": child_validation,
        "parent_plan_valid_on_child": stale_validation["valid"],
        "parent_plan_errors_on_child": stale_validation["errors"],
        "counterfactual_validation": counterfactual,
        "decision_metrics": counterfactual["decision_metrics"],
        "counterfactual_required": requires_counterfactual,
        "counterfactual_gate_passed": counterfactual_valid,
        "action_ablation": ablation,
        "necessary_action_gate_passed": necessary_action_gate,
    }
    return EvolutionEvaluation(product=product, report=report)
