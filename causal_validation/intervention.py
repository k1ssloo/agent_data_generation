"""Validate counterfactual state variants and policy adaptation."""

from __future__ import annotations

import copy
from dataclasses import replace
import math
from typing import Any

from rollout import run_reference_plan
from runtime.executor import CausalRuntime
from task_factory.bundle import TaskBundle

from .validator import validate_episode


def bundle_with_state_overrides(
    bundle: TaskBundle, overrides: dict[str, Any]
) -> TaskBundle:
    environment = copy.deepcopy(bundle.environment)
    for path, value in overrides.items():
        if not path.startswith("$state."):
            raise ValueError(f"counterfactual override must target $state: {path!r}")
        current = environment["initial_state"]
        parts = path[len("$state.") :].split(".")
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                raise ValueError(f"counterfactual path does not exist: {path!r}")
            current = current[part]
        if parts[-1] not in current:
            raise ValueError(f"counterfactual path does not exist: {path!r}")
        current[parts[-1]] = copy.deepcopy(value)
    return replace(bundle, environment=environment)


def _action_signature(actions: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [
        (
            str(action.get("tool")),
            repr(sorted(action.get("arguments", {}).items())),
        )
        for action in actions
    ]


def _recovery_strategies(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the public action that first resolves each observed failure."""
    failures: dict[str, int] = {}
    strategies = []
    for step in report.get("trace", []):
        error = step.get("error_code")
        if isinstance(error, str) and error:
            failures[error] = int(step.get("step", 0))
        for resolved in step.get("resolves_errors", []):
            if resolved not in failures:
                continue
            strategies.append(
                {
                    "error_code": resolved,
                    "failure_step": failures.pop(resolved),
                    "recovery_step": int(step.get("step", 0)),
                    "recovery_tool": str(step.get("public_tool", "")),
                    "capability_id": str(step.get("capability_id", "")),
                    "selected_branch": str(step.get("selected_branch", "")),
                    "write_set": sorted(str(path) for path in step.get("write_set", [])),
                }
            )
    return strategies


def _first_strategy_divergence(
    baseline: list[dict[str, Any]], adapted: list[dict[str, Any]]
) -> int:
    baseline_signature = _action_signature(baseline)
    adapted_signature = _action_signature(adapted)
    limit = min(len(baseline_signature), len(adapted_signature))
    for index in range(limit):
        if baseline_signature[index] != adapted_signature[index]:
            return index
    return limit


def _stale_policy_timing(
    bundle: TaskBundle,
    actions: list[dict[str, Any]],
    *,
    divergence: int,
) -> dict[str, Any]:
    """Locate goal satisfaction and the first post-decision stale-policy failure."""
    runtime = CausalRuntime(bundle)
    first_goal_step: int | None = None
    first_failure_step: int | None = None
    failure_reason: str | None = None
    for step, action in enumerate(actions, start=1):
        try:
            result = runtime.execute(action["tool"], action.get("arguments", {}))
        except Exception as exc:  # Runtime/schema failures are stale-policy failures.
            if step > divergence and first_failure_step is None:
                first_failure_step = step
                failure_reason = str(exc)
            break

        if (
            step > divergence
            and result.trace.get("error_code")
            and first_failure_step is None
        ):
            first_failure_step = step
            failure_reason = str(result.trace["error_code"])
        invalid_invariants = [
            item for item in runtime.evaluate_invariants() if not item.get("valid")
        ]
        if step > divergence and invalid_invariants and first_failure_step is None:
            first_failure_step = step
            failure_reason = "invariant_violation"

        goals = runtime.evaluate_goals()
        if (
            first_goal_step is None
            and goals
            and all(item.get("valid") for item in goals)
        ):
            first_goal_step = step

    # A recoverable error is not a rejection witness if the unchanged policy
    # later reaches the goal anyway. Any earlier goal satisfaction also means
    # the intervention did not force adaptation.
    failed_before_goal = first_failure_step is not None and first_goal_step is None
    return {
        "first_goal_satisfaction_step": first_goal_step,
        "first_post_divergence_failure_step": first_failure_step,
        "first_post_divergence_failure_reason": failure_reason,
        "failed_before_goal_satisfaction": failed_before_goal,
        "goal_reached_before_failure": first_goal_step is not None
        and (first_failure_step is None or first_goal_step <= first_failure_step),
    }


def _path_covered(path: str, observed: set[str]) -> bool:
    return any(path.startswith(item) or item.startswith(path) for item in observed)


def _decision_grounding(
    bundle: TaskBundle,
    *,
    variant: dict[str, Any],
    baseline_report: dict[str, Any],
    adapted_report: dict[str, Any],
) -> dict[str, Any]:
    adapted_actions = variant.get("actions", [])
    divergence = _first_strategy_divergence(
        bundle.reference_plan["actions"], adapted_actions
    )
    axis_paths = {
        axis.get("state_path")
        for axis in bundle.contract.get("counterfactual_axes", [])
        if isinstance(axis, dict) and isinstance(axis.get("state_path"), str)
    }
    changed_axes = sorted(set(variant.get("state_overrides", {})) & axis_paths)

    def observed_before(report: dict[str, Any]) -> set[str]:
        return {
            path
            for step in report.get("trace", [])[:divergence]
            for path in step.get("observed_state_paths", [])
        }

    baseline_observed = observed_before(baseline_report)
    adapted_observed = observed_before(adapted_report)
    missing_baseline = [
        path for path in changed_axes if not _path_covered(path, baseline_observed)
    ]
    missing_adapted = [
        path for path in changed_axes if not _path_covered(path, adapted_observed)
    ]
    return {
        "valid": bool(changed_axes) and not missing_baseline and not missing_adapted,
        "first_strategy_divergence": divergence,
        "changed_axes": changed_axes,
        "baseline_observed_state_paths": sorted(baseline_observed),
        "adapted_observed_state_paths": sorted(adapted_observed),
        "missing_baseline_axes": missing_baseline,
        "missing_adapted_axes": missing_adapted,
    }


def evaluate_counterfactuals(bundle: TaskBundle) -> dict[str, Any]:
    baseline_actions = bundle.reference_plan["actions"]
    baseline_report = run_reference_plan(bundle, actions=baseline_actions)
    baseline_recovery_strategies = _recovery_strategies(baseline_report)
    variants = []
    for variant in bundle.reference_plan.get("counterfactuals", []):
        variant_id = variant.get("id") or variant.get("name") or "variant"
        try:
            variant_bundle = bundle_with_state_overrides(
                bundle, variant.get("state_overrides", {})
            )
        except (KeyError, TypeError, ValueError) as exc:
            variants.append(
                {
                    "id": variant_id,
                    "valid": False,
                    "strategy_changed": False,
                    "adapted_valid": False,
                    "stale_strategy_valid": False,
                    "adapted_errors": [str(exc)],
                    "stale_strategy_errors": ["state intervention could not be applied"],
                }
            )
            continue
        initial_invariants = CausalRuntime(variant_bundle).evaluate_invariants()
        violated_initial_invariants = [
            item for item in initial_invariants if not item.get("valid")
        ]
        if violated_initial_invariants:
            invariant_ids = [
                str(item.get("id", "invariant"))
                for item in violated_initial_invariants
            ]
            variants.append(
                {
                    "id": variant_id,
                    "valid": False,
                    "strategy_changed": False,
                    "adapted_valid": False,
                    "stale_strategy_valid": False,
                    "adapted_errors": [
                        "counterfactual initial state violates invariants: "
                        + ", ".join(invariant_ids)
                    ],
                    "stale_strategy_errors": [
                        "counterfactual intervention is internally inconsistent"
                    ],
                    "initial_invariant_violations": violated_initial_invariants,
                }
            )
            continue
        adapted_actions = variant.get("actions", [])
        adapted_report = run_reference_plan(variant_bundle, actions=adapted_actions)
        adapted_validation = validate_episode(
            variant_bundle,
            adapted_report,
            require_semantic_recovery=False,
        )
        stale_report = run_reference_plan(variant_bundle, actions=baseline_actions)
        stale_validation = validate_episode(
            variant_bundle,
            stale_report,
            min_delayed_handle_distance=0,
            min_handle_chain_depth=0,
            require_semantic_recovery=False,
        )
        strategy_changed = _action_signature(adapted_actions) != _action_signature(baseline_actions)
        decision_grounding = _decision_grounding(
            bundle,
            variant=variant,
            baseline_report=baseline_report,
            adapted_report=adapted_report,
        )
        stale_policy_timing = _stale_policy_timing(
            variant_bundle,
            baseline_actions,
            divergence=int(decision_grounding["first_strategy_divergence"]),
        )
        valid = (
            adapted_validation["valid"]
            and stale_policy_timing["failed_before_goal_satisfaction"]
            and strategy_changed
            and decision_grounding["valid"]
        )
        variants.append(
            {
                "id": variant_id,
                "valid": valid,
                "strategy_changed": strategy_changed,
                "adapted_valid": adapted_validation["valid"],
                "stale_strategy_valid": not stale_policy_timing[
                    "failed_before_goal_satisfaction"
                ],
                "adapted_errors": adapted_validation["errors"],
                "stale_strategy_errors": stale_validation["errors"],
                "stale_policy_timing": stale_policy_timing,
                "decision_grounding": decision_grounding,
                "recovery_strategies": _recovery_strategies(adapted_report),
            }
        )
    result = {
        "task_id": bundle.task_id,
        "counterfactual_count": len(variants),
        "valid": bool(variants) and all(item["valid"] for item in variants),
        "variants": variants,
        "baseline_recovery_strategies": baseline_recovery_strategies,
    }
    result["decision_metrics"] = counterfactual_decision_metrics(bundle, result)
    return result


def counterfactual_decision_metrics(
    bundle: TaskBundle, evaluation: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Measure policy alternatives, not deterministic state transitions.

    A pending-to-ready poll changes runtime branches but contributes no strategy
    entropy by itself.  Entropy is credited only when a valid state intervention
    forces a grounded divergence in the public action policy.
    """
    evaluation = evaluation or evaluate_counterfactuals(bundle)
    variants_by_id = {
        str(item.get("id") or item.get("name") or "variant"): item
        for item in bundle.reference_plan.get("counterfactuals", [])
    }
    baseline = bundle.reference_plan.get("actions", [])
    groups: dict[tuple[int, tuple[str, ...]], dict[str, Any]] = {}
    for audit in evaluation.get("variants", []):
        grounding = audit.get("decision_grounding", {})
        if not audit.get("valid") or not grounding.get("valid"):
            continue
        variant = variants_by_id.get(str(audit.get("id")))
        if not isinstance(variant, dict):
            continue
        divergence = int(grounding["first_strategy_divergence"])
        axes = tuple(sorted(grounding.get("changed_axes", [])))
        key = (divergence, axes)
        group = groups.setdefault(
            key,
            {
                "first_strategy_divergence": divergence,
                "changed_axes": list(axes),
                "alternatives": set(),
                "variant_ids": [],
            },
        )

        def decision_at(actions: list[dict[str, Any]]) -> tuple[str, str]:
            if divergence >= len(actions):
                return ("<end>", "")
            action = actions[divergence]
            return (
                str(action.get("tool")),
                repr(sorted(action.get("arguments", {}).items())),
            )

        group["alternatives"].add(decision_at(baseline))
        group["alternatives"].add(decision_at(variant.get("actions", [])))
        group["variant_ids"].append(audit.get("id"))

    decisions = []
    entropy = 0.0
    observation_dependent_count = 0
    for (_key, group) in sorted(groups.items()):
        alternatives = sorted(group.pop("alternatives"))
        alternative_count = len(alternatives)
        distinct_tools = {tool for tool, _arguments in alternatives}
        decision_type = (
            "policy_branch"
            if alternative_count > 1 and len(distinct_tools) > 1
            else "argument_binding"
        )
        if alternative_count > 1:
            observation_dependent_count += 1
        bits = (
            math.log2(alternative_count)
            if decision_type == "policy_branch"
            else 0.0
        )
        entropy += bits
        decisions.append(
            {
                **group,
                "alternative_count": alternative_count,
                "decision_type": decision_type,
                "counts_as_meaningful_planning": decision_type == "policy_branch",
                "entropy_bits": round(bits, 4),
                "alternatives": [
                    {"tool": tool, "arguments": arguments}
                    for tool, arguments in alternatives
                ],
            }
        )
    return {
        "meaningful_planning_decision_count": sum(
            item["counts_as_meaningful_planning"] for item in decisions
        ),
        "observation_dependent_decision_count": observation_dependent_count,
        "decision_entropy_bits": round(entropy, 4),
        "decisions": decisions,
    }
