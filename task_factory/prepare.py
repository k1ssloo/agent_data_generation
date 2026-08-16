"""Prepare a validated generated bundle for recursive task evolution."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

from causal_validation.intervention import evaluate_counterfactuals
from rollout import run_reference_plan

from .bundle import TaskBundle
from .hooks import attach_inferred_evolution_hooks


def admit_valid_counterfactuals(
    bundle: TaskBundle,
    *,
    evaluation: dict[str, Any] | None = None,
) -> tuple[TaskBundle, dict[str, Any]]:
    """Retain only counterfactual witnesses that pass the full adaptation gate."""
    source = bundle.reference_plan.get("counterfactuals", [])
    accepted: list[dict[str, Any]] = []
    audits = []
    evaluated_by_id = {
        str(item.get("id")): item
        for item in (evaluation or {}).get("variants", [])
        if isinstance(item, dict)
    }
    for index, variant in enumerate(source):
        variant_id = str(variant.get("id") or variant.get("name") or f"variant_{index}")
        result = evaluated_by_id.get(variant_id)
        if result is None:
            reference_plan = copy.deepcopy(bundle.reference_plan)
            reference_plan["counterfactuals"] = [copy.deepcopy(variant)]
            candidate = replace(bundle, reference_plan=reference_plan)
            variant_evaluation = evaluate_counterfactuals(candidate)
            result = variant_evaluation["variants"][0]
        audits.append({"id": variant_id, **result})
        if result["valid"]:
            accepted.append(copy.deepcopy(variant))
    reference_plan = copy.deepcopy(bundle.reference_plan)
    reference_plan["counterfactuals"] = accepted
    return replace(bundle, reference_plan=reference_plan), {
        "input_count": len(source),
        "accepted_count": len(accepted),
        "rejected_count": len(source) - len(accepted),
        "variants": audits,
    }


def prepare_recursive_parent(
    bundle: TaskBundle,
    *,
    counterfactual_evaluation: dict[str, Any] | None = None,
    episode_report: dict[str, Any] | None = None,
) -> tuple[TaskBundle, dict[str, Any]]:
    admitted, counterfactual_audit = admit_valid_counterfactuals(
        bundle, evaluation=counterfactual_evaluation
    )
    semantic_plan_unchanged = (
        counterfactual_audit["accepted_count"] == counterfactual_audit["input_count"]
    )
    report = (
        episode_report
        if semantic_plan_unchanged and episode_report is not None
        else run_reference_plan(admitted)
    )
    prepared = attach_inferred_evolution_hooks(admitted, report)
    return prepared, {
        "counterfactual_admission": counterfactual_audit,
        "evolution_hooks": copy.deepcopy(prepared.manifest.get("evolution_hooks", {})),
        "reused_execution_evidence": bool(
            semantic_plan_unchanged and episode_report is not None
        ),
    }


__all__ = ["admit_valid_counterfactuals", "prepare_recursive_parent"]
