"""Evaluate whether reference actions are necessary under causal validation."""

from __future__ import annotations

from typing import Any

from rollout.episode_runner import run_reference_plan
from task_factory.bundle import TaskBundle
from .validator import validate_episode


def evaluate_action_ablation(bundle: TaskBundle) -> dict[str, Any]:
    actions = bundle.reference_plan["actions"]
    items = []
    necessary = 0
    for index, action in enumerate(actions):
        ablated = actions[:index] + actions[index + 1 :]
        report = run_reference_plan(bundle, actions=ablated)
        validation = validate_episode(
            bundle,
            report,
            min_delayed_handle_distance=0,
            min_handle_chain_depth=0,
            require_semantic_recovery=False,
        )
        is_necessary = not validation["valid"]
        necessary += int(is_necessary)
        items.append(
            {
                "removed_index": index,
                "removed_tool": action["tool"],
                "necessary": is_necessary,
                "episode_status": report["status"],
                "validation_errors": validation["errors"],
            }
        )
    return {
        "task_id": bundle.task_id,
        "actions": len(actions),
        "necessary_actions": necessary,
        "necessary_action_ratio": round(necessary / len(actions), 4) if actions else 0.0,
        "items": items,
    }


def minimize_action_plan(bundle: TaskBundle) -> dict[str, Any]:
    """Find a deterministic plan irreducible under contiguous-range deletion.

    Single-action ablation can miss a redundant dependency chain: deleting its
    producer alone breaks its consumer, and deleting its consumer alone can
    leave an otherwise unnecessary producer. Workflow chains are normally
    contiguous, so delete the longest passing range and repeat. This is a
    diagnostic local minimum, not a claim of global minimum cardinality.
    """
    indexed = list(enumerate(bundle.reference_plan["actions"]))
    removals = []
    while len(indexed) > 1:
        removed = False
        for length in range(len(indexed) - 1, 0, -1):
            for start in range(0, len(indexed) - length + 1):
                candidate = indexed[:start] + indexed[start + length :]
                report = run_reference_plan(
                    bundle, actions=[action for _index, action in candidate]
                )
                validation = validate_episode(
                    bundle,
                    report,
                    min_delayed_handle_distance=0,
                    min_handle_chain_depth=0,
                    require_semantic_recovery=False,
                )
                if not validation["valid"]:
                    continue
                deleted = indexed[start : start + length]
                removals.append(
                    {
                        "original_indices": [index for index, _action in deleted],
                        "tools": [action["tool"] for _index, action in deleted],
                    }
                )
                indexed = candidate
                removed = True
                break
            if removed:
                break
        if not removed:
            break
    retained_indices = [index for index, _action in indexed]
    removed_indices = sorted(
        set(range(len(bundle.reference_plan["actions"]))) - set(retained_indices)
    )
    return {
        "task_id": bundle.task_id,
        "original_actions": len(bundle.reference_plan["actions"]),
        "irreducible_actions": len(indexed),
        "irreducible_action_ratio": round(
            len(indexed) / len(bundle.reference_plan["actions"]), 4
        )
        if bundle.reference_plan["actions"]
        else 0.0,
        "retained_indices": retained_indices,
        "removed_indices": removed_indices,
        "removals": removals,
    }
