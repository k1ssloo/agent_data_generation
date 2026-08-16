"""Validation and normalization for semantic task-evolution patches."""

from __future__ import annotations

from typing import Any


def normalize_contract_patch(
    patch: dict[str, Any],
    *,
    parent_task_id: str,
    child_task_id: str,
    generation: int,
) -> dict[str, Any]:
    operator_id = patch.get("operator_id")
    semantic_changes = patch.get("semantic_changes")
    added_goal_paths = patch.get("added_goal_paths", [])
    if not isinstance(operator_id, str) or not operator_id:
        raise ValueError("contract patch requires operator_id")
    if (
        not isinstance(semantic_changes, list)
        or not semantic_changes
        or any(not isinstance(item, str) or not item for item in semantic_changes)
    ):
        raise ValueError("contract patch requires non-empty semantic_changes")
    if not isinstance(added_goal_paths, list) or any(
        not isinstance(item, str) or not item.startswith("$state.") for item in added_goal_paths
    ):
        raise ValueError("contract patch added_goal_paths must target $state")
    return {
        "contract_patch_version": "contract-patch-v1",
        "parent_task_id": parent_task_id,
        "child_task_id": child_task_id,
        "generation": generation,
        "operator_id": operator_id,
        "semantic_changes": semantic_changes,
        "added_goal_paths": added_goal_paths,
    }
