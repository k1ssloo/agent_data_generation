"""Infer portable recursive-evolution hooks from validated runtime provenance."""

from __future__ import annotations

import copy
from dataclasses import replace
import re
from typing import Any

from runtime.predicates import predicate_paths

from .bundle import TaskBundle


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "workflow_commit"


def _goal_paths(bundle: TaskBundle) -> set[str]:
    return {
        path
        for item in bundle.contract.get("goal_predicates", [])
        for path in predicate_paths(item.get("predicate", item))
    }


def _covers_goal_paths(reads: list[str], goals: set[str]) -> bool:
    return bool(goals) and all(
        any(goal.startswith(read) or read.startswith(goal) for read in reads)
        for goal in goals
    )


def _missing_goal_paths(reads: list[str], goals: set[str]) -> list[str]:
    return sorted(
        goal
        for goal in goals
        if not any(goal.startswith(read) or read.startswith(goal) for read in reads)
    )


def infer_audit_checkpoint_hook(
    bundle: TaskBundle, report: dict[str, Any]
) -> dict[str, Any]:
    """Identify a consequential commit and final observation without domain rules."""
    trace = report.get("trace", [])
    if report.get("status") != "goal_satisfied" or not trace:
        raise ValueError("evolution hooks require a goal-satisfying execution trace")
    goals = _goal_paths(bundle)
    last_goal_write_step = max(
        (
            int(step.get("step", 0))
            for step in trace
            if any(
                goal.startswith(write) or write.startswith(goal)
                for goal in goals
                for write in step.get("write_set", [])
            )
        ),
        default=0,
    )
    verify_candidates = [
        step
        for step in trace
        if isinstance(step.get("selected_branch"), str)
        and not step.get("write_set")
        and int(step.get("step", 0)) > last_goal_write_step
        and _covers_goal_paths(step.get("read_set", []), goals)
    ]
    if not verify_candidates:
        observable = [
            step
            for step in trace
            if isinstance(step.get("selected_branch"), str)
            and step.get("read_set")
            and not step.get("error_code")
        ]
        closest = min(
            observable,
            key=lambda step: (
                len(_missing_goal_paths(step.get("read_set", []), goals)),
                -int(step.get("step", 0)),
            ),
            default=None,
        )
        detail = ""
        if closest is not None:
            detail = (
                f"; closest step {closest.get('step')} "
                f"{closest.get('public_tool')!r} is missing "
                f"{_missing_goal_paths(closest.get('read_set', []), goals)}"
            )
        raise ValueError("no final domain observation covers every goal path" + detail)
    verify = verify_candidates[-1]
    def sourced_target_arguments(step: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            detail
            for detail in step.get("arguments", {}).values()
            if isinstance(detail, dict)
            and detail.get("required")
            and isinstance(detail.get("value"), str)
            and isinstance(detail.get("source"), dict)
        ]

    commit_candidates = [
        step
        for step in trace
        if step.get("step", 0) < verify.get("step", 0)
        and step.get("write_set")
        and not step.get("error_code")
        and isinstance(step.get("selected_branch"), str)
        and sourced_target_arguments(step)
        and any(
            goal.startswith(write) or write.startswith(goal)
            for goal in goals
            for write in step.get("write_set", [])
        )
    ]
    if not commit_candidates:
        raise ValueError("no successful pre-verification action writes goal state")
    tool_counts: dict[str, int] = {}
    for step in trace:
        tool = step.get("public_tool")
        if isinstance(tool, str):
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

    def commit_score(step: dict[str, Any]) -> tuple[int, int, int, int]:
        writes = step.get("write_set", [])
        covered = sum(
            1
            for goal in goals
            if any(goal.startswith(write) or write.startswith(goal) for write in writes)
        )
        unique_tool = int(tool_counts.get(str(step.get("public_tool")), 0) == 1)
        return unique_tool, covered, len(writes), int(step.get("step", 0))

    commit = max(commit_candidates, key=commit_score)
    sourced_arguments = sourced_target_arguments(commit)
    target = min(
        sourced_arguments,
        key=lambda item: int(item["source"].get("step", commit["step"])),
    )["value"]
    prior_same_tool = any(
        step.get("public_tool") == commit.get("public_tool")
        and step.get("step", 0) < commit.get("step", 0)
        for step in trace
    )
    return {
        "scope": _slug(str(commit["public_tool"])),
        "commit_tool": commit["public_tool"],
        "commit_capability": commit["capability_id"],
        "commit_branch": commit["selected_branch"],
        "commit_last": prior_same_tool,
        "target_value": target,
        "verify_capability": verify["capability_id"],
        "verify_branch": verify["selected_branch"],
    }


def attach_inferred_evolution_hooks(
    bundle: TaskBundle, report: dict[str, Any]
) -> TaskBundle:
    manifest = copy.deepcopy(bundle.manifest)
    hooks = copy.deepcopy(manifest.get("evolution_hooks", {}))
    hooks["audit_checkpoint"] = infer_audit_checkpoint_hook(bundle, report)
    manifest["evolution_hooks"] = hooks
    return replace(bundle, manifest=manifest)


__all__ = ["attach_inferred_evolution_hooks", "infer_audit_checkpoint_hook"]
