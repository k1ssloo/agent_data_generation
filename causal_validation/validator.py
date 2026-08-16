"""Validate outcome evidence and causal structure from runtime provenance."""

from __future__ import annotations

from typing import Any

from runtime.predicates import predicate_paths
from task_factory.bundle import TaskBundle


def _producer_steps(trace: list[dict[str, Any]]) -> dict[str, int]:
    producers: dict[str, int] = {}
    for step in trace:
        for handle in step.get("produced_handles", []):
            # A stable UI/object handle can be observed repeatedly. Delayed use
            # is measured from its first public discovery, not its latest echo.
            producers.setdefault(handle["value"], step["step"])
    return producers


def _delayed_handles(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    producers = _producer_steps(trace)
    delayed = []
    for step in trace:
        for handle in step.get("consumed_handles", []):
            producer = producers.get(handle)
            if producer is not None:
                delayed.append(
                    {
                        "handle": handle,
                        "producer_step": producer,
                        "consumer_step": step["step"],
                        "distance": step["step"] - producer,
                    }
                )
    return delayed


def _handle_chain_depth(trace: list[dict[str, Any]]) -> int:
    depth_by_step: dict[int, int] = {}
    maximum = 0
    for step in trace:
        source_steps = [
            detail["source"]["step"]
            for detail in step.get("arguments", {}).values()
            if isinstance(detail.get("source"), dict) and isinstance(detail["source"].get("step"), int)
        ]
        parent_depth = max((depth_by_step.get(source_step, 1) for source_step in source_steps), default=0)
        depth = parent_depth + 1 if source_steps else 1
        depth_by_step[step["step"]] = depth
        maximum = max(maximum, depth)
    return maximum


def _goal_paths(bundle: TaskBundle) -> set[str]:
    paths: set[str] = set()
    for item in bundle.contract.get("goal_predicates", []):
        paths |= predicate_paths(item.get("predicate", item))
    return paths


def _paths_overlap(left: str, right: str) -> bool:
    return left.startswith(right) or right.startswith(left)


def _final_goal_observation(
    bundle: TaskBundle, trace: list[dict[str, Any]]
) -> dict[str, Any]:
    """Find one read-only observation after the final goal-state mutation."""
    expected = _goal_paths(bundle)
    goal_writes = [
        step
        for step in trace
        if any(
            _paths_overlap(goal, write)
            for goal in expected
            for write in step.get("write_set", [])
        )
    ]
    last_goal_write_step = max(
        (int(step.get("step", 0)) for step in goal_writes), default=0
    )
    candidates = [
        step
        for step in trace
        if int(step.get("step", 0)) > last_goal_write_step
        and not step.get("write_set")
        and not step.get("error_code")
        and isinstance(step.get("selected_branch"), str)
    ]

    def covered(step: dict[str, Any]) -> set[str]:
        observable_paths = step.get("observed_state_paths", [])
        return {
            goal
            for goal in expected
            if any(_paths_overlap(goal, path) for path in observable_paths)
        }

    best = max(
        candidates,
        key=lambda step: (len(covered(step)), int(step.get("step", 0))),
        default=None,
    )
    observed = covered(best) if best is not None else set()
    return {
        "last_goal_write_step": last_goal_write_step,
        "step": best.get("step") if best is not None else None,
        "tool": best.get("public_tool") if best is not None else None,
        "observed_goal_paths": observed,
    }


def _semantic_recoveries(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: dict[str, int] = {}
    recoveries = []
    for step in trace:
        error = step.get("error_code")
        if isinstance(error, str) and error:
            failures[error] = step["step"]
        for resolved in step.get("resolves_errors", []):
            if resolved in failures:
                recoveries.append(
                    {"error_code": resolved, "failure_step": failures[resolved], "recovery_step": step["step"]}
                )
                del failures[resolved]
    return recoveries


def _observation_dependent_branches(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find public tools whose observed inputs led to different runtime branches."""
    by_tool: dict[str, list[dict[str, Any]]] = {}
    for step in trace:
        tool = step.get("public_tool")
        branch = step.get("selected_branch")
        if not isinstance(tool, str) or not isinstance(branch, str):
            continue
        observed_arguments = sorted(
            name
            for name, detail in step.get("arguments", {}).items()
            if isinstance(detail.get("source"), dict)
        )
        by_tool.setdefault(tool, []).append(
            {
                "step": step.get("step"),
                "branch": branch,
                "observed_arguments": observed_arguments,
            }
        )

    decisions = []
    for tool, calls in by_tool.items():
        branches = sorted({call["branch"] for call in calls})
        if len(branches) < 2 or not any(call["observed_arguments"] for call in calls):
            continue
        decisions.append(
            {
                "public_tool": tool,
                "branches": branches,
                "calls": calls,
            }
        )
    return decisions


def _argument_source_fanout(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[int, dict[str, Any]] = {}
    for step in trace:
        for name, detail in step.get("arguments", {}).items():
            source = detail.get("source")
            classes = set(detail.get("provenance_classes", []))
            if not isinstance(source, dict) or classes & {"user_grounded", "schema_grounded"}:
                continue
            source_step = source.get("step")
            if not isinstance(source_step, int):
                continue
            row = by_source.setdefault(
                source_step,
                {
                    "source_step": source_step,
                    "source_tool": source.get("tool"),
                    "distinct_values": set(),
                    "consumers": [],
                },
            )
            row["distinct_values"].add(repr(detail.get("value")))
            row["consumers"].append(
                {"step": step.get("step"), "tool": step.get("public_tool"), "argument": name}
            )
    result = []
    for _step, row in sorted(by_source.items()):
        distinct_values = sorted(row["distinct_values"])
        result.append(
            {
                "source_step": row["source_step"],
                "source_tool": row["source_tool"],
                "distinct_values": distinct_values,
                "distinct_value_count": len(distinct_values),
                "consumers": row["consumers"],
                "consumer_count": len(row["consumers"]),
            }
        )
    return result


def validate_episode(
    bundle: TaskBundle,
    report: dict[str, Any],
    *,
    min_delayed_handle_distance: int = 4,
    min_handle_chain_depth: int = 3,
    require_semantic_recovery: bool | None = None,
) -> dict[str, Any]:
    trace = report.get("trace", [])
    errors: list[str] = []
    warnings: list[str] = []
    goals = report.get("goal_results", [])
    if report.get("status") != "goal_satisfied" or not goals or not all(item.get("valid") for item in goals):
        errors.append("episode did not satisfy all goal predicates")
    violated = [
        {"step": item["step"], "result": result}
        for item in report.get("invariant_history", [])
        for result in item.get("results", [])
        if not result.get("valid")
    ]
    if violated:
        errors.append("one or more invariants were violated")
    missing_tool_provenance = [
        {
            "step": step["step"],
            "tool": step.get("public_tool"),
            "argument": name,
            "value": detail.get("value"),
            "reason": "tool_observation_required",
        }
        for step in trace
        for name, detail in step.get("arguments", {}).items()
        if detail.get("required") and detail.get("source") is None
    ]
    unexplained_arguments = [
        {
            "step": step["step"],
            "tool": step.get("public_tool"),
            "argument": name,
            "value": detail.get("value"),
            "reason": "unexplained_argument",
        }
        for step in trace
        for name, detail in step.get("arguments", {}).items()
        if detail.get("provenance_kind") == "unexplained"
    ]
    missing_keys = {
        (item["step"], item["argument"]) for item in missing_tool_provenance
    }
    missing_provenance = missing_tool_provenance + [
        item
        for item in unexplained_arguments
        if (item["step"], item["argument"]) not in missing_keys
    ]
    if missing_tool_provenance:
        errors.append("required arguments lack tool-output provenance")
    if unexplained_arguments:
        errors.append("one or more tool arguments lack an admissible provenance source")
    delayed = _delayed_handles(trace)
    max_distance = max((item["distance"] for item in delayed), default=0)
    if max_distance < min_delayed_handle_distance:
        errors.append(
            f"maximum delayed handle distance {max_distance} is below {min_delayed_handle_distance}"
        )
    chain_depth = _handle_chain_depth(trace)
    if chain_depth < min_handle_chain_depth:
        errors.append(f"handle chain depth {chain_depth} is below {min_handle_chain_depth}")
    goal_paths = _goal_paths(bundle)
    final_observation = _final_goal_observation(bundle, trace)
    observed = final_observation["observed_goal_paths"]
    missing_goal_evidence = sorted(goal_paths - observed)
    if missing_goal_evidence:
        errors.append("final observations do not cover every goal predicate path")
    recoveries = _semantic_recoveries(trace)
    expected_recovery = (
        bool(bundle.contract.get("requirements", {}).get("semantic_recovery"))
        if require_semantic_recovery is None
        else require_semantic_recovery
    )
    if expected_recovery and not recoveries:
        errors.append("task requires semantic recovery but no matching recovery was observed")
    observation_branches = _observation_dependent_branches(trace)
    if bundle.contract.get("requirements", {}).get("async_decision") and not observation_branches:
        errors.append("task requires an observation-dependent async decision but none was observed")
    source_fanout = _argument_source_fanout(trace)
    concentrated_sources = [
        item
        for item in source_fanout
        if item["distinct_value_count"] > 3
    ]
    if concentrated_sources:
        errors.append(
            "one observation over-concentrates otherwise undiscoverable future arguments"
        )
    if not trace:
        errors.append("episode trace is empty")
    elif len(trace) < 6:
        warnings.append("episode has fewer than six executed actions")
    return {
        "task_id": bundle.task_id,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "metrics": {
            "steps": len(trace),
            "max_delayed_handle_distance": max_distance,
            "handle_chain_depth": chain_depth,
            "semantic_recoveries": recoveries,
            "observation_dependent_branches": observation_branches,
            "observation_dependent_branch_count": len(observation_branches),
            "state_dependent_transition_count": len(observation_branches),
            "distinct_selected_branches": len(
                {
                    (step.get("public_tool"), step.get("selected_branch"))
                    for step in trace
                    if step.get("selected_branch")
                }
            ),
            "goal_paths": sorted(goal_paths),
            "observed_goal_paths": sorted(observed),
            # This measures evidence for the declared executable contract. It
            # does not establish that natural-language user subgoals were all
            # translated into that contract.
            "contract_goal_evidence_coverage": round(len(observed) / len(goal_paths), 4) if goal_paths else 0.0,
            "goal_evidence_coverage": round(len(observed) / len(goal_paths), 4) if goal_paths else 0.0,
            "goal_evidence_scope": "declared_contract_predicates_only",
            "last_goal_write_step": final_observation["last_goal_write_step"],
            "final_goal_observation_step": final_observation["step"],
            "final_goal_observation_tool": final_observation["tool"],
            "missing_provenance": missing_provenance,
            "unexplained_arguments": unexplained_arguments,
            "argument_provenance_counts": {
                kind: sum(
                    detail.get("provenance_kind") == kind
                    for step in trace
                    for detail in step.get("arguments", {}).values()
                )
                for kind in (
                    "user_grounded",
                    "tool_observation_grounded",
                    "schema_grounded",
                    "derived",
                    "unexplained",
                )
            },
            "argument_source_fanout": source_fanout,
            "overconcentrated_argument_sources": concentrated_sources,
            "invariant_violations": violated,
        },
    }
