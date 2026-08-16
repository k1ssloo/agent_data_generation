"""Validate instruction decomposition and per-subgoal executable evidence."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any

from rollout import EpisodeRunner, run_reference_plan
from runtime.predicates import (
    EvaluationError,
    evaluate_predicate,
    predicate_paths,
    validate_predicate_syntax,
)
from task_factory.bundle import TaskBundle


def _sentences(text: str) -> list[str]:
    # Sentence boundaries require following whitespace so versions, decimals,
    # email addresses, and URLs remain intact.
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+", text.strip())
        if item.strip()
    ]


def _overlap(left: str, right: str) -> bool:
    return left.startswith(right) or right.startswith(left)


def _eval(predicate: Any, state: dict[str, Any]) -> bool:
    try:
        return evaluate_predicate(
            predicate, {"state": state, "args": {}, "response": {}}
        )
    except EvaluationError:
        return False


def _clause_bundle(bundle: TaskBundle, clause: dict[str, Any]) -> TaskBundle:
    contract = dict(bundle.contract)
    contract["goal_predicates"] = [
        {"id": str(clause.get("id", "clause")), "predicate": clause["predicate"]}
    ]
    contract["requirements"] = {
        **contract.get("requirements", {}),
        "semantic_recovery": False,
        "async_decision": False,
    }
    return replace(bundle, contract=contract)


def _predicate_history(
    bundle: TaskBundle, predicate: dict[str, Any]
) -> list[dict[str, Any]]:
    """Replay the public plan and record when a conditional invariant breaks."""
    runner = EpisodeRunner(bundle)
    history = [
        {
            "step": 0,
            "tool": None,
            "satisfied": _eval(predicate, runner.runtime.state),
        }
    ]
    for action in bundle.reference_plan.get("actions", []):
        try:
            runner.tool_call(action["tool"], action.get("arguments", {}))
        except ValueError:
            break
        history.append(
            {
                "step": runner.runtime.step,
                "tool": action["tool"],
                "satisfied": _eval(predicate, runner.runtime.state),
            }
        )
        if runner.status != "running":
            break
    return history


def validate_goal_alignment(
    bundle: TaskBundle, report: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Prove every declared instruction clause against executable state.

    Completeness is represented by an exhaustive sentence decomposition. Goal
    clauses must be initially false, become true, be independently observed,
    have a real witness transition, and fail when all named witness actions are
    removed. Semantic correctness of the decomposition remains explicit data;
    validators can prove it was not silently omitted or bypassed.
    """
    claims = bundle.contract.get("instruction_claims")
    clauses = bundle.contract.get("goal_clauses")
    errors: list[str] = []
    if not isinstance(claims, list) or not isinstance(clauses, list) or not clauses:
        return {
            "task_id": bundle.task_id,
            "valid": False,
            "errors": ["contract does not declare complete instruction claims and goal clauses"],
            "metrics": {},
            "clauses": [],
        }

    instruction_sentences = _sentences(bundle.instruction)
    claim_spans = [item.get("evidence_span") for item in claims if isinstance(item, dict)]
    if claim_spans != instruction_sentences:
        errors.append("instruction claims must exhaustively match instruction sentences in order")
    clause_by_id: dict[str, dict[str, Any]] = {}
    for index, clause in enumerate(clauses):
        if not isinstance(clause, dict):
            errors.append(f"goal clause {index} must be an object")
            continue
        clause_id = clause.get("id")
        if not isinstance(clause_id, str) or not clause_id or clause_id in clause_by_id:
            errors.append(f"goal clause {index} has missing or duplicate id")
            continue
        clause_by_id[clause_id] = clause

    referenced: set[str] = set()
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            errors.append(f"instruction claim {index} must be an object")
            continue
        kind = claim.get("kind")
        clause_ids = claim.get("clause_ids", [])
        if kind not in {"context", "goal", "constraint", "synthetic_constraint"}:
            errors.append(f"instruction claim {index} has invalid kind")
        if not isinstance(clause_ids, list) or any(
            not isinstance(item, str) or item not in clause_by_id for item in clause_ids
        ):
            errors.append(f"instruction claim {index} references unknown goal clauses")
            continue
        if kind != "context" and not clause_ids:
            errors.append(f"non-context instruction claim {index} has no goal clause")
        referenced.update(clause_ids)
    unreferenced = sorted(set(clause_by_id) - referenced)
    if unreferenced:
        errors.append(f"goal clauses are not linked to instruction claims: {unreferenced}")
    executable_goal_paths = {
        path
        for item in bundle.contract.get("goal_predicates", [])
        if isinstance(item, dict)
        for path in predicate_paths(item.get("predicate", item))
    }
    executable_constraint_paths = {
        path
        for item in bundle.contract.get("invariants", [])
        if isinstance(item, dict)
        for path in predicate_paths(item.get("predicate", item))
    }
    executable_instruction_paths = (
        executable_goal_paths | executable_constraint_paths
    )
    clause_goal_paths = {
        path
        for clause in clause_by_id.values()
        for path in predicate_paths(clause.get("predicate"))
        if not validate_predicate_syntax(clause.get("predicate"))
    }
    uncovered_executable = sorted(
        path
        for path in executable_goal_paths
        if not any(_overlap(path, clause_path) for clause_path in clause_goal_paths)
    )
    non_executable_clauses = sorted(
        path
        for path in clause_goal_paths
        if not any(
            _overlap(path, executable_path)
            for executable_path in executable_instruction_paths
        )
    )
    if uncovered_executable:
        errors.append(
            f"executable goal paths lack instruction clauses: {uncovered_executable}"
        )
    if non_executable_clauses:
        errors.append(
            f"instruction clauses are absent from executable goals: {non_executable_clauses}"
        )

    report = report or run_reference_plan(bundle)
    trace = report.get("trace", [])
    initial_state = bundle.environment.get("initial_state", {})
    tool_names = {tool.get("name") for tool in bundle.tools}
    clause_results = []
    for clause_id, clause in clause_by_id.items():
        clause_errors = []
        predicate = clause.get("predicate")
        syntax_errors = validate_predicate_syntax(predicate)
        if syntax_errors:
            clause_errors.extend(syntax_errors)
        paths = predicate_paths(predicate) if not syntax_errors else set()
        transition_paths = clause.get("transition_paths", [])
        evidence_paths = clause.get("evidence_paths", [])
        witness_tools = clause.get("witness_tools", [])
        if not isinstance(transition_paths, list) or not transition_paths:
            clause_errors.append("transition_paths must be non-empty")
            transition_paths = []
        if not isinstance(evidence_paths, list) or not evidence_paths:
            clause_errors.append("evidence_paths must be non-empty")
            evidence_paths = []
        if not isinstance(witness_tools, list) or not witness_tools:
            clause_errors.append("witness_tools must be non-empty")
            witness_tools = []
        elif any(tool not in tool_names for tool in witness_tools):
            clause_errors.append("witness_tools reference unknown public tools")
        if any(not isinstance(path, str) or not path.startswith("$state.") for path in transition_paths + evidence_paths):
            clause_errors.append("transition and evidence paths must target $state")
        if paths and any(not any(_overlap(path, goal) for goal in paths) for path in evidence_paths):
            clause_errors.append("evidence_paths do not cover clause predicate paths")

        initially_satisfied = _eval(predicate, initial_state) if not syntax_errors else False
        clause_bundle = _clause_bundle(bundle, clause) if not syntax_errors else None
        final_report = run_reference_plan(clause_bundle) if clause_bundle else {"goal_results": []}
        finally_satisfied = bool(final_report.get("goal_results")) and all(
            item.get("valid") for item in final_report["goal_results"]
        )
        predicate_history = (
            _predicate_history(clause_bundle, predicate) if clause_bundle else []
        )
        first_violation = next(
            (item for item in predicate_history[1:] if not item["satisfied"]), None
        )
        restored_invariant = bool(
            initially_satisfied and first_violation and finally_satisfied
        )
        maintained_invariant = bool(
            initially_satisfied and not first_violation and finally_satisfied
        )
        if not finally_satisfied:
            clause_errors.append("reference plan does not satisfy clause")

        repair_witness_tools = set(witness_tools)
        if restored_invariant and first_violation:
            repair_witness_tools.discard(first_violation.get("tool"))
            if not repair_witness_tools:
                clause_errors.append(
                    "restored invariant has no repair witness distinct from its trigger"
                )
        elif maintained_invariant:
            repair_witness_tools = set()
        witness_steps = [
            step
            for step in trace
            if step.get("public_tool") in repair_witness_tools
        ]
        transition_witnessed = any(
            any(_overlap(path, write) for path in transition_paths for write in step.get("write_set", []))
            for step in witness_steps
        )
        if not transition_witnessed and not maintained_invariant:
            clause_errors.append("witness actions do not write a declared transition path")
        last_transition = max(
            (
                int(step.get("step", 0))
                for step in trace
                if any(
                    _overlap(path, write)
                    for path in transition_paths
                    for write in step.get("write_set", [])
                )
            ),
            default=0,
        )
        evidence_steps = [
            step
            for step in trace
            if int(step.get("step", 0)) > last_transition
            and not step.get("write_set")
            and all(
                any(_overlap(path, read) for read in step.get("read_set", []))
                for path in evidence_paths
            )
        ]
        if not evidence_steps:
            clause_errors.append("no post-transition read-only observation covers clause evidence")

        if maintained_invariant:
            witness_necessary = True
        else:
            ablated_actions = [
                action
                for action in bundle.reference_plan.get("actions", [])
                if action.get("tool") not in repair_witness_tools
            ]
            ablated_report = (
                run_reference_plan(clause_bundle, actions=ablated_actions)
                if clause_bundle
                else {"goal_results": []}
            )
            witness_necessary = not (
                ablated_report.get("goal_results")
                and all(item.get("valid") for item in ablated_report["goal_results"])
            )
            if not witness_necessary:
                clause_errors.append(
                    "clause remains satisfied after removing all witness actions"
                )
        clause_results.append(
            {
                "id": clause_id,
                "valid": not clause_errors,
                "errors": clause_errors,
                "initially_satisfied": initially_satisfied,
                "finally_satisfied": finally_satisfied,
                "temporal_mode": (
                    "restored_invariant"
                    if restored_invariant
                    else "maintained_invariant"
                    if maintained_invariant
                    else "outcome"
                ),
                "first_violation_step": (
                    first_violation.get("step")
                    if restored_invariant and first_violation
                    else None
                ),
                "first_violation_tool": (
                    first_violation.get("tool")
                    if restored_invariant and first_violation
                    else None
                ),
                "transition_witnessed": transition_witnessed,
                "evidence_step": evidence_steps[-1]["step"] if evidence_steps else None,
                "witness_necessary": witness_necessary,
            }
        )
        errors.extend(f"{clause_id}: {error}" for error in clause_errors)
    return {
        "task_id": bundle.task_id,
        "valid": not errors,
        "errors": errors,
        "metrics": {
            "instruction_sentence_count": len(instruction_sentences),
            "instruction_claim_count": len(claims),
            "goal_clause_count": len(clause_results),
            "valid_goal_clause_count": sum(item["valid"] for item in clause_results),
            "instruction_goal_coverage": round(
                sum(item["valid"] for item in clause_results) / len(clause_results), 4
            )
            if clause_results
            else 0.0,
        },
        "clauses": clause_results,
    }
