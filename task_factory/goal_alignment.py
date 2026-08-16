"""Generate and compile compact instruction-to-contract alignment plans."""

from __future__ import annotations

import copy
import re
from typing import Any

from causal_validation import validate_goal_alignment
from rollout import EpisodeRunner
from runtime.predicates import predicate_paths, validate_predicate_syntax
from task_factory.bundle import TaskBundle, validate_bundle


ALIGNMENT_VERSION = "goal-alignment-v1"


def normalize_alignment_plan(
    bundle: TaskBundle, plan: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Normalize harmless provider aliases without changing task semantics.

    Alignment is metadata over an already executable contract. A model may
    decompose that contract, but it must not strengthen a conditional goal and
    thereby invalidate an otherwise correct counterfactual policy.
    """
    normalized = copy.deepcopy(plan)
    changes: list[str] = []
    if "alignment_version" not in normalized:
        normalized["alignment_version"] = ALIGNMENT_VERSION
        changes.append("defaulted alignment_version")
    if "rejection_reasons" not in normalized:
        normalized["rejection_reasons"] = []
        changes.append("defaulted rejection_reasons")
    if "goal_clauses" not in normalized and isinstance(
        normalized.get("clauses"), list
    ):
        normalized["goal_clauses"] = normalized.pop("clauses")
        changes.append("renamed clauses to goal_clauses")

    claims = normalized.get("instruction_claims")
    if isinstance(claims, list):
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            if "kind" not in claim and isinstance(claim.get("classification"), str):
                claim["kind"] = claim.pop("classification")
                changes.append("renamed instruction claim classification to kind")

    invariants = normalized.get("domain_invariants")
    if isinstance(invariants, list):
        for index, invariant in enumerate(invariants, 1):
            if not isinstance(invariant, dict) or invariant.get("id"):
                continue
            raw = str(invariant.get("relation") or f"domain_invariant_{index}")
            slug = re.sub(r"[^a-z0-9]+", "_", raw.casefold()).strip("_")
            invariant["id"] = slug or f"domain_invariant_{index}"
            changes.append(f"assigned domain invariant id {invariant['id']}")

    original_by_paths: dict[frozenset[str], list[dict[str, Any]]] = {}
    executable_paths: set[str] = set()
    for goal in bundle.contract.get("goal_predicates", []):
        if not isinstance(goal, dict):
            continue
        predicate = goal.get("predicate", goal)
        if validate_predicate_syntax(predicate):
            continue
        paths = predicate_paths(predicate)
        executable_paths.update(paths)
        original_by_paths.setdefault(frozenset(paths), []).append(predicate)
    for invariant in bundle.contract.get("invariants", []):
        if not isinstance(invariant, dict):
            continue
        predicate = invariant.get("predicate", invariant)
        if not validate_predicate_syntax(predicate):
            executable_paths.update(predicate_paths(predicate))

    def path_allowed(path: str) -> bool:
        return any(
            path.startswith(allowed) or allowed.startswith(path)
            for allowed in executable_paths
        )

    def prune_unexecutable_conjuncts(predicate: Any) -> Any | None:
        if not isinstance(predicate, dict) or len(predicate) != 1:
            return None
        operator, value = next(iter(predicate.items()))
        if operator == "all" and isinstance(value, list):
            retained = [
                pruned
                for item in value
                if (pruned := prune_unexecutable_conjuncts(item)) is not None
            ]
            return {"all": retained} if retained else None
        paths = predicate_paths(predicate)
        return predicate if paths and all(path_allowed(path) for path in paths) else None

    clauses = normalized.get("goal_clauses")
    if isinstance(clauses, list):
        for clause in clauses:
            if not isinstance(clause, dict):
                continue
            predicate = clause.get("predicate")
            if validate_predicate_syntax(predicate):
                continue
            originals = original_by_paths.get(frozenset(predicate_paths(predicate)), [])
            if len(originals) == 1 and predicate != originals[0]:
                clause["predicate"] = copy.deepcopy(originals[0])
                changes.append(
                    f"restored executable predicate for clause {clause.get('id', '<unknown>')}"
                )
                predicate = clause["predicate"]
            pruned = prune_unexecutable_conjuncts(predicate)
            if pruned is not None and pruned != predicate:
                clause["predicate"] = pruned
                changes.append(
                    f"removed non-contract conjuncts from clause {clause.get('id', '<unknown>')}"
                )
            if pruned is not None:
                paths = sorted(predicate_paths(pruned))
                clause["evidence_paths"] = paths
                clause["transition_paths"] = [
                    path
                    for path in clause.get("transition_paths", [])
                    if path in paths
                ] or paths
    return normalized, changes


def _expose_alignment_evidence(
    bundle: TaskBundle, clauses: list[dict[str, Any]]
) -> tuple[TaskBundle, list[str]]:
    """Expose declared contract state on the existing final read-only view."""
    report = EpisodeRunner(bundle)
    for action in bundle.reference_plan.get("actions", []):
        report.tool_call(action["tool"], action.get("arguments", {}))
    if not report.runtime.trace:
        raise ValueError("alignment evidence requires a non-empty reference plan")
    final = report.runtime.trace[-1]
    if final.get("write_set"):
        raise ValueError("alignment evidence requires a final read-only observation")
    evidence_paths = sorted(
        {
            path
            for clause in clauses
            if isinstance(clause, dict)
            for path in clause.get("evidence_paths", [])
            if isinstance(path, str) and path.startswith("$state.")
        }
    )
    missing = [path for path in evidence_paths if path not in final.get("read_set", [])]
    if not missing:
        return bundle, []
    environment = copy.deepcopy(bundle.environment)
    capability = environment["capabilities"][final["capability_id"]]
    branch = next(
        item
        for item in capability.get("branches", [])
        if item.get("id") == final["selected_branch"]
    )
    response = branch.setdefault("response", {})
    reads = branch.setdefault("reads", [])
    for path in missing:
        key = re.sub(r"[^a-z0-9]+", "_", path.rsplit(".", 1)[-1].casefold()).strip(
            "_"
        ) or "constraint_status"
        base = key
        suffix = 2
        while key in response:
            key = f"{base}_{suffix}"
            suffix += 1
        response[key] = path
        if path not in reads:
            reads.append(path)
    branch["reads"] = sorted(reads)
    return type(bundle)(
        root=bundle.root,
        manifest=copy.deepcopy(bundle.manifest),
        instruction=bundle.instruction,
        contract=copy.deepcopy(bundle.contract),
        environment=environment,
        bindings=copy.deepcopy(bundle.bindings),
        reference_plan=copy.deepcopy(bundle.reference_plan),
    ), missing


def instruction_sentences(instruction: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+", instruction.strip())
        if item.strip()
    ]


def alignment_context(bundle: TaskBundle) -> dict[str, Any]:
    runner = EpisodeRunner(bundle)
    transitions = []
    for action in bundle.reference_plan["actions"]:
        response = runner.tool_call(action["tool"], action.get("arguments", {}))
        step = runner.runtime.trace[-1]
        transitions.append(
            {
                "step": step["step"],
                "tool": action["tool"],
                "arguments": action.get("arguments", {}),
                "response": response,
                "reads": step.get("read_set", []),
                "writes": step.get("write_set", []),
            }
        )
    relevant_paths = sorted(
        {
            path
            for field in ("goal_predicates", "invariants")
            for item in bundle.contract.get(field, [])
            if isinstance(item, dict)
            for path in predicate_paths(item.get("predicate", item))
        }
    )

    def state_slice(state: dict[str, Any]) -> dict[str, Any]:
        from runtime.predicates import EvaluationError, resolve_path

        context = {"state": state, "args": {}, "response": {}}
        result: dict[str, Any] = {}
        for path in relevant_paths:
            try:
                result[path] = copy.deepcopy(resolve_path(path, context))
            except EvaluationError:
                result[path] = {"missing": True}
        return result

    return {
        "instruction_sentences": instruction_sentences(bundle.instruction),
        "executable_goal_predicates": copy.deepcopy(
            bundle.contract.get("goal_predicates", [])
        ),
        "executable_domain_invariants": copy.deepcopy(
            bundle.contract.get("invariants", [])
        ),
        "initial_state": state_slice(bundle.environment["initial_state"]),
        "final_state": state_slice(runner.runtime.state),
        "transitions": transitions,
        "public_tools": [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
            }
            for tool in bundle.tools
        ],
    }


def compile_alignment_plan(
    bundle: TaskBundle, plan: dict[str, Any]
) -> tuple[TaskBundle | None, dict[str, Any]]:
    plan, normalization = normalize_alignment_plan(bundle, plan)
    if plan.get("alignment_version") != ALIGNMENT_VERSION:
        return None, {
            "valid": False,
            "errors": ["invalid alignment_version"],
            "normalization": normalization,
        }
    if plan.get("supported") is not True:
        reasons = plan.get("rejection_reasons", [])
        return None, {
            "valid": False,
            "errors": ["model rejected semantic alignment", *map(str, reasons)],
        }
    contract = copy.deepcopy(bundle.contract)
    claims = copy.deepcopy(plan.get("instruction_claims"))
    clauses = copy.deepcopy(plan.get("goal_clauses"))
    proposed_invariants = copy.deepcopy(plan.get("domain_invariants", []))
    if not isinstance(claims, list) or not isinstance(clauses, list) or not clauses:
        return None, {
            "valid": False,
            "errors": ["alignment plan must contain claims and clauses"],
        }
    contract["instruction_claims"] = claims
    contract["goal_clauses"] = clauses
    # The alignment plan explains the existing executable goals; it does not
    # replace them. Counterfactual validity depends on preserving conditional
    # goal semantics such as "no failure OR recovered failure".
    # Runtime invariants are part of the already validated environment
    # contract. Alignment may describe them but cannot add stronger temporal
    # assumptions or replace their executable semantics.
    evidence_bundle, exposed_paths = _expose_alignment_evidence(bundle, clauses)
    aligned = type(bundle)(
        root=bundle.root,
        manifest={
            **copy.deepcopy(bundle.manifest),
            "goal_alignment_version": ALIGNMENT_VERSION,
        },
        instruction=bundle.instruction,
        contract=contract,
        environment=copy.deepcopy(evidence_bundle.environment),
        bindings=copy.deepcopy(bundle.bindings),
        reference_plan=copy.deepcopy(bundle.reference_plan),
    )
    structural = validate_bundle(aligned)
    semantic = validate_goal_alignment(aligned)
    errors = [*structural, *semantic["errors"]]
    return (aligned if not errors else None), {
        "valid": not errors,
        "errors": errors,
        "goal_alignment": semantic,
        "normalization": normalization,
        "proposed_domain_invariant_count": len(proposed_invariants),
        "exposed_evidence_paths": exposed_paths,
    }


__all__ = [
    "ALIGNMENT_VERSION",
    "alignment_context",
    "compile_alignment_plan",
    "instruction_sentences",
    "normalize_alignment_plan",
]
