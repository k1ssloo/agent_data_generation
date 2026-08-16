"""MAP-Elites style recursive candidate search over task bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rollout import run_reference_plan

from .api_graph import build_api_dependency_graph
from .bundle import TaskBundle
from .evolve import EvolutionEvaluation, evolve_once
from .fingerprint import semantic_fingerprint
from .operators import OPERATORS


def _bucket(value: int, boundaries: tuple[int, ...]) -> str:
    for boundary in boundaries:
        if value <= boundary:
            return f"le_{boundary}"
    return f"gt_{boundaries[-1]}"


def archive_cell(bundle: TaskBundle, report: dict[str, Any]) -> tuple[str, ...]:
    profile = report["child_profile"]
    lineage = bundle.manifest.get("lineage", {})
    domain = str(bundle.manifest.get("domain", lineage.get("domain", "unknown")))
    operators = "+".join(sorted(lineage.get("operators", []))) or "root"
    return (
        domain,
        operators,
        _bucket(int(profile["handle_chain_depth"]), (3, 6, 10, 15)),
        _bucket(int(profile["observation_dependent_branches"]), (0, 1, 3, 6)),
        _bucket(int(profile["counterfactual_variants"]), (0, 1, 2, 4)),
    )


def candidate_score(evaluation: EvolutionEvaluation) -> tuple[int, ...]:
    report = evaluation.report
    profile = report["child_profile"]
    delta = report["complexity_delta"]
    return (
        int(report["counterfactual_gate_passed"]),
        int(report["decision_metrics"]["meaningful_planning_decision_count"]),
        int(round(report["decision_metrics"]["decision_entropy_bits"] * 1000)),
        int(profile["counterfactual_variants"]),
        int(profile["observation_dependent_branches"]),
        int(profile["handle_chain_depth"]),
        int(profile["goal_paths"]),
        sum(max(0, int(value)) for value in delta.values()),
        int(profile["steps"]),
    )


@dataclass(frozen=True)
class SearchCandidate:
    parent: TaskBundle
    evaluation: EvolutionEvaluation
    fingerprint: str
    cell: tuple[str, ...]
    score: tuple[int, ...]


def generate_candidates(
    parents: list[TaskBundle],
    operator_ids: list[str] | None = None,
    *,
    objective: str = "semantic",
) -> tuple[list[SearchCandidate], list[dict[str, Any]]]:
    operator_ids = operator_ids or sorted(OPERATORS)
    candidates = []
    rejected = []
    for parent in parents:
        for operator_id in operator_ids:
            try:
                evaluation = evolve_once(parent, operator_id, objective=objective)
            except (KeyError, TypeError, ValueError) as exc:
                rejected.append(
                    {
                        "parent_task_id": parent.task_id,
                        "operator_id": operator_id,
                        "reason": "operator_not_applicable",
                        "detail": str(exc),
                    }
                )
                continue
            if not evaluation.report["accepted"]:
                rejected.append(
                    {
                        "parent_task_id": parent.task_id,
                        "operator_id": operator_id,
                        "reason": "quality_gate_failed",
                        "detail": evaluation.report["errors"],
                    }
                )
                continue
            child = evaluation.product.bundle
            candidates.append(
                SearchCandidate(
                    parent=parent,
                    evaluation=evaluation,
                    fingerprint=semantic_fingerprint(child),
                    cell=archive_cell(child, evaluation.report),
                    score=candidate_score(evaluation),
                )
            )
    return candidates, rejected


def select_candidates(
    candidates: list[SearchCandidate],
    *,
    existing_fingerprints: set[str] | None = None,
    max_per_cell: int = 1,
    max_per_parent: int = 2,
    max_per_operator: int = 100,
) -> tuple[list[SearchCandidate], list[dict[str, Any]]]:
    existing_fingerprints = existing_fingerprints or set()
    selected = []
    rejected = []
    seen = set(existing_fingerprints)
    cell_counts: dict[tuple[str, ...], int] = {}
    parent_counts: dict[str, int] = {}
    operator_counts: dict[str, int] = {}
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        operator_id = candidate.evaluation.report["operator_id"]
        parent_id = candidate.parent.task_id
        reason = None
        if candidate.fingerprint in seen:
            reason = "semantic_duplicate"
        elif cell_counts.get(candidate.cell, 0) >= max_per_cell:
            reason = "archive_cell_full"
        elif parent_counts.get(parent_id, 0) >= max_per_parent:
            reason = "parent_quota"
        elif operator_counts.get(operator_id, 0) >= max_per_operator:
            reason = "operator_quota"
        if reason:
            rejected.append(
                {
                    "child_task_id": candidate.evaluation.product.bundle.task_id,
                    "parent_task_id": parent_id,
                    "operator_id": operator_id,
                    "reason": reason,
                    "cell": list(candidate.cell),
                }
            )
            continue
        selected.append(candidate)
        seen.add(candidate.fingerprint)
        cell_counts[candidate.cell] = cell_counts.get(candidate.cell, 0) + 1
        parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
        operator_counts[operator_id] = operator_counts.get(operator_id, 0) + 1
    return selected, rejected


def candidate_metadata(candidate: SearchCandidate) -> dict[str, Any]:
    child = candidate.evaluation.product.bundle
    graph = build_api_dependency_graph(run_reference_plan(child))
    return {
        "task_id": child.task_id,
        "parent_task_id": candidate.parent.task_id,
        "operator_id": candidate.evaluation.report["operator_id"],
        "semantic_fingerprint": candidate.fingerprint,
        "archive_cell": list(candidate.cell),
        "score": list(candidate.score),
        "api_graph": graph,
        "complexity_profile": candidate.evaluation.report["child_profile"],
    }
