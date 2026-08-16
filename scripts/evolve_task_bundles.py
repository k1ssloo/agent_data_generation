#!/usr/bin/env python3
"""Recursively evolve a bundle through semantic operators and hard quality gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task_factory import load_task_bundle
from task_factory.archive import TaskArchive
from task_factory.evolve import evolve_once
from task_factory.materialize import materialize_candidate
from task_factory.operators import OPERATORS
from task_factory.operators.base import manifest_metadata


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--operators",
        nargs="+",
        default=["policy_freshness_coupling_v1", "capacity_reservation_branch_v1"],
        help=f"Ordered operator IDs. Available: {', '.join(sorted(OPERATORS))}",
    )
    parser.add_argument(
        "--objective",
        choices=("decision_nodes", "semantic"),
        default="decision_nodes",
        help="Recursive acceptance target. Production defaults to grounded decision-node gain.",
    )
    args = parser.parse_args()
    unknown = sorted(set(args.operators) - set(OPERATORS))
    if unknown:
        raise SystemExit(f"unknown operators: {unknown}")

    current = load_task_bundle(args.parent)
    archive = TaskArchive(args.output_dir / "archive.jsonl")
    generations = []
    for operator_id in args.operators:
        evaluation = evolve_once(current, operator_id, objective=args.objective)
        report = evaluation.report
        generation = evaluation.product.patch["generation"]
        audit_dir = args.output_dir / "audits" / f"generation_{generation:02d}"
        write_json(audit_dir / "contract_patch.json", evaluation.product.patch)
        write_json(audit_dir / "evaluation.json", report)
        if not report["accepted"]:
            generations.append(report)
            break

        child = evaluation.product.bundle
        bundle_path = materialize_candidate(
            args.output_dir / "bundles",
            task_id=child.task_id,
            contract=child.contract,
            candidate={
                "instruction": child.instruction,
                "environment": child.environment,
                "bindings": child.bindings,
                "reference_plan": child.reference_plan,
            },
            lineage=child.manifest["lineage"],
            manifest_metadata=manifest_metadata(child),
        )
        archive.add(
            {
                "task_id": child.task_id,
                "parent_task_id": current.task_id,
                "generation": generation,
                "operator_id": operator_id,
                "bundle": str(bundle_path),
                "evaluation": str(audit_dir / "evaluation.json"),
                "complexity_profile": report["child_profile"],
            }
        )
        generations.append(report)
        current = load_task_bundle(bundle_path)

    accepted = sum(1 for report in generations if report["accepted"])
    summary = {
        "root_task_id": load_task_bundle(args.parent).task_id,
        "requested_generations": len(args.operators),
        "accepted_generations": accepted,
        "final_task_id": current.task_id,
        "generations": [
            {
                "operator_id": report["operator_id"],
                "child_task_id": report["child_task_id"],
                "accepted": report["accepted"],
                "errors": report["errors"],
                "complexity_delta": report["complexity_delta"],
                "evolution_objective": report["evolution_objective"],
                "decision_node_delta": report["decision_node_delta"],
                "parent_plan_valid_on_child": report["parent_plan_valid_on_child"],
                "counterfactual_required": report["counterfactual_required"],
                "counterfactual_gate_passed": report["counterfactual_gate_passed"],
            }
            for report in generations
        ],
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if accepted != len(args.operators):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
