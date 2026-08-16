#!/usr/bin/env python3
"""Prepare executable WikiHow parents for semantic goal alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_validation import (
    evaluate_action_ablation,
    validate_episode,
    validate_tool_identifiability,
)
from rollout import run_reference_plan
from task_factory import (
    load_task_bundle,
    totalize_public_capabilities,
    validate_public_executability,
)
from task_factory.materialize import materialize_candidate
from task_factory.operators.base import manifest_metadata


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feasibility-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-necessary-action-ratio", type=float, default=0.6)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    source = json.loads(args.feasibility_report.read_text(encoding="utf-8"))
    candidates = [
        item
        for item in source.get("results", [])
        if item.get("parent_relaxed_valid") is True and item.get("parent_bundle")
    ]
    if args.limit > 0:
        candidates = candidates[: args.limit]

    audits = []
    for item in candidates:
        bundle = totalize_public_capabilities(
            load_task_bundle(Path(item["parent_bundle"]))
        )
        episode = run_reference_plan(bundle)
        causal = validate_episode(
            bundle,
            episode,
            min_delayed_handle_distance=0,
            min_handle_chain_depth=0,
            require_semantic_recovery=False,
        )
        public = validate_public_executability(bundle)
        identifiable = validate_tool_identifiability(bundle)
        ablation = evaluate_action_ablation(bundle)
        errors = []
        if episode["status"] != "goal_satisfied" or not causal["valid"]:
            errors.append("source workflow is not causally executable")
            errors.extend(causal["errors"])
        if not public["valid"]:
            errors.append("public interface is not structurally total")
            errors.extend(public["errors"])
        if not identifiable["valid"]:
            errors.append("public tools are not identifiable")
            errors.extend(identifiable["errors"])
        if causal["metrics"].get("goal_evidence_coverage") != 1.0:
            errors.append("final observation does not cover every declared goal path")
        if causal["metrics"].get("unexplained_arguments"):
            errors.append("one or more arguments lack public provenance")
        if ablation["necessary_action_ratio"] < args.min_necessary_action_ratio:
            errors.append("necessary action ratio below threshold")

        audit: dict[str, Any] = {
            "task_id": bundle.task_id,
            "accepted_for_alignment": not errors,
            "semantic_alignment_status": "pending" if not errors else "not_admitted",
            "errors": errors,
            "metrics": {
                "steps": causal["metrics"]["steps"],
                "goal_evidence_coverage": causal["metrics"][
                    "goal_evidence_coverage"
                ],
                "unexplained_argument_count": len(
                    causal["metrics"].get("unexplained_arguments", [])
                ),
                "necessary_action_ratio": ablation["necessary_action_ratio"],
            },
        }
        if not errors:
            path = materialize_candidate(
                args.output_dir / "admitted",
                task_id=bundle.task_id,
                contract=bundle.contract,
                candidate={
                    "instruction": bundle.instruction,
                    "environment": bundle.environment,
                    "bindings": bundle.bindings,
                    "reference_plan": bundle.reference_plan,
                },
                lineage=bundle.manifest.get("lineage", {}),
                manifest_metadata={
                    **manifest_metadata(bundle),
                    "semantic_alignment_status": "pending",
                    "public_interface_normalized": True,
                },
            )
            audit["bundle"] = str(path)
        audits.append(audit)
        write_json(args.output_dir / "audits" / f"{bundle.task_id}.json", audit)

    summary = {
        "considered": len(candidates),
        "admitted_for_alignment": sum(
            item["accepted_for_alignment"] for item in audits
        ),
        "rejected": sum(not item["accepted_for_alignment"] for item in audits),
        "semantic_alignment_status": "pending",
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
