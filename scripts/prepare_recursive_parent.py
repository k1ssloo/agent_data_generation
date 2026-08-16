#!/usr/bin/env python3
"""Admit valid interventions and infer portable hooks for recursive evolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_validation import evaluate_action_ablation, validate_episode
from rollout import run_reference_plan
from task_factory import load_task_bundle
from task_factory.materialize import materialize_candidate
from task_factory.operators.base import manifest_metadata
from task_factory.prepare import prepare_recursive_parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = load_task_bundle(args.bundle)
    prepared, audit = prepare_recursive_parent(source)
    report = run_reference_plan(prepared)
    validation = validate_episode(prepared, report)
    ablation = evaluate_action_ablation(prepared)
    if not validation["valid"] or ablation["necessary_action_ratio"] < 0.6:
        raise SystemExit(
            "prepared parent failed causal admission: "
            + json.dumps(
                {
                    "errors": validation["errors"],
                    "necessary_action_ratio": ablation["necessary_action_ratio"],
                },
                ensure_ascii=False,
            )
        )
    output = materialize_candidate(
        args.output_dir,
        task_id=prepared.task_id,
        contract=prepared.contract,
        candidate={
            "instruction": prepared.instruction,
            "environment": prepared.environment,
            "bindings": prepared.bindings,
            "reference_plan": prepared.reference_plan,
        },
        lineage={**prepared.manifest.get("lineage", {}), "prepared_for_recursion": True},
        manifest_metadata=manifest_metadata(prepared),
    )
    result = {
        "task_id": prepared.task_id,
        "output": str(output),
        "audit": audit,
        "causal_metrics": validation["metrics"],
        "necessary_action_ratio": ablation["necessary_action_ratio"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "preparation_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
