#!/usr/bin/env python3
"""Synthesize validated oracle trajectories with episode-specific public APIs."""

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
    evaluate_counterfactuals,
    validate_episode,
    validate_tool_identifiability,
)
from rollout import run_reference_plan
from runtime.tool_renderer import render_alternate_api
from scripts.export_task_first_sft import build_sft_row
from task_factory import load_task_bundle


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--renderer-prefix", default="demo_api")
    args = parser.parse_args()
    if args.count < 1:
        raise SystemExit("--count must be >= 1")

    source = load_task_bundle(args.bundle)
    counterfactual = evaluate_counterfactuals(source)
    decision_metrics = counterfactual["decision_metrics"]
    accepted: list[dict[str, Any]] = []
    summaries = []
    for index in range(args.count):
        renderer_seed = None if index == 0 else f"{args.renderer_prefix}_{index}"
        bundle = source if renderer_seed is None else render_alternate_api(source, seed=renderer_seed)
        episode = run_reference_plan(bundle)
        validation = validate_episode(bundle, episode)
        identifiability = validate_tool_identifiability(bundle)
        if not identifiability["valid"]:
            validation["valid"] = False
            validation["errors"].extend(identifiability["errors"])
        validation["metrics"].update(
            {
                "meaningful_planning_decision_count": decision_metrics[
                    "meaningful_planning_decision_count"
                ],
                "decision_entropy_bits": decision_metrics["decision_entropy_bits"],
            }
        )
        result = {
            "task_id": bundle.task_id,
            "renderer_seed": renderer_seed,
            "episode": episode,
            "validation": validation,
            "tool_identifiability": identifiability,
        }
        result_path = args.output_dir / "results" / f"sample_{index + 1:03d}.json"
        write_json(result_path, result)
        row = build_sft_row(result)
        if row is not None:
            row["metadata"]["renderer_seed"] = renderer_seed or "canonical"
            grounding = source.manifest.get("source_grounding", {})
            lineage = source.manifest.get("lineage", {})
            row["metadata"]["source_id"] = grounding.get("source_id")
            row["metadata"]["source_sha256"] = grounding.get("source_sha256")
            row["metadata"]["semantic_episode_id"] = source.task_id
            row["metadata"]["assigned_operator"] = source.manifest.get(
                "assigned_operator"
            )
            row["metadata"]["operator_family"] = source.manifest.get(
                "operator_family"
            )
            row["metadata"]["recursive_generation"] = int(
                lineage.get("generation", 0)
            )
            row["metadata"]["recursive_operators"] = list(
                lineage.get("operators", [])
            )
            row["metadata"]["counterfactual_count"] = len(
                source.reference_plan.get("counterfactuals", [])
            )
            row["metadata"]["decision_metrics"] = decision_metrics
            row["metadata"]["tool_identifiability"] = identifiability["metrics"]
            accepted.append(row)
        metrics = validation["metrics"]
        summaries.append(
            {
                "id": bundle.task_id,
                "renderer_seed": renderer_seed or "canonical",
                "valid": validation["valid"],
                "steps": metrics["steps"],
                "max_delayed_handle_distance": metrics["max_delayed_handle_distance"],
                "handle_chain_depth": metrics["handle_chain_depth"],
                "observation_dependent_branch_count": metrics[
                    "observation_dependent_branch_count"
                ],
                "meaningful_planning_decision_count": decision_metrics[
                    "meaningful_planning_decision_count"
                ],
                "decision_entropy_bits": decision_metrics["decision_entropy_bits"],
                "tool_identifiability": identifiability["metrics"],
                "tool_names": [tool["function"]["name"] for tool in episode["public_tools"]],
                "result": str(result_path),
            }
        )

    write_jsonl(args.output_dir / "accepted_sft.jsonl", accepted)
    summary = {
        "requested": args.count,
        "accepted": len(accepted),
        "rejected": args.count - len(accepted),
        "samples": summaries,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if len(accepted) != args.count:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
