#!/usr/bin/env python3
"""Audit semantic workflow quality beyond structural trajectory validity."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_validation import (
    evaluate_action_ablation,
    minimize_action_plan,
    validate_episode,
)
from causal_validation.intervention import evaluate_counterfactuals
from rollout import run_reference_plan
from task_factory import load_task_bundle


SCAFFOLD_CAPABILITY_PREFIXES = ("audit.",)
SCAFFOLD_TOOLS = {"observe_workflow_context", "observe_workflow_outcome"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def is_scaffolding(step: dict[str, Any]) -> bool:
    tool = step.get("public_tool")
    capability = str(step.get("capability_id", ""))
    return tool in SCAFFOLD_TOOLS or capability.startswith(
        SCAFFOLD_CAPABILITY_PREFIXES
    )


def audit_bundle(path: Path) -> dict[str, Any]:
    bundle = load_task_bundle(path)
    episode = run_reference_plan(bundle)
    causal = validate_episode(bundle, episode)
    ablation = evaluate_action_ablation(bundle)
    minimization = minimize_action_plan(bundle)
    counterfactual = evaluate_counterfactuals(bundle)
    optional_indices = {
        int(item["removed_index"])
        for item in ablation["items"]
        if not item["necessary"]
    }
    optional_steps = []
    for index in sorted(optional_indices):
        step = episode["trace"][index]
        optional_steps.append(
            {
                "index": index,
                "step": step["step"],
                "tool": step["public_tool"],
                "capability_id": step["capability_id"],
                "kind": "scaffolding" if is_scaffolding(step) else "domain",
                "mutates_state": bool(step.get("write_set")),
                "write_set": step.get("write_set", []),
            }
        )
    optional_domain = [item for item in optional_steps if item["kind"] == "domain"]
    minimized_away_steps = []
    for index in minimization["removed_indices"]:
        step = episode["trace"][index]
        minimized_away_steps.append(
            {
                "index": index,
                "step": step["step"],
                "tool": step["public_tool"],
                "capability_id": step["capability_id"],
                "kind": "scaffolding" if is_scaffolding(step) else "domain",
                "mutates_state": bool(step.get("write_set")),
                "write_set": step.get("write_set", []),
            }
        )
    minimized_away_domain = [
        item for item in minimized_away_steps if item["kind"] == "domain"
    ]
    decision = counterfactual["decision_metrics"]
    semantic_recovery_count = len(
        causal.get("metrics", {}).get("semantic_recoveries", [])
    )
    base_valid = causal["valid"]
    mutation_complete = base_valid and not any(
        item["mutates_state"] for item in minimized_away_domain
    )
    strict_workflow_valid = base_valid and not optional_domain
    adaptive_valid = strict_workflow_valid and (
        decision["meaningful_planning_decision_count"] > 0
        and semantic_recovery_count > 0
    )
    return {
        "id": bundle.task_id,
        "bundle": str(bundle.root),
        "tiers": {
            "base_contract_valid": base_valid,
            "no_redundant_domain_mutations": mutation_complete,
            "strict_workflow_valid": strict_workflow_valid,
            "adaptive_valid": adaptive_valid,
        },
        "raw_steps": len(episode["trace"]),
        "necessary_steps": ablation["necessary_actions"],
        "necessary_action_ratio": ablation["necessary_action_ratio"],
        "irreducible_steps": minimization["irreducible_actions"],
        "irreducible_action_ratio": minimization["irreducible_action_ratio"],
        "minimized_away_steps": minimized_away_steps,
        "minimized_away_domain_step_count": len(minimized_away_domain),
        "minimized_away_domain_mutation_count": sum(
            item["mutates_state"] for item in minimized_away_domain
        ),
        "optional_steps": optional_steps,
        "optional_domain_step_count": len(optional_domain),
        "optional_domain_mutation_count": sum(
            item["mutates_state"] for item in optional_domain
        ),
        "semantic_recovery_count": semantic_recovery_count,
        "meaningful_planning_decision_count": decision[
            "meaningful_planning_decision_count"
        ],
        "decision_entropy_bits": decision["decision_entropy_bits"],
        "contract_goal_evidence_coverage": causal["metrics"].get(
            "contract_goal_evidence_coverage",
            causal["metrics"].get("goal_evidence_coverage", 0.0),
        ),
        "instruction_goal_coverage": "requires_semantic_audit",
        "environment_semantic_consistency": "requires_declared_domain_invariants",
        "errors": causal["errors"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--ids-from",
        type=Path,
        help="Optional OpenAI-message JSONL used to select bundle task IDs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    selected = None
    if args.ids_from:
        selected = {str(row["id"]) for row in load_jsonl(args.ids_from)}
    rows = []
    for manifest in sorted(args.input_dir.rglob("manifest.json")):
        bundle = load_task_bundle(manifest)
        if selected is not None and bundle.task_id not in selected:
            continue
        rows.append(audit_bundle(manifest))
    if selected is not None:
        found = {row["id"] for row in rows}
        missing = sorted(selected - found)
        if missing:
            raise SystemExit(f"could not find {len(missing)} selected bundles")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    operator_distribution = Counter()
    for row in rows:
        manifest = load_task_bundle(Path(row["bundle"])).manifest
        operator_distribution.update(manifest.get("lineage", {}).get("operators", []))
    total_steps = sum(row["raw_steps"] for row in rows)
    necessary_steps = sum(row["necessary_steps"] for row in rows)
    irreducible_steps = sum(row["irreducible_steps"] for row in rows)
    summary = {
        "rows": len(rows),
        "tiers": {
            tier: sum(row["tiers"][tier] for row in rows)
            for tier in (
                "base_contract_valid",
                "no_redundant_domain_mutations",
                "strict_workflow_valid",
                "adaptive_valid",
            )
        },
        "raw_steps": total_steps,
        "necessary_steps": necessary_steps,
        "aggregate_necessary_action_ratio": round(
            necessary_steps / total_steps, 4
        )
        if total_steps
        else 0.0,
        "irreducible_steps": irreducible_steps,
        "aggregate_irreducible_action_ratio": round(
            irreducible_steps / total_steps, 4
        )
        if total_steps
        else 0.0,
        "minimized_away_domain_steps": sum(
            row["minimized_away_domain_step_count"] for row in rows
        ),
        "minimized_away_domain_mutations": sum(
            row["minimized_away_domain_mutation_count"] for row in rows
        ),
        "rows_with_optional_domain_steps": sum(
            row["optional_domain_step_count"] > 0 for row in rows
        ),
        "optional_domain_steps": sum(
            row["optional_domain_step_count"] for row in rows
        ),
        "optional_domain_mutations": sum(
            row["optional_domain_mutation_count"] for row in rows
        ),
        "with_meaningful_planning_decision": sum(
            row["meaningful_planning_decision_count"] > 0 for row in rows
        ),
        "with_semantic_recovery": sum(
            row["semantic_recovery_count"] > 0 for row in rows
        ),
        "operator_distribution": dict(sorted(operator_distribution.items())),
        "metric_scope": {
            "contract_goal_evidence_coverage": (
                "Coverage of declared executable predicates only."
            ),
            "instruction_goal_coverage": (
                "Not inferred by deterministic validation; requires semantic audit."
            ),
        },
        "output": str(args.output),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
