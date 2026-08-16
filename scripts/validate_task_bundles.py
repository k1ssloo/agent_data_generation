#!/usr/bin/env python3
"""Validate task-first bundles with oracle, causal, rendering, and ablation gates."""

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
    minimize_action_plan,
    validate_episode,
    validate_goal_alignment,
    validate_tool_identifiability,
    validate_adaptive_profile,
    validate_vnext_adaptive_profile,
)
from causal_validation.intervention import evaluate_counterfactuals
from rollout import run_reference_plan
from runtime.tool_renderer import render_alternate_api
from task_factory import load_task_bundle, validate_public_executability
from task_factory.bundle import BundleError


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-necessary-action-ratio", type=float, default=0.6)
    parser.add_argument("--renderer-seed", default="validation_api_v1")
    parser.add_argument("--require-goal-alignment", action="store_true")
    parser.add_argument("--require-adaptive", action="store_true")
    parser.add_argument("--require-vnext-adaptive", action="store_true")
    parser.add_argument("--require-public-executability", action="store_true")
    args = parser.parse_args()
    manifests = sorted(args.input_dir.rglob("manifest.json"))
    rows = []
    for manifest in manifests:
        try:
            bundle = load_task_bundle(manifest)
            report = run_reference_plan(bundle)
            causal = validate_episode(bundle, report)
            rendered = render_alternate_api(bundle, seed=args.renderer_seed)
            rendered_causal = validate_episode(rendered, run_reference_plan(rendered))
            rendered_identifiability = validate_tool_identifiability(rendered)
            ablation = evaluate_action_ablation(bundle)
            counterfactual = evaluate_counterfactuals(bundle)
            alignment = validate_goal_alignment(bundle, report)
            minimization = minimize_action_plan(bundle)
            public_executability = validate_public_executability(bundle)
            errors = list(causal["errors"])
            if not rendered_causal["valid"]:
                errors.append("alternate API rendering failed")
            if not rendered_identifiability["valid"]:
                errors.append("alternate API rendering is not publicly identifiable")
            if ablation["necessary_action_ratio"] < args.min_necessary_action_ratio:
                errors.append("necessary action ratio below threshold")
            if bundle.reference_plan.get("counterfactuals") and not counterfactual["valid"]:
                errors.append("counterfactual strategy adaptation failed")
            if args.require_goal_alignment and not alignment["valid"]:
                errors.append("instruction-goal alignment failed")
            if args.require_public_executability and not public_executability["valid"]:
                errors.append("public tool interface is not runtime-total")
                errors.extend(public_executability["errors"])
            has_recovery = bool(causal["metrics"].get("semantic_recoveries"))
            adaptive_profile = validate_adaptive_profile(
                bundle,
                report,
                counterfactual,
                semantic_recovery_count=len(
                    causal["metrics"].get("semantic_recoveries", [])
                ),
            )
            vnext_profile = validate_vnext_adaptive_profile(
                bundle,
                report,
                causal,
                counterfactual,
                ablation=ablation,
            )
            if args.require_adaptive and not adaptive_profile["valid"]:
                errors.append(
                    "adaptive gate requires validated planning plus semantic recovery, "
                    "evidence-backed closed-loop control, or temporal provenance"
                )
            if args.require_vnext_adaptive and not vnext_profile["valid"]:
                errors.append("vNext adaptive quality gate failed")
                errors.extend(vnext_profile["errors"])
            rows.append(
                {
                    "id": bundle.task_id,
                    "valid": not errors,
                    "errors": errors,
                    "causal_metrics": causal["metrics"],
                    "rendered_valid": rendered_causal["valid"],
                    "rendered_identifiable": rendered_identifiability["valid"],
                    "necessary_action_ratio": ablation["necessary_action_ratio"],
                    "counterfactual_valid": counterfactual["valid"],
                    "goal_alignment": alignment,
                    "plan_minimization": minimization,
                    "public_executability": public_executability,
                    "adaptive": adaptive_profile["valid"],
                    "adaptive_profile": adaptive_profile,
                    "vnext_adaptive": vnext_profile["valid"],
                    "vnext_adaptive_profile": vnext_profile,
                    "has_planning": adaptive_profile["has_planning"],
                    "has_semantic_recovery": has_recovery,
                    "bundle": str(bundle.root),
                }
            )
        except (BundleError, OSError, ValueError) as exc:
            rows.append({"id": manifest.parent.name, "valid": False, "errors": [str(exc)], "bundle": str(manifest.parent)})
    write_jsonl(args.output, rows)
    summary = {"checked": len(rows), "valid": sum(row["valid"] for row in rows), "output": str(args.output)}
    print(json.dumps(summary, indent=2))
    if any(not row["valid"] for row in rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
