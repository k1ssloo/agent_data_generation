#!/usr/bin/env python3
"""Run oracle, causal, API-rendering, and action-ablation checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_validation import (
    evaluate_action_ablation,
    validate_episode,
    validate_tool_identifiability,
)
from causal_validation.intervention import evaluate_counterfactuals
from rollout import run_reference_plan
from runtime.tool_renderer import render_alternate_api
from task_factory import load_task_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--renderer-seed", default="heldout_api_v1")
    parser.add_argument("--min-necessary-action-ratio", type=float, default=0.6)
    args = parser.parse_args()

    bundle = load_task_bundle(args.bundle)
    baseline_report = run_reference_plan(bundle)
    baseline_validation = validate_episode(bundle, baseline_report)
    rendered = render_alternate_api(bundle, seed=args.renderer_seed)
    rendered_report = run_reference_plan(rendered)
    rendered_validation = validate_episode(rendered, rendered_report)
    rendered_identifiability = validate_tool_identifiability(rendered)
    ablation = evaluate_action_ablation(bundle)
    counterfactual = evaluate_counterfactuals(bundle)
    errors = []
    if not baseline_validation["valid"]:
        errors.append("baseline oracle/causal validation failed")
    if not rendered_validation["valid"]:
        errors.append("alternate API rendering changed task validity")
    if not rendered_identifiability["valid"]:
        errors.append("alternate API rendering is not publicly identifiable")
    if ablation["necessary_action_ratio"] < args.min_necessary_action_ratio:
        errors.append("necessary action ratio is below threshold")
    if bundle.reference_plan.get("counterfactuals") and not counterfactual["valid"]:
        errors.append("counterfactual strategy adaptation failed")
    result = {
        "task_id": bundle.task_id,
        "valid": not errors,
        "errors": errors,
        "baseline_validation": baseline_validation,
        "rendered_validation": rendered_validation,
        "rendered_identifiability": rendered_identifiability,
        "rendered_public_tools": [tool["function"]["name"] for tool in rendered_report["public_tools"]],
        "ablation": ablation,
        "counterfactual_validation": counterfactual,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"task_id": bundle.task_id, "valid": not errors, "errors": errors, "necessary_action_ratio": ablation["necessary_action_ratio"], "counterfactual_valid": counterfactual["valid"], "output": str(args.output)}, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
