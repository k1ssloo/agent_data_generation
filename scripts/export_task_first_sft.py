#!/usr/bin/env python3
"""Export successful hidden rollouts without private task data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def public_causal_metrics(
    metrics: dict[str, Any],
    decision_metrics: dict[str, Any] | None = None,
) -> dict[str, int | float]:
    """Keep aggregate quality signals without exporting verifier internals."""
    decision_metrics = decision_metrics or {}
    return {
        "steps": int(metrics.get("steps", 0)),
        "max_delayed_handle_distance": int(metrics.get("max_delayed_handle_distance", 0)),
        "handle_chain_depth": int(metrics.get("handle_chain_depth", 0)),
        "semantic_recovery_count": len(metrics.get("semantic_recoveries", [])),
        "observation_dependent_branch_count": int(
            metrics.get("observation_dependent_branch_count", 0)
        ),
        "state_dependent_transition_count": int(
            metrics.get(
                "state_dependent_transition_count",
                metrics.get("observation_dependent_branch_count", 0),
            )
        ),
        "meaningful_planning_decision_count": int(
            decision_metrics.get(
                "meaningful_planning_decision_count",
                metrics.get("meaningful_planning_decision_count", 0),
            )
        ),
        "decision_entropy_bits": float(
            decision_metrics.get(
                "decision_entropy_bits", metrics.get("decision_entropy_bits", 0.0)
            )
        ),
        "distinct_transition_count": int(metrics.get("distinct_selected_branches", 0)),
        "contract_goal_evidence_coverage": float(
            metrics.get(
                "contract_goal_evidence_coverage",
                metrics.get("goal_evidence_coverage", 0.0),
            )
        ),
        # Compatibility alias. This is contract coverage, not an
        # instruction-to-contract semantic alignment score.
        "goal_evidence_coverage": float(metrics.get("goal_evidence_coverage", 0.0)),
        "missing_provenance_count": len(metrics.get("missing_provenance", [])),
        "invariant_violation_count": len(metrics.get("invariant_violations", [])),
    }


def build_sft_row(value: dict[str, Any]) -> dict[str, Any] | None:
    episode = value.get("episode", value)
    validation = value.get("validation", {})
    if episode.get("status") != "goal_satisfied" or validation.get("valid") is not True:
        return None
    strict_rollout = value.get("generation_mode") in {
        "model_policy_hidden_environment_rollout",
        "subagent_policy_hidden_environment_rollout",
    }
    if strict_rollout and (
        value.get("goal_alignment", {}).get("valid") is not True
        or value.get("counterfactual_validation", {}).get("valid") is not True
        or value.get("adaptive") is not True
    ):
        return None
    messages = episode.get("messages")
    tools = episode.get("public_tools")
    if not isinstance(messages, list) or not isinstance(tools, list):
        return None
    adaptive_profile = value.get("adaptive_profile", {})
    adaptive_profiles = (
        [
            str(item)
            for item in adaptive_profile.get("profiles", [])
            if isinstance(item, str)
        ]
        if isinstance(adaptive_profile, dict)
        else []
    )
    if strict_rollout and not adaptive_profiles:
        metrics = validation.get("metrics", {})
        if metrics.get("semantic_recoveries"):
            adaptive_profiles = ["planning_with_semantic_recovery"]
    metadata = {
        "generation_mode": value.get(
            "generation_mode", "hidden_environment_rollout"
        ),
        "causal_metrics": public_causal_metrics(
            validation.get("metrics", {}),
            value.get("counterfactual_validation", {}).get(
                "decision_metrics", {}
            ),
        ),
        "validation_scope": (
            "instruction_aligned_adaptive_hidden_environment"
            if strict_rollout
            else "declared_executable_contract_only"
        ),
        "instruction_goal_coverage": (
            float(
                value.get("goal_alignment", {})
                .get("metrics", {})
                .get("instruction_goal_coverage", 0.0)
            )
            if strict_rollout
            else "not_evaluated_requires_semantic_audit"
        ),
        "adaptive_profiles": adaptive_profiles,
    }
    for key in (
        "source_id",
        "source_sha256",
        "semantic_episode_id",
        "recursive_generation",
        "recursive_operators",
        "renderer_seed",
        "operator_family",
    ):
        if key in value:
            metadata[key] = value[key]
    metadata.setdefault("semantic_episode_id", episode["task_id"])
    counterfactual = value.get("counterfactual_validation", {})
    metadata["counterfactual_count"] = int(
        counterfactual.get("counterfactual_count", 0)
    )
    identifiability = value.get("tool_identifiability")
    if isinstance(identifiability, dict) and identifiability.get("valid") is True:
        metadata["tool_identifiability"] = identifiability.get("metrics", {})
    return {
        "id": episode["task_id"],
        "tools": tools,
        "messages": messages,
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="Episode or rollout result JSON files.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    skipped = 0
    for path in args.input:
        value = load_json(path)
        row = build_sft_row(value)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
    write_jsonl(args.output, rows)
    print(json.dumps({"written": len(rows), "skipped": skipped, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
