#!/usr/bin/env python3
"""Assemble validated task-first SFT shards and report diversity coverage."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain an object")
        rows.append(value)
    return rows


def validate_row(row: dict[str, Any]) -> list[str]:
    errors = []
    if not isinstance(row.get("id"), str) or not row.get("id"):
        errors.append("id must be a non-empty string")
    tools = row.get("tools")
    if not isinstance(tools, list) or not tools:
        errors.append("tools must be a non-empty list")
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        errors.append("messages must be a non-empty list")
        return errors
    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    if len(roles) != len(messages) or roles[0] != "system" or "user" not in roles:
        errors.append("messages must be object-valued and begin with system plus a user turn")
    call_ids = []
    result_ids = []
    for message in messages:
        if message.get("role") == "assistant":
            calls = message.get("tool_calls", [])
            if not isinstance(calls, list):
                errors.append("assistant tool_calls must be a list")
                continue
            call_ids.extend(
                call.get("id") for call in calls if isinstance(call, dict)
            )
        elif message.get("role") == "tool":
            result_ids.append(message.get("tool_call_id"))
    if not call_ids or any(not isinstance(item, str) or not item for item in call_ids):
        errors.append("assistant tool calls must have non-empty IDs")
    if call_ids != result_ids:
        errors.append("assistant tool calls and tool results must be ordered and balanced")
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("metadata must be an object")
    return errors


def validate_adaptive_row(row: dict[str, Any]) -> list[str]:
    errors = validate_row(row)
    metadata = row.get("metadata", {})
    metrics = metadata.get("causal_metrics", {}) if isinstance(metadata, dict) else {}
    if metadata.get("validation_scope") != (
        "instruction_aligned_adaptive_hidden_environment"
    ):
        errors.append("adaptive row lacks strict instruction-aligned validation scope")
    if float(metadata.get("instruction_goal_coverage", 0.0)) != 1.0:
        errors.append("adaptive row must have complete instruction goal coverage")
    if int(metrics.get("meaningful_planning_decision_count", 0)) < 1:
        errors.append("adaptive row must contain a meaningful planning decision")
    if float(metrics.get("decision_entropy_bits", 0.0)) <= 0.0:
        errors.append("adaptive row must have positive decision entropy")
    if int(metrics.get("missing_provenance_count", 0)) != 0:
        errors.append("adaptive row contains missing argument provenance")
    if int(metrics.get("invariant_violation_count", 0)) != 0:
        errors.append("adaptive row contains invariant violations")
    if float(metrics.get("contract_goal_evidence_coverage", 0.0)) != 1.0:
        errors.append("adaptive row lacks complete final goal evidence")
    profiles = metadata.get("adaptive_profiles", [])
    allowed_profiles = {
        "planning_with_semantic_recovery",
        "planning_with_closed_loop_control",
        "planning_with_temporal_provenance",
    }
    if (
        not isinstance(profiles, list)
        or not profiles
        or any(profile not in allowed_profiles for profile in profiles)
    ):
        errors.append("adaptive row lacks a verified adaptive evidence profile")
    identifiability = metadata.get("tool_identifiability")
    if not isinstance(identifiability, dict):
        errors.append("adaptive row lacks tool-identifiability evidence")
    elif (
        int(identifiability.get("described_tool_count", -1))
        != int(identifiability.get("tool_count", 0))
        or bool(identifiability.get("indistinguishable_groups"))
    ):
        errors.append("adaptive row contains publicly unidentifiable tools")
    return errors


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--require-adaptive",
        action="store_true",
        help="Reject rows that do not pass the strict adaptive corpus gate.",
    )
    args = parser.parse_args()

    rows = []
    seen_ids: set[str] = set()
    for path in args.input:
        for row in load_jsonl(path):
            errors = (
                validate_adaptive_row(row)
                if args.require_adaptive
                else validate_row(row)
            )
            if errors:
                raise SystemExit(f"{path}: row {row.get('id')!r}: {'; '.join(errors)}")
            if row["id"] in seen_ids:
                raise SystemExit(f"duplicate trajectory id: {row['id']}")
            seen_ids.add(row["id"])
            rows.append(row)

    metrics = [row["metadata"].get("causal_metrics", {}) for row in rows]
    steps = [int(item.get("steps", 0)) for item in metrics]
    chain_depths = [int(item.get("handle_chain_depth", 0)) for item in metrics]
    delayed = [int(item.get("max_delayed_handle_distance", 0)) for item in metrics]
    decision_entropy = [float(item.get("decision_entropy_bits", 0.0)) for item in metrics]
    source_identities = {
        row["metadata"].get("source_id")
        or row["metadata"].get("source_sha256")
        for row in rows
        if row["metadata"].get("source_id")
        or row["metadata"].get("source_sha256")
    }
    report = {
        "rendered_training_rows": len(rows),
        "unique_source_tasks": len(source_identities),
        "unique_semantic_episodes": len(
            {
                row["metadata"].get("semantic_episode_id", row["id"])
                for row in rows
            }
        ),
        "recursive_descendants": len(
            {
                row["metadata"].get("semantic_episode_id", row["id"])
                for row in rows
                if int(row["metadata"].get("recursive_generation", 0)) > 0
            }
        ),
        # Compatibility fields for older report consumers.
        "rows": len(rows),
        "distinct_source_hashes": len(source_identities),
        "renderer_distribution": dict(
            sorted(
                Counter(
                    "canonical"
                    if row["metadata"].get("renderer_seed") == "canonical"
                    else "renamed_api"
                    for row in rows
                ).items()
            )
        ),
        "operator_family_distribution": dict(
            sorted(Counter(row["metadata"].get("operator_family") or "unknown" for row in rows).items())
        ),
        "recursive_operator_distribution": dict(
            sorted(
                Counter(
                    operator
                    for row in rows
                    for operator in row["metadata"].get("recursive_operators", [])
                ).items()
            )
        ),
        "with_counterfactual_strategy": sum(
            int(row["metadata"].get("counterfactual_count", 0)) > 0
            for row in rows
        ),
        "tool_identifiability": {
            "validated_rows": sum(
                isinstance(row["metadata"].get("tool_identifiability"), dict)
                for row in rows
            ),
            "fully_described_rendered_rows": sum(
                row["metadata"].get("renderer_seed") != "canonical"
                and row["metadata"].get("tool_identifiability", {}).get(
                    "described_tool_count"
                )
                == row["metadata"].get("tool_identifiability", {}).get(
                    "tool_count"
                )
                and row["metadata"].get("tool_identifiability", {}).get(
                    "described_parameter_count"
                )
                == row["metadata"].get("tool_identifiability", {}).get(
                    "parameter_count"
                )
                for row in rows
            ),
            "rows_with_indistinguishable_groups": sum(
                bool(
                    row["metadata"].get("tool_identifiability", {}).get(
                        "indistinguishable_groups"
                    )
                )
                for row in rows
            ),
        },
        "complexity": {
            "steps": {"min": min(steps), "mean": round(mean(steps), 3), "max": max(steps)},
            "handle_chain_depth": {
                "min": min(chain_depths),
                "mean": round(mean(chain_depths), 3),
                "max": max(chain_depths),
            },
            "max_delayed_handle_distance": {
                "min": min(delayed),
                "mean": round(mean(delayed), 3),
                "max": max(delayed),
            },
            "with_observation_dependent_branch": sum(
                int(item.get("observation_dependent_branch_count", 0)) > 0
                for item in metrics
            ),
            "with_meaningful_planning_decision": sum(
                int(item.get("meaningful_planning_decision_count", 0)) > 0
                for item in metrics
            ),
            "decision_entropy_bits": {
                "min": min(decision_entropy),
                "mean": round(mean(decision_entropy), 3),
                "max": max(decision_entropy),
            },
            "with_semantic_recovery": sum(
                int(item.get("semantic_recovery_count", 0)) > 0 for item in metrics
            ),
        },
        "inputs": [str(path) for path in args.input],
        "output": str(args.output),
    }
    write_jsonl(args.output, rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
