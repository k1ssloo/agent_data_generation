#!/usr/bin/env python3
"""Select distinct validated WikiHow bundles and export canonical SFT rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
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
from scripts.export_task_first_sft import build_sft_row
from task_factory import load_task_bundle


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def source_id(task_id: str) -> str:
    match = re.search(r"wikihow_computer_\d+", task_id)
    if match:
        return match.group(0)
    return task_id.split("__task_first", 1)[0]


def quality_key(row: dict[str, Any]) -> tuple[float, ...]:
    metrics = row.get("causal_metrics", {})
    return (
        float(metrics.get("observation_dependent_branch_count", 0)),
        float(len(metrics.get("semantic_recoveries", []))),
        float(metrics.get("handle_chain_depth", 0)),
        float(metrics.get("steps", 0)),
        float(row.get("necessary_action_ratio", 0.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--bundle",
        type=Path,
        action="append",
        default=[],
        help="Additional validated bundle to include in candidate selection.",
    )
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        help="Optional source corpus used to export the selected process texts.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    if args.count < 1:
        raise SystemExit("--count must be >= 1")

    started = time.monotonic()
    validation_rows = [
        row
        for validation_path in args.validation
        for row in load_jsonl(validation_path)
    ]
    for bundle_path in args.bundle:
        bundle = load_task_bundle(bundle_path)
        episode = run_reference_plan(bundle)
        validation = validate_episode(bundle, episode)
        identifiability = validate_tool_identifiability(bundle)
        ablation = evaluate_action_ablation(bundle)
        if (
            validation["valid"]
            and identifiability["valid"]
            and ablation["necessary_action_ratio"] >= 0.6
        ):
            validation_rows.append(
                {
                    "id": bundle.task_id,
                    "bundle": str(bundle.root.resolve()),
                    "valid": True,
                    "causal_metrics": validation["metrics"],
                    "necessary_action_ratio": ablation["necessary_action_ratio"],
                }
            )
    candidates = sorted(
        (row for row in validation_rows if row.get("valid") is True),
        key=quality_key,
        reverse=True,
    )
    selected = []
    seen_sources: set[str] = set()
    for row in candidates:
        task_source = source_id(str(row.get("id", "")))
        if not task_source or task_source in seen_sources:
            continue
        seen_sources.add(task_source)
        selected.append(row)
        if len(selected) == args.count:
            break
    if len(selected) != args.count:
        raise SystemExit(
            f"need {args.count} distinct valid sources, found {len(selected)}"
        )

    exported = []
    audits = []
    for row in selected:
        bundle = load_task_bundle(Path(row["bundle"]))
        episode = run_reference_plan(bundle)
        validation = validate_episode(bundle, episode)
        identifiability = validate_tool_identifiability(bundle)
        if not validation["valid"] or not identifiability["valid"]:
            raise SystemExit(
                f"selected bundle {bundle.task_id!r} failed canonical export gate"
            )
        result = {
            "task_id": bundle.task_id,
            "generation_mode": "wikihow_model_workflow_plus_deterministic_environment_patch",
            "episode": episode,
            "validation": validation,
        }
        sft = build_sft_row(result)
        if sft is None:
            raise SystemExit(f"selected bundle {bundle.task_id!r} could not export")
        lineage = bundle.manifest.get("lineage", {})
        sft["metadata"].update(
            {
                "source_id": source_id(bundle.task_id),
                "semantic_episode_id": bundle.task_id,
                "recursive_generation": int(lineage.get("generation", 0)),
                "recursive_operators": list(lineage.get("operators", [])),
                "renderer_seed": "canonical",
                "tool_identifiability": identifiability["metrics"],
                "necessary_action_ratio": row.get("necessary_action_ratio"),
                "validation_scope": "declared_executable_contract_only",
                "instruction_goal_coverage": (
                    "not_evaluated_requires_semantic_audit"
                ),
                "environment_semantic_consistency": (
                    "not_evaluated_requires_declared_domain_invariants"
                ),
            }
        )
        exported.append(sft)
        audit = {
            "source_id": source_id(bundle.task_id),
            "task_id": bundle.task_id,
            "bundle": str(bundle.root.resolve()),
            "episode": episode,
            "validation": validation,
            "tool_identifiability": identifiability,
            "necessary_action_ratio": row.get("necessary_action_ratio"),
        }
        audits.append(audit)
        write_json(args.output_dir / "rollouts" / f"{bundle.task_id}.json", audit)

    output = args.output_dir / "openai_messages.jsonl"
    write_jsonl(output, exported)
    source_text_output = None
    if args.source_jsonl:
        selected_source_ids = {
            row["metadata"]["source_id"] for row in exported
        }
        source_rows = [
            row
            for row in load_jsonl(args.source_jsonl)
            if str(row.get("id", "")) in selected_source_ids
        ]
        if len(source_rows) != len(selected_source_ids):
            raise SystemExit(
                "source corpus does not contain every selected source task"
            )
        source_text_output = args.output_dir / "source_texts.jsonl"
        write_jsonl(source_text_output, source_rows)
    metrics = [row["metadata"]["causal_metrics"] for row in exported]
    summary = {
        "requested": args.count,
        "written": len(exported),
        "unique_source_tasks": len({row["metadata"]["source_id"] for row in exported}),
        "unique_semantic_episodes": len(
            {row["metadata"]["semantic_episode_id"] for row in exported}
        ),
        "rendered_training_rows": len(exported),
        "steps": {
            "min": min(item["steps"] for item in metrics),
            "max": max(item["steps"] for item in metrics),
            "mean": round(sum(item["steps"] for item in metrics) / len(metrics), 3),
        },
        "handle_chain_depth": {
            "min": min(item["handle_chain_depth"] for item in metrics),
            "max": max(item["handle_chain_depth"] for item in metrics),
        },
        "max_delayed_handle_distance": {
            "min": min(item["max_delayed_handle_distance"] for item in metrics),
            "max": max(item["max_delayed_handle_distance"] for item in metrics),
        },
        "with_observation_dependent_branch": sum(
            item["observation_dependent_branch_count"] > 0 for item in metrics
        ),
        "with_semantic_recovery": sum(
            item["semantic_recovery_count"] > 0 for item in metrics
        ),
        "all_contract_goal_evidence_coverage": min(
            item.get(
                "contract_goal_evidence_coverage",
                item["goal_evidence_coverage"],
            )
            for item in metrics
        ),
        "instruction_goal_coverage": "not_evaluated_requires_semantic_audit",
        "all_missing_provenance_count": max(
            item["missing_provenance_count"] for item in metrics
        ),
        "elapsed_sec": round(time.monotonic() - started, 3),
        "output": str(output),
        "source_texts": str(source_text_output) if source_text_output else None,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
