#!/usr/bin/env python3
"""Probe whether WikiHow workflows can seed executable recursive tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_validation import evaluate_action_ablation, validate_episode
from rollout import run_reference_plan
from task_factory import load_task_bundle
from task_factory.bundle import TaskBundle, validate_bundle
from task_factory.evolve import evolve_once
from task_factory.materialize import materialize_candidate
from task_factory.operators.base import manifest_metadata
from task_factory.wikihow_compiler import WikiHowCompileError, compile_wikihow_row


SIGNALS = {
    "observe": r"\b(open|list|view|check|find|search|select|inspect|look|read|click)\b",
    "mutate": r"\b(create|add|set|save|upload|download|delete|change|update|enter|type|paste|send|install|enable|disable|export|import|format|rename)\b",
    "branch": r"\b(if|otherwise|either|option|depending|unless|when|alternative)\b",
    "failure": r"\b(error|fail|wrong|problem|cannot|retry|undo|recover|fix|troubleshoot)\b",
    "async": r"\b(wait|progress|complete|finish|download|upload|scan|processing|restart|reopen)\b",
    "verify": r"\b(verify|confirm|make sure|should see|check|test|available|appears|display|view)\b",
    "artifact": r"\b(file|document|email|account|map|signature|setting|record|link|photo|video|track|spreadsheet|folder|app)\b",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    signal_counts = {name: 0 for name in SIGNALS}
    core = recursive = 0
    step_counts = []
    for row in rows:
        text = str(row.get("text", "")).lower()
        signals = {name: bool(re.search(pattern, text)) for name, pattern in SIGNALS.items()}
        for name, present in signals.items():
            signal_counts[name] += int(present)
        summary = text.split("summary:", 1)[-1]
        steps = [item for item in re.split(r"\s+\.\s+", summary) if len(item.split()) >= 2]
        step_counts.append(len(steps))
        compilable = (
            len(steps) >= 4
            and signals["observe"]
            and signals["mutate"]
            and signals["artifact"]
            and signals["verify"]
        )
        core += int(compilable)
        recursive += int(compilable and any(signals[name] for name in ("branch", "failure", "async")))
    ordered = sorted(step_counts)
    return {
        "rows": len(rows),
        "signal_counts": signal_counts,
        "step_count": {
            "minimum": min(ordered, default=0),
            "median": ordered[len(ordered) // 2] if ordered else 0,
            "maximum": max(ordered, default=0),
        },
        "core_compilable_heuristic": core,
        "recursive_affordance_heuristic": recursive,
        "heuristic_note": "Signals estimate candidate coverage; executable gates below determine acceptance.",
    }


def materialize_seed(seed: Any, root: Path) -> Path:
    candidate = TaskBundle(
        root=root,
        manifest={
            "bundle_version": "task-bundle-v1",
            "task_id": seed.task_id,
            **seed.manifest_metadata,
        },
        instruction=seed.instruction,
        contract=seed.contract,
        environment=seed.environment,
        bindings=seed.bindings,
        reference_plan=seed.reference_plan,
    )
    errors = validate_bundle(candidate)
    if errors:
        raise WikiHowCompileError("compiled bundle preflight failed: " + "; ".join(errors))
    return materialize_candidate(
        root,
        task_id=seed.task_id,
        contract=seed.contract,
            candidate=seed.candidate,
        lineage={"generation": 0, "root_task_id": seed.task_id, "operators": []},
        manifest_metadata={
            **seed.manifest_metadata,
            "provenance_class": "wikihow_source_grounded",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    source_rows = load_jsonl(args.source)
    trajectory_rows = load_jsonl(args.trajectories)
    if args.limit > 0:
        trajectory_rows = trajectory_rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for row in trajectory_rows:
        result: dict[str, Any] = {"source_id": row.get("id"), "compiled": False, "recursive_accepted": False}
        try:
            seed = compile_wikihow_row(row)
            parent_path = materialize_seed(seed, args.output_dir / "parents")
            parent = load_task_bundle(parent_path)
            parent_report = run_reference_plan(parent)
            parent_relaxed = validate_episode(
                parent,
                parent_report,
                min_delayed_handle_distance=0,
                min_handle_chain_depth=0,
                require_semantic_recovery=False,
            )
            parent_strict = validate_episode(parent, parent_report)
            ablation = evaluate_action_ablation(parent)
            result.update(
                {
                    "compiled": True,
                    "parent_bundle": str(parent_path),
                    "parent_oracle_status": parent_report["status"],
                    "parent_relaxed_valid": parent_relaxed["valid"],
                    "parent_strict_valid": parent_strict["valid"],
                    "parent_strict_errors": parent_strict["errors"],
                    "parent_metrics": parent_relaxed["metrics"],
                    "parent_necessary_action_ratio": ablation["necessary_action_ratio"],
                }
            )
            evolution = evolve_once(parent, "audit_checkpoint_v1")
            result["recursive_accepted"] = evolution.report["accepted"]
            result["recursive_errors"] = evolution.report["errors"]
            result["parent_solution_valid_on_child"] = evolution.report["parent_plan_valid_on_child"]
            result["complexity_delta"] = evolution.report["complexity_delta"]
            result["child_metrics"] = evolution.report["child_validation"]["metrics"]
            if evolution.report["accepted"]:
                child = evolution.product.bundle
                child_path = materialize_candidate(
                    args.output_dir / "children",
                    task_id=child.task_id,
                    contract=child.contract,
                    candidate={
                        "instruction": child.instruction,
                        "environment": child.environment,
                        "bindings": child.bindings,
                        "reference_plan": child.reference_plan,
                    },
                    lineage=child.manifest["lineage"],
                    manifest_metadata={
                        **manifest_metadata(child),
                        "provenance_class": "wikihow_plus_synthetic_extension",
                        "synthetic_extension": {
                            "operator": "audit_checkpoint_v1",
                            "source_claim": "not asserted by the WikiHow article",
                        },
                    },
                )
                result["child_bundle"] = str(child_path)
        except (KeyError, OSError, TypeError, ValueError, WikiHowCompileError) as exc:
            result["errors"] = [str(exc)]
        results.append(result)

    summary = {
        "source_feasibility": source_audit(source_rows),
        "trajectory_probe": {
            "attempted": len(results),
            "compiled": sum(item["compiled"] for item in results),
            "parent_relaxed_valid": sum(item.get("parent_relaxed_valid", False) for item in results),
            "parent_strict_valid": sum(item.get("parent_strict_valid", False) for item in results),
            "recursive_accepted": sum(item["recursive_accepted"] for item in results),
            "old_solution_rejected_by_child": sum(
                item.get("recursive_accepted", False)
                and not item.get("parent_solution_valid_on_child", True)
                for item in results
            ),
        },
        "interpretation": {
            "source_grounded": "Normal workflow, objects, and user-visible goal come from WikiHow and its replayed trajectory.",
            "synthetic_extension": "Recursive hidden state is generated as an explicit operator patch and is not attributed to WikiHow.",
            "acceptance": "A child is accepted only if its new solution passes strict causal validation and the parent solution fails unchanged.",
        },
        "results": results,
    }
    report = args.output_dir / "feasibility_report.json"
    report.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({**summary["trajectory_probe"], "report": str(report)}, indent=2))


if __name__ == "__main__":
    main()
