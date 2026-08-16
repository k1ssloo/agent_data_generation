#!/usr/bin/env python3
"""Rank admitted WikiHow seeds for strict adaptive environment generation."""

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

from task_factory import load_task_bundle


TOPOLOGIES = {
    "failure_diagnosis_recovery": {
        "source_terms": (
            "error",
            "fail",
            "problem",
            "unable",
            "retry",
            "reset",
            "repair",
            "remove",
            "prevent",
        ),
        "observe_terms": ("status", "inspect", "read", "verify", "list", "get"),
        "act_terms": (
            "update",
            "set",
            "delete",
            "reset",
            "start",
            "install",
            "format",
        ),
    },
    "closed_loop_feedback": {
        "source_terms": (
            "monitor",
            "usage",
            "power",
            "test",
            "calibrat",
            "diagnos",
            "check",
            "measure",
            "compare",
        ),
        "observe_terms": (
            "status",
            "inspect",
            "read",
            "verify",
            "list",
            "get",
            "poll",
            "search",
        ),
        "act_terms": ("update", "set", "delete", "start", "install", "configure"),
    },
    "temporal_revision_provenance": {
        "source_terms": (
            "file",
            "version",
            "revision",
            "backup",
            "restore",
            "download",
            "upload",
            "transfer",
            "attachment",
            "export",
        ),
        "observe_terms": ("locate", "list", "get", "search", "inspect", "read"),
        "act_terms": ("download", "upload", "transfer", "export", "send", "restore"),
    },
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def term_hits(text: str, terms: tuple[str, ...]) -> list[str]:
    lowered = text.casefold()
    return sorted({term for term in terms if term in lowered})


def capability_is_mutating(capability: dict[str, Any]) -> bool:
    return any(
        branch.get("effects") or branch.get("writes")
        for branch in capability.get("branches", [])
        if isinstance(branch, dict)
    )


def audit_bundle(path: Path, source_text: str) -> dict[str, Any]:
    bundle = load_task_bundle(path)
    capabilities = bundle.environment.get("capabilities", {})
    observations: list[str] = []
    actions: list[str] = []
    for tool in bundle.tools:
        surface = f"{tool.get('name', '')} {tool.get('description', '')}".casefold()
        capability = capabilities.get(tool.get("capability_id"), {})
        target = actions if capability_is_mutating(capability) else observations
        target.append(surface)
    observation_surface = " ".join(observations)
    action_surface = " ".join(actions)
    final_observation = any(
        token in observation_surface for token in ("outcome", "status", "verify")
    )

    topology_rows = []
    for topology, spec in TOPOLOGIES.items():
        source_hits = term_hits(source_text, spec["source_terms"])
        observation_hits = term_hits(observation_surface, spec["observe_terms"])
        action_hits = term_hits(action_surface, spec["act_terms"])
        missing = []
        if not source_hits:
            missing.append("source_semantic_anchor")
        if not observation_hits:
            missing.append("observable_decision_operand")
        if not action_hits:
            missing.append("state_changing_action")
        if not final_observation:
            missing.append("final_observation_surface")
        topology_rows.append(
            {
                "topology": topology,
                "ready_for_strict_generation": not missing,
                "score": (
                    min(len(source_hits), 3)
                    + int(bool(observation_hits))
                    + int(bool(action_hits))
                    + int(final_observation)
                ),
                "source_hits": source_hits,
                "observation_hits": observation_hits,
                "action_hits": action_hits,
                "missing": missing,
            }
        )
    ranked = sorted(
        topology_rows,
        key=lambda item: (
            item["ready_for_strict_generation"],
            item["score"],
            item["topology"],
        ),
        reverse=True,
    )
    ready = [item for item in ranked if item["ready_for_strict_generation"]]
    return {
        "task_id": bundle.task_id,
        "source_id": bundle.manifest.get("source_id", bundle.task_id.split("__", 1)[0]),
        "recommended_operator": ready[0]["topology"] if ready else None,
        "ready_operator_count": len(ready),
        "topologies": ranked,
        "existing_counterfactual_axis_count": len(
            bundle.contract.get("counterfactual_axes", [])
        ),
        "note": (
            "Readiness selects a topology for fresh strict generation; it does not "
            "promote this base environment to adaptive data."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    source = {str(row["id"]): str(row.get("text", "")) for row in load_jsonl(args.source_jsonl)}
    rows = []
    for manifest in sorted(args.input_dir.rglob("manifest.json")):
        bundle = load_task_bundle(manifest)
        source_id = str(bundle.manifest.get("source_id") or bundle.task_id.split("__", 1)[0])
        rows.append(audit_bundle(manifest.parent, source.get(source_id, "")))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    distribution = Counter(
        row["recommended_operator"] or "not_ready" for row in rows
    )
    summary = {
        "considered": len(rows),
        "ready_for_strict_generation": sum(
            row["recommended_operator"] is not None for row in rows
        ),
        "not_ready": sum(row["recommended_operator"] is None for row in rows),
        "recommended_operator_distribution": dict(sorted(distribution.items())),
        "existing_adaptive_environments": sum(
            row["existing_counterfactual_axis_count"] > 0 for row in rows
        ),
        "metric_scope": (
            "Topology readiness for fresh environment generation, not adaptive "
            "episode acceptance."
        ),
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
