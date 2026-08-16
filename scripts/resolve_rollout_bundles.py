#!/usr/bin/env python3
"""Resolve an exported SFT corpus back to its exact audited task bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_validation import validate_episode, validate_tool_identifiability
from rollout import run_reference_plan
from scripts.export_task_first_sft import build_sft_row
from task_factory import load_task_bundle


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number} must contain an object")
        rows.append(value)
    return rows


def public_trace(row: dict[str, Any]) -> str:
    value = {"messages": row.get("messages"), "tools": row.get("tools")}
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audits: dict[str, dict[str, Any]] = {}
    for path in sorted(args.rollout_dir.glob("*.json")):
        audit = load_json(path)
        task_id = str(audit.get("task_id", ""))
        if not task_id:
            raise SystemExit(f"rollout audit has no task_id: {path}")
        if task_id in audits:
            raise SystemExit(f"duplicate rollout audit for {task_id!r}")
        audits[task_id] = audit

    resolved = []
    seen: set[str] = set()
    for row in load_jsonl(args.sft):
        metadata = row.get("metadata", {})
        task_id = str(metadata.get("semantic_episode_id", ""))
        if not task_id or task_id in seen:
            raise SystemExit(f"missing or duplicate semantic episode ID: {task_id!r}")
        seen.add(task_id)
        audit = audits.get(task_id)
        if audit is None or not isinstance(audit.get("bundle"), str):
            raise SystemExit(f"no exact bundle audit for {task_id!r}")
        bundle = load_task_bundle(Path(audit["bundle"]))
        if bundle.task_id != task_id:
            raise SystemExit(f"audit bundle ID mismatch for {task_id!r}")
        episode = run_reference_plan(bundle)
        validation = validate_episode(bundle, episode)
        identifiability = validate_tool_identifiability(bundle)
        rebuilt = build_sft_row(
            {
                "episode": episode,
                "validation": validation,
                "semantic_episode_id": task_id,
            }
        )
        if not validation["valid"] or not identifiability["valid"] or rebuilt is None:
            raise SystemExit(f"audited bundle no longer passes export gates: {task_id}")
        if public_trace(row) != public_trace(rebuilt):
            raise SystemExit(f"SFT public trace does not match audited bundle: {task_id}")
        resolved.append(
            {
                "id": task_id,
                "valid": True,
                "bundle": str(bundle.root.resolve()),
                "source": "exact_rollout_audit",
            }
        )

    if set(audits) != seen:
        extra = sorted(set(audits) - seen)
        raise SystemExit(f"rollout directory has audits absent from SFT: {extra[:5]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
            for item in resolved
        ),
        encoding="utf-8",
    )
    print(json.dumps({"resolved": len(resolved), "output": str(args.output)}))


if __name__ == "__main__":
    main()
