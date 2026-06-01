#!/usr/bin/env python3
"""Merge LLM outputs for the legacy no-validation GEM pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def materialize_stage2(base: dict[str, dict[str, Any]], output: dict[str, Any]) -> dict[str, Any]:
    parsed = output["json_response"]
    row = dict(base[output["id"]])
    for key in ["multi_step", "summary", "domain", "platform", "task", "workflow", "tools"]:
        if key in parsed:
            row[key] = parsed[key]
    return row


def materialize_stage3(base: dict[str, dict[str, Any]], output: dict[str, Any]) -> dict[str, Any]:
    parsed = output["json_response"]
    row = dict(base[output["id"]])
    row["messages"] = parsed["messages"]
    row["refinement_patterns"] = parsed.get("refinement_patterns", [])
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--llm-output", type=Path, required=True)
    parser.add_argument("--stage", choices=["stage2", "stage3"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_by_id = {row["id"]: row for row in load_jsonl(args.base)}
    rows = []
    for output in load_jsonl(args.llm_output):
        if not output.get("ok"):
            continue
        if output["id"] not in base_by_id:
            continue
        if args.stage == "stage2":
            rows.append(materialize_stage2(base_by_id, output))
        else:
            rows.append(materialize_stage3(base_by_id, output))
    write_jsonl(args.output, rows)
    print(json.dumps({"stage": args.stage, "written": len(rows), "output": str(args.output)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
