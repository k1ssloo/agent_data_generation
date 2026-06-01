#!/usr/bin/env python3
"""Convert validated GEM trajectories into SFT JSONL records.

The output keeps the OpenAI-style message structure and tools. This is a common
intermediate format that can later be adapted to LLaMA-Factory, TRL, Axolotl, or
a model-specific tool-call template.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def load_valid_ids(paths: list[Path]) -> set[str] | None:
    if not paths:
        return None
    valid_ids = set()
    for index, path in enumerate(paths):
        current = {row["id"] for row in load_jsonl(path) if row.get("valid")}
        if index == 0:
            valid_ids = current
        else:
            valid_ids &= current
    return valid_ids


def to_sft_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "source_text": row["text"],
        "tools": row["tools"],
        "messages": row["messages"],
        "metadata": {
            "summary": row.get("summary"),
            "domain": row.get("domain"),
            "platform": row.get("platform"),
            "task": row.get("task"),
            "workflow": row.get("workflow"),
            "environment": row.get("environment"),
            "execution_validation": row.get("execution_validation"),
            "refinement_patterns": row.get("refinement_patterns", []),
            "refinement_summary": row.get("refinement_summary"),
            "complexity_changes": row.get("complexity_changes", []),
            "stage4_complexity": row.get("stage4_complexity"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, default=PROJECT_ROOT / "outputs/toy/trajectories.jsonl")
    parser.add_argument("--validation", type=Path, default=PROJECT_ROOT / "outputs/toy/validation.jsonl")
    parser.add_argument("--extra-validation", action="append", type=Path, default=[], help="Additional validation JSONL files; only ids valid in every file are exported.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/sft/sft_openai_messages.jsonl")
    args = parser.parse_args()

    validation_paths = []
    if args.validation:
        validation_paths.append(args.validation)
    validation_paths.extend(args.extra_validation)
    valid_ids = load_valid_ids(validation_paths)
    rows = []
    skipped = 0
    for row in load_jsonl(args.trajectories):
        if valid_ids is not None and row["id"] not in valid_ids:
            skipped += 1
            continue
        if row.get("missing_tool_requirements") or "messages" not in row or "tools" not in row:
            skipped += 1
            continue
        rows.append(to_sft_record(row))
    write_jsonl(args.output, rows)
    print(json.dumps({"written": len(rows), "skipped": skipped, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
