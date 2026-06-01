#!/usr/bin/env python3
"""Convert legacy GEM trajectories to SFT JSONL without mandatory validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def load_valid_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    valid_ids: set[str] = set()
    for row in load_jsonl(path):
        if row.get("valid"):
            valid_ids.add(row["id"])
    return valid_ids


def to_sft_record(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    return {
        "id": row["id"],
        "source_text": row.get("text", row.get("source_text")),
        "tools": row["tools"],
        "messages": row["messages"],
        "metadata": {
            "summary": row.get("summary", metadata.get("summary")),
            "domain": row.get("domain", metadata.get("domain")),
            "platform": row.get("platform", metadata.get("platform")),
            "task": row.get("task", metadata.get("task")),
            "workflow": row.get("workflow", metadata.get("workflow")),
            "refinement_patterns": row.get(
                "refinement_patterns",
                metadata.get("refinement_patterns", []),
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=PROJECT_ROOT / "legacy_no_validation/fixtures/qwen32b_stage3_smoke_legacy.jsonl",
    )
    parser.add_argument("--validation", type=Path, help="Optional old schema-validation JSONL filter.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "legacy_no_validation/outputs/qwen32b_sft_openai_messages_smoke.jsonl",
    )
    args = parser.parse_args()

    valid_ids = load_valid_ids(args.validation)
    sft_rows = []
    for row in load_jsonl(args.trajectories):
        if valid_ids is not None and row["id"] not in valid_ids:
            continue
        sft_rows.append(to_sft_record(row))
    write_jsonl(args.output, sft_rows)
    print(json.dumps({"written": len(sft_rows), "output": str(args.output), "validation": args.validation is not None}, indent=2))


if __name__ == "__main__":
    main()
