#!/usr/bin/env python3
"""Validate grounded WikiHow seed model outputs and emit approved contract inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task_factory.wikihow_seed import validate_wikihow_seed


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    source = {row["id"]: row for row in load_jsonl(args.source)}
    accepted = []
    rejected = []
    for row in load_jsonl(args.model_output):
        task_id = row.get("id")
        source_row = source.get(task_id)
        seed = row.get("json_response")
        assigned = row.get("metadata", {}).get("assigned_operator")
        if not row.get("ok") or not isinstance(source_row, dict) or not isinstance(seed, dict):
            rejected.append({"id": task_id, "errors": [row.get("error") or "missing source or seed"]})
            continue
        errors = validate_wikihow_seed(
            seed,
            str(source_row.get("text", "")),
            assigned_operator=assigned,
            source_id=task_id,
        )
        if errors:
            rejected.append({"id": task_id, "errors": errors})
            continue
        accepted.append(
            {
                "id": task_id,
                "seed": seed,
                "metadata": {
                    "source_dataset": source_row.get("metadata", {}).get("source_dataset"),
                    "source_index": source_row.get("metadata", {}).get("source_index"),
                    "assigned_operator": assigned,
                },
            }
        )
    write_jsonl(args.output, accepted)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps({"accepted": len(accepted), "rejected": rejected}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"accepted": len(accepted), "rejected": len(rejected), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
