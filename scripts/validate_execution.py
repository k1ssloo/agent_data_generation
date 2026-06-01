#!/usr/bin/env python3
"""Replay trajectory tool calls against executable toy environments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from executable_environment import replay_row


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-final-state", action="store_true")
    args = parser.parse_args()

    results = []
    for row in load_jsonl(args.input):
        result = replay_row(row)
        if not args.include_final_state:
            result.pop("final_state", None)
        results.append(result)
    write_jsonl(args.output, results)
    print(
        json.dumps(
            {
                "checked": len(results),
                "valid": sum(item["valid"] for item in results),
                "output": str(args.output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
