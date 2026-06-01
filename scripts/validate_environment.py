#!/usr/bin/env python3
"""Validate LLM-generated executable environment specs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from executable_environment import validate_environment_spec


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


def tool_names(row: dict[str, Any]) -> set[str]:
    return {tool.get("function", {}).get("name", "") for tool in row.get("tools", [])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = []
    for row in load_jsonl(args.input):
        errors = validate_environment_spec(row.get("environment", {}), tool_names=tool_names(row))
        results.append({"id": row.get("id"), "valid": not errors, "errors": errors})
    write_jsonl(args.output, results)
    print(json.dumps({"checked": len(results), "valid": sum(item["valid"] for item in results), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
