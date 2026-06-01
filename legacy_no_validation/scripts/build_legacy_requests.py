#!/usr/bin/env python3
"""Build JSONL requests for the legacy no-validation GEM pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / "prompts"


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


def render_template(template: str, variables: dict[str, str]) -> str:
    rendered = template
    for key, value in variables.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    return rendered


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_stage2(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    template = (PROMPT_DIR / "stage2_workflow_tools_legacy.txt").read_text(encoding="utf-8")
    requests = []
    for row in rows:
        requests.append(
            {
                "id": row["id"],
                "stage": "stage2_workflow_tools_legacy",
                "messages": [
                    {
                        "role": "user",
                        "content": render_template(template, {"text": row["text"]}),
                    }
                ],
            }
        )
    return requests


def build_stage3(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    template = (PROMPT_DIR / "stage3_trajectory_legacy.txt").read_text(encoding="utf-8")
    requests = []
    for row in rows:
        requests.append(
            {
                "id": row["id"],
                "stage": "stage3_trajectory_legacy",
                "messages": [
                    {
                        "role": "user",
                        "content": render_template(
                            template,
                            {
                                "text": row["text"],
                                "workflow_json": compact_json(row["workflow"]),
                                "tools_json": compact_json(row["tools"]),
                            },
                        ),
                    }
                ],
            }
        )
    return requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["stage2", "stage3"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if args.stage == "stage2":
        requests = build_stage2(rows)
    else:
        requests = build_stage3(rows)
    write_jsonl(args.output, requests)
    print(json.dumps({"stage": args.stage, "requests": len(requests), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
