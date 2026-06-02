#!/usr/bin/env python3
"""Merge parsed LLM stage outputs back into GEM JSONL artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def message_stats(messages: list[dict[str, Any]]) -> dict[str, Any]:
    tool_calls = [
        message.get("tool_call", {}).get("name", "")
        for message in messages
        if message.get("role") == "assistant" and "tool_call" in message
    ]
    return {
        "messages": len(messages),
        "user_turns": sum(1 for message in messages if message.get("role") == "user"),
        "assistant_natural_turns": sum(
            1 for message in messages if message.get("role") == "assistant" and "content" in message
        ),
        "tool_calls": len(tool_calls),
        "unique_called_tools": len(set(tool_calls)),
    }


def complexity_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        key: after.get(key, 0) - before.get(key, 0)
        for key in ("messages", "user_turns", "assistant_natural_turns", "tool_calls", "unique_called_tools")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True, help="Base JSONL records keyed by id.")
    parser.add_argument("--llm-output", type=Path, required=True, help="Output from execute_llm_requests.py.")
    parser.add_argument("--stage", choices=["stage1", "stage2", "stage3", "stage4"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_by_id = {row["id"]: row for row in load_jsonl(args.base)}
    rows = []
    for output in load_jsonl(args.llm_output):
        if not output.get("ok"):
            continue
        row = dict(base_by_id[output["id"]])
        parsed = output["json_response"]
        if args.stage == "stage1":
            if parsed.get("multi_step"):
                row.update(parsed)
                rows.append(row)
        elif args.stage == "stage2":
            if parsed.get("missing_tool_requirements") and not all(key in parsed for key in ("workflow", "tools", "environment")):
                row["stage2_status"] = "missing_tool"
                row["missing_tool_requirements"] = parsed["missing_tool_requirements"]
                if "workflow" in parsed:
                    row["workflow"] = parsed["workflow"]
                if "rationale" in parsed:
                    row["missing_tool_rationale"] = parsed["rationale"]
                rows.append(row)
                continue
            row["workflow"] = parsed["workflow"]
            row["tools"] = parsed["tools"]
            if "environment" in parsed:
                row["environment"] = parsed["environment"]
            if "missing_tool_requirements" in parsed:
                row["missing_tool_requirements"] = parsed["missing_tool_requirements"]
            row["stage2_status"] = "ready"
            rows.append(row)
        elif args.stage == "stage3":
            for key in ("workflow", "tools", "environment"):
                if key in parsed:
                    row[key] = parsed[key]
            row["messages"] = parsed["messages"]
            if "refinement_patterns" in parsed:
                row["refinement_patterns"] = parsed["refinement_patterns"]
            rows.append(row)
        else:
            if not isinstance(parsed.get("messages"), list):
                continue
            before = message_stats(row.get("messages", []))
            after = message_stats(parsed["messages"])
            row["messages"] = parsed["messages"]
            row["refinement_patterns"] = parsed.get("refinement_patterns", [])
            row["refinement_summary"] = parsed.get("refinement_summary")
            row["complexity_changes"] = parsed.get("complexity_changes", [])
            row["stage4_complexity"] = {
                "before": before,
                "after": after,
                "delta": complexity_delta(before, after),
            }
            ignored = [key for key in ("workflow", "tools", "environment") if key in parsed]
            if ignored:
                row["stage4_ignored_fields"] = ignored
            rows.append(row)
    write_jsonl(args.output, rows)
    print(json.dumps({"written": len(rows), "output": str(args.output)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
