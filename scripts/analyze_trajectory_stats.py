#!/usr/bin/env python3
"""Summarize trajectory length, user turns, and canonical tool reuse."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOOL_BANK = PROJECT_ROOT / "config/tool_bank.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_tool_bank_names(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return {tool.get("function", {}).get("name", "") for tool in data.get("tools", [])}


def is_failure_response(content: Any) -> bool:
    lowered = json.dumps(content, ensure_ascii=False).lower()
    return any(marker in lowered for marker in ("error", "failed", "failure", "not found", "not available", "cannot"))


def is_verification_tool(name: str) -> bool:
    tokens = name.split("_")
    return (
        name.startswith(("verify_", "read_", "get_", "list_", "search_", "poll_"))
        or "verify" in tokens
        or "status" in tokens
    )


def mean(values: list[int | float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def percentile(values: list[int], pct: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * pct)
    return ordered[index]


def analyze_row(row: dict[str, Any], canonical_names: set[str]) -> dict[str, Any]:
    messages = row.get("messages") or []
    selected_tools = [tool.get("function", {}).get("name", "") for tool in row.get("tools", [])]
    tool_calls = []
    failure_count = 0
    for message in messages:
        if message.get("role") == "assistant" and "tool_call" in message:
            tool_calls.append(message["tool_call"].get("name", ""))
        elif message.get("role") == "tool" and is_failure_response(message.get("content")):
            failure_count += 1
    user_turns = sum(1 for message in messages if message.get("role") == "user")
    assistant_natural_turns = sum(1 for message in messages if message.get("role") == "assistant" and "content" in message)
    noncanonical_selected = sorted({name for name in selected_tools if canonical_names and name not in canonical_names})
    noncanonical_called = sorted({name for name in tool_calls if canonical_names and name not in canonical_names})
    final_tool = tool_calls[-1] if tool_calls else None
    return {
        "id": row.get("id"),
        "messages": len(messages),
        "user_turns": user_turns,
        "assistant_natural_turns": assistant_natural_turns,
        "tool_calls": len(tool_calls),
        "unique_called_tools": len(set(tool_calls)),
        "selected_tools": selected_tools,
        "tool_sequence": tool_calls,
        "failure_tool_responses": failure_count,
        "final_tool": final_tool,
        "final_tool_is_verification": bool(final_tool and is_verification_tool(final_tool)),
        "noncanonical_selected_tools": noncanonical_selected,
        "noncanonical_called_tools": noncanonical_called,
        "missing_tool_requirements": row.get("missing_tool_requirements", []),
    }


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    tool_call_counts = [item["tool_calls"] for item in items]
    user_turn_counts = [item["user_turns"] for item in items]
    message_counts = [item["messages"] for item in items]
    all_called = [name for item in items for name in item["tool_sequence"]]
    noncanonical_items = [
        item["id"]
        for item in items
        if item["noncanonical_selected_tools"] or item["noncanonical_called_tools"]
    ]
    missing_tool_items = [item["id"] for item in items if item["missing_tool_requirements"]]
    return {
        "items": len(items),
        "avg_messages": mean(message_counts),
        "avg_tool_calls": mean(tool_call_counts),
        "median_tool_calls": percentile(tool_call_counts, 0.5),
        "p90_tool_calls": percentile(tool_call_counts, 0.9),
        "avg_user_turns": mean(user_turn_counts),
        "max_user_turns": max(user_turn_counts) if user_turn_counts else 0,
        "final_verification_rate": round(
            sum(1 for item in items if item["final_tool_is_verification"]) / len(items),
            3,
        )
        if items
        else 0,
        "items_with_failures": sum(1 for item in items if item["failure_tool_responses"]),
        "unique_called_tools": len(set(all_called)),
        "called_tool_frequencies": dict(Counter(all_called).most_common()),
        "noncanonical_items": noncanonical_items,
        "missing_tool_items": missing_tool_items,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tool-bank", type=Path, default=DEFAULT_TOOL_BANK)
    args = parser.parse_args()

    canonical_names = load_tool_bank_names(args.tool_bank)
    items = [analyze_row(row, canonical_names) for row in load_jsonl(args.input)]
    result = {"summary": summarize(items), "items": items}
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
