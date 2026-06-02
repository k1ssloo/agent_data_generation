#!/usr/bin/env python3
"""Rewrite tool messages with executable environment replay responses."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from executable_environment import build_environment_for_row, execute_tool, validate_environment_spec


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


def canonicalize_row(row: dict[str, Any]) -> dict[str, Any]:
    environment = build_environment_for_row(row)
    tool_names = {tool.get("function", {}).get("name", "") for tool in row.get("tools", [])}
    errors = validate_environment_spec(environment, tool_names=tool_names)
    state = copy.deepcopy(environment.get("initial_state", {}))
    messages = copy.deepcopy(row.get("messages", []))
    replacements = 0
    executable_calls = 0

    if not errors:
        for index, message in enumerate(messages):
            if message.get("role") != "assistant" or "tool_call" not in message:
                continue
            call = message["tool_call"]
            name = call.get("name")
            args = call.get("arguments", {})
            if index + 1 >= len(messages) or messages[index + 1].get("role") != "tool":
                errors.append(f"message {index}: assistant tool_call has no following tool response")
                continue
            tool_message = messages[index + 1]
            if tool_message.get("name") != name:
                errors.append(f"message {index + 1}: tool response name {tool_message.get('name')!r} does not match {name!r}")
                continue
            expected, step_errors = execute_tool(name, args, state, environment)
            executable_calls += 1
            if step_errors:
                errors.extend(f"message {index}: {error}" for error in step_errors)
            if tool_message.get("content") != expected:
                tool_message["content"] = expected
                replacements += 1

    updated = dict(row)
    updated["messages"] = messages
    updated["tool_response_canonicalization"] = {
        "enabled": True,
        "replacements": replacements,
        "executable_calls": executable_calls,
        "errors": errors,
    }
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [canonicalize_row(row) for row in load_jsonl(args.input)]
    write_jsonl(args.output, rows)
    print(
        json.dumps(
            {
                "written": len(rows),
                "rows_with_replacements": sum(1 for row in rows if row["tool_response_canonicalization"]["replacements"]),
                "replacements": sum(row["tool_response_canonicalization"]["replacements"] for row in rows),
                "rows_with_errors": sum(1 for row in rows if row["tool_response_canonicalization"]["errors"]),
                "output": str(args.output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
