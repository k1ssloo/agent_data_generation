#!/usr/bin/env python3
"""Check whether Stage 2 artifacts are ready for hidden-environment rollout.

Environment DSL validation proves a tool can execute when exact arguments are
known. Rollout readiness checks a different property: whether Stage 3 can
discover or infer those exact arguments without seeing hidden environment JSON.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from executable_environment import validate_environment_spec


CONTROL_ARG_NAMES = {
    "channel",
    "collection",
    "constraint_type",
    "credential_type",
    "destination",
    "expected_status",
    "format",
    "job_type",
    "provider",
    "relationship",
    "resource_type",
    "service",
    "target_collection",
    "view",
    "visibility",
}
RISKY_CONTROL_ARG_NAMES = {"destination", "provider"}

DISCOVERY_TOOL_PREFIXES = ("list_", "search_", "get_", "read_", "locate_", "poll_", "verify_")
ID_ARG_NAMES = {"record_id", "resource_id", "session_id", "job_id", "target_id", "account_id"}


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


def workflow_sequence(row: dict[str, Any]) -> list[str]:
    graph = row.get("workflow", {}).get("execution_graph", "")
    return re.findall(r"\(([a-zA-Z_][a-zA-Z0-9_]*)\)", graph)


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def visible_text(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "text": row.get("text", ""),
            "workflow": row.get("workflow", {}),
        },
        ensure_ascii=False,
    ).lower()


def literal_visible(value: Any, text: str, exposed: set[str]) -> bool:
    if not isinstance(value, str) or not value:
        return True
    lowered = value.lower()
    normalized = normalize(value)
    if lowered in text or (normalized and normalized in normalize(text)):
        return True
    return lowered in exposed or normalized in {normalize(item) for item in exposed}


def is_catch_all(branch: dict[str, Any]) -> bool:
    return branch.get("if", True) is True


def walk(value: Any):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)


def string_literals(value: Any) -> set[str]:
    literals: set[str] = set()
    if isinstance(value, str):
        if not value.startswith("$") and value:
            literals.add(value)
    elif isinstance(value, dict):
        for item in value.values():
            literals |= string_literals(item)
    elif isinstance(value, list):
        for item in value:
            literals |= string_literals(item)
    return literals


def state_collection_names(value: Any) -> set[str]:
    names: set[str] = set()
    text = json.dumps(value, ensure_ascii=False)
    for match in re.finditer(r"\$state\.([A-Za-z_][A-Za-z0-9_]*)", text):
        names.add(match.group(1))
    return names


def collect_state_strings(value: Any) -> set[str]:
    strings: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                strings.add(key)
            strings |= collect_state_strings(item)
    elif isinstance(value, list):
        for item in value:
            strings |= collect_state_strings(item)
    elif isinstance(value, str):
        strings.add(value)
    return strings


def branch_response_exposures(rule: dict[str, Any], initial_state: dict[str, Any]) -> set[str]:
    exposed: set[str] = set()
    for branch in rule.get("branches", []):
        response = branch.get("response", {})
        exposed |= string_literals(response)
        for collection in state_collection_names(response):
            collection_value = initial_state.get(collection)
            if collection_value is not None:
                exposed |= collect_state_strings(collection_value)
    return exposed


def arg_literal_constraints(condition: Any) -> list[tuple[str, Any]]:
    constraints: list[tuple[str, Any]] = []
    if isinstance(condition, bool) or not isinstance(condition, dict):
        return constraints
    if "all" in condition or "any" in condition:
        key = "all" if "all" in condition else "any"
        for item in condition.get(key, []):
            constraints.extend(arg_literal_constraints(item))
    if "not" in condition:
        constraints.extend(arg_literal_constraints(condition["not"]))
    for op in ("eq", "ne"):
        values = condition.get(op)
        if not isinstance(values, list) or len(values) != 2:
            continue
        left, right = values
        pair = arg_literal_pair(left, right) or arg_literal_pair(right, left)
        if pair:
            constraints.append(pair)
    return constraints


def arg_literal_pair(arg_side: Any, literal_side: Any) -> tuple[str, Any] | None:
    if not isinstance(arg_side, str):
        return None
    match = re.fullmatch(r"\$args\.([A-Za-z_][A-Za-z0-9_]*)", arg_side)
    if not match:
        return None
    if isinstance(literal_side, str) and literal_side.startswith("$"):
        return None
    if isinstance(literal_side, (str, int, float, bool)) or literal_side is None:
        return match.group(1), literal_side
    return None


def validate_row(row: dict[str, Any], warnings_as_errors: bool) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    seen_messages: set[str] = set()
    environment = row.get("environment", {})
    initial_state = environment.get("initial_state", {})
    rules = environment.get("tool_rules", {})
    text = visible_text(row)

    env_errors = validate_environment_spec(environment, tool_names=tool_names(row))
    errors.extend(f"environment: {error}" for error in env_errors)
    if not isinstance(initial_state, dict) or not isinstance(rules, dict):
        return errors, warnings

    sequence = workflow_sequence(row)
    if not sequence:
        warnings.append("workflow execution_graph does not expose a tool sequence")

    for name, rule in rules.items():
        branches = rule.get("branches", [])
        if not any(is_catch_all(branch) for branch in branches):
            errors.append(f"{name}: no catch-all failure branch; wrong arguments become interpreter errors")

    exposed: set[str] = set()
    for step_index, name in enumerate(sequence):
        rule = rules.get(name)
        if not isinstance(rule, dict):
            continue
        for branch_index, branch in enumerate(rule.get("branches", [])):
            if is_catch_all(branch):
                continue
            for arg_name, literal in arg_literal_constraints(branch.get("if", True)):
                if literal is None or literal == "":
                    continue
                if literal_visible(literal, text, exposed):
                    continue
                message = (
                    f"{name}.branches[{branch_index}]: arg {arg_name} requires hidden literal {literal!r} "
                    f"before it is visible from source/workflow or prior tool responses"
                )
                if arg_name in CONTROL_ARG_NAMES:
                    if arg_name in RISKY_CONTROL_ARG_NAMES and message not in seen_messages:
                        warnings.append(message)
                        seen_messages.add(message)
                else:
                    if message not in seen_messages:
                        errors.append(message)
                        seen_messages.add(message)
        exposed |= branch_response_exposures(rule, initial_state)

    # ID-like values should be produced by a previous response, not required as
    # a hidden constant in an early tool condition.
    seen_tools: set[str] = set()
    exposed = set()
    for name in sequence:
        rule = rules.get(name, {})
        for branch_index, branch in enumerate(rule.get("branches", [])):
            if is_catch_all(branch):
                continue
            for arg_name, literal in arg_literal_constraints(branch.get("if", True)):
                if arg_name not in ID_ARG_NAMES:
                    continue
                if literal_visible(literal, text, exposed):
                    continue
                if not any(tool.startswith(DISCOVERY_TOOL_PREFIXES) for tool in seen_tools):
                    message = (
                        f"{name}.branches[{branch_index}]: id-like arg {arg_name}={literal!r} is required before any discovery tool can expose it"
                    )
                    if message not in seen_messages:
                        errors.append(message)
                        seen_messages.add(message)
        exposed |= branch_response_exposures(rule, initial_state)
        seen_tools.add(name)

    if warnings_as_errors:
        errors.extend(warnings)
        warnings = []
    return errors, warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warnings-as-errors", action="store_true")
    args = parser.parse_args()

    results = []
    for row in load_jsonl(args.input):
        errors, warnings = validate_row(row, warnings_as_errors=args.warnings_as_errors)
        results.append({"id": row.get("id"), "valid": not errors, "errors": errors, "warnings": warnings})
    write_jsonl(args.output, results)
    print(
        json.dumps(
            {
                "checked": len(results),
                "valid": sum(item["valid"] for item in results),
                "warnings": sum(bool(item["warnings"]) for item in results),
                "output": str(args.output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
