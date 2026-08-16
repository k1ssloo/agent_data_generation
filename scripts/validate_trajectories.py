#!/usr/bin/env python3
"""Validate GEM trajectories with schema and lightweight grounding checks."""

from __future__ import annotations

import argparse
import json
import re
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


def expected_type_matches(expected: str, value: Any) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def message_text(message: dict[str, Any]) -> str:
    pieces = []
    if "content" in message:
        pieces.append(json.dumps(message["content"], ensure_ascii=False))
    if "tool_call" in message:
        pieces.append(json.dumps(message["tool_call"], ensure_ascii=False))
    return " ".join(pieces)


def environment_identifiers(row: dict[str, Any]) -> set[str]:
    identifiers: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if is_identifier_like(key):
                    identifiers.add(key)
                if key in {"id", "name", "title"} and isinstance(item, str) and len(item) >= 3:
                    identifiers.add(item)
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(row.get("environment", {}).get("initial_state", {}))
    return identifiers


def is_identifier_like(value: str) -> bool:
    return len(value) >= 3 and bool(re.search(r"[_/\d]", value))


def workflow_tool_names(row: dict[str, Any]) -> list[str]:
    graph = row.get("workflow", {}).get("execution_graph", "")
    return re.findall(r"\(([a-zA-Z_][a-zA-Z0-9_]*)\)", graph)


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


def is_verification_tool(name: str) -> bool:
    tokens = name.split("_")
    return (
        name.startswith(("verify_", "read_", "get_", "list_", "search_", "poll_"))
        or "verify" in tokens
        or "status" in tokens
    )


def recovery_expected(row: dict[str, Any]) -> bool:
    text = json.dumps({"text": row.get("text", ""), "workflow": row.get("workflow", {})}, ensure_ascii=False).lower()
    return any(marker in text for marker in ("fail", "fallback", "retry", "try another", "not available", "offline"))


def is_failure_response(content: Any) -> bool:
    lowered = json.dumps(content, ensure_ascii=False).lower()
    return any(marker in lowered for marker in ("failed", "failure", "error", "not available", "offline"))


def is_success_response(content: Any) -> bool:
    lowered = json.dumps(content, ensure_ascii=False).lower()
    return any(marker in lowered for marker in ('"success"', '"queued"', '"scheduled"', '"enrolled"', '"ok"'))


def collect_tool_schemas(row: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    errors: list[str] = []
    tools: dict[str, dict[str, Any]] = {}
    raw_tools = row.get("tools", [])
    if not isinstance(raw_tools, list):
        return tools, [f"tools must be a list, got {type(raw_tools).__name__}"]
    for index, tool in enumerate(raw_tools):
        if not isinstance(tool, dict):
            errors.append(f"tools[{index}]: tool must be an object, got {type(tool).__name__}")
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            errors.append(f"tools[{index}].function must be an object, got {type(function).__name__}")
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            errors.append(f"tools[{index}].function.name must be a non-empty string")
            continue
        parameters = function.get("parameters", {})
        if not isinstance(parameters, dict):
            errors.append(f"tools[{index}].function.parameters must be an object, got {type(parameters).__name__}")
            parameters = {}
        if name in tools:
            errors.append(f"tools[{index}]: duplicate tool {name!r}")
        tools[name] = parameters
    return tools, errors


def object_at(messages: list[Any], index: int) -> dict[str, Any]:
    if 0 <= index < len(messages) and isinstance(messages[index], dict):
        return messages[index]
    return {}


def validate(
    row: dict[str, Any],
    strict_grounding: bool,
    require_workflow_tools: bool,
    require_error_recovery: bool,
    min_tool_calls: int,
    max_user_turns: int,
    require_final_verification: bool,
    allow_control_arg_literals: bool,
) -> list[str]:
    errors: list[str] = []
    tools, tool_errors = collect_tool_schemas(row)
    errors.extend(tool_errors)
    messages = row.get("messages", [])
    if not isinstance(messages, list):
        errors.append(f"messages must be a list, got {type(messages).__name__}")
        messages = []
    if not messages or not isinstance(messages[0], dict) or messages[0].get("role") != "system":
        errors.append("missing system message")
    used_tools = []
    user_turns = 0
    saw_failure = False
    saw_success_after_failure = False
    env_identifiers = environment_identifiers(row)
    prior_context = ""
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"message {index}: message must be an object, got {type(message).__name__}")
            continue
        role = message.get("role")
        if role == "user":
            user_turns += 1
        if role == "tool":
            previous = object_at(messages, index - 1)
            if index == 0 or previous.get("role") != "assistant" or "tool_call" not in previous:
                errors.append(f"message {index}: tool response not preceded by assistant tool call")
            elif not isinstance(previous.get("tool_call"), dict):
                errors.append(f"message {index}: tool response not preceded by assistant tool call")
            elif message.get("name") != previous["tool_call"].get("name"):
                errors.append(f"message {index}: tool response name does not match preceding tool call")
            if is_failure_response(message.get("content")):
                saw_failure = True
            elif saw_failure and is_success_response(message.get("content")):
                saw_success_after_failure = True
            prior_context += " " + message_text(message)
            continue
        if role != "assistant" or "tool_call" not in message or not isinstance(message.get("tool_call"), dict):
            if strict_grounding and role == "assistant":
                current_text = message_text(message).lower()
                prior_lower = prior_context.lower()
                for identifier in sorted(env_identifiers):
                    if identifier.lower() in current_text and identifier.lower() not in prior_lower:
                        errors.append(f"message {index}: assistant mentioned environment identifier {identifier!r} before it was grounded")
            prior_context += " " + message_text(message)
            continue
        call = message["tool_call"]
        name = call.get("name")
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            errors.append(f"message {index}: tool_call.arguments must be an object, got {type(args).__name__}")
            args = {}
        used_tools.append(name)
        if name not in tools:
            errors.append(f"message {index}: unknown tool {name}")
            prior_context += " " + message_text(message)
            continue
        schema = tools[name]
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            errors.append(f"message {index}: parameters.properties for {name} must be an object")
            properties = {}
        required_args = schema.get("required", [])
        if not isinstance(required_args, list):
            errors.append(f"message {index}: parameters.required for {name} must be a list")
            required_args = []
        for required in required_args:
            if required not in args:
                errors.append(f"message {index}: missing required arg {required} for {name}")
        for arg_name, value in args.items():
            if arg_name not in properties:
                errors.append(f"message {index}: unexpected arg {arg_name} for {name}")
                continue
            expected = properties[arg_name].get("type", "")
            if not expected_type_matches(expected, value):
                errors.append(f"message {index}: arg {arg_name} expected {expected}, got {type(value).__name__}")
            if strict_grounding and isinstance(value, str):
                if allow_control_arg_literals and arg_name in CONTROL_ARG_NAMES:
                    continue
                value_text = str(value)
                if value_text and value_text.lower() not in prior_context.lower():
                    errors.append(f"message {index}: arg {arg_name}={value_text!r} is not grounded in prior context")
        next_message = object_at(messages, index + 1)
        if index + 1 >= len(messages) or next_message.get("role") != "tool":
            errors.append(f"message {index}: tool call not followed by tool response")
        prior_context += " " + message_text(message)
    if require_workflow_tools:
        expected = set(workflow_tool_names(row))
        missing = sorted(expected - set(used_tools))
        if missing:
            errors.append(f"workflow tools not used in trajectory: {missing}")
    if min_tool_calls and len(used_tools) < min_tool_calls:
        errors.append(f"tool call count {len(used_tools)} is below required minimum {min_tool_calls}")
    if max_user_turns and user_turns > max_user_turns:
        errors.append(f"user turn count {user_turns} exceeds maximum {max_user_turns}")
    if require_final_verification and used_tools:
        if not is_verification_tool(used_tools[-1]):
            errors.append(f"last tool call {used_tools[-1]!r} is not a verification/read/get/list/search/poll tool")
    if require_error_recovery and recovery_expected(row):
        if not saw_failure:
            errors.append("expected an error/failure recovery pattern, but no failed tool response was present")
        elif not saw_success_after_failure:
            errors.append("expected recovery after failure, but no later successful tool response was present")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict-grounding", action="store_true")
    parser.add_argument("--require-workflow-tools", action="store_true")
    parser.add_argument("--require-error-recovery", action="store_true")
    parser.add_argument("--min-tool-calls", type=int, default=0, help="Require at least this many assistant tool calls.")
    parser.add_argument("--max-user-turns", type=int, default=0, help="Require at most this many user messages; 0 disables the check.")
    parser.add_argument("--require-final-verification", action="store_true", help="Require the final tool call to be a verification/read/get/list/search/poll tool.")
    parser.add_argument("--allow-control-arg-literals", action="store_true", help="Allow canonical control arguments such as collection/resource_type/expected_status without prior textual grounding.")
    args = parser.parse_args()

    results = []
    for row in load_jsonl(args.input):
        errors = validate(
            row,
            strict_grounding=args.strict_grounding,
            require_workflow_tools=args.require_workflow_tools,
            require_error_recovery=args.require_error_recovery,
            min_tool_calls=args.min_tool_calls,
            max_user_turns=args.max_user_turns,
            require_final_verification=args.require_final_verification,
            allow_control_arg_literals=args.allow_control_arg_literals,
        )
        results.append({"id": row["id"], "valid": not errors, "errors": errors})
    write_jsonl(args.output, results)
    print(json.dumps({"checked": len(results), "valid": sum(item["valid"] for item in results), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
