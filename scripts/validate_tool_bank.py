#!/usr/bin/env python3
"""Validate Stage 2 artifacts against the canonical atomic tool bank."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOOL_BANK = PROJECT_ROOT / "config/tool_bank.json"
ALLOWED_SCHEMA_TYPES = {"string", "integer", "number", "boolean", "array", "object"}
GENERIC_NAMES = {"print", "search", "update", "run", "call", "execute", "process", "handle", "manage"}
TASK_SPECIFIC_HIGH_LEVEL_NAMES = {
    "login_student",
    "authenticate_scanner",
    "dreambox_login",
    "scan_to_email",
    "upload_file_to_cloud",
    "create_assignment_with_link",
    "download_offline_map",
    "export_gpx_track",
}
BUNDLED_PATTERNS = [
    "and",
    "if",
    "workflow",
    "complete",
    "process",
    "handle",
    "manage",
    "orchestrate",
    "all",
]


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


def load_tool_bank(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tools = data.get("tools", [])
    return {tool.get("function", {}).get("name", ""): tool for tool in tools}


def canonical_parameters(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function")
    if not isinstance(function, dict):
        return {}
    parameters = function.get("parameters", {})
    return schema_signature(parameters) if isinstance(parameters, dict) else {}


def schema_signature(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: schema_signature(item) for key, item in value.items() if key != "description"}
    if isinstance(value, list):
        return [schema_signature(item) for item in value]
    return value


def lint_tool(tool: dict[str, Any]) -> list[str]:
    errors = []
    function = tool.get("function")
    if not isinstance(function, dict):
        return [f"tool.function must be an object, got {type(function).__name__}"]
    name = function.get("name", "")
    if not isinstance(name, str):
        return [f"function.name must be a string, got {type(name).__name__}"]
    tokens = [token for token in name.split("_") if token]
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        errors.append(f"{name}: function name must be snake_case")
    if name in GENERIC_NAMES:
        errors.append(f"{name}: function name is too generic")
    if name in TASK_SPECIFIC_HIGH_LEVEL_NAMES:
        errors.append(f"{name}: function name is task-specific/high-level; decompose into canonical atomic tools")
    if len(tokens) > 4:
        errors.append(f"{name}: function name is too long; prefer 2 to 4 words")
    for pattern in BUNDLED_PATTERNS:
        if pattern in tokens:
            errors.append(f"{name}: name suggests bundled workflow logic via token {pattern!r}")
            break
    parameters = function.get("parameters", {})
    if parameters.get("type") != "object":
        errors.append(f"{name}: parameters.type must be object")
    properties = parameters.get("properties", {})
    required = parameters.get("required", [])
    if not isinstance(properties, dict):
        errors.append(f"{name}: properties must be an object")
    if not isinstance(required, list):
        errors.append(f"{name}: required must be a list")
    for arg_name, arg_schema in properties.items():
        arg_type = arg_schema.get("type")
        if arg_type not in ALLOWED_SCHEMA_TYPES:
            errors.append(f"{name}.{arg_name}: unsupported type {arg_type!r}")
        if arg_type == "array" and "items" not in arg_schema:
            errors.append(f"{name}.{arg_name}: array type must include items schema")
    for arg_name in required:
        if arg_name not in properties:
            errors.append(f"{name}: required arg {arg_name!r} missing from properties")
    return errors


def lint_tool_bank(tool_bank: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if not tool_bank:
        return ["tool bank contains no tools"]
    for name, tool in sorted(tool_bank.items()):
        if not name:
            errors.append("tool bank contains a tool without function.name")
            continue
        errors.extend(f"tool_bank.{error}" for error in lint_tool(tool))
    return errors


def validate_missing_tool_requirements(value: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, list) or not value:
        return ["missing_tool_requirements must be a non-empty list"]
    for index, item in enumerate(value):
        prefix = f"missing_tool_requirements[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix}: item must be an object")
            continue
        for field in ("capability", "reason", "suggested_tool_name", "suggested_parameters", "reusability"):
            if field not in item:
                errors.append(f"{prefix}: missing field {field!r}")
        for field in ("capability", "reason", "suggested_tool_name", "reusability"):
            if field in item and (not isinstance(item[field], str) or not item[field].strip()):
                errors.append(f"{prefix}.{field}: must be a non-empty string")
        suggested_name = item.get("suggested_tool_name")
        if isinstance(suggested_name, str):
            if not re.fullmatch(r"[a-z][a-z0-9_]*", suggested_name):
                errors.append(f"{prefix}.suggested_tool_name: must be snake_case")
        params = item.get("suggested_parameters")
        if "suggested_parameters" in item and not isinstance(params, dict):
            errors.append(f"{prefix}.suggested_parameters: must be an object")
        elif isinstance(params, dict):
            for arg_name, schema in params.items():
                if not isinstance(arg_name, str) or not arg_name:
                    errors.append(f"{prefix}.suggested_parameters: argument names must be non-empty strings")
                    continue
                if not isinstance(schema, dict):
                    errors.append(f"{prefix}.suggested_parameters.{arg_name}: schema must be an object")
                    continue
                arg_type = schema.get("type")
                if arg_type not in ALLOWED_SCHEMA_TYPES:
                    errors.append(f"{prefix}.suggested_parameters.{arg_name}: unsupported type {arg_type!r}")
                if not isinstance(schema.get("description", ""), str) or not schema.get("description", "").strip():
                    errors.append(f"{prefix}.suggested_parameters.{arg_name}: description must be a non-empty string")
    return errors


def validate_row(
    row: dict[str, Any],
    tool_bank: dict[str, dict[str, Any]],
    *,
    allow_missing_tool: bool,
    strict_description: bool,
    require_discoverable_record_ids: bool,
) -> list[str]:
    errors: list[str] = []
    missing = row.get("missing_tool_requirements")
    if missing:
        if not allow_missing_tool:
            errors.append("row has missing_tool_requirements and must not be used for Stage3/SFT until the tool bank is extended")
        errors.extend(validate_missing_tool_requirements(missing))
        if row.get("messages"):
            errors.append("row has missing_tool_requirements but already contains Stage3 messages")
        return errors

    tools = row.get("tools", [])
    if not isinstance(tools, list) or not tools:
        return ["row has no tools and no missing_tool_requirements"]

    seen: set[str] = set()
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            errors.append(f"tools[{index}]: tool must be an object, got {type(tool).__name__}")
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            errors.append(f"tools[{index}].function must be an object, got {type(function).__name__}")
            errors.extend(lint_tool(tool))
            continue
        name = function.get("name", "")
        if not isinstance(name, str):
            errors.append(f"tools[{index}].function.name must be a string, got {type(name).__name__}")
            continue
        if not name:
            errors.append("tool without function.name")
            continue
        if name in seen:
            errors.append(f"{name}: duplicate tool")
        seen.add(name)
        if name not in tool_bank:
            errors.append(f"{name}: not found in canonical tool bank")
            continue
        errors.extend(lint_tool(tool))
        expected = tool_bank[name]
        if canonical_parameters(tool) != canonical_parameters(expected):
            errors.append(f"{name}: parameters do not match canonical tool bank definition")
        if strict_description and tool.get("function", {}).get("description") != expected.get("function", {}).get("description"):
            errors.append(f"{name}: description does not match canonical tool bank definition")

    rules = row.get("environment", {}).get("tool_rules")
    if isinstance(rules, dict):
        rule_names = set(rules)
        if rule_names != seen:
            errors.append(f"environment.tool_rules names {sorted(rule_names)} do not match selected tools {sorted(seen)}")
        if require_discoverable_record_ids:
            errors.extend(validate_discoverable_record_ids(row))
    return errors


def rule_text(rule: Any) -> str:
    return json.dumps(rule, ensure_ascii=False, separators=(",", ":"))


def collect_strings(value: Any) -> set[str]:
    strings: set[str] = set()
    if isinstance(value, str):
        strings.add(value)
    elif isinstance(value, list):
        for item in value:
            strings.update(collect_strings(item))
    elif isinstance(value, dict):
        for item in value.values():
            strings.update(collect_strings(item))
    return strings


def exposes_collection(collection: str, records: dict[str, Any], rules: dict[str, Any]) -> bool:
    """Return whether a retrieval rule can expose ids for this collection."""
    retrieval_rules = [rules.get("list_records", {}), rules.get("read_state", {}), rules.get("search_records", {})]
    retrieval_text = "".join(rule_text(rule) for rule in retrieval_rules)
    if f"$state.{collection}" in retrieval_text:
        return True

    record_ids = {str(record_id) for record_id in records}
    if not record_ids:
        return False
    for rule in retrieval_rules:
        branches = rule.get("branches", []) if isinstance(rule, dict) else []
        for branch in branches:
            if not isinstance(branch, dict):
                continue
            branch_text = rule_text(branch)
            if collection not in branch_text:
                continue
            response_strings = collect_strings(branch.get("response", {}))
            if record_ids & response_strings:
                return True
    return False


def validate_discoverable_record_ids(row: dict[str, Any]) -> list[str]:
    """Heuristically catch hidden record_id requirements in canonical get/update rules."""
    errors: list[str] = []
    environment = row.get("environment", {})
    initial_state = environment.get("initial_state", {})
    rules = environment.get("tool_rules", {})
    if not isinstance(initial_state, dict) or not isinstance(rules, dict):
        return errors

    mutation_rule_text = rule_text(rules.get("get_record", {})) + rule_text(rules.get("update_record", {})) + rule_text(rules.get("delete_record", {}))
    authenticate_rule_text = rule_text(rules.get("authenticate", {}))
    for collection, value in initial_state.items():
        if not isinstance(value, dict) or not value:
            continue
        dynamic_record_path = f"$state.{collection}[$args.record_id]"
        dynamic_target_path = f"$state.{collection}[$args.target_id]"
        needs_record_id = dynamic_record_path in mutation_rule_text or dynamic_target_path in mutation_rule_text
        needs_account_id = "$args.account_id" in authenticate_rule_text and f"$state.{collection}" in authenticate_rule_text
        if (needs_record_id or needs_account_id) and not exposes_collection(collection, value, rules):
            errors.append(
                f"{collection}: ids are required by get/update/delete/authenticate rules but no list/read/search rule exposes this collection"
            )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tool-bank", type=Path, default=DEFAULT_TOOL_BANK)
    parser.add_argument("--allow-missing-tool", action="store_true", help="Treat missing_tool_requirements rows as reviewable instead of invalid.")
    parser.add_argument("--strict-description", action="store_true", help="Require selected tool descriptions to exactly match the bank.")
    parser.add_argument("--require-discoverable-record-ids", action="store_true", help="Require hidden record ids used by get/update/delete rules to be exposed by list/read/search rules.")
    args = parser.parse_args()

    tool_bank = load_tool_bank(args.tool_bank)
    bank_errors = lint_tool_bank(tool_bank)
    results = []
    for row in load_jsonl(args.input):
        errors = [f"tool_bank: {error}" for error in bank_errors]
        errors.extend(
            validate_row(
                row,
                tool_bank,
                allow_missing_tool=args.allow_missing_tool,
                strict_description=args.strict_description,
                require_discoverable_record_ids=args.require_discoverable_record_ids,
            )
        )
        results.append({"id": row.get("id"), "valid": not errors, "errors": errors})

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
