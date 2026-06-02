#!/usr/bin/env python3
"""Generic executable DSL for GEM-style trajectory replay.

Stage 2 should synthesize the environment spec. This module only interprets the
spec; it does not implement open-ended tools by name. The toy builders at the
bottom are regression fixtures that emit the same DSL format expected from an
LLM-generated environment.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any


DSL_VERSION = "text-exec-dsl-v0"


class ExecutionError(ValueError):
    """Raised when an executable environment or rule cannot be interpreted."""


def build_environment(task: str, tool_names: list[str] | None = None) -> dict[str, Any]:
    tool_set = set(tool_names or [])
    if task == "multimedia_processing":
        return build_photo_environment(tool_set)
    if task == "ecommerce_and_retail":
        return build_return_environment()
    if task == "education_elearning":
        return build_course_environment()
    return {"version": DSL_VERSION, "initial_state": {}, "tool_rules": {}}


def build_environment_for_row(row: dict[str, Any]) -> dict[str, Any]:
    tool_names = [tool.get("function", {}).get("name", "") for tool in row.get("tools", [])]
    return row.get("environment") or build_environment(row.get("task", ""), tool_names)


def validate_environment_spec(environment: dict[str, Any], tool_names: set[str] | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(environment, dict):
        return ["environment must be an object"]
    if environment.get("version") != DSL_VERSION:
        errors.append(f"environment.version must be {DSL_VERSION!r}")
    if not isinstance(environment.get("initial_state"), dict):
        errors.append("environment.initial_state must be an object")
    rules = environment.get("tool_rules")
    if not isinstance(rules, dict):
        errors.append("environment.tool_rules must be an object")
        return errors
    if tool_names is not None:
        missing = sorted(tool_names - set(rules))
        if missing:
            errors.append(f"missing tool_rules for tools: {missing}")
        extra = sorted(set(rules) - tool_names)
        if extra:
            errors.append(f"tool_rules without matching tools: {extra}")
    for tool_name, rule in rules.items():
        if not isinstance(rule, dict):
            errors.append(f"{tool_name}: rule must be an object")
            continue
        branches = rule.get("branches")
        if not isinstance(branches, list) or not branches:
            errors.append(f"{tool_name}: branches must be a non-empty list")
            continue
        for index, branch in enumerate(branches):
            if not isinstance(branch, dict):
                errors.append(f"{tool_name}.branches[{index}]: branch must be an object")
                continue
            if "if" in branch:
                errors.extend(f"{tool_name}.branches[{index}].if: {error}" for error in validate_condition_syntax(branch["if"]))
            if "response" not in branch:
                errors.append(f"{tool_name}.branches[{index}]: missing response")
            else:
                errors.extend(f"{tool_name}.branches[{index}].response: {error}" for error in validate_value_syntax(branch["response"]))
            effects = branch.get("effects", [])
            if not isinstance(effects, list):
                errors.append(f"{tool_name}.branches[{index}]: effects must be a list")
            else:
                for effect_index, effect in enumerate(effects):
                    errors.extend(f"{tool_name}.branches[{index}].effects[{effect_index}]: {error}" for error in validate_effect_syntax(effect))
                    errors.extend(f"{tool_name}.branches[{index}].effects[{effect_index}]: {error}" for error in validate_effect_semantics(effect, environment.get("initial_state", {})))
    return errors


def validate_condition_syntax(condition: Any) -> list[str]:
    if isinstance(condition, bool):
        return []
    if not isinstance(condition, dict):
        return [f"condition must be object or boolean, got {type(condition).__name__}"]
    errors: list[str] = []
    supported = {"all", "any", "not", "exists", "not_exists", "eq", "ne", "in", "range"}
    unknown = set(condition) - supported
    if unknown:
        errors.append(f"unsupported condition keys {sorted(unknown)}")
    if "all" in condition or "any" in condition:
        key = "all" if "all" in condition else "any"
        items = condition[key]
        if not isinstance(items, list):
            errors.append(f"{key} must be a list")
        else:
            for item in items:
                errors.extend(validate_condition_syntax(item))
    if "not" in condition:
        errors.extend(validate_condition_syntax(condition["not"]))
    for key in ("exists", "not_exists"):
        if key in condition and not is_path(condition[key]):
            errors.append(f"{key} must be a DSL path starting with $")
    for key in ("eq", "ne", "in"):
        if key in condition:
            values = condition[key]
            if not isinstance(values, list) or len(values) != 2:
                errors.append(f"{key} must be a two-item list")
            else:
                errors.extend(validate_value_syntax(values[0]))
                errors.extend(validate_value_syntax(values[1]))
    if "range" in condition:
        spec = condition["range"]
        if not isinstance(spec, dict) or "value" not in spec:
            errors.append("range must be an object with value")
        else:
            errors.extend(validate_value_syntax(spec["value"]))
    return errors


def validate_value_syntax(value: Any) -> list[str]:
    errors: list[str] = []
    if isinstance(value, str):
        return []
    if isinstance(value, list):
        for item in value:
            errors.extend(validate_value_syntax(item))
        return errors
    if isinstance(value, dict):
        special = {"literal", "get", "template", "filter_values"}
        present = special & set(value)
        if len(present) > 1:
            errors.append(f"value object has multiple special operators {sorted(present)}")
        if "get" in value and not is_path(value["get"]):
            errors.append("get must be a DSL path starting with $")
        if "filter_values" in value:
            spec = value["filter_values"]
            if not isinstance(spec, dict):
                errors.append("filter_values must be an object")
            else:
                if not is_path(spec.get("from")):
                    errors.append("filter_values.from must be a DSL path starting with $")
                errors.extend(validate_condition_syntax(spec.get("where", True)))
                errors.extend(validate_value_syntax(spec.get("select", "$item")))
        if present:
            return errors
        for item in value.values():
            errors.extend(validate_value_syntax(item))
    return errors


def validate_effect_syntax(effect: Any) -> list[str]:
    if not isinstance(effect, dict):
        return ["effect must be an object"]
    keys = {"set", "append", "delete"} & set(effect)
    if len(keys) != 1:
        return ["effect must contain exactly one of set, append, delete"]
    key = next(iter(keys))
    target = effect[key]
    errors = []
    if not is_path(target) or not str(target).startswith("$state"):
        errors.append(f"{key} target must be a $state path")
    if key in {"set", "append"}:
        if "value" not in effect:
            errors.append(f"{key} effect must include value")
        else:
            errors.extend(validate_value_syntax(effect["value"]))
    return errors


def validate_effect_semantics(effect: Any, initial_state: dict[str, Any]) -> list[str]:
    if not isinstance(effect, dict):
        return []
    errors: list[str] = []
    if "append" in effect:
        target = resolve_static_state_path(effect["append"], initial_state)
        if target is not MISSING and not isinstance(target, list):
            errors.append("append target must resolve to an array in initial_state")
    return errors


def resolve_static_state_path(path: Any, initial_state: dict[str, Any]) -> Any:
    if not isinstance(path, str) or not path.startswith("$state"):
        return MISSING
    try:
        tokens = parse_path(path[1:])
    except ExecutionError:
        return MISSING
    if not tokens or tokens[0] != "state":
        return MISSING
    current: Any = initial_state
    for token in tokens[1:]:
        if isinstance(token, dict):
            return MISSING
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and isinstance(token, int) and 0 <= token < len(current):
            current = current[token]
        else:
            return MISSING
    return current


def is_path(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("$")


def execute_tool(name: str, args: dict[str, Any], state: dict[str, Any], environment: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors = validate_environment_spec(environment)
    if errors:
        return {"status": "failed", "error": "invalid executable environment"}, errors
    rule = environment.get("tool_rules", {}).get(name)
    if not rule:
        return {"status": "failed", "error": f"missing executable rule for {name}"}, [f"missing executable rule for {name}"]
    context = {"state": state, "args": args, "response": None}
    for branch in rule.get("branches", []):
        condition = branch.get("if", True)
        try:
            matches = eval_condition(condition, context)
        except ExecutionError as exc:
            return {"status": "failed", "error": str(exc)}, [f"{name}: {exc}"]
        if not matches:
            continue
        try:
            response = eval_value(branch.get("response", {}), context)
        except ExecutionError as exc:
            return {"status": "failed", "error": str(exc)}, [f"{name}: {exc}"]
        context["response"] = response
        for effect in branch.get("effects", []):
            try:
                apply_effect(effect, context)
            except ExecutionError as exc:
                return response, [f"{name}: {exc}"]
        return response, []
    return {"status": "failed", "error": "no executable branch matched"}, [f"{name}: no executable branch matched"]


def eval_condition(condition: Any, context: dict[str, Any]) -> bool:
    if isinstance(condition, bool):
        return condition
    if not isinstance(condition, dict):
        raise ExecutionError(f"invalid condition {condition!r}")
    if "all" in condition:
        return all(eval_condition(item, context) for item in condition["all"])
    if "any" in condition:
        return any(eval_condition(item, context) for item in condition["any"])
    if "not" in condition:
        return not eval_condition(condition["not"], context)
    if "exists" in condition:
        return resolve_path(condition["exists"], context, missing_ok=True) is not MISSING
    if "not_exists" in condition:
        return resolve_path(condition["not_exists"], context, missing_ok=True) is MISSING
    if "eq" in condition:
        left, right = condition["eq"]
        left_value = eval_condition_value(left, context)
        right_value = eval_condition_value(right, context)
        if left_value is MISSING or right_value is MISSING:
            return False
        return left_value == right_value
    if "ne" in condition:
        left, right = condition["ne"]
        left_value = eval_condition_value(left, context)
        right_value = eval_condition_value(right, context)
        if left_value is MISSING or right_value is MISSING:
            return False
        return left_value != right_value
    if "in" in condition:
        value, container = condition["in"]
        resolved_container = eval_condition_value(container, context)
        resolved_value = eval_condition_value(value, context)
        if resolved_container is MISSING or resolved_value is MISSING:
            return False
        return resolved_value in resolved_container
    if "range" in condition:
        spec = condition["range"]
        value = eval_value(spec["value"], context)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        return spec.get("min", value) <= value <= spec.get("max", value)
    raise ExecutionError(f"unsupported condition op {condition!r}")


def eval_condition_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        return resolve_path(value, context, missing_ok=True)
    try:
        return eval_value(value, context)
    except ExecutionError:
        return MISSING


def eval_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        if value.startswith("$"):
            resolved = resolve_path(value, context, missing_ok=False)
            return resolved
        return value
    if isinstance(value, list):
        return [eval_value(item, context) for item in value]
    if isinstance(value, dict):
        if "literal" in value:
            return value["literal"]
        if "get" in value:
            return resolve_path(value["get"], context, missing_ok=False)
        if "template" in value:
            return render_template(value["template"], context)
        if "filter_values" in value:
            return filter_values(value["filter_values"], context)
        return {key: eval_value(item, context) for key, item in value.items()}
    return value


def render_template(template: str, context: dict[str, Any]) -> Any:
    concat_match = re.fullmatch(r"\$\{([^}]+)\}\.concat\(\[([^]]+)\]\)", template)
    if concat_match:
        base_path = concat_match.group(1)
        item_expr = concat_match.group(2).strip()
        if not base_path.startswith("$"):
            base_path = f"${base_path}"
        base = resolve_path(base_path, context, missing_ok=False)
        if not isinstance(base, list):
            raise ExecutionError(f"template concat base {base_path!r} is not a list")
        item = eval_value(item_expr, context) if item_expr.startswith("$") else item_expr
        return base + [item]

    def replace(match: re.Match[str]) -> str:
        return stringify_template_value(eval_template_expr(match.group(1), context))

    rendered = re.sub(r"\$\{([^}]+)\}", replace, template)
    stripped = rendered.strip()
    if stripped.startswith(("{", "[")):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return rendered
    return rendered


def eval_template_expr(expr: str, context: dict[str, Any], *, missing_ok: bool = False) -> Any:
    expr = expr.strip()
    if expr.startswith("json_encode(") and expr.endswith(")"):
        inner = expr[len("json_encode(") : -1]
        return json_dumps_compact(eval_template_expr(inner, context, missing_ok=missing_ok))

    ternary = re.fullmatch(r"(.+?)\s*!=\s*null\s*\?\s*(.+?)\s*:\s*(.+)", expr)
    if ternary:
        condition_path, true_expr, false_expr = (part.strip() for part in ternary.groups())
        condition_value = eval_template_expr(condition_path, context, missing_ok=True)
        return eval_template_expr(true_expr if condition_value is not MISSING and condition_value is not None else false_expr, context, missing_ok=missing_ok)

    if expr in {"true", "false", "null"}:
        return {"true": True, "false": False, "null": None}[expr]
    if re.fullmatch(r"-?\d+", expr):
        return int(expr)
    if re.fullmatch(r"-?\d+\.\d+", expr):
        return float(expr)
    if (expr.startswith('"') and expr.endswith('"')) or (expr.startswith("'") and expr.endswith("'")):
        return expr[1:-1]

    path = expr if expr.startswith("$") else f"${expr}"
    return resolve_path(path, context, missing_ok=missing_ok)


def stringify_template_value(value: Any) -> str:
    if value is MISSING:
        raise ExecutionError("template expression resolved to missing value")
    if isinstance(value, (dict, list)):
        return json_dumps_compact(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def json_dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def filter_values(spec: dict[str, Any], context: dict[str, Any]) -> list[Any]:
    source = eval_value(spec["from"], context)
    if not isinstance(source, (dict, list)):
        raise ExecutionError("filter_values.from must resolve to an object or array")
    where = spec.get("where", True)
    select = spec.get("select", "$item")
    results = []
    iterator = source.items() if isinstance(source, dict) else enumerate(source)
    for key, item in iterator:
        child_context = {**context, "item": item, "key": key}
        if eval_condition(where, child_context):
            results.append(eval_value(select, child_context))
    return results


class _Missing:
    pass


MISSING = _Missing()


def resolve_path(path: str, context: dict[str, Any], missing_ok: bool) -> Any:
    if not isinstance(path, str) or not path.startswith("$"):
        raise ExecutionError(f"invalid path {path!r}")
    tokens = parse_path(path[1:])
    if not tokens:
        raise ExecutionError(f"empty path {path!r}")
    root = tokens[0]
    if root not in context:
        raise ExecutionError(f"unknown path root {root!r}")
    current = context[root]
    for token in tokens[1:]:
        key = eval_path_token(token, context)
        if isinstance(current, dict) and is_hashable_key(key) and key in current:
            current = current[key]
        elif isinstance(current, list) and isinstance(key, int) and 0 <= key < len(current):
            current = current[key]
        else:
            if missing_ok:
                return MISSING
            raise ExecutionError(f"path {path!r} missing at {key!r}")
    return current


def set_path(path: str, context: dict[str, Any], value: Any) -> None:
    if not path.startswith("$"):
        raise ExecutionError(f"invalid set path {path!r}")
    tokens = parse_path(path[1:])
    if not tokens or tokens[0] != "state":
        raise ExecutionError("effects may only write under $state")
    current = context["state"]
    for token in tokens[1:-1]:
        key = eval_path_token(token, context)
        if not is_hashable_key(key):
            raise ExecutionError(f"unhashable path key {key!r} in {path!r}")
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    final_key = eval_path_token(tokens[-1], context)
    if not is_hashable_key(final_key):
        raise ExecutionError(f"unhashable final path key {final_key!r} in {path!r}")
    if isinstance(current, dict):
        if isinstance(current.get(final_key), dict) and isinstance(value, dict):
            current[final_key].update(value)
        else:
            current[final_key] = value
        return
    if isinstance(current, list) and isinstance(final_key, int) and 0 <= final_key < len(current):
        current[final_key] = value
        return
    raise ExecutionError(f"cannot set path {path!r}")


def append_path(path: str, context: dict[str, Any], value: Any) -> None:
    target = resolve_path(path, context, missing_ok=False)
    if not isinstance(target, list):
        raise ExecutionError(f"append target {path!r} is not a list")
    target.append(value)


def apply_effect(effect: dict[str, Any], context: dict[str, Any]) -> None:
    if "set" in effect:
        set_path(effect["set"], context, eval_value(effect.get("value"), context))
        return
    if "append" in effect:
        append_path(effect["append"], context, eval_value(effect.get("value"), context))
        return
    if "delete" in effect:
        delete_path(effect["delete"], context)
        return
    raise ExecutionError(f"unsupported effect {effect!r}")


def delete_path(path: str, context: dict[str, Any]) -> None:
    tokens = parse_path(path[1:])
    if not tokens or tokens[0] != "state":
        raise ExecutionError("effects may only delete under $state")
    current = context["state"]
    for token in tokens[1:-1]:
        key = eval_path_token(token, context)
        if not is_hashable_key(key):
            raise ExecutionError(f"unhashable path key {key!r} in {path!r}")
        current = current[key]
    final_key = eval_path_token(tokens[-1], context)
    if isinstance(current, dict) and is_hashable_key(final_key):
        current.pop(final_key, None)


def is_hashable_key(value: Any) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


def parse_path(path: str) -> list[Any]:
    tokens: list[Any] = []
    index = 0
    buffer = ""
    while index < len(path):
        char = path[index]
        if char == ".":
            if buffer:
                tokens.append(buffer)
                buffer = ""
            index += 1
            continue
        if char == "[":
            if buffer:
                tokens.append(buffer)
                buffer = ""
            end = path.find("]", index)
            if end < 0:
                raise ExecutionError(f"unclosed bracket in path {path!r}")
            tokens.append({"expr": path[index + 1 : end]})
            index = end + 1
            continue
        buffer += char
        index += 1
    if buffer:
        tokens.append(buffer)
    return tokens


def eval_path_token(token: Any, context: dict[str, Any]) -> Any:
    if isinstance(token, dict) and "expr" in token:
        expr = token["expr"]
        if expr.startswith("$"):
            return resolve_path(expr, context, missing_ok=False)
        if "${" in expr:
            return render_template(expr, context)
        if expr.split(".", 1)[0] in context:
            return resolve_path(f"${expr}", context, missing_ok=False)
        if expr.isdigit():
            return int(expr)
        return expr.strip("\"'")
    return token


def replay_row(row: dict[str, Any]) -> dict[str, Any]:
    environment = build_environment_for_row(row)
    tool_names = {tool.get("function", {}).get("name", "") for tool in row.get("tools", [])}
    errors = validate_environment_spec(environment, tool_names=tool_names)
    state = copy.deepcopy(environment.get("initial_state", {}))
    steps = []
    messages = row.get("messages", [])
    for index, message in enumerate(messages):
        if message.get("role") != "assistant" or "tool_call" not in message:
            continue
        call = message["tool_call"]
        name = call.get("name")
        args = call.get("arguments", {})
        if index + 1 >= len(messages) or messages[index + 1].get("role") != "tool":
            errors.append(f"message {index}: assistant tool_call has no following tool response")
            continue
        actual_tool = messages[index + 1]
        if actual_tool.get("name") != name:
            errors.append(f"message {index + 1}: tool response name {actual_tool.get('name')!r} does not match {name!r}")
            continue
        expected, step_errors = execute_tool(name, args, state, environment)
        actual = actual_tool.get("content")
        if step_errors:
            errors.extend(f"message {index}: {error}" for error in step_errors)
        if expected != actual:
            errors.append(f"message {index + 1}: tool response mismatch; expected {expected!r}, got {actual!r}")
        steps.append({"message_index": index, "tool": name, "arguments": args, "expected_response": expected, "actual_response": actual})
    return {
        "id": row.get("id"),
        "valid": not errors,
        "errors": errors,
        "steps": steps,
        "final_state": state,
    }


def build_photo_environment(tool_set: set[str]) -> dict[str, Any]:
    if "copy_photo" in tool_set:
        return build_qwen32b_promptopt_photo_environment()
    if "copy_picture" in tool_set:
        return build_qwen_smoke_photo_environment()
    return {
        "version": DSL_VERSION,
        "kind": "photo_printing",
        "initial_state": {
            "images": {
                "/home/vacation.jpg": {"copied_from": None, "edited": False},
            },
            "editor_sessions": {},
            "printers": {
                "printer_A": {"status": "offline"},
                "printer_B": {"status": "online"},
            },
            "print_jobs": {},
        },
        "tool_rules": {
            "copy_image": {
                "branches": [
                    {
                        "if": {"exists": "$state.images[$args.image_path]"},
                        "response": {"copy_path": "/tmp/vacation_copy.jpg"},
                        "effects": [{"set": "$state.images[$response.copy_path]", "value": {"copied_from": "$args.image_path", "edited": False}}],
                    }
                ]
            },
            "open_editor": {
                "branches": [
                    {
                        "if": {"exists": "$state.images[$args.copy_path]"},
                        "response": {"editor_id": "editor_41"},
                        "effects": [{"set": "$state.editor_sessions[$response.editor_id]", "value": {"image_path": "$args.copy_path"}}],
                    }
                ]
            },
            "add_text": {
                "branches": [
                    {
                        "if": {"all": [{"exists": "$state.editor_sessions[$args.editor_id]"}, {"range": {"value": "$args.font_size", "min": 8, "max": 96}}]},
                        "response": {"image_path": "/tmp/vacation_copy_edited.jpg"},
                        "effects": [{"set": "$state.images[$response.image_path]", "value": {"copied_from": "$state.editor_sessions[$args.editor_id].image_path", "edited": True}}],
                    }
                ]
            },
            "print_image": {
                "branches": [
                    {
                        "if": {"eq": ["$state.printers[$args.printer_id].status", "offline"]},
                        "response": {"status": "failed", "error": "printer offline"},
                    },
                    {
                        "if": {"eq": ["$state.printers[$args.printer_id].status", "online"]},
                        "response": {"status": "queued", "job_id": "job_778"},
                        "effects": [{"set": "$state.print_jobs[$response.job_id]", "value": {"image_path": "$args.image_path", "printer_id": "$args.printer_id"}}],
                    },
                ]
            },
            "list_available_printers": {
                "branches": [
                    {
                        "if": True,
                        "response": {
                            "available_printers": {
                                "filter_values": {
                                    "from": "$state.printers",
                                    "where": {"eq": ["$item.status", "online"]},
                                    "select": "$key",
                                }
                            }
                        },
                    }
                ]
            },
        },
    }


def build_qwen_smoke_photo_environment() -> dict[str, Any]:
    return {
        "version": DSL_VERSION,
        "kind": "photo_printing",
        "initial_state": {
            "images": {
                "/home/user/photos/birthday.jpg": {"edited": False},
                "/home/user/photos/birthday_copy.jpg": {"edited": False},
                "/home/user/photos/birthday_copy_edited.jpg": {"edited": True},
            },
            "printers": {
                "HP_LaserJet_4000": {"status": "offline"},
                "Canon_MF269dw": {"status": "online"},
                "Epson_Expression_3450": {"status": "online"},
            },
        },
        "tool_rules": {
            "copy_picture": {
                "branches": [{"if": {"exists": "$state.images[$args.image_path]"}, "response": {"copied_path": "/home/user/photos/birthday_copy.jpg"}}]
            },
            "edit_image": {
                "branches": [{"if": {"exists": "$state.images[$args.image_path]"}, "response": {"edited_path": "/home/user/photos/birthday_copy_edited.jpg"}}]
            },
            "print": {
                "branches": [
                    {"if": {"eq": ["$state.printers[$args.printer].status", "offline"]}, "response": {"status": "failed", "error": {"template": "Printer ${args.printer} is offline."}}},
                    {"if": {"eq": ["$state.printers[$args.printer].status", "online"]}, "response": {"status": "success", "message": {"template": "Print job completed successfully on ${args.printer}."}}},
                ]
            },
            "find_available_printer": {
                "branches": [{"if": True, "response": {"available_printers": {"filter_values": {"from": "$state.printers", "where": {"eq": ["$item.status", "online"]}, "select": "$key"}}}}]
            },
        },
    }


def build_qwen32b_promptopt_photo_environment() -> dict[str, Any]:
    return {
        "version": DSL_VERSION,
        "kind": "photo_printing",
        "initial_state": {
            "images": {
                "/photos/birthday.jpg": {"edited": False},
                "/photos/birthday_copy.jpg": {"edited": False},
            },
            "printers": {
                "printer_01": {"status": "offline", "name": "HP LaserJet Pro"},
                "printer_02": {"status": "online", "name": "Canon PIXMA"},
            },
        },
        "tool_rules": {
            "copy_photo": {
                "branches": [{"if": {"exists": "$state.images[$args.photo_path]"}, "response": {"copied_photo_path": "/photos/birthday_copy.jpg"}}]
            },
            "edit_photo": {
                "branches": [{"if": {"exists": "$state.images[$args.photo_path]"}, "response": {"status": "success", "photo_path": "$args.photo_path"}}]
            },
            "set_text_properties": {
                "branches": [{"if": {"exists": "$state.images[$args.photo_path]"}, "response": {"status": "success", "photo_path": "$args.photo_path"}}]
            },
            "print_photo": {
                "branches": [
                    {"if": {"eq": ["$state.printers[$args.printer_id].status", "offline"]}, "response": {"status": "failed", "error": "Printer offline"}},
                    {"if": {"eq": ["$state.printers[$args.printer_id].status", "online"]}, "response": {"status": "success", "printer_id": "$args.printer_id"}},
                ]
            },
            "list_available_printers": {
                "branches": [
                    {
                        "if": True,
                        "response": {
                            "printers": [
                                {"id": "printer_01", "name": "HP LaserJet Pro"},
                                {"id": "printer_02", "name": "Canon PIXMA"},
                            ]
                        },
                    }
                ]
            },
        },
    }


def build_return_environment() -> dict[str, Any]:
    return {
        "version": DSL_VERSION,
        "kind": "online_return",
        "initial_state": {
            "customers": {"alex@example.com": {"customer_id": "cust_18"}},
            "orders": {
                "R100": {
                    "customer_id": "cust_18",
                    "status": "delivered",
                    "days_since_delivery": 12,
                    "items": [{"item_id": "i9", "name": "moisturizer", "category": "cosmetics", "opened": True}],
                }
            },
            "items": {"i9": {"category": "cosmetics", "opened": True}},
            "return_labels": {},
            "pickups": {},
        },
        "tool_rules": {
            "sign_in": {"branches": [{"if": {"exists": "$state.customers[$args.email]"}, "response": {"customer_id": "$state.customers[$args.email].customer_id"}}]},
            "get_order": {
                "branches": [
                    {
                        "if": {"exists": "$state.orders[$args.order_id]"},
                        "response": {
                            "status": "$state.orders[$args.order_id].status",
                            "days_since_delivery": "$state.orders[$args.order_id].days_since_delivery",
                            "items": "$state.orders[$args.order_id].items",
                        },
                    }
                ]
            },
            "check_return_eligibility": {
                "branches": [
                    {
                        "if": {"all": [{"eq": ["$state.items[$args.item_id].category", "cosmetics"]}, {"eq": ["$state.items[$args.item_id].opened", True]}, {"not": {"eq": ["$args.reason", "damaged"]}}]},
                        "response": {"eligible": False, "reason": "opened cosmetics require damage reason"},
                    },
                    {"if": True, "response": {"eligible": True, "reason": "eligible"}},
                ]
            },
            "create_return_label": {
                "branches": [{"if": True, "response": {"label_id": "label_42"}, "effects": [{"set": "$state.return_labels[$response.label_id]", "value": {"order_id": "$args.order_id", "item_id": "$args.item_id"}}]}]
            },
            "schedule_pickup": {
                "branches": [{"if": {"exists": "$state.return_labels[$args.label_id]"}, "response": {"status": "scheduled", "pickup_id": "pickup_42"}, "effects": [{"set": "$state.pickups[$response.pickup_id]", "value": {"label_id": "$args.label_id", "pickup_date": "$args.pickup_date"}}]}]
            },
        },
    }


def build_course_environment() -> dict[str, Any]:
    return {
        "version": DSL_VERSION,
        "kind": "course_enrollment",
        "initial_state": {
            "students": {"S123": {"completed_courses": []}},
            "courses": {"CS540": {"title": "Advanced Databases", "seats_available": 3, "prerequisites": ["CS340"]}},
            "approvals": {},
            "enrollments": [],
        },
        "tool_rules": {
            "portal_login": {"branches": [{"if": {"exists": "$state.students[$args.student_id]"}, "response": {"status": "ok"}}]},
            "search_course": {"branches": [{"if": {"eq": ["$args.query", "Advanced Databases"]}, "response": {"course_id": "CS540", "seats_available": 3}}]},
            "check_prerequisites": {"branches": [{"if": True, "response": {"met": False, "missing": ["CS340"]}}]},
            "request_instructor_approval": {"branches": [{"if": True, "response": {"approved": True, "approval_id": "ap_77"}, "effects": [{"set": "$state.approvals[$response.approval_id]", "value": {"student_id": "$args.student_id", "course_id": "$args.course_id", "approved": True}}]}]},
            "confirm_enrollment": {"branches": [{"if": True, "response": {"status": "enrolled"}, "effects": [{"append": "$state.enrollments", "value": {"student_id": "$args.student_id", "course_id": "$args.course_id"}}]}]},
        },
    }
