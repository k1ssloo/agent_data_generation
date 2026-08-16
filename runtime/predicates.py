"""Small declarative expression evaluator used by the causal runtime."""

from __future__ import annotations

import re
from typing import Any


class EvaluationError(ValueError):
    pass


class _Missing:
    pass


MISSING = _Missing()
_TOKEN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)|\[([^\]]+)\]")
PREDICATE_OPERATORS = {
    "all",
    "any",
    "not",
    "exists",
    "not_exists",
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "in",
}


def _tokens(path: str) -> list[tuple[str, bool]]:
    if not path.startswith("$"):
        raise EvaluationError(f"invalid path {path!r}")
    return [(plain or bracket, bool(bracket)) for plain, bracket in _TOKEN.findall(path[1:])]


def resolve_path(path: str, context: dict[str, Any], *, missing_ok: bool = False) -> Any:
    tokens = _tokens(path)
    if not tokens or tokens[0][0] not in context:
        raise EvaluationError(f"unknown path root in {path!r}")
    current: Any = context[tokens[0][0]]
    for raw, dynamic in tokens[1:]:
        key: Any
        if dynamic and raw.startswith("$"):
            key = resolve_path(raw, context, missing_ok=missing_ok)
        elif dynamic and raw in context.get("args", {}):
            key = context["args"][raw]
        elif raw.isdigit() and isinstance(current, list):
            key = int(raw)
        else:
            key = raw.strip("\"'")
        if isinstance(current, dict):
            try:
                found = key in current
            except TypeError as exc:
                raise EvaluationError(
                    f"path {path!r} resolved non-scalar map key {key!r}"
                ) from exc
            if found:
                current = current[key]
            elif missing_ok:
                return MISSING
            else:
                raise EvaluationError(f"path {path!r} missing at {key!r}")
        elif isinstance(current, list) and isinstance(key, int) and 0 <= key < len(current):
            current = current[key]
        elif missing_ok:
            return MISSING
        else:
            raise EvaluationError(f"path {path!r} missing at {key!r}")
    return current


def evaluate_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        return resolve_path(value, context)
    if isinstance(value, list):
        return [evaluate_value(item, context) for item in value]
    if isinstance(value, dict):
        return {key: evaluate_value(item, context) for key, item in value.items()}
    return value


def evaluate_predicate(predicate: Any, context: dict[str, Any]) -> bool:
    if isinstance(predicate, bool):
        return predicate
    if not isinstance(predicate, dict) or len(predicate) != 1:
        raise EvaluationError(f"invalid predicate {predicate!r}")
    operator, value = next(iter(predicate.items()))
    if operator == "all":
        return all(evaluate_predicate(item, context) for item in value)
    if operator == "any":
        return any(evaluate_predicate(item, context) for item in value)
    if operator == "not":
        return not evaluate_predicate(value, context)
    if operator in {"exists", "not_exists"}:
        found = resolve_path(value, context, missing_ok=True) is not MISSING
        return found if operator == "exists" else not found
    if operator in {"eq", "ne", "gt", "gte", "lt", "lte", "in"}:
        if not isinstance(value, list) or len(value) != 2:
            raise EvaluationError(f"{operator} requires two values")
        left = evaluate_value(value[0], context)
        right = evaluate_value(value[1], context)
        operations = {
            "eq": lambda: left == right,
            "ne": lambda: left != right,
            "gt": lambda: left > right,
            "gte": lambda: left >= right,
            "lt": lambda: left < right,
            "lte": lambda: left <= right,
            "in": lambda: left in right,
        }
        try:
            return bool(operations[operator]())
        except (TypeError, ValueError) as exc:
            raise EvaluationError(
                f"cannot evaluate {operator} for {left!r} and {right!r}"
            ) from exc
    raise EvaluationError(f"unsupported predicate operator {operator!r}")


def validate_predicate_syntax(predicate: Any) -> list[str]:
    """Validate predicate structure without resolving state or argument paths."""
    errors: list[str] = []

    def visit(value: Any, location: str) -> None:
        if isinstance(value, bool):
            return
        if not isinstance(value, dict) or len(value) != 1:
            errors.append(f"{location} must be a boolean or one-operator object")
            return
        operator, operand = next(iter(value.items()))
        if operator not in PREDICATE_OPERATORS:
            errors.append(f"{location} uses unsupported predicate operator {operator!r}")
            return
        if operator in {"all", "any"}:
            if not isinstance(operand, list) or not operand:
                errors.append(f"{location}.{operator} must be a non-empty list")
                return
            for index, item in enumerate(operand):
                visit(item, f"{location}.{operator}[{index}]")
            return
        if operator == "not":
            visit(operand, f"{location}.not")
            return
        if operator in {"exists", "not_exists"}:
            if not isinstance(operand, str) or not operand.startswith("$"):
                errors.append(f"{location}.{operator} must be a $-rooted path")
            return
        if not isinstance(operand, list) or len(operand) != 2:
            errors.append(f"{location}.{operator} must contain exactly two values")

    visit(predicate, "predicate")
    return errors


def predicate_paths(value: Any) -> set[str]:
    paths: set[str] = set()
    if isinstance(value, str) and value.startswith("$state"):
        paths.add(value)
    elif isinstance(value, list):
        for item in value:
            paths |= predicate_paths(item)
    elif isinstance(value, dict):
        for item in value.values():
            paths |= predicate_paths(item)
    return paths
