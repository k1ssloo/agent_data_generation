"""Conservatively complete declared causal-runtime state shape."""

from __future__ import annotations

import copy
import re
from typing import Any

from runtime.predicates import MISSING, predicate_paths, resolve_path


_SIMPLE_STATE_PATH = re.compile(r"^\$state(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")


def _existence_paths(value: Any, operators: set[str] | None = None) -> set[str]:
    operators = operators or {"exists", "not_exists"}
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in operators and isinstance(item, str):
                paths.add(item)
            else:
                paths |= _existence_paths(item, operators)
    elif isinstance(value, list):
        for item in value:
            paths |= _existence_paths(item, operators)
    return paths


def _state_references(contract: dict[str, Any], environment: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for item in contract.get("goal_predicates", []):
        if isinstance(item, dict):
            paths |= predicate_paths(item.get("predicate", item))
    for item in contract.get("invariants", []):
        if isinstance(item, dict):
            paths |= predicate_paths(item.get("predicate", item))
    for capability in environment.get("capabilities", {}).values():
        if not isinstance(capability, dict):
            continue
        for branch in capability.get("branches", []):
            if not isinstance(branch, dict):
                continue
            paths |= predicate_paths(branch.get("when", True))
            paths |= predicate_paths(branch.get("response", {}))
    return paths


def _written_paths(environment: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for capability in environment.get("capabilities", {}).values():
        if not isinstance(capability, dict):
            continue
        for branch in capability.get("branches", []):
            if not isinstance(branch, dict):
                continue
            for effect in branch.get("effects", []):
                if not isinstance(effect, dict):
                    continue
                for operation in ("set", "increment", "delete"):
                    target = effect.get(operation)
                    if isinstance(target, str) and target.startswith("$state."):
                        paths.add(target)
    return paths


def _has_writer(path: str, writers: set[str]) -> bool:
    return any(path == writer or path.startswith(writer + ".") for writer in writers)


def _existence_sensitive(path: str, existence_paths: set[str]) -> bool:
    return any(
        path == item or path.startswith(item + ".") or item.startswith(path + ".")
        for item in existence_paths
    )


def _set_null(initial_state: dict[str, Any], path: str) -> bool:
    parts = path[len("$state.") :].split(".")
    current = initial_state
    for part in parts[:-1]:
        existing = current.get(part)
        if existing is None and part not in current:
            current[part] = {}
            existing = current[part]
        if not isinstance(existing, dict):
            return False
        current = existing
    if parts[-1] in current:
        return False
    current[parts[-1]] = None
    return True


def _delete_null(initial_state: dict[str, Any], path: str) -> bool:
    parts = path[len("$state.") :].split(".")
    current: Any = initial_state
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    if not isinstance(current, dict) or current.get(parts[-1], MISSING) is not None:
        return False
    del current[parts[-1]]
    return True


def complete_initial_state_schema(
    contract: dict[str, Any], candidate: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Add null placeholders only for referenced paths created by later effects."""
    result = copy.deepcopy(candidate)
    environment = result.get("environment", {})
    initial_state = environment.get("initial_state")
    if not isinstance(initial_state, dict):
        return result, []
    references = _state_references(contract, environment)
    writers = _written_paths(environment)
    existence_paths = _existence_paths(contract) | _existence_paths(
        environment.get("capabilities", {})
    )
    absent_until_created = _existence_paths(
        contract, {"exists", "not_exists"}
    ) | _existence_paths(
        environment.get("capabilities", {}), {"exists", "not_exists"}
    )
    completed: list[str] = []
    for path in sorted(absent_until_created):
        if (
            _SIMPLE_STATE_PATH.fullmatch(path)
            and _has_writer(path, writers)
            and _delete_null(initial_state, path)
        ):
            completed.append(f"delete-null:{path}")
    for path in sorted(references):
        if not _SIMPLE_STATE_PATH.fullmatch(path):
            continue
        if resolve_path(path, {"state": initial_state}, missing_ok=True) is not MISSING:
            continue
        if not _has_writer(path, writers) or _existence_sensitive(path, existence_paths):
            continue
        if _set_null(initial_state, path):
            completed.append(path)
    return result, completed


__all__ = ["complete_initial_state_schema"]
