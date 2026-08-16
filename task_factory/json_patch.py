"""Apply a small, restricted JSON Patch document to generated task artifacts."""

from __future__ import annotations

import copy
from typing import Any


class JsonPatchError(ValueError):
    pass


def _parts(path: Any) -> list[str]:
    if not isinstance(path, str) or not path.startswith("/"):
        raise JsonPatchError("patch path must be a non-root JSON Pointer")
    return [part.replace("~1", "/").replace("~0", "~") for part in path[1:].split("/")]


def _list_index(raw: str, size: int, *, allow_end: bool) -> int:
    if raw == "-" and allow_end:
        return size
    if not raw.isdigit() or (len(raw) > 1 and raw.startswith("0")):
        raise JsonPatchError(f"invalid list index {raw!r}")
    index = int(raw)
    limit = size if allow_end else size - 1
    if index < 0 or index > limit:
        raise JsonPatchError(f"list index {index} is out of range")
    return index


def _parent(document: Any, parts: list[str]) -> tuple[Any, str]:
    current = document
    for part in parts[:-1]:
        if isinstance(current, dict):
            if part not in current:
                raise JsonPatchError(f"patch parent path does not exist at {part!r}")
            current = current[part]
        elif isinstance(current, list):
            current = current[_list_index(part, len(current), allow_end=False)]
        else:
            raise JsonPatchError("patch path traverses a scalar value")
    return current, parts[-1]


def apply_json_patch(document: Any, operations: Any) -> Any:
    """Apply add/replace/remove atomically and return a deep-copied result."""
    if not isinstance(operations, list) or not operations:
        raise JsonPatchError("operations must be a non-empty list")
    if len(operations) > 64:
        raise JsonPatchError("patch contains more than 64 operations")
    result = copy.deepcopy(document)
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            raise JsonPatchError(f"operation {index} must be an object")
        kind = operation.get("op")
        if kind not in {"add", "replace", "remove"}:
            raise JsonPatchError(f"operation {index} uses unsupported op {kind!r}")
        parts = _parts(operation.get("path"))
        if parts[0] not in {"instruction", "environment", "bindings", "reference_plan"}:
            raise JsonPatchError(f"operation {index} targets a forbidden top-level field")
        parent, key = _parent(result, parts)
        if isinstance(parent, dict):
            exists = key in parent
            if kind in {"replace", "remove"} and not exists:
                raise JsonPatchError(f"operation {index} target does not exist")
            if kind == "remove":
                del parent[key]
            else:
                if "value" not in operation:
                    raise JsonPatchError(f"operation {index} requires value")
                parent[key] = copy.deepcopy(operation["value"])
        elif isinstance(parent, list):
            item_index = _list_index(key, len(parent), allow_end=kind == "add")
            if kind == "add":
                if "value" not in operation:
                    raise JsonPatchError(f"operation {index} requires value")
                parent.insert(item_index, copy.deepcopy(operation["value"]))
            elif kind == "replace":
                if "value" not in operation:
                    raise JsonPatchError(f"operation {index} requires value")
                parent[item_index] = copy.deepcopy(operation["value"])
            else:
                del parent[item_index]
        else:
            raise JsonPatchError(f"operation {index} target parent is scalar")
    return result


__all__ = ["JsonPatchError", "apply_json_patch"]
