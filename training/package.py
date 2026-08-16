"""Portable training-package helpers with no rLLM dependency."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def tree_digest(root: Path) -> str:
    """Hash file names and bytes for one task bundle directory."""

    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def portable_bundle_name(task_id: str) -> str:
    """Return a safe single-directory bundle name or reject the task ID."""

    if (
        not task_id
        or task_id in {".", ".."}
        or "/" in task_id
        or "\\" in task_id
        or Path(task_id).name != task_id
    ):
        raise ValueError(f"task ID is not a safe portable directory name: {task_id!r}")
    return task_id


def portable_relative_path(value: str, *, field: str) -> Path:
    """Return a normalized package-relative path without parent traversal."""

    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field} is not a safe relative path: {value!r}")
    return path


def anchor_portable_rows(rows: list[Any], dataset_path: str | Path) -> None:
    """Attach an in-memory package root to portable RL rows.

    The absolute path is deliberately injected only after loading on the target
    machine. It is never serialized into ``rl_tasks.jsonl``.
    """

    package_root = str(Path(dataset_path).resolve().parent)
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("every RL dataset row must be an object")
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("every RL dataset row must contain metadata")
        if metadata.get("bundle_path_base") == "training_package":
            metadata["training_package_root"] = package_root


def resolve_bundle_path(metadata: dict[str, Any], *, fallback_root: Path) -> Path:
    """Resolve one bundle locator without allowing package-directory escape."""

    value = metadata.get("bundle_path")
    if not isinstance(value, str) or not value:
        raise ValueError("rLLM task metadata must contain bundle_path")
    path = Path(value)
    base_mode = metadata.get("bundle_path_base", "project_root")
    if base_mode == "training_package":
        if path.is_absolute():
            raise ValueError("portable training-package bundle_path must be relative")
        root_value = metadata.get("training_package_root")
        if not isinstance(root_value, str) or not root_value:
            raise ValueError(
                "portable rLLM task metadata requires training_package_root; "
                "load it through training.train_rllm"
            )
        root = Path(root_value).resolve()
    elif base_mode == "project_root":
        if path.is_absolute():
            return path
        root = Path(metadata.get("project_root", fallback_root)).resolve()
    else:
        raise ValueError(f"unknown bundle_path_base: {base_mode!r}")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"bundle_path escapes its declared base: {value!r}") from exc
    return resolved
