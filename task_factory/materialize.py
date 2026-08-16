"""Materialize generated task bundle candidates into an isolated directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .bundle import TaskBundle, validate_bundle


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def materialize_candidate(
    output_dir: Path,
    *,
    task_id: str,
    contract: dict[str, Any],
    candidate: dict[str, Any],
    lineage: dict[str, Any] | None = None,
    manifest_metadata: dict[str, Any] | None = None,
) -> Path:
    required = {"instruction", "environment", "bindings", "reference_plan"}
    missing = sorted(required - set(candidate))
    if missing:
        raise ValueError(f"candidate missing fields: {missing}")
    manifest = {
        "bundle_version": "task-bundle-v1",
        "task_id": task_id,
        "instruction_file": "instruction.md",
        "contract_file": "contract.json",
        "environment_file": "environment/environment.json",
        "bindings_file": "capabilities/bindings.json",
        "reference_plan_file": "solution/reference_plan.json",
        "lineage": lineage or {},
    }
    reserved = set(manifest)
    for key, value in (manifest_metadata or {}).items():
        if key not in reserved:
            manifest[key] = value
    instruction = candidate["instruction"]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("candidate instruction must be a non-empty string")
    root = output_dir / task_id
    bundle = TaskBundle(
        root=root,
        manifest=manifest,
        instruction=instruction,
        contract=contract,
        environment=candidate["environment"],
        bindings=candidate["bindings"],
        reference_plan=candidate["reference_plan"],
    )
    errors = validate_bundle(bundle)
    if errors:
        raise ValueError("candidate bundle is invalid: " + "; ".join(errors))
    root.mkdir(parents=True, exist_ok=False)
    _write_json(root / "manifest.json", manifest)
    (root / "instruction.md").write_text(instruction.strip() + "\n", encoding="utf-8")
    _write_json(root / "contract.json", contract)
    _write_json(root / "environment" / "environment.json", candidate["environment"])
    _write_json(root / "capabilities" / "bindings.json", candidate["bindings"])
    _write_json(root / "solution" / "reference_plan.json", candidate["reference_plan"])
    return root
