#!/usr/bin/env python3
"""Validate a portable GEM rLLM package before reserving GPUs."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_validation import validate_episode, validate_tool_identifiability
from rollout import run_reference_plan
from task_factory import load_task_bundle, validate_public_executability
from training import GemTaskEnvironment
from training.package import anchor_portable_rows, resolve_bundle_path, tree_digest


PRIVATE_KEYS = {
    "initial_state",
    "reference_plan",
    "goal_predicates",
    "invariants",
    "environment",
    "contract",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number} must contain an object")
        rows.append(value)
    return rows


def private_keys(value: Any, prefix: str = "$") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}"
            if key in PRIVATE_KEYS:
                found.append(path)
            found.extend(private_keys(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(private_keys(item, f"{prefix}[{index}]"))
    return found


def package_digest(sft_path: Path, rl_path: Path, digests: dict[str, str]) -> str:
    digest = hashlib.sha256()
    digest.update(sft_path.read_bytes())
    digest.update(rl_path.read_bytes())
    for task_id, value in sorted(digests.items()):
        digest.update(task_id.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


def check_rllm_contract() -> dict[str, Any]:
    try:
        import rllm
        from rllm.trainer import AgentTrainer
    except ImportError as exc:
        raise ValueError(f"rLLM import failed: {exc}") from exc
    signature = inspect.signature(AgentTrainer)
    required = {"agent_flow", "evaluator", "train_dataset"}
    missing = sorted(required - set(signature.parameters))
    if missing:
        raise ValueError(f"rLLM AgentTrainer lacks parameters: {missing}")
    from training.rllm_adapter import gem_causal_evaluator, gem_tool_rollout

    if gem_tool_rollout is None or gem_causal_evaluator is None:
        raise ValueError("GEM rLLM decorators did not materialize")
    return {
        "version": getattr(rllm, "__version__", "unknown"),
        "agent_trainer_parameters": sorted(signature.parameters),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--require-rllm", action="store_true")
    parser.add_argument("--max-bundles", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    package_dir = args.package_dir.resolve()
    manifest_path = package_dir / "manifest.json"
    sft_path = package_dir / "sft.jsonl"
    rl_path = package_dir / "rl_tasks.jsonl"
    manifest = load_json(manifest_path)
    sft_rows = load_jsonl(sft_path)
    rl_rows = load_jsonl(rl_path)
    errors: list[str] = []
    if len(sft_rows) != len(rl_rows):
        errors.append(f"SFT/RL row mismatch: {len(sft_rows)} != {len(rl_rows)}")
    if manifest.get("sft_rows") != len(sft_rows):
        errors.append("manifest.sft_rows does not match sft.jsonl")
    if manifest.get("rl_tasks") != len(rl_rows):
        errors.append("manifest.rl_tasks does not match rl_tasks.jsonl")
    leaked = [
        {"row": index, "paths": private_keys(row)}
        for index, row in enumerate(sft_rows)
        if private_keys(row)
    ]
    if leaked:
        errors.append(f"private fields found in SFT rows: {leaked[:3]}")

    anchor_portable_rows(rl_rows, rl_path)
    expected_digests = manifest.get("bundle_digests", {})
    if not isinstance(expected_digests, dict):
        expected_digests = {}
        errors.append("manifest.bundle_digests must be an object")
    checked = []
    seen_ids: set[str] = set()
    for row in rl_rows[: args.max_bundles or None]:
        task_id = str(row.get("id", ""))
        if not task_id or task_id in seen_ids:
            errors.append(f"missing or duplicate RL task ID: {task_id!r}")
            continue
        seen_ids.add(task_id)
        metadata = row.get("metadata", {})
        try:
            path = resolve_bundle_path(metadata, fallback_root=PROJECT_ROOT)
            actual_digest = tree_digest(path)
            expected_digest = expected_digests.get(task_id)
            if expected_digest and actual_digest != expected_digest:
                raise ValueError("bundle digest does not match manifest")
            bundle = load_task_bundle(path)
            report = run_reference_plan(bundle)
            causal = validate_episode(bundle, report)
            identifiability = validate_tool_identifiability(bundle)
            public = validate_public_executability(bundle)
            environment = GemTaskEnvironment(bundle)
            context = environment.reset()
            for action in environment.bundle.reference_plan["actions"]:
                environment.step(action["tool"], action.get("arguments", {}))
            score = environment.score(environment.finish())
            if (
                not causal["valid"]
                or not identifiability["valid"]
                or not public["valid"]
                or not score["is_correct"]
            ):
                raise ValueError(
                    "reference execution failed causal/identifiability/public/reward validation"
                )
            if set(context) != {"task_id", "messages", "tools", "remaining_steps"}:
                raise ValueError("policy context has an unexpected visibility boundary")
            checked.append(
                {
                    "id": task_id,
                    "bundle": str(path.relative_to(package_dir)),
                    "reference_steps": len(bundle.reference_plan["actions"]),
                    "reward": score["reward"],
                }
            )
        except (OSError, ValueError) as exc:
            errors.append(f"{task_id}: {exc}")

    if not args.max_bundles:
        actual_package_digest = package_digest(sft_path, rl_path, expected_digests)
        if manifest.get("package_sha256") != actual_package_digest:
            errors.append("package_sha256 does not match package contents")
    rllm_contract = None
    if args.require_rllm:
        try:
            rllm_contract = check_rllm_contract()
        except ValueError as exc:
            errors.append(str(exc))
    result = {
        "valid": not errors,
        "errors": errors,
        "package": str(package_dir),
        "dataset_tier": manifest.get("dataset_tier"),
        "sft_rows": len(sft_rows),
        "rl_tasks": len(rl_rows),
        "bundles_checked": len(checked),
        "checks": checked,
        "rllm_contract": rllm_contract,
    }
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
