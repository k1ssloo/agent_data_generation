#!/usr/bin/env python3
"""Export validated bundles for rLLM SFT and online agentic RL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_validation import validate_episode, validate_tool_identifiability
from rollout import run_reference_plan
from scripts.export_task_first_sft import build_sft_row
from task_factory import (
    load_task_bundle,
    totalize_public_capabilities,
    validate_public_executability,
)
from training.package import portable_bundle_name, portable_relative_path, tree_digest


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def tool_catalog(tools: list[dict[str, Any]]) -> str:
    return json.dumps(tools, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def inject_tools(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = [dict(message) for message in messages]
    catalog = (
        "\n\nAvailable tools (OpenAI function schema; use only these tools):\n"
        + tool_catalog(tools)
    )
    if result and result[0].get("role") == "system":
        result[0]["content"] = str(result[0].get("content", "")) + catalog
    else:
        result.insert(0, {"role": "system", "content": catalog.strip()})
    return result


def relative_bundle_path(bundle_root: Path, project_root: Path) -> str:
    try:
        return str(bundle_root.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(bundle_root.resolve())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def copy_bundle(bundle: Any, output_dir: Path) -> tuple[str, str]:
    bundle_dir = output_dir / "bundles"
    portable_name = portable_bundle_name(bundle.task_id)
    target = bundle_dir / portable_name
    expected_files = {
        Path("manifest.json"): bundle.manifest,
        Path("contract.json"): bundle.contract,
        portable_relative_path(
            str(bundle.manifest["environment_file"]), field="environment_file"
        ): bundle.environment,
        portable_relative_path(
            str(bundle.manifest["bindings_file"]), field="bindings_file"
        ): bundle.bindings,
        portable_relative_path(
            str(bundle.manifest["reference_plan_file"]), field="reference_plan_file"
        ): bundle.reference_plan,
    }
    instruction_file = str(bundle.manifest["instruction_file"])
    instruction_relative = portable_relative_path(
        instruction_file, field="instruction_file"
    )
    if target.exists():
        shutil.rmtree(target)
    else:
        bundle_dir.mkdir(parents=True, exist_ok=True)
    for relative, value in expected_files.items():
        write_json(target / relative, value)
    instruction_path = target / instruction_relative
    instruction_path.parent.mkdir(parents=True, exist_ok=True)
    instruction_path.write_text(bundle.instruction.strip() + "\n", encoding="utf-8")
    return str(Path("bundles") / portable_name), tree_digest(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--limit", type=int, default=0, help="0 exports every valid row.")
    parser.add_argument(
        "--dataset-tier",
        choices=["base", "adaptive", "vnext"],
        default="base",
        help="Validated quality tier represented by the supplied validation files.",
    )
    parser.add_argument(
        "--no-inject-tools",
        action="store_true",
        help="Do not add tool schemas to the SFT system message.",
    )
    parser.add_argument(
        "--no-copy-bundles",
        action="store_true",
        help=(
            "Keep bundle locators pointed at the source repository instead of "
            "building a self-contained training package."
        ),
    )
    args = parser.parse_args()

    candidates = [
        row
        for path in args.validation
        for row in load_jsonl(path)
        if row.get("valid") is True and isinstance(row.get("bundle"), str)
    ]
    seen: set[str] = set()
    sft_rows = []
    rl_rows = []
    rejected = []
    bundle_digests: dict[str, str] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle_output = args.output_dir / "bundles"
    if bundle_output.exists() and not args.no_copy_bundles:
        shutil.rmtree(bundle_output)
    for candidate in candidates:
        bundle = totalize_public_capabilities(
            load_task_bundle(Path(candidate["bundle"]))
        )
        if bundle.task_id in seen:
            continue
        report = run_reference_plan(bundle)
        validation = validate_episode(bundle, report)
        identifiability = validate_tool_identifiability(bundle)
        public = validate_public_executability(bundle)
        if (
            not validation["valid"]
            or not identifiability["valid"]
            or not public["valid"]
        ):
            rejected.append(bundle.task_id)
            continue
        value = {
            "generation_mode": "hidden_environment_rollout",
            "episode": report,
            "validation": validation,
            "source_id": bundle.manifest.get("source_id", bundle.task_id),
            "semantic_episode_id": bundle.task_id,
            "recursive_generation": int(
                bundle.manifest.get("lineage", {}).get("generation", 0)
            ),
            "recursive_operators": list(
                bundle.manifest.get("lineage", {}).get("operators", [])
            ),
            "renderer_seed": "canonical",
            "tool_identifiability": identifiability,
        }
        sft = build_sft_row(value)
        if sft is None:
            rejected.append(bundle.task_id)
            continue
        seen.add(bundle.task_id)
        if not args.no_inject_tools:
            sft["messages"] = inject_tools(sft["messages"], sft["tools"])
            sft["metadata"]["sft_tool_schema_transport"] = "system_catalog"
        sft["metadata"]["dataset_tier"] = args.dataset_tier
        if args.no_copy_bundles:
            bundle_path = relative_bundle_path(bundle.root, args.project_root)
            locator = {
                "bundle_path": bundle_path,
                "project_root": str(args.project_root.resolve()),
                "bundle_path_base": "project_root",
            }
        else:
            bundle_path, bundle_digest = copy_bundle(bundle, args.output_dir)
            bundle_digests[bundle.task_id] = bundle_digest
            locator = {
                "bundle_path": bundle_path,
                "bundle_path_base": "training_package",
            }
        rl = {
            "id": bundle.task_id,
            "instruction": bundle.instruction.strip(),
            "metadata": {
                **locator,
                "source_id": value["source_id"],
                "semantic_episode_id": bundle.task_id,
                "reference_steps": len(bundle.reference_plan["actions"]),
                "causal_metrics": sft["metadata"]["causal_metrics"],
                "dataset_tier": args.dataset_tier,
            },
        }
        sft_rows.append(sft)
        rl_rows.append(rl)
        if args.limit and len(sft_rows) >= args.limit:
            break

    if not sft_rows:
        raise SystemExit("no bundle passed the export gate")
    sft_path = args.output_dir / "sft.jsonl"
    rl_path = args.output_dir / "rl_tasks.jsonl"
    write_jsonl(sft_path, sft_rows)
    write_jsonl(rl_path, rl_rows)
    digest = hashlib.sha256(sft_path.read_bytes() + rl_path.read_bytes()).hexdigest()
    package_digest = hashlib.sha256()
    package_digest.update(sft_path.read_bytes())
    package_digest.update(rl_path.read_bytes())
    for task_id, bundle_digest in sorted(bundle_digests.items()):
        package_digest.update(task_id.encode("utf-8"))
        package_digest.update(bundle_digest.encode("ascii"))
    summary = {
        "format": "rllm-gem-v1",
        "dataset_tier": args.dataset_tier,
        "sft_rows": len(sft_rows),
        "rl_tasks": len(rl_rows),
        "rejected": sorted(set(rejected) - seen),
        "sft": sft_path.name,
        "rl": rl_path.name,
        "sha256": digest,
        "package_sha256": package_digest.hexdigest(),
        "portable_bundle_count": len(bundle_digests),
        "bundle_digests": bundle_digests,
        "private_fields_in_sft": [],
        "rl_policy_visibility": ["instruction", "public_messages", "public_tools"],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
