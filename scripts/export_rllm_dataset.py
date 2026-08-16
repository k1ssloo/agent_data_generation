#!/usr/bin/env python3
"""Export validated bundles for rLLM SFT and online agentic RL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_validation import validate_episode
from rollout import run_reference_plan
from scripts.export_task_first_sft import build_sft_row
from task_factory import (
    load_task_bundle,
    totalize_public_capabilities,
    validate_public_executability,
)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--limit", type=int, default=0, help="0 exports every valid row.")
    parser.add_argument(
        "--no-inject-tools",
        action="store_true",
        help="Do not add tool schemas to the SFT system message.",
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
    for candidate in candidates:
        bundle = totalize_public_capabilities(
            load_task_bundle(Path(candidate["bundle"]))
        )
        if bundle.task_id in seen:
            continue
        seen.add(bundle.task_id)
        report = run_reference_plan(bundle)
        validation = validate_episode(bundle, report)
        public = validate_public_executability(bundle)
        if not validation["valid"] or not public["valid"]:
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
        }
        sft = build_sft_row(value)
        if sft is None:
            rejected.append(bundle.task_id)
            continue
        if not args.no_inject_tools:
            sft["messages"] = inject_tools(sft["messages"], sft["tools"])
            sft["metadata"]["sft_tool_schema_transport"] = "system_catalog"
        bundle_path = relative_bundle_path(bundle.root, args.project_root)
        rl = {
            "id": bundle.task_id,
            "instruction": bundle.instruction.strip(),
            "metadata": {
                "bundle_path": bundle_path,
                "project_root": str(args.project_root.resolve()),
                "source_id": value["source_id"],
                "semantic_episode_id": bundle.task_id,
                "reference_steps": len(bundle.reference_plan["actions"]),
                "causal_metrics": sft["metadata"]["causal_metrics"],
            },
        }
        sft_rows.append(sft)
        rl_rows.append(rl)
        if args.limit and len(sft_rows) >= args.limit:
            break

    if not sft_rows:
        raise SystemExit("no bundle passed the export gate")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sft_path = args.output_dir / "sft.jsonl"
    rl_path = args.output_dir / "rl_tasks.jsonl"
    write_jsonl(sft_path, sft_rows)
    write_jsonl(rl_path, rl_rows)
    digest = hashlib.sha256(sft_path.read_bytes() + rl_path.read_bytes()).hexdigest()
    summary = {
        "format": "rllm-gem-v1",
        "sft_rows": len(sft_rows),
        "rl_tasks": len(rl_rows),
        "rejected": rejected,
        "sft": str(sft_path),
        "rl": str(rl_path),
        "sha256": digest,
        "private_fields_in_sft": [],
        "rl_policy_visibility": ["instruction", "public_messages", "public_tools"],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
