#!/usr/bin/env python3
"""Align natural-language task goals to executable contracts with one compact LLM call."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.llm_client import PROVIDERS, call_chat, parse_json_object, render_template
from task_factory import load_task_bundle
from task_factory.goal_alignment import alignment_context, compile_alignment_plan
from task_factory.materialize import materialize_candidate
from task_factory.operators.base import manifest_metadata


PROMPT = PROJECT_ROOT / "prompts" / "goal_alignment_plan.txt"
SCHEMA = PROJECT_ROOT / "schemas" / "goal_alignment_plan_v1.json"


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=PROVIDERS, default="codex")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--request-retries", type=int, default=2)
    args = parser.parse_args()
    if args.provider == "codex" and args.config:
        os.environ["GEM_CODEX_CONFIG"] = str(args.config.resolve())

    manifests = sorted(args.input_dir.rglob("manifest.json"))
    if args.limit > 0:
        manifests = manifests[: args.limit]
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    prompt_template = PROMPT.read_text(encoding="utf-8")
    audits = []
    for manifest in manifests:
        bundle = load_task_bundle(manifest)
        context = alignment_context(bundle)
        prompt = render_template(
            prompt_template, {"alignment_context_json": compact(context)}
        )
        try:
            failures = []
            for attempt in range(args.request_retries + 1):
                try:
                    raw, usage = call_chat(
                        [{"role": "user", "content": prompt}],
                        max_tokens=args.max_tokens,
                        temperature=0.0,
                        provider=args.provider,
                        reasoning_effort=args.reasoning_effort,
                        response_schema=schema if args.provider == "codex" else None,
                    )
                    break
                except RuntimeError as exc:
                    failures.append(str(exc))
                    if attempt >= args.request_retries:
                        raise
                    time.sleep(min(2.0 * (attempt + 1), 5.0))
            plan = parse_json_object(raw)
            aligned, report = compile_alignment_plan(bundle, plan)
            audit = {
                "id": bundle.task_id,
                "accepted": aligned is not None,
                "usage": usage,
                "transport_failures": failures,
                "plan": plan,
                "report": report,
            }
            if aligned is not None:
                task_id = f"{bundle.task_id}__goal_aligned"
                aligned.manifest["task_id"] = task_id
                path = materialize_candidate(
                    args.output_dir / "accepted",
                    task_id=task_id,
                    contract=aligned.contract,
                    candidate={
                        "instruction": aligned.instruction,
                        "environment": aligned.environment,
                        "bindings": aligned.bindings,
                        "reference_plan": aligned.reference_plan,
                    },
                    lineage={
                        **aligned.manifest.get("lineage", {}),
                        "parent_task_id": bundle.task_id,
                    },
                    manifest_metadata={
                        **manifest_metadata(aligned),
                        "goal_alignment_version": "goal-alignment-v1",
                    },
                )
                audit["bundle"] = str(path)
        except Exception as exc:  # preserve every failed model/compiler audit
            audit = {
                "id": bundle.task_id,
                "accepted": False,
                "errors": [str(exc)],
            }
        audits.append(audit)
        write_json(args.output_dir / "audits" / f"{bundle.task_id}.json", audit)

    summary = {
        "attempted": len(audits),
        "accepted": sum(item["accepted"] for item in audits),
        "rejected": sum(not item["accepted"] for item in audits),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
