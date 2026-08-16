#!/usr/bin/env python3
"""Execute and validate a task-first bundle reference plan offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_validation import validate_episode
from rollout import run_reference_plan
from task_factory import load_task_bundle


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--min-delayed-handle-distance", type=int, default=4)
    parser.add_argument("--min-handle-chain-depth", type=int, default=3)
    args = parser.parse_args()
    bundle = load_task_bundle(args.bundle)
    report = run_reference_plan(bundle, max_steps=args.max_steps)
    validation = validate_episode(
        bundle,
        report,
        min_delayed_handle_distance=args.min_delayed_handle_distance,
        min_handle_chain_depth=args.min_handle_chain_depth,
    )
    result = {"task_id": bundle.task_id, "episode": report, "validation": validation}
    if args.output_dir:
        write_json(args.output_dir / "episode.json", report)
        write_json(args.output_dir / "validation.json", validation)
        write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not validation["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
