#!/usr/bin/env python3
"""Run the task-first WikiHow factory in resumable validated shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def run(command: list[str], *, dry_run: bool) -> int:
    print(json.dumps({"command": command}, ensure_ascii=False), flush=True)
    if dry_run:
        return 0
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--provider", choices=["codex", "responses"], default="responses")
    parser.add_argument("--limit", type=int, default=0, help="0 processes all rows.")
    parser.add_argument("--shard-size", type=int, default=25)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bundle-candidates", type=int, default=1)
    parser.add_argument("--repair-rounds", type=int, default=3)
    parser.add_argument("--recursive-generations", type=int, default=2)
    parser.add_argument("--beam-size", type=int, default=6)
    parser.add_argument("--cache-namespace", default="batch_v1")
    parser.add_argument("--strict-vnext", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.shard_size < 1 or args.workers < 1 or args.bundle_candidates < 1:
        raise SystemExit("shard-size, workers, and bundle-candidates must be >= 1")

    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    shards = [
        rows[index : index + args.shard_size]
        for index in range(0, len(rows), args.shard_size)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    validations = []
    items = []
    for index, shard in enumerate(shards):
        name = f"shard_{index:05d}"
        shard_dir = args.output_dir / "shards" / name
        shard_input = args.output_dir / "inputs" / f"{name}.jsonl"
        root_validation = args.output_dir / "validations" / f"{name}_roots.jsonl"
        recursive_validation = (
            args.output_dir / "validations" / f"{name}_recursive.jsonl"
        )
        write_jsonl(shard_input, shard)
        factory_command = [
            sys.executable,
            str(SCRIPT_DIR / "run_wikihow_task_factory.py"),
            "--input", str(shard_input),
            "--output-dir", str(shard_dir),
            "--config", str(args.config),
            "--provider", args.provider,
            "--workers", str(args.workers),
            "--bundle-candidates", str(args.bundle_candidates),
            "--repair-rounds", str(args.repair_rounds),
            "--recursive-generations", str(args.recursive_generations),
            "--beam-size", str(args.beam_size),
            "--limit", str(len(shard)),
            "--cache-namespace", args.cache_namespace,
            "--strict-adaptive",
        ]
        if args.resume:
            factory_command.append("--resume")
        if args.strict_vnext:
            factory_command.append("--strict-vnext")
        factory_code = run(factory_command, dry_run=args.dry_run)
        validation_codes: list[int] = []
        if factory_code == 0:
            validation_targets = [(shard_dir / "roots", root_validation)]
            recursive_root = shard_dir / "recursive" / "bundles"
            if args.dry_run or recursive_root.is_dir():
                validation_targets.append((recursive_root, recursive_validation))
            for input_dir, output_path in validation_targets:
                validation_command = [
                    sys.executable,
                    str(SCRIPT_DIR / "validate_task_bundles.py"),
                    "--input-dir", str(input_dir),
                    "--output", str(output_path),
                    "--require-goal-alignment",
                    "--require-public-executability",
                    "--require-adaptive",
                ]
                if args.strict_vnext:
                    validation_command.append("--require-vnext-adaptive")
                validation_codes.append(
                    run(validation_command, dry_run=args.dry_run)
                )
                if output_path.exists():
                    validations.append(output_path)
        items.append(
            {
                "shard": name,
                "input": len(shard),
                "factory_returncode": factory_code,
                "validation_returncodes": validation_codes,
            }
        )
        if factory_code != 0 and not args.continue_on_error:
            break

    export_code = None
    if validations:
        export_command = [
            sys.executable,
            str(SCRIPT_DIR / "export_rllm_dataset.py"),
            "--validation",
            *[str(path) for path in validations],
            "--output-dir", str(args.output_dir / "training"),
            "--project-root", str(PROJECT_ROOT),
        ]
        export_code = run(export_command, dry_run=args.dry_run)
    summary = {
        "input_rows": len(rows),
        "shards": len(shards),
        "completed_factory_shards": sum(
            item["factory_returncode"] == 0 for item in items
        ),
        "validation_files": len(validations),
        "export_returncode": export_code,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "items": items,
    }
    (args.output_dir / "batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.dry_run and (not validations or export_code != 0):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
