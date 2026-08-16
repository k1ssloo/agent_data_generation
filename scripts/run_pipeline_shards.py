#!/usr/bin/env python3
"""Run the GEM pipeline over a large JSONL input in resumable shards."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_jsonl(output_path: Path, input_path: Path) -> int:
    if not input_path.exists():
        return 0
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as source, output_path.open("a", encoding="utf-8") as target:
        for line in source:
            if line.strip():
                target.write(line)
                count += 1
    return count


def shard_rows(rows: list[dict[str, Any]], shard_size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + shard_size] for index in range(0, len(rows), shard_size)]


def run_command(command: list[str], *, dry_run: bool) -> int:
    print(" ".join(command), flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True)
    return completed.returncode


def load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(output_dir: Path, shard_dirs: list[Path]) -> None:
    sft_output = output_dir / "sft_openai_messages.jsonl"
    if sft_output.exists():
        sft_output.unlink()
    summaries = []
    total_sft = 0
    for shard_dir in shard_dirs:
        summary = load_summary(shard_dir / "summary.json")
        sft_count = append_jsonl(sft_output, shard_dir / "sft_openai_messages.jsonl")
        summaries.append(
            {
                "shard": shard_dir.name,
                "summary": summary,
                "sft_records": sft_count,
            }
        )
        total_sft += sft_count
    report = {
        "shards": len(shard_dirs),
        "completed_shards": sum(1 for item in summaries if item["summary"]),
        "sft_records": total_sft,
        "sft_output": str(sft_output),
        "items": summaries,
    }
    (output_dir / "shard_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-limit", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=0, help="0 means all remaining shards.")
    parser.add_argument("--force", action="store_true", help="Re-run shards even if summary.json already exists.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--min-shard-final-valid", type=int, default=0, help="Stop if a completed shard has fewer valid SFT rows. 0 disables.")
    parser.add_argument("--continue-on-low-yield", action="store_true")
    parser.add_argument("--provider", choices=["openai", "gemini", "codex"], default="gemini")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--gemini-thinking-budget", type=int, default=0)
    parser.add_argument("--stage1-max-tokens", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--repair-max-tokens", type=int, default=12288)
    parser.add_argument("--stage2-repair-rounds", type=int, default=1)
    parser.add_argument("--trajectory-repair-rounds", type=int, default=2)
    parser.add_argument("--min-tool-calls", type=int, default=0)
    parser.add_argument("--max-user-turns", type=int, default=0)
    parser.add_argument("--require-final-verification", action="store_true")
    parser.add_argument("--skip-stage4", action="store_true")
    parser.add_argument("--no-canonicalize-tool-responses", action="store_true")
    args = parser.parse_args()

    if args.shard_size < 1:
        raise SystemExit("--shard-size must be >= 1")
    if args.provider == "codex" and args.workers != 1:
        raise SystemExit("--provider codex requires --workers 1")

    rows = load_jsonl(args.input)
    if args.input_limit:
        rows = rows[: args.input_limit]
    shards = shard_rows(rows, args.shard_size)
    selected_indexes = list(range(args.start_shard, len(shards)))
    if args.max_shards:
        selected_indexes = selected_indexes[: args.max_shards]

    output_dir = args.output_dir
    shard_input_dir = output_dir / "shard_inputs"
    shard_output_dir = output_dir / "shards"
    all_shard_dirs = [shard_output_dir / f"shard_{shard_index:05d}" for shard_index in range(len(shards))]

    for shard_index in selected_indexes:
        shard_id = f"shard_{shard_index:05d}"
        shard_input = shard_input_dir / f"{shard_id}.jsonl"
        shard_dir = shard_output_dir / shard_id
        write_jsonl(shard_input, shards[shard_index])
        if (shard_dir / "summary.json").exists() and not args.force:
            print(json.dumps({"shard": shard_id, "status": "skip_existing"}, ensure_ascii=False), flush=True)
            continue

        command = [
            sys.executable,
            str(SCRIPT_DIR / "run_pipeline.py"),
            "--input",
            str(shard_input),
            "--output-dir",
            str(shard_dir),
            "--provider",
            args.provider,
            "--workers",
            str(args.workers),
            "--retries",
            str(args.retries),
            "--retry-backoff",
            str(args.retry_backoff),
            "--checkpoint-every",
            str(args.checkpoint_every),
            "--temperature",
            str(args.temperature),
            "--gemini-thinking-budget",
            str(args.gemini_thinking_budget),
            "--stage1-max-tokens",
            str(args.stage1_max_tokens),
            "--max-tokens",
            str(args.max_tokens),
            "--repair-max-tokens",
            str(args.repair_max_tokens),
            "--stage2-repair-rounds",
            str(args.stage2_repair_rounds),
            "--trajectory-repair-rounds",
            str(args.trajectory_repair_rounds),
        ]
        if args.min_tool_calls:
            command.extend(["--min-tool-calls", str(args.min_tool_calls)])
        if args.max_user_turns:
            command.extend(["--max-user-turns", str(args.max_user_turns)])
        if args.require_final_verification:
            command.append("--require-final-verification")
        if args.skip_stage4:
            command.append("--skip-stage4")
        if args.no_canonicalize_tool_responses:
            command.append("--no-canonicalize-tool-responses")

        code = run_command(command, dry_run=args.dry_run)
        if code != 0:
            print(json.dumps({"shard": shard_id, "status": "failed", "returncode": code}, ensure_ascii=False), flush=True)
            if not args.continue_on_error:
                raise SystemExit(code)
        if code == 0 and args.min_shard_final_valid and not args.dry_run:
            summary = load_summary(shard_dir / "summary.json")
            final_valid = int(summary.get("final_valid", 0)) if summary else 0
            if final_valid < args.min_shard_final_valid:
                print(
                    json.dumps(
                        {
                            "shard": shard_id,
                            "status": "low_yield",
                            "final_valid": final_valid,
                            "min_shard_final_valid": args.min_shard_final_valid,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if not args.continue_on_low_yield:
                    raise SystemExit(2)
        aggregate(output_dir, all_shard_dirs)

    aggregate(output_dir, all_shard_dirs)


if __name__ == "__main__":
    main()
