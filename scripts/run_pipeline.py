#!/usr/bin/env python3
"""Run the GEM synthesis pipeline with validation-guided repair loops."""

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
    if not path.exists():
        return []
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


def script(name: str) -> str:
    return str(SCRIPT_DIR / name)


def run_command(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(" ".join(args), flush=True)
    return subprocess.run(args, cwd=PROJECT_ROOT, check=check, text=True)


def subset_jsonl(input_path: Path, output_path: Path, limit: int) -> None:
    rows = load_jsonl(input_path)
    if limit:
        rows = rows[:limit]
    write_jsonl(output_path, rows)


def filter_by_ids(input_path: Path, output_path: Path, ids: set[str], *, limit: int = 0) -> None:
    rows = [row for row in load_jsonl(input_path) if row.get("id") in ids]
    if limit:
        rows = rows[:limit]
    write_jsonl(output_path, rows)


def merge_by_id(base_path: Path, replacement_path: Path, output_path: Path) -> None:
    replacements = {row["id"]: row for row in load_jsonl(replacement_path)}
    rows = [replacements.get(row["id"], row) for row in load_jsonl(base_path)]
    write_jsonl(output_path, rows)


def valid_ids(validation_path: Path) -> set[str]:
    return {row["id"] for row in load_jsonl(validation_path) if row.get("valid")}


def invalid_ids(*validation_paths: Path) -> set[str]:
    ids: set[str] = set()
    for path in validation_paths:
        ids.update(row["id"] for row in load_jsonl(path) if not row.get("valid"))
    return ids


def combine_validation(validation_paths: list[Path], output_path: Path) -> None:
    combined: dict[str, dict[str, Any]] = {}
    for path in validation_paths:
        for row in load_jsonl(path):
            item = combined.setdefault(row["id"], {"id": row["id"], "valid": True, "errors": []})
            if not row.get("valid"):
                item["valid"] = False
                item["errors"].extend(f"{path.stem}: {error}" for error in row.get("errors", []))
    write_jsonl(output_path, [row for row in combined.values() if not row["valid"]])


def passed_rows(input_path: Path, output_path: Path, validation_paths: list[Path]) -> set[str]:
    if not validation_paths:
        ids = {row["id"] for row in load_jsonl(input_path)}
    else:
        ids = set.intersection(*(valid_ids(path) for path in validation_paths))
    filter_by_ids(input_path, output_path, ids)
    return ids


def execute_requests(args: argparse.Namespace, request_path: Path, output_path: Path, *, max_tokens: int) -> None:
    command = [
        sys.executable,
        script("execute_llm_requests.py"),
        "--input",
        str(request_path),
        "--output",
        str(output_path),
        "--provider",
        args.provider,
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        str(args.temperature),
        "--workers",
        str(args.workers),
        "--retries",
        str(args.retries),
        "--retry-backoff",
        str(args.retry_backoff),
        "--resume",
        "--checkpoint-every",
        str(args.checkpoint_every),
    ]
    run_command(command)


def build_requests(stage: str, input_path: Path, output_path: Path, *, validation: Path | None = None, execution_validation: Path | None = None) -> None:
    command = [
        sys.executable,
        script("build_llm_requests.py"),
        "--stage",
        stage,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]
    if validation is not None:
        command.extend(["--validation", str(validation)])
    if execution_validation is not None:
        command.extend(["--execution-validation", str(execution_validation)])
    run_command(command)


def materialize(stage: str, base_path: Path, llm_output_path: Path, output_path: Path) -> None:
    run_command(
        [
            sys.executable,
            script("materialize_llm_outputs.py"),
            "--base",
            str(base_path),
            "--llm-output",
            str(llm_output_path),
            "--stage",
            stage,
            "--output",
            str(output_path),
        ]
    )


def validate_stage2(artifact_path: Path, validation_dir: Path) -> list[Path]:
    validation_dir.mkdir(parents=True, exist_ok=True)
    environment_path = validation_dir / "environment.jsonl"
    tool_bank_path = validation_dir / "tool_bank.jsonl"
    run_command([sys.executable, script("validate_environment.py"), "--input", str(artifact_path), "--output", str(environment_path)])
    run_command(
        [
            sys.executable,
            script("validate_tool_bank.py"),
            "--input",
            str(artifact_path),
            "--output",
            str(tool_bank_path),
            "--require-discoverable-record-ids",
        ]
    )
    return [environment_path, tool_bank_path]


def validate_trajectories(artifact_path: Path, validation_dir: Path, args: argparse.Namespace) -> list[Path]:
    validation_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = validation_dir / "trajectory.jsonl"
    execution_path = validation_dir / "execution.jsonl"
    tool_bank_path = validation_dir / "tool_bank.jsonl"
    command = [
        sys.executable,
        script("validate_trajectories.py"),
        "--input",
        str(artifact_path),
        "--output",
        str(trajectory_path),
        "--strict-grounding",
        "--require-workflow-tools",
        "--require-error-recovery",
        "--allow-control-arg-literals",
    ]
    if args.require_final_verification:
        command.append("--require-final-verification")
    if args.min_tool_calls:
        command.extend(["--min-tool-calls", str(args.min_tool_calls)])
    if args.max_user_turns:
        command.extend(["--max-user-turns", str(args.max_user_turns)])
    run_command(command)
    run_command([sys.executable, script("validate_execution.py"), "--input", str(artifact_path), "--output", str(execution_path)])
    run_command(
        [
            sys.executable,
            script("validate_tool_bank.py"),
            "--input",
            str(artifact_path),
            "--output",
            str(tool_bank_path),
            "--require-discoverable-record-ids",
        ]
    )
    return [trajectory_path, execution_path, tool_bank_path]


def repair_stage2(current_path: Path, output_dir: Path, args: argparse.Namespace) -> tuple[Path, list[Path]]:
    current = current_path
    validations = validate_stage2(current, output_dir / "validation")
    for round_index in range(1, args.stage2_repair_rounds + 1):
        failed_ids = invalid_ids(*validations)
        if not failed_ids:
            break
        round_dir = output_dir / f"repair_round{round_index}"
        repair_input = round_dir / "input.jsonl"
        repair_validation = round_dir / "validation_errors.jsonl"
        repair_requests = round_dir / "requests.jsonl"
        repair_responses = round_dir / "responses.jsonl"
        repair_artifacts = round_dir / "artifacts.jsonl"
        merged = round_dir / "merged.jsonl"
        filter_by_ids(current, repair_input, failed_ids)
        combine_validation(validations, repair_validation)
        build_requests("stage2_repair", repair_input, repair_requests, validation=repair_validation)
        execute_requests(args, repair_requests, repair_responses, max_tokens=args.max_tokens)
        materialize("stage2", repair_input, repair_responses, repair_artifacts)
        merge_by_id(current, repair_artifacts, merged)
        current = merged
        validations = validate_stage2(current, round_dir / "validation")
    return current, validations


def repair_trajectories(current_path: Path, output_dir: Path, args: argparse.Namespace) -> tuple[Path, list[Path]]:
    current = current_path
    validations = validate_trajectories(current, output_dir / "validation", args)
    for round_index in range(1, args.trajectory_repair_rounds + 1):
        trajectory_path, execution_path, tool_bank_path = validations
        failed_ids = invalid_ids(trajectory_path, execution_path)
        failed_ids |= invalid_ids(tool_bank_path) & {row["id"] for row in load_jsonl(current)}
        if not failed_ids:
            break
        round_dir = output_dir / f"repair_round{round_index}"
        repair_input = round_dir / "input.jsonl"
        repair_requests = round_dir / "requests.jsonl"
        repair_responses = round_dir / "responses.jsonl"
        repair_artifacts = round_dir / "artifacts.jsonl"
        merged = round_dir / "merged.jsonl"
        filter_by_ids(current, repair_input, failed_ids)
        build_requests("stage3_repair", repair_input, repair_requests, validation=trajectory_path, execution_validation=execution_path)
        execute_requests(args, repair_requests, repair_responses, max_tokens=args.max_tokens)
        materialize("stage3", repair_input, repair_responses, repair_artifacts)
        merge_by_id(current, repair_artifacts, merged)
        current = merged
        validations = validate_trajectories(current, round_dir / "validation", args)
    return current, validations


def summarize(output_dir: Path, final_artifacts: Path, validations: list[Path]) -> None:
    summary = {
        "final_artifacts": str(final_artifacts),
        "total": len(load_jsonl(final_artifacts)),
        "validations": {},
    }
    valid_sets = []
    for path in validations:
        rows = load_jsonl(path)
        valid = [row for row in rows if row.get("valid")]
        summary["validations"][path.stem] = {"checked": len(rows), "valid": len(valid), "invalid": len(rows) - len(valid)}
        valid_sets.append({row["id"] for row in valid})
    final_ids = set.intersection(*valid_sets) if valid_sets else set()
    summary["final_valid"] = len(final_ids)
    summary["final_valid_ids"] = sorted(final_ids)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, default=0, help="Limit raw Stage1 candidates. 0 means all input rows.")
    parser.add_argument("--target", type=int, default=0, help="Keep the first N Stage1-positive rows. 0 means keep all positives.")
    parser.add_argument("--provider", choices=["openai", "gemini"], default="gemini")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--stage1-max-tokens", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--stage2-repair-rounds", type=int, default=1)
    parser.add_argument("--trajectory-repair-rounds", type=int, default=1)
    parser.add_argument("--min-tool-calls", type=int, default=0)
    parser.add_argument("--max-user-turns", type=int, default=0)
    parser.add_argument("--require-final-verification", action="store_true")
    parser.add_argument("--skip-stage4", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stage1_dir = output_dir / "stage1"
    stage1_input = stage1_dir / "input.jsonl"
    stage1_requests = stage1_dir / "requests.jsonl"
    stage1_responses = stage1_dir / "responses.jsonl"
    stage1_passed = stage1_dir / "passed.jsonl"
    selected = output_dir / "selected_stage1.jsonl"
    subset_jsonl(args.input, stage1_input, args.candidate_limit)
    build_requests("stage1", stage1_input, stage1_requests)
    execute_requests(args, stage1_requests, stage1_responses, max_tokens=args.stage1_max_tokens)
    materialize("stage1", stage1_input, stage1_responses, stage1_passed)
    selected_ids = {row["id"] for row in load_jsonl(stage1_passed)}
    filter_by_ids(stage1_passed, selected, selected_ids, limit=args.target)

    stage2_dir = output_dir / "stage2"
    stage2_requests = stage2_dir / "requests.jsonl"
    stage2_responses = stage2_dir / "responses.jsonl"
    stage2_artifacts = stage2_dir / "artifacts.jsonl"
    build_requests("stage2", selected, stage2_requests)
    execute_requests(args, stage2_requests, stage2_responses, max_tokens=args.max_tokens)
    materialize("stage2", selected, stage2_responses, stage2_artifacts)
    stage2_final, stage2_validations = repair_stage2(stage2_artifacts, stage2_dir, args)
    stage2_passed = stage2_dir / "passed.jsonl"
    passed_rows(stage2_final, stage2_passed, stage2_validations)

    stage3_dir = output_dir / "stage3"
    stage3_requests = stage3_dir / "requests.jsonl"
    stage3_responses = stage3_dir / "responses.jsonl"
    stage3_artifacts = stage3_dir / "artifacts.jsonl"
    build_requests("stage3", stage2_passed, stage3_requests)
    execute_requests(args, stage3_requests, stage3_responses, max_tokens=args.max_tokens)
    materialize("stage3", stage2_passed, stage3_responses, stage3_artifacts)
    stage3_final, stage3_validations = repair_trajectories(stage3_artifacts, stage3_dir, args)
    stage3_passed = stage3_dir / "passed.jsonl"
    passed_rows(stage3_final, stage3_passed, stage3_validations)

    if args.skip_stage4:
        final_artifacts = stage3_final
        final_validations = stage3_validations
    else:
        stage4_dir = output_dir / "stage4"
        stage4_requests = stage4_dir / "requests.jsonl"
        stage4_responses = stage4_dir / "responses.jsonl"
        stage4_artifacts = stage4_dir / "artifacts.jsonl"
        build_requests("stage4", stage3_passed, stage4_requests, validation=stage3_validations[0], execution_validation=stage3_validations[1])
        execute_requests(args, stage4_requests, stage4_responses, max_tokens=args.max_tokens)
        materialize("stage4", stage3_passed, stage4_responses, stage4_artifacts)
        final_artifacts, final_validations = repair_trajectories(stage4_artifacts, stage4_dir, args)

    quality_report = output_dir / "quality_report.json"
    run_command(
        [
            sys.executable,
            script("quality_gate.py"),
            "--input",
            str(final_artifacts),
            "--trajectory-validation",
            str(final_validations[0]),
            "--execution-validation",
            str(final_validations[1]),
            "--tool-bank-validation",
            str(final_validations[2]),
            "--output",
            str(quality_report),
        ],
        check=False,
    )
    sft_output = output_dir / "sft_openai_messages.jsonl"
    run_command(
        [
            sys.executable,
            script("convert_to_sft.py"),
            "--trajectories",
            str(final_artifacts),
            "--validation",
            str(final_validations[0]),
            "--extra-validation",
            str(final_validations[1]),
            "--extra-validation",
            str(final_validations[2]),
            "--output",
            str(sft_output),
        ]
    )
    summarize(output_dir, final_artifacts, final_validations)


if __name__ == "__main__":
    main()
