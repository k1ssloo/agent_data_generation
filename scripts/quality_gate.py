#!/usr/bin/env python3
"""Summarize pipeline quality and fail fast when a batch is not scalable."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def valid_id_set(rows: list[dict[str, Any]]) -> set[str]:
    return {row["id"] for row in rows if row.get("valid")}


def first_error_bucket(errors: list[str]) -> str:
    if not errors:
        return "none"
    text = errors[0]
    lowered = text.lower()
    if "json" in lowered or "unterminated string" in lowered or "expecting" in lowered:
        return "json_parse"
    if "no executable branch matched" in lowered:
        return "environment_branch_miss"
    if "duplicate tool call" in lowered:
        return "duplicate_tool_call"
    if "already been called" in lowered:
        return "tool_loop_guard"
    if "finalization rejected" in lowered or "last tool call" in lowered:
        return "missing_final_verification"
    if "grounded" in lowered:
        return "grounding"
    if "workflow tools not used" in lowered:
        return "workflow_coverage"
    return "other"


def summarize_validation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    invalid = [row for row in rows if not row.get("valid")]
    return {
        "checked": len(rows),
        "valid": len(rows) - len(invalid),
        "invalid": len(invalid),
        "invalid_ids": [row.get("id") for row in invalid],
        "error_buckets": dict(Counter(first_error_bucket(row.get("errors", [])) for row in invalid)),
    }


def is_completed(row: dict[str, Any]) -> bool:
    return row.get("rollout_status", "completed") == "completed"


def tool_call_count(row: dict[str, Any]) -> int:
    if "rollout_tool_calls" in row:
        return int(row.get("rollout_tool_calls") or 0)
    return sum(
        1
        for message in row.get("messages", [])
        if message.get("role") == "assistant" and "tool_call" in message
    )


def summarize_generation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if is_completed(row)]
    failed = [row for row in rows if not is_completed(row)]
    return {
        "checked": len(rows),
        "completed": len(completed),
        "failed": len(failed),
        "completion_rate": round(len(completed) / len(rows), 4) if rows else 0.0,
        "avg_tool_calls_completed": round(sum(tool_call_count(row) for row in completed) / len(completed), 3) if completed else 0.0,
        "failed_ids": [row.get("id") for row in failed],
        "error_buckets": dict(Counter(first_error_bucket(row.get("rollout_errors", [])) for row in failed)),
    }


def ratio(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "--rollout", dest="trajectories", type=Path, required=True, help="Stage 3 trajectory JSONL. Rollout artifacts may include failed rows.")
    parser.add_argument("--trajectory-validation", type=Path, help="Strict trajectory validation JSONL.")
    parser.add_argument("--execution-validation", type=Path, help="Execution validation JSONL.")
    parser.add_argument("--tool-bank-validation", type=Path, help="Tool-bank validation JSONL.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-completion-rate", type=float, default=0.8)
    parser.add_argument("--min-strict-rate", type=float, default=0.8)
    parser.add_argument("--min-execution-rate", type=float, default=0.95)
    parser.add_argument("--min-final-yield-rate", type=float, default=0.7)
    args = parser.parse_args()

    trajectory_artifacts = load_jsonl(args.trajectories)
    trajectory_rows = load_jsonl(args.trajectory_validation)
    execution_rows = load_jsonl(args.execution_validation)
    tool_bank_rows = load_jsonl(args.tool_bank_validation)

    completed_ids = {row["id"] for row in trajectory_artifacts if is_completed(row)}
    valid_sets = []
    if trajectory_rows:
        valid_sets.append(valid_id_set(trajectory_rows))
    if execution_rows:
        valid_sets.append(valid_id_set(execution_rows))
    if tool_bank_rows:
        valid_sets.append(valid_id_set(tool_bank_rows))
    final_valid_ids = set.intersection(*valid_sets) if valid_sets else completed_ids
    final_valid_ids &= completed_ids

    total = len(trajectory_artifacts)
    completed = len(completed_ids)
    strict_valid = len(valid_id_set(trajectory_rows) & completed_ids) if trajectory_rows else 0
    execution_valid = len(valid_id_set(execution_rows) & completed_ids) if execution_rows else 0
    final_valid = len(final_valid_ids)

    summary = {
        "generation": summarize_generation(trajectory_artifacts),
        "trajectory_validation": summarize_validation(trajectory_rows) if trajectory_rows else None,
        "execution_validation": summarize_validation(execution_rows) if execution_rows else None,
        "tool_bank_validation": summarize_validation(tool_bank_rows) if tool_bank_rows else None,
        "rates": {
            "completion_rate": ratio(completed, total),
            "strict_rate_over_completed": ratio(strict_valid, completed),
            "execution_rate_over_completed": ratio(execution_valid, completed),
            "final_yield_rate": ratio(final_valid, total),
        },
        "final_valid_ids": sorted(final_valid_ids),
        "thresholds": {
            "min_completion_rate": args.min_completion_rate,
            "min_strict_rate": args.min_strict_rate,
            "min_execution_rate": args.min_execution_rate,
            "min_final_yield_rate": args.min_final_yield_rate,
        },
    }

    failures = []
    rates = summary["rates"]
    if rates["completion_rate"] < args.min_completion_rate:
        failures.append("completion_rate")
    if trajectory_rows and rates["strict_rate_over_completed"] < args.min_strict_rate:
        failures.append("strict_rate_over_completed")
    if execution_rows and rates["execution_rate_over_completed"] < args.min_execution_rate:
        failures.append("execution_rate_over_completed")
    if rates["final_yield_rate"] < args.min_final_yield_rate:
        failures.append("final_yield_rate")
    summary["passed"] = not failures
    summary["failed_thresholds"] = failures

    write_json(args.output, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
