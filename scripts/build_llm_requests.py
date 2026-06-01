#!/usr/bin/env python3
"""Build JSONL LLM requests for GEM stages.

This script prepares prompts for batch generation. It does not call any model.
Use it to inspect or submit stage requests to an OpenAI-compatible endpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm_client import render_template


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = PROJECT_ROOT / "prompts"
DEFAULT_TOOL_BANK = PROJECT_ROOT / "config/tool_bank.json"


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
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def message_stats(messages: list[dict[str, Any]]) -> dict[str, Any]:
    tool_calls = [
        message.get("tool_call", {}).get("name", "")
        for message in messages
        if message.get("role") == "assistant" and "tool_call" in message
    ]
    failure_responses = 0
    for message in messages:
        if message.get("role") != "tool":
            continue
        lowered = json.dumps(message.get("content"), ensure_ascii=False).lower()
        if any(marker in lowered for marker in ("error", "failed", "failure", "not found", "not available", "cannot")):
            failure_responses += 1
    return {
        "messages": len(messages),
        "user_turns": sum(1 for message in messages if message.get("role") == "user"),
        "assistant_natural_turns": sum(
            1 for message in messages if message.get("role") == "assistant" and "content" in message
        ),
        "tool_calls": len(tool_calls),
        "unique_called_tools": len(set(tool_calls)),
        "tool_sequence": tool_calls,
        "failure_tool_responses": failure_responses,
        "final_tool": tool_calls[-1] if tool_calls else None,
    }


def complexity_targets(row: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    tool_count = len(row.get("tools", []))
    preferred_min_unique_tools = min(tool_count, max(stats["unique_called_tools"] + 1, 3)) if tool_count else 0
    return {
        "preferred_min_tool_calls": max(stats["tool_calls"] + 2, 6),
        "preferred_min_unique_called_tools": preferred_min_unique_tools,
        "preferred_min_refinement_patterns": 2,
        "preferred_max_user_turns": 4,
        "required_final_tool_type": "verification/read/get/list/search/poll when such a tool is available",
        "priority_patterns": [
            "long_horizon_dependency",
            "state_inspection",
            "clarification",
            "error_recovery",
            "conditional_branch",
            "constraint_refusal",
            "final_verification",
        ],
        "validity_over_complexity": True,
    }


def load_tool_bank(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_tool_bank_view(tool_bank: dict[str, Any]) -> dict[str, Any]:
    """Keep Stage2 prompt context focused on canonical tool definitions."""
    return {
        "name": tool_bank.get("name", "canonical_tool_bank"),
        "description": tool_bank.get("description", ""),
        "tools": tool_bank.get("tools", []),
    }


def build_stage1(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    template = (PROMPT_DIR / "stage1_filter.txt").read_text(encoding="utf-8")
    return [
        {
            "id": row["id"],
            "stage": "stage1_filter",
            "messages": [{"role": "user", "content": render_template(template, {"text": row["text"]})}],
        }
        for row in rows
    ]


def build_stage2(rows: list[dict[str, Any]], tool_bank: dict[str, Any]) -> list[dict[str, Any]]:
    template = (PROMPT_DIR / "stage2_workflow_tools.txt").read_text(encoding="utf-8")
    tool_bank_json = compact_json(selected_tool_bank_view(tool_bank))
    return [
        {
            "id": row["id"],
            "stage": "stage2_workflow_tools",
            "messages": [
                {
                    "role": "user",
                    "content": render_template(
                        template,
                        {
                            "text": row["text"],
                            "tool_bank_json": tool_bank_json,
                        },
                    ),
                }
            ],
        }
        for row in rows
    ]


def build_stage2_repair(
    rows: list[dict[str, Any]],
    validation_by_id: dict[str, dict[str, Any]],
    tool_bank: dict[str, Any],
) -> list[dict[str, Any]]:
    template = (PROMPT_DIR / "stage2_repair_environment.txt").read_text(encoding="utf-8")
    tool_bank_json = compact_json(selected_tool_bank_view(tool_bank))
    requests = []
    for row in rows:
        validation = validation_by_id.get(row["id"], {})
        requests.append(
            {
                "id": row["id"],
                "stage": "stage2_repair_environment",
                "messages": [
                    {
                        "role": "user",
                        "content": render_template(
                            template,
                            {
                                "text": row["text"],
                                "workflow_json": compact_json(row.get("workflow", {})),
                                "tools_json": compact_json(row.get("tools", [])),
                                "environment_json": compact_json(row.get("environment", {})),
                                "validation_errors_json": compact_json(validation.get("errors", [])),
                                "tool_bank_json": tool_bank_json,
                            },
                        ),
                    }
                ],
            }
        )
    return requests


def build_stage3(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    template = (PROMPT_DIR / "stage3_trajectory.txt").read_text(encoding="utf-8")
    requests = []
    for row in rows:
        if row.get("missing_tool_requirements") or row.get("stage2_status") == "missing_tool":
            continue
        if not all(key in row for key in ("workflow", "tools", "environment")):
            continue
        requests.append(
            {
                "id": row["id"],
                "stage": "stage3_trajectory",
                "messages": [
                    {
                        "role": "user",
                        "content": render_template(
                            template,
                            {
                                "text": row["text"],
                                "workflow_json": compact_json(row["workflow"]),
                                "tools_json": compact_json(row["tools"]),
                                "environment_json": compact_json(row.get("environment", {})),
                            },
                        ),
                    }
                ],
            }
        )
    return requests


def build_stage3_repair(
    rows: list[dict[str, Any]],
    trajectory_validation_by_id: dict[str, dict[str, Any]],
    execution_validation_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    template = (PROMPT_DIR / "stage3_repair_trajectory.txt").read_text(encoding="utf-8")
    requests = []
    for row in rows:
        trajectory_validation = trajectory_validation_by_id.get(row["id"], {})
        execution_validation = execution_validation_by_id.get(row["id"], {})
        requests.append(
            {
                "id": row["id"],
                "stage": "stage3_repair_trajectory",
                "messages": [
                    {
                        "role": "user",
                        "content": render_template(
                            template,
                            {
                                "text": row["text"],
                                "workflow_json": compact_json(row.get("workflow", {})),
                                "tools_json": compact_json(row.get("tools", [])),
                                "environment_json": compact_json(row.get("environment", {})),
                                "messages_json": compact_json(row.get("messages", [])),
                                "trajectory_errors_json": compact_json(trajectory_validation.get("errors", [])),
                                "execution_errors_json": compact_json(execution_validation.get("errors", [])),
                            },
                        ),
                    }
                ],
            }
        )
    return requests


def build_stage4(
    rows: list[dict[str, Any]],
    trajectory_validation_by_id: dict[str, dict[str, Any]],
    execution_validation_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    template = (PROMPT_DIR / "stage4_refine.txt").read_text(encoding="utf-8")
    requests = []
    for row in rows:
        if not all(key in row for key in ("tools", "environment", "messages")):
            continue
        stats = message_stats(row.get("messages", []))
        trajectory_validation = trajectory_validation_by_id.get(row["id"], {})
        execution_validation = execution_validation_by_id.get(row["id"], {})
        requests.append(
            {
                "id": row["id"],
                "stage": "stage4_refine",
                "messages": [
                    {
                        "role": "user",
                        "content": render_template(
                            template,
                            {
                                "text": row["text"],
                                "workflow_json": compact_json(row.get("workflow", {})),
                                "tools_json": compact_json(row["tools"]),
                                "environment_json": compact_json(row.get("environment", {})),
                                "messages_json": compact_json(row["messages"]),
                                "trajectory_stats_json": compact_json(stats),
                                "complexity_targets_json": compact_json(complexity_targets(row, stats)),
                                "trajectory_errors_json": compact_json(trajectory_validation.get("errors", [])),
                                "execution_errors_json": compact_json(execution_validation.get("errors", [])),
                            },
                        ),
                    }
                ],
            }
        )
    return requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["stage1", "stage2", "stage2_repair", "stage3", "stage3_repair", "stage4"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation", type=Path, help="Trajectory/environment validation JSONL used by repair or refinement stages.")
    parser.add_argument("--execution-validation", type=Path, help="Execution validation JSONL used by trajectory repair or refinement.")
    parser.add_argument("--tool-bank", type=Path, default=DEFAULT_TOOL_BANK, help="Canonical tool bank injected into Stage 2 prompts.")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if args.stage == "stage1":
        requests = build_stage1(rows)
    elif args.stage == "stage2":
        requests = build_stage2(rows, load_tool_bank(args.tool_bank))
    elif args.stage == "stage2_repair":
        validation_by_id = {row["id"]: row for row in load_jsonl(args.validation)} if args.validation else {}
        requests = build_stage2_repair(rows, validation_by_id, load_tool_bank(args.tool_bank))
    elif args.stage == "stage3":
        requests = build_stage3(rows)
    elif args.stage == "stage3_repair":
        trajectory_validation_by_id = {row["id"]: row for row in load_jsonl(args.validation)} if args.validation else {}
        execution_validation_by_id = {row["id"]: row for row in load_jsonl(args.execution_validation)} if args.execution_validation else {}
        requests = build_stage3_repair(rows, trajectory_validation_by_id, execution_validation_by_id)
    else:
        trajectory_validation_by_id = {row["id"]: row for row in load_jsonl(args.validation)} if args.validation else {}
        execution_validation_by_id = {row["id"]: row for row in load_jsonl(args.execution_validation)} if args.execution_validation else {}
        requests = build_stage4(rows, trajectory_validation_by_id, execution_validation_by_id)
    write_jsonl(args.output, requests)
    print(json.dumps({"stage": args.stage, "requests": len(requests), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
