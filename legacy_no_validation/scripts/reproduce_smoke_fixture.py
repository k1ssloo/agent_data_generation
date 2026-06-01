#!/usr/bin/env python3
"""Rebuild the historical no-validation smoke SFT artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from convert_legacy_to_sft import to_sft_record, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sft_to_legacy_stage3(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    return {
        "id": row["id"],
        "text": row["source_text"],
        "summary": metadata.get("summary"),
        "domain": metadata.get("domain"),
        "platform": metadata.get("platform"),
        "task": metadata.get("task"),
        "workflow": metadata.get("workflow"),
        "tools": row["tools"],
        "messages": row["messages"],
        "refinement_patterns": metadata.get("refinement_patterns", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-sft",
        type=Path,
        default=PROJECT_ROOT / "outputs/sft/qwen32b_sft_openai_messages_smoke.jsonl",
        help="Historical SFT smoke file used to bootstrap the legacy fixture.",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=PROJECT_ROOT / "legacy_no_validation/fixtures/qwen32b_stage3_smoke_legacy.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "legacy_no_validation/outputs/qwen32b_sft_openai_messages_smoke.jsonl",
    )
    args = parser.parse_args()

    legacy_rows = [sft_to_legacy_stage3(row) for row in load_jsonl(args.source_sft)]
    write_jsonl(args.fixture, legacy_rows)
    write_jsonl(args.output, [to_sft_record(row) for row in legacy_rows])
    print(
        json.dumps(
            {
                "fixture_rows": len(legacy_rows),
                "fixture": str(args.fixture),
                "sft_output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
