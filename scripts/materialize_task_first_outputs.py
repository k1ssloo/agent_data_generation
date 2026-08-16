#!/usr/bin/env python3
"""Materialize task-first model outputs after deterministic contract checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task_factory.contracts import normalize_contract, validate_contract
from task_factory.materialize import materialize_candidate


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contracts", type=Path, required=True, help="Successful contract model outputs JSONL.")
    parser.add_argument("--bundles", type=Path, required=True, help="Successful bundle model outputs JSONL.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    contracts = {
        row["id"]: normalize_contract(row["json_response"])
        for row in load_jsonl(args.contracts)
        if row.get("ok")
    }
    written = []
    rejected = []
    for row in load_jsonl(args.bundles):
        if not row.get("ok") or row.get("id") not in contracts:
            continue
        task_id = row["id"]
        contract = contracts[task_id]
        candidate = row["json_response"]
        initial_state = candidate.get("environment", {}).get("initial_state")
        errors = validate_contract(contract, initial_state if isinstance(initial_state, dict) else None)
        if errors:
            rejected.append({"id": task_id, "errors": errors})
            continue
        try:
            path = materialize_candidate(
                args.output_dir,
                task_id=task_id,
                contract=contract,
                candidate=candidate,
                lineage={"generation_stage": "task_first_v1"},
            )
            written.append(str(path))
        except (OSError, ValueError) as exc:
            rejected.append({"id": task_id, "errors": [str(exc)]})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "materialization_report.json").write_text(
        json.dumps({"written": written, "rejected": rejected}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"written": len(written), "rejected": len(rejected), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
