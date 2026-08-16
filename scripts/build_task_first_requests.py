#!/usr/bin/env python3
"""Build model requests for task-first contract and bundle generation."""

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

from scripts.llm_client import render_template


DEFAULT_OPERATORS = [
    "discovery_and_evidence",
    "configuration_consistency",
    "permission_authorization",
    "asynchronous_lifecycle",
    "failure_diagnosis_recovery",
    "resource_budget_constraints",
    "artifact_provenance",
    "rollback_idempotency",
    "multi_resource_coordination",
    "alternative_plan_affordance",
]
DEFAULT_OPERATOR_CATALOG = PROJECT_ROOT / "config" / "task_rewrite_operators.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def request_inputs(row: dict[str, Any], stage: str) -> tuple[Any, dict[str, Any]]:
    if stage == "contract":
        return row.get("seed", row), {}
    contract = row.get("contract")
    if not isinstance(contract, dict):
        contract = row.get("json_response")
    if not isinstance(contract, dict):
        raise ValueError("bundle input requires contract or json_response object")
    metadata = row.get("metadata", {})
    seed = row.get("seed")
    if seed is None and isinstance(metadata, dict):
        seed = metadata.get("seed")
    return seed if seed is not None else {}, contract


def rows_by_id(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    return {row["id"]: row for row in load_jsonl(path) if isinstance(row.get("id"), str)}


def validation_errors_by_id(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    rejected = value.get("rejected", []) if isinstance(value, dict) else []
    indexed = {
        row["id"]: row.get("errors", [])
        for row in rejected
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    if isinstance(value, dict) and isinstance(value.get("task_id"), str):
        feedback = list(value.get("errors", []))
        metrics = value.get("metrics", {})
        if isinstance(metrics, dict):
            feedback.append("metrics=" + json.dumps(metrics, ensure_ascii=False, separators=(",", ":")))
        counterfactual = value.get("counterfactual_validation")
        if isinstance(counterfactual, dict):
            feedback.append(
                "counterfactual_validation="
                + json.dumps(counterfactual, ensure_ascii=False, separators=(",", ":"))
            )
        indexed[value["task_id"]] = feedback
    nested = value.get("validation") if isinstance(value, dict) else None
    if isinstance(nested, dict) and isinstance(nested.get("task_id"), str):
        feedback = list(nested.get("errors", []))
        metrics = nested.get("metrics", {})
        if isinstance(metrics, dict):
            feedback.append("metrics=" + json.dumps(metrics, ensure_ascii=False, separators=(",", ":")))
        indexed[nested["task_id"]] = feedback
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["seed", "contract", "contract_repair", "bundle", "bundle_repair"],
        required=True,
    )
    parser.add_argument("--input", type=Path, required=True, help="JSONL seed summaries or approved contracts.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operators", type=Path, help="Optional JSON array of allowed operator IDs.")
    parser.add_argument("--operator-catalog", type=Path, default=DEFAULT_OPERATOR_CATALOG)
    parser.add_argument(
        "--contracts", type=Path, help="Contract model outputs JSONL for bundle_repair."
    )
    parser.add_argument(
        "--bundles", type=Path, help="Bundle model outputs JSONL for contract_repair."
    )
    parser.add_argument(
        "--validation", type=Path, help="Materialization report JSON for bundle_repair."
    )
    args = parser.parse_args()
    rows = load_jsonl(args.input)
    operators = json.loads(args.operators.read_text(encoding="utf-8")) if args.operators else DEFAULT_OPERATORS
    if not isinstance(operators, list) or any(not isinstance(item, str) for item in operators):
        raise SystemExit("--operators must contain a JSON string array")
    catalog_value = json.loads(args.operator_catalog.read_text(encoding="utf-8"))
    catalog = {
        item["id"]: item
        for item in catalog_value.get("operators", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    missing_cards = sorted(set(operators) - set(catalog))
    if missing_cards:
        raise SystemExit(f"operators missing from catalog: {missing_cards}")
    if args.stage == "bundle_repair" and (not args.contracts or not args.validation):
        raise SystemExit("bundle_repair requires --contracts and --validation")
    if args.stage == "contract_repair" and (not args.bundles or not args.validation):
        raise SystemExit("contract_repair requires --bundles and --validation")
    template_name = {
        "seed": "wikihow_seed_compile.txt",
        "contract": "task_contract_generate.txt",
        "contract_repair": "task_contract_repair.txt",
        "bundle": "task_bundle_generate.txt",
        "bundle_repair": "task_bundle_repair.txt",
    }[args.stage]
    template = (PROJECT_ROOT / "prompts" / template_name).read_text(encoding="utf-8")
    contracts = rows_by_id(args.contracts)
    bundles = rows_by_id(args.bundles)
    validation_errors = validation_errors_by_id(args.validation)
    requests = []
    for row in rows:
        task_id = row.get("id") or row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise SystemExit("every input row requires id or task_id")
        assigned_operator = operators[len(requests) % len(operators)]
        if args.stage == "seed":
            seed = {}
            contract = {}
            candidate = {}
            errors = []
        elif args.stage == "contract_repair":
            contract = row.get("json_response") or row.get("contract")
            candidate = bundles.get(task_id, {}).get("json_response")
            seed = row.get("metadata", {}).get("seed", {})
            errors = validation_errors.get(task_id, [])
            if not isinstance(contract, dict) or not isinstance(candidate, dict) or not errors:
                raise SystemExit(
                    f"{task_id}: contract repair requires contract, candidate, and validation errors"
                )
        elif args.stage == "bundle_repair":
            contract_row = contracts.get(task_id, {})
            contract = contract_row.get("json_response") or contract_row.get("contract")
            candidate = row.get("json_response")
            seed = row.get("metadata", {}).get("seed", {})
            errors = validation_errors.get(task_id, [])
            if not isinstance(contract, dict) or not isinstance(candidate, dict) or not errors:
                raise SystemExit(
                    f"{task_id}: repair requires contract, candidate, and validation errors"
                )
        else:
            try:
                seed, contract = request_inputs(row, args.stage)
            except ValueError as exc:
                raise SystemExit(f"{task_id}: {exc}") from exc
            candidate = {}
            errors = []
        variables = {
            "seed_json": json.dumps(seed, ensure_ascii=False, separators=(",", ":")),
            "operators_json": json.dumps(operators, ensure_ascii=False, separators=(",", ":")),
            "contract_json": json.dumps(contract, ensure_ascii=False, separators=(",", ":")),
            "candidate_json": json.dumps(candidate, ensure_ascii=False, separators=(",", ":")),
            "validation_errors_json": json.dumps(errors, ensure_ascii=False, separators=(",", ":")),
            "source_id": task_id,
            "source_text": str(row.get("text", "")),
            "source_sha256": hashlib.sha256(
                str(row.get("text", "")).encode("utf-8")
            ).hexdigest(),
            "assigned_operator": assigned_operator,
            "assigned_operator_card_json": json.dumps(
                catalog[assigned_operator], ensure_ascii=False, separators=(",", ":")
            ),
        }
        requests.append(
            {
                "id": task_id,
                "stage": f"task_first_{args.stage}",
                "metadata": {
                    "seed": seed,
                    "assigned_operator": assigned_operator,
                },
                "messages": [{"role": "user", "content": render_template(template, variables)}],
            }
        )
    write_jsonl(args.output, requests)
    print(json.dumps({"written": len(requests), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
