#!/usr/bin/env python3
"""Search multiple recursive task candidates and retain diverse archive elites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task_factory import load_task_bundle
from task_factory.archive import TaskArchive
from task_factory.materialize import materialize_candidate
from task_factory.operators.base import manifest_metadata
from task_factory.operators import OPERATORS
from task_factory.search import candidate_metadata, generate_candidates, select_candidates


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--max-per-cell", type=int, default=1)
    parser.add_argument("--max-per-parent", type=int, default=2)
    parser.add_argument("--max-per-operator", type=int, default=100)
    parser.add_argument("--operators", nargs="+", default=sorted(OPERATORS))
    args = parser.parse_args()
    if args.generations < 1 or args.beam_size < 1:
        raise SystemExit("--generations and --beam-size must be >= 1")
    unknown = sorted(set(args.operators) - set(OPERATORS))
    if unknown:
        raise SystemExit(f"unknown operators: {unknown}")

    parents = [load_task_bundle(path) for path in args.roots]
    archive = TaskArchive(args.output_dir / "archive.jsonl")
    existing_fingerprints: set[str] = set()
    generation_reports = []
    total_selected = 0
    for generation in range(1, args.generations + 1):
        candidates, generation_rejections = generate_candidates(parents, args.operators)
        selected, selection_rejections = select_candidates(
            candidates,
            existing_fingerprints=existing_fingerprints,
            max_per_cell=args.max_per_cell,
            max_per_parent=args.max_per_parent,
            max_per_operator=args.max_per_operator,
        )
        selected = selected[: args.beam_size]
        next_parents = []
        selected_rows = []
        for candidate in selected:
            child = candidate.evaluation.product.bundle
            bundle_path = materialize_candidate(
                args.output_dir / "bundles",
                task_id=child.task_id,
                contract=child.contract,
                candidate={
                    "instruction": child.instruction,
                    "environment": child.environment,
                    "bindings": child.bindings,
                    "reference_plan": child.reference_plan,
                },
                lineage=child.manifest["lineage"],
                manifest_metadata=manifest_metadata(child),
            )
            metadata = candidate_metadata(candidate)
            metadata["bundle"] = str(bundle_path)
            metadata["generation"] = generation
            archive.add(metadata)
            write_json(
                args.output_dir / "audits" / child.task_id / "contract_patch.json",
                candidate.evaluation.product.patch,
            )
            write_json(
                args.output_dir / "audits" / child.task_id / "evaluation.json",
                candidate.evaluation.report,
            )
            existing_fingerprints.add(candidate.fingerprint)
            next_parents.append(load_task_bundle(bundle_path))
            selected_rows.append(metadata)
        generation_report = {
            "generation": generation,
            "parents": len(parents),
            "candidates": len(candidates),
            "selected": len(selected_rows),
            "selected_tasks": selected_rows,
            "generation_rejections": generation_rejections,
            "selection_rejections": selection_rejections,
        }
        generation_reports.append(generation_report)
        write_json(args.output_dir / "reports" / f"generation_{generation:02d}.json", generation_report)
        total_selected += len(selected_rows)
        if not next_parents:
            break
        parents = next_parents

    summary = {
        "root_count": len(args.roots),
        "requested_generations": args.generations,
        "completed_generations": len(generation_reports),
        "selected_tasks": total_selected,
        "distinct_semantic_fingerprints": len(existing_fingerprints),
        "generations": [
            {
                "generation": report["generation"],
                "parents": report["parents"],
                "candidates": report["candidates"],
                "selected": report["selected"],
            }
            for report in generation_reports
        ],
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
