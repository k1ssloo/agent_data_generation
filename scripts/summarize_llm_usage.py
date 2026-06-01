#!/usr/bin/env python3
"""Summarize LLM request output usage and elapsed time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def add_usage(target: dict[str, Any], usage: dict[str, Any]) -> None:
    for key, value in usage.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            target[key] = target.get(key, 0) + value


def summarize(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    ok_rows = [row for row in rows if row.get("ok")]
    failed_rows = [row for row in rows if not row.get("ok")]
    elapsed = [row.get("elapsed_sec") for row in rows if isinstance(row.get("elapsed_sec"), int | float)]
    summary: dict[str, Any] = {
        "path": str(path),
        "requests": len(rows),
        "ok": len(ok_rows),
        "failed": len(failed_rows),
        "elapsed_sec_sum": round(sum(elapsed), 3),
        "elapsed_sec_avg": round(sum(elapsed) / len(elapsed), 3) if elapsed else 0,
        "usage": {},
        "ok_usage": {},
        "failed_usage": {},
        "failed_ids": [row.get("id") for row in failed_rows[:20]],
    }
    for row in rows:
        usage = row.get("usage")
        if isinstance(usage, dict):
            add_usage(summary["usage"], usage)
            add_usage(summary["ok_usage"] if row.get("ok") else summary["failed_usage"], usage)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summaries = [summarize(path) for path in args.paths]
    combined: dict[str, Any] = {
        "files": summaries,
        "requests": sum(item["requests"] for item in summaries),
        "ok": sum(item["ok"] for item in summaries),
        "failed": sum(item["failed"] for item in summaries),
        "elapsed_sec_sum": round(sum(item["elapsed_sec_sum"] for item in summaries), 3),
        "usage": {},
        "ok_usage": {},
        "failed_usage": {},
    }
    for item in summaries:
        for key, value in item["usage"].items():
            combined["usage"][key] = combined["usage"].get(key, 0) + value
        for key, value in item["ok_usage"].items():
            combined["ok_usage"][key] = combined["ok_usage"].get(key, 0) + value
        for key, value in item["failed_usage"].items():
            combined["failed_usage"][key] = combined["failed_usage"].get(key, 0) + value
    result = {"combined": combined, "files": summaries}
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
