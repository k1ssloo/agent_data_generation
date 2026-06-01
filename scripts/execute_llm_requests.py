#!/usr/bin/env python3
"""Execute GEM LLM request JSONL files against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
from pathlib import Path
from typing import Any

from llm_client import call_chat, parse_json_object


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means all requests.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--provider", choices=["openai", "gemini"], default=os.environ.get("GEM_LLM_PROVIDER", "openai"))
    args = parser.parse_args()

    requests = load_jsonl(args.input)
    if args.limit:
        requests = requests[: args.limit]
    outputs = []
    for index, request in enumerate(requests, start=1):
        result = {
            "id": request["id"],
            "stage": request.get("stage"),
            "ok": False,
            "raw_response": None,
            "json_response": None,
            "usage": None,
            "elapsed_sec": None,
            "error": None,
        }
        started = time.monotonic()
        try:
            raw, usage = call_chat(request["messages"], max_tokens=args.max_tokens, temperature=args.temperature, provider=args.provider)
            result["elapsed_sec"] = round(time.monotonic() - started, 3)
            result["raw_response"] = raw
            result["usage"] = usage
            result["json_response"] = parse_json_object(raw)
            result["ok"] = True
        except (RuntimeError, urllib.error.URLError, json.JSONDecodeError, KeyError) as exc:
            result["elapsed_sec"] = round(time.monotonic() - started, 3)
            result["error"] = str(exc)
        outputs.append(result)
        print(json.dumps({"done": index, "id": request["id"], "ok": result["ok"]}, ensure_ascii=False))
        if args.sleep:
            time.sleep(args.sleep)
    write_jsonl(args.output, outputs)
    ok_count = sum(1 for row in outputs if row["ok"])
    print(json.dumps({"written": len(outputs), "ok": ok_count, "output": str(args.output)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
