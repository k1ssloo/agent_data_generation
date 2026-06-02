#!/usr/bin/env python3
"""Execute GEM LLM request JSONL files against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def empty_result(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": request["id"],
        "stage": request.get("stage"),
        "ok": False,
        "raw_response": None,
        "json_response": None,
        "usage": None,
        "elapsed_sec": None,
        "attempts": 0,
        "errors": [],
        "error": None,
    }


def execute_request(request: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    result = empty_result(request)
    started = time.monotonic()
    max_attempts = args.retries + 1
    for attempt in range(1, max_attempts + 1):
        result["attempts"] = attempt
        try:
            raw, usage = call_chat(
                request["messages"],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                provider=args.provider,
            )
            result["raw_response"] = raw
            result["usage"] = usage
            result["json_response"] = parse_json_object(raw)
            result["ok"] = True
            result["error"] = None
            break
        except (RuntimeError, urllib.error.URLError, json.JSONDecodeError, KeyError) as exc:
            message = str(exc)
            result["error"] = message
            result["errors"].append({"attempt": attempt, "error": message})
            if attempt < max_attempts:
                time.sleep(args.retry_backoff * attempt)
    result["elapsed_sec"] = round(time.monotonic() - started, 3)
    if args.sleep:
        time.sleep(args.sleep)
    return result


def resumable_outputs(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    return {row["id"]: row for row in rows if row.get("ok") and "id" in row}


def checkpoint(path: Path, ordered_outputs: list[dict[str, Any] | None]) -> None:
    ready = [row for row in ordered_outputs if row is not None]
    if ready:
        write_jsonl(path, ready)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means all requests.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent requests.")
    parser.add_argument("--retries", type=int, default=0, help="Retries per request after network or JSON parse failures.")
    parser.add_argument("--retry-backoff", type=float, default=2.0, help="Seconds multiplied by attempt index between retries.")
    parser.add_argument("--resume", action="store_true", help="Reuse successful rows already present in --output.")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Write partial results every N completed requests. 0 disables.")
    parser.add_argument("--provider", choices=["openai", "gemini"], default=os.environ.get("GEM_LLM_PROVIDER", "openai"))
    args = parser.parse_args()

    requests = load_jsonl(args.input)
    if args.limit:
        requests = requests[: args.limit]
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.retries < 0:
        raise SystemExit("--retries must be >= 0")

    completed_by_id = resumable_outputs(args.output) if args.resume else {}
    outputs: list[dict[str, Any] | None] = [None] * len(requests)
    pending: list[tuple[int, dict[str, Any]]] = []
    for index, request in enumerate(requests):
        previous = completed_by_id.get(request["id"])
        if previous:
            outputs[index] = previous
        else:
            pending.append((index, request))

    if completed_by_id:
        print(json.dumps({"resume_ok": len(requests) - len(pending), "pending": len(pending)}, ensure_ascii=False))

    done_count = len(requests) - len(pending)
    if args.workers == 1:
        for index, request in pending:
            result = execute_request(request, args)
            outputs[index] = result
            done_count += 1
            print(json.dumps({"done": done_count, "id": request["id"], "ok": result["ok"], "attempts": result["attempts"]}, ensure_ascii=False))
            if args.checkpoint_every and done_count % args.checkpoint_every == 0:
                checkpoint(args.output, outputs)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(execute_request, request, args): (index, request) for index, request in pending}
            for future in as_completed(futures):
                index, request = futures[future]
                result = future.result()
                outputs[index] = result
                done_count += 1
                print(json.dumps({"done": done_count, "id": request["id"], "ok": result["ok"], "attempts": result["attempts"]}, ensure_ascii=False))
                if args.checkpoint_every and done_count % args.checkpoint_every == 0:
                    checkpoint(args.output, outputs)

    final_outputs = [row for row in outputs if row is not None]
    write_jsonl(args.output, final_outputs)
    ok_count = sum(1 for row in final_outputs if row["ok"])
    print(json.dumps({"written": len(final_outputs), "ok": ok_count, "output": str(args.output)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
