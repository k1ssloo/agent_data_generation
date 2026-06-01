#!/usr/bin/env python3
"""Execute legacy GEM LLM requests with bounded client-side concurrency."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_json_object(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def call_chat(messages: list[dict[str, str]], max_tokens: int, temperature: float, timeout: int) -> str:
    base_url = os.environ.get("GEM_LLM_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("GEM_LLM_API_KEY", "")
    model = os.environ.get("GEM_LLM_MODEL", "")
    if not base_url or not api_key or not model:
        raise RuntimeError("Set GEM_LLM_BASE_URL, GEM_LLM_API_KEY, and GEM_LLM_MODEL.")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def execute_one(request: dict[str, Any], max_tokens: int, temperature: float, timeout: int) -> dict[str, Any]:
    result = {
        "id": request["id"],
        "stage": request.get("stage"),
        "ok": False,
        "raw_response": None,
        "json_response": None,
        "error": None,
    }
    try:
        raw = call_chat(request["messages"], max_tokens=max_tokens, temperature=temperature, timeout=timeout)
        result["raw_response"] = raw
        result["json_response"] = parse_json_object(raw)
        result["ok"] = True
    except (RuntimeError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
        result["error"] = str(exc)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="0 means all requests.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--resume", action="store_true", help="Skip ids already present in the output JSONL.")
    args = parser.parse_args()

    requests = load_jsonl(args.input)
    if args.limit:
        requests = requests[: args.limit]

    existing_ids: set[str] = set()
    if args.resume:
        existing_ids = {row["id"] for row in load_jsonl(args.output)}
        requests = [row for row in requests if row["id"] not in existing_ids]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    mode = "a" if args.resume and args.output.exists() else "w"
    started = time.time()
    completed = 0
    ok_count = 0

    with args.output.open(mode, encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_id = {
                executor.submit(execute_one, request, args.max_tokens, args.temperature, args.timeout): request["id"]
                for request in requests
            }
            for future in concurrent.futures.as_completed(future_to_id):
                row = future.result()
                completed += 1
                if row["ok"]:
                    ok_count += 1
                with lock:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                print(
                    json.dumps(
                        {
                            "done": completed,
                            "remaining": len(requests) - completed,
                            "id": row["id"],
                            "ok": row["ok"],
                            "elapsed_sec": round(time.time() - started, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    total_existing = len(existing_ids)
    print(
        json.dumps(
            {
                "requested": len(requests),
                "skipped_existing": total_existing,
                "written": len(requests),
                "ok": ok_count,
                "output": str(args.output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
