#!/usr/bin/env python3
"""Extract procedural WikiHow rows from a HuggingFace dataset."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal envs.
    load_dataset = None


COMPUTER_KEYWORDS = [
    "app",
    "browser",
    "click",
    "computer",
    "download",
    "email",
    "export",
    "file",
    "folder",
    "google",
    "install",
    "internet",
    "keyboard",
    "login",
    "log in",
    "online",
    "open",
    "password",
    "photo",
    "print",
    "printer",
    "save",
    "search",
    "settings",
    "software",
    "upload",
    "url",
    "website",
]

STRONG_COMPUTER_KEYWORDS = [
    "app",
    "browser",
    "computer",
    "download",
    "email",
    "export",
    "file",
    "folder",
    "google",
    "install",
    "internet",
    "login",
    "log in",
    "online",
    "password",
    "printer",
    "software",
    "upload",
    "url",
    "website",
]

CORE_COMPUTER_KEYWORDS = [
    "app",
    "browser",
    "click",
    "computer",
    "download",
    "email",
    "export",
    "facebook",
    "file",
    "folder",
    "google",
    "install",
    "ipad",
    "iphone",
    "keyboard",
    "login",
    "log in",
    "password",
    "printer",
    "software",
    "upload",
    "url",
    "website",
    "youtube",
]

PROCEDURAL_MARKERS = [
    " first ",
    " then ",
    " after ",
    " before ",
    " next ",
    " select ",
    " choose ",
    " click ",
    " open ",
    " go to ",
    " type ",
]


def normalize(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\n", " ").strip()


def procedural_score(text: str) -> int:
    lowered = f" {text.lower()} "
    keyword_hits = sum(1 for keyword in COMPUTER_KEYWORDS if contains_phrase(lowered, keyword))
    marker_hits = sum(1 for marker in PROCEDURAL_MARKERS if marker in lowered)
    step_like = len(re.findall(r"\b(step|steps|method)\b", lowered))
    return keyword_hits * 2 + marker_hits + step_like


def contains_phrase(text: str, phrase: str) -> bool:
    if " " in phrase:
        return phrase in text
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def strong_hit_count(text: str) -> int:
    lowered = f" {text.lower()} "
    return sum(1 for keyword in STRONG_COMPUTER_KEYWORDS if contains_phrase(lowered, keyword))


def core_hit_count(text: str) -> int:
    lowered = f" {text.lower()} "
    return sum(1 for keyword in CORE_COMPUTER_KEYWORDS if contains_phrase(lowered, keyword))


def row_text(row: dict[str, Any]) -> str:
    title = normalize(row.get("title"))
    body = normalize(row.get("text"))
    summary = normalize(row.get("summary"))
    pieces = []
    if title:
        pieces.append(f"Title: {title}")
    if body:
        pieces.append(f"Steps: {body}")
    if summary:
        pieces.append(f"Summary: {summary}")
    return "\n".join(pieces)


def too_short_or_long(text: str, *, min_chars: int, max_chars: int) -> bool:
    if min_chars and len(text) < min_chars:
        return True
    return bool(max_chars and len(text) > max_chars)


def stable_row_id(
    prefix: str,
    *,
    source_index: int,
    selected_index: int,
    legacy_sequential: bool = False,
) -> str:
    row_number = selected_index if legacy_sequential else source_index
    return f"{prefix}_{row_number:09d}"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gursi26/wikihow-cleaned")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--scan-limit", type=int, default=20000)
    parser.add_argument("--start-index", type=int, default=0, help="Skip source rows before this streaming index.")
    parser.add_argument("--filter", choices=["computer", "none"], default="computer")
    parser.add_argument("--id-prefix", default="wikihow_computer")
    parser.add_argument(
        "--legacy-sequential-ids",
        action="store_true",
        help="Use selected-row ordinals for backward reproduction; these IDs can collide across extracts.",
    )
    parser.add_argument("--min-text-chars", type=int, default=0)
    parser.add_argument("--max-text-chars", type=int, default=0, help="0 disables the upper bound.")
    parser.add_argument("--min-score", type=int, default=8)
    parser.add_argument("--min-strong-hits", type=int, default=1)
    parser.add_argument("--min-core-hits", type=int, default=1)
    parser.add_argument("--require-title-hit", action="store_true")
    parser.add_argument("--progress-every", type=int, default=0)
    args = parser.parse_args()

    if load_dataset is None:
        raise SystemExit("Missing dependency: install HuggingFace datasets with `python3 -m pip install datasets`.")

    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    rows: list[dict[str, Any]] = []
    scanned = 0
    for source_index, source_row in enumerate(dataset):
        if source_index < args.start_index:
            continue
        scanned += 1
        text = row_text(source_row)
        if too_short_or_long(text, min_chars=args.min_text_chars, max_chars=args.max_text_chars):
            if scanned >= args.scan_limit:
                break
            continue
        title = normalize(source_row.get("title"))
        score = procedural_score(text)
        strong_hits = strong_hit_count(text)
        core_hits = core_hit_count(text)
        title_hits = core_hit_count(title)
        is_selected = args.filter == "none" or (
            score >= args.min_score
            and strong_hits >= args.min_strong_hits
            and core_hits >= args.min_core_hits
            and (not args.require_title_hit or title_hits > 0)
        )
        if is_selected:
            rows.append(
                {
                    "id": stable_row_id(
                        args.id_prefix,
                        source_index=source_index,
                        selected_index=len(rows),
                        legacy_sequential=args.legacy_sequential_ids,
                    ),
                    "text": text,
                    "metadata": {
                        "source_dataset": args.dataset,
                        "source_split": args.split,
                        "source_index": source_index,
                        "computer_use_score": score,
                        "strong_computer_hits": strong_hits,
                        "core_computer_hits": core_hits,
                        "title_core_hits": title_hits,
                        "title": title,
                    },
                }
            )
        if args.progress_every and scanned % args.progress_every == 0:
            print(json.dumps({"scanned": scanned, "written": len(rows)}, ensure_ascii=False), flush=True)
        if len(rows) >= args.target or scanned >= args.scan_limit:
            break

    write_jsonl(args.output, rows)
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "split": args.split,
                "scanned": scanned,
                "written": len(rows),
                "target": args.target,
                "output": str(args.output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
