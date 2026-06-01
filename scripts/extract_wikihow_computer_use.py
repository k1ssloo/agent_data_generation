#!/usr/bin/env python3
"""Extract computer-use procedural WikiHow rows from a HuggingFace dataset."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset


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
    parser.add_argument("--min-score", type=int, default=8)
    parser.add_argument("--min-strong-hits", type=int, default=1)
    parser.add_argument("--min-core-hits", type=int, default=1)
    parser.add_argument("--require-title-hit", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    rows: list[dict[str, Any]] = []
    scanned = 0
    for source_index, source_row in enumerate(dataset):
        scanned += 1
        text = row_text(source_row)
        title = normalize(source_row.get("title"))
        score = procedural_score(text)
        strong_hits = strong_hit_count(text)
        core_hits = core_hit_count(text)
        title_hits = core_hit_count(title)
        if (
            score >= args.min_score
            and strong_hits >= args.min_strong_hits
            and core_hits >= args.min_core_hits
            and (not args.require_title_hit or title_hits > 0)
        ):
            rows.append(
                {
                    "id": f"wikihow_computer_{len(rows):06d}",
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
