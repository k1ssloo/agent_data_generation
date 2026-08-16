"""Append-only metadata archive for accepted evolved task bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TaskArchive:
    def __init__(self, path: Path):
        self.path = path

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def add(self, row: dict[str, Any]) -> None:
        existing = self.rows()
        if any(item.get("task_id") == row.get("task_id") for item in existing):
            raise ValueError(f"task {row.get('task_id')!r} already exists in archive")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
