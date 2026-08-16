"""Build and sample public API dependency graphs from runtime provenance."""

from __future__ import annotations

import random
from typing import Any


def build_api_dependency_graph(report: dict[str, Any]) -> dict[str, Any]:
    trace = report.get("trace", [])
    nodes = sorted({step.get("public_tool") for step in trace if step.get("public_tool")})
    edge_counts: dict[tuple[str, str], int] = {}
    for step in trace:
        consumer = step.get("public_tool")
        for detail in step.get("arguments", {}).values():
            source = detail.get("source")
            producer = source.get("tool") if isinstance(source, dict) else None
            if producer and consumer and producer != consumer:
                edge_counts[(producer, consumer)] = edge_counts.get((producer, consumer), 0) + 1
    edges = [
        {"source": source, "target": target, "provenance_count": count}
        for (source, target), count in sorted(edge_counts.items())
    ]
    return {"nodes": nodes, "edges": edges}


def sample_dependency_walks(
    graph: dict[str, Any], *, count: int, min_length: int, max_length: int, seed: int
) -> list[list[str]]:
    if min_length < 1 or max_length < min_length:
        raise ValueError("invalid dependency walk length")
    adjacency: dict[str, list[str]] = {node: [] for node in graph.get("nodes", [])}
    for edge in graph.get("edges", []):
        adjacency.setdefault(edge["source"], []).append(edge["target"])
    starts = sorted(node for node, targets in adjacency.items() if targets)
    if not starts:
        return []
    rng = random.Random(seed)
    walks: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    attempts = max(count * 20, 20)
    for _ in range(attempts):
        current = rng.choice(starts)
        walk = [current]
        target_length = rng.randint(min_length, max_length)
        while len(walk) < target_length and adjacency.get(current):
            current = rng.choice(sorted(adjacency[current]))
            walk.append(current)
        key = tuple(walk)
        if len(walk) >= min_length and key not in seen:
            walks.append(walk)
            seen.add(key)
        if len(walks) >= count:
            break
    return walks
