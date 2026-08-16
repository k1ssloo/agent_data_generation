"""Deterministic runtime for task-first episodes."""

from __future__ import annotations

from typing import Any


__all__ = ["CausalRuntime", "ExecutionResult"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .executor import CausalRuntime, ExecutionResult

        return {"CausalRuntime": CausalRuntime, "ExecutionResult": ExecutionResult}[name]
    raise AttributeError(name)
