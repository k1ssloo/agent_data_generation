"""Resettable hidden environment used by evaluation and agentic RL."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from causal_validation import validate_episode
from rollout import EpisodeRunner
from runtime.executor import RuntimeError
from task_factory import load_task_bundle, totalize_public_capabilities
from task_factory.bundle import TaskBundle


class GemTaskEnvironment:
    """Expose only public policy context while retaining a private verifier."""

    def __init__(self, bundle: str | Path | TaskBundle, *, max_steps: int = 64):
        loaded = (
            bundle if isinstance(bundle, TaskBundle) else load_task_bundle(Path(bundle))
        )
        self.bundle = totalize_public_capabilities(loaded)
        self.max_steps = max_steps
        self.runner = EpisodeRunner(self.bundle, max_steps=max_steps)
        self._finished = False

    def reset(self) -> dict[str, Any]:
        self.runner = EpisodeRunner(self.bundle, max_steps=self.max_steps)
        self._finished = False
        return copy.deepcopy(self.runner.policy_context())

    def policy_context(self) -> dict[str, Any]:
        return copy.deepcopy(self.runner.policy_context())

    def step(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._finished:
            raise RuntimeError("environment is already finished; call reset()")
        try:
            observation = self.runner.tool_call(tool_name, arguments)
        except RuntimeError as exc:
            observation = {"ok": False, "error": str(exc)}
        return {
            "observation": copy.deepcopy(observation),
            "done": self.runner.status != "running",
            "status": self.runner.status,
            "policy_context": self.policy_context(),
        }

    def finish(self, content: str = "Task complete.") -> dict[str, Any]:
        if self._finished:
            return self.runner.report()
        self._finished = True
        return self.runner.finish(content)

    def score(self, report: dict[str, Any] | None = None) -> dict[str, Any]:
        return score_report(self.bundle, report or self.runner.report())


def score_report(bundle: TaskBundle, report: dict[str, Any]) -> dict[str, Any]:
    """Compute a deterministic outcome reward without plan imitation."""

    goals = report.get("goal_results", [])
    goal_fraction = (
        sum(bool(item.get("valid")) for item in goals) / len(goals) if goals else 0.0
    )
    invariant_results = [
        result
        for item in report.get("invariant_history", [])
        for result in item.get("results", [])
    ]
    invariant_fraction = (
        sum(bool(item.get("valid")) for item in invariant_results)
        / len(invariant_results)
        if invariant_results
        else 1.0
    )
    validation = validate_episode(
        bundle,
        report,
        min_delayed_handle_distance=0,
        min_handle_chain_depth=0,
        require_semantic_recovery=False,
    )
    missing_provenance = len(validation["metrics"].get("missing_provenance", []))
    provenance_score = 1.0 if missing_provenance == 0 else 0.0
    steps = len(report.get("trace", []))
    reference_steps = max(1, len(bundle.reference_plan.get("actions", [])))
    efficiency = min(1.0, reference_steps / max(1, steps))
    completed = validation["valid"]
    reward = (
        0.6 * goal_fraction
        + 0.15 * invariant_fraction
        + 0.15 * provenance_score
        + 0.1 * efficiency
    )
    if not completed:
        reward = min(reward, 0.89)
    return {
        "reward": round(reward, 6),
        "is_correct": completed,
        "signals": {
            "goal_fraction": round(goal_fraction, 6),
            "invariant_fraction": round(invariant_fraction, 6),
            "provenance": provenance_score,
            "efficiency": round(efficiency, 6),
            "steps": steps,
        },
        "validation": validation,
    }
