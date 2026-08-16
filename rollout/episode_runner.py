"""Run actions without exposing contracts, hidden state, or oracle plans."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from runtime.executor import CausalRuntime, RuntimeError
from task_factory.bundle import TaskBundle


@dataclass
class EpisodeRunner:
    bundle: TaskBundle
    max_steps: int = 64
    runtime: CausalRuntime = field(init=False)
    messages: list[dict[str, Any]] = field(init=False)
    status: str = field(default="running", init=False)
    errors: list[str] = field(default_factory=list, init=False)
    invariant_history: list[dict[str, Any]] = field(default_factory=list, init=False)
    attempt_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.runtime = CausalRuntime(self.bundle)
        self.messages = [
            {
                "role": "system",
                "content": (
                    "Complete the user's task using only the provided tools. "
                    "Infer tool behavior from public schemas and observations. "
                    "Do not assume hidden state or claim success without evidence."
                ),
            },
            {"role": "user", "content": self.bundle.instruction.strip()},
        ]

    def policy_context(self) -> dict[str, Any]:
        """Return the complete and only context an external policy may receive."""
        return {
            "task_id": self.bundle.task_id,
            "messages": self.messages,
            "tools": self.runtime.public_tools(),
            "remaining_steps": max(0, self.max_steps - self.attempt_count),
        }

    def tool_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.status != "running":
            raise RuntimeError(f"episode is already terminal: {self.status}")
        if self.attempt_count >= self.max_steps:
            self.status = "budget_exhausted"
            raise RuntimeError("episode step budget exhausted")
        self.attempt_count += 1
        call_id = f"call_{self.attempt_count:04d}"
        assistant_call = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(
                            arguments,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                }
            ],
        }
        try:
            result = self.runtime.execute(name, arguments)
        except RuntimeError as exc:
            self.errors.append(str(exc))
            self.messages.append(assistant_call)
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(
                        {"ok": False, "error": str(exc)},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
            raise
        self.messages.append(assistant_call)
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(
                    result.response,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
        invariants = self.runtime.evaluate_invariants()
        self.invariant_history.append({"step": result.trace["step"], "results": invariants})
        if any(not item["valid"] for item in invariants):
            self.status = "invariant_violation"
        return result.response

    def assistant_message(self, content: str) -> None:
        if self.status != "running":
            raise RuntimeError(f"episode is already terminal: {self.status}")
        self.messages.append({"role": "assistant", "content": content})

    def finish(self, content: str = "Task complete.") -> dict[str, Any]:
        if self.status == "running":
            goals = self.runtime.evaluate_goals()
            self.status = "goal_satisfied" if goals and all(item["valid"] for item in goals) else "agent_stopped_incomplete"
        else:
            goals = self.runtime.evaluate_goals()
        self.messages.append({"role": "assistant", "content": content})
        return self.report(goals=goals)

    def report(self, *, goals: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return {
            "task_id": self.bundle.task_id,
            "status": self.status,
            "messages": self.messages,
            "public_tools": self.runtime.public_tools(),
            "trace": self.runtime.trace,
            "goal_results": goals if goals is not None else self.runtime.evaluate_goals(),
            "invariant_history": self.invariant_history,
            "errors": self.errors,
        }


def run_reference_plan(
    bundle: TaskBundle,
    *,
    max_steps: int = 64,
    actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute the private plan as an oracle proof; never expose it in policy context."""
    runner = EpisodeRunner(bundle=bundle, max_steps=max_steps)
    for action in actions if actions is not None else bundle.reference_plan["actions"]:
        try:
            runner.tool_call(action["tool"], action.get("arguments", {}))
        except RuntimeError as exc:
            runner.status = "terminal_failure"
            runner.errors.append(str(exc))
            break
        if runner.status != "running":
            break
    return runner.finish("Task completed and final evidence was inspected.")
