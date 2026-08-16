"""rLLM rollout and evaluator for GEM task bundles.

Install rLLM separately. Keeping this module outside the core runtime lets the
offline validators and data factory remain dependency-free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training.package import resolve_bundle_path


def _task_value(task: Any, name: str, default: Any = None) -> Any:
    if isinstance(task, dict):
        return task.get(name, default)
    return getattr(task, name, default)


def _bundle_path(task: Any) -> Path:
    metadata = _task_value(task, "metadata", {}) or {}
    nested = metadata.get("metadata")
    if isinstance(nested, dict):
        metadata = {**metadata, **nested}
    return resolve_bundle_path(metadata, fallback_root=Path.cwd())


try:
    import rllm
    from openai import AsyncOpenAI
    from rllm.eval.types import EvalOutput, Signal
    from rllm.types import AgentConfig, Episode, Step, Task, Trajectory
except ImportError:  # pragma: no cover - exercised only in training environment
    rllm = None


if rllm is not None:
    from training.gem_environment import GemTaskEnvironment

    @rllm.rollout(name="gem-tool-agent")
    async def gem_tool_rollout(task: Task, config: AgentConfig) -> Episode:
        """Run a policy against one fresh hidden causal environment."""

        environment = GemTaskEnvironment(_bundle_path(task))
        context = environment.reset()
        client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
        steps: list[Step] = []
        final_text = ""
        while context["remaining_steps"] > 0:
            request_messages = list(context["messages"])
            response = await client.chat.completions.create(
                model=config.model,
                messages=request_messages,
                tools=context["tools"],
                tool_choice="auto",
            )
            message = response.choices[0].message
            output = message.model_dump(exclude_none=True)
            observations = []
            for call in message.tool_calls or []:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                transition = environment.step(call.function.name, arguments)
                observations.append(transition["observation"])
                if transition["done"]:
                    break
            steps.append(
                Step(
                    input=request_messages,
                    output=output,
                    observation=observations,
                    done=(
                        not bool(message.tool_calls)
                        or environment.runner.status != "running"
                    ),
                )
            )
            if not message.tool_calls:
                final_text = message.content or ""
                break
            context = environment.policy_context()
            if environment.runner.status != "running":
                break
        report = environment.finish(final_text or "Task complete.")
        return Episode(
            task=task,
            trajectories=[Trajectory(name="gem-tool-agent", task=task, steps=steps)],
            artifacts={"environment_report": report},
            metadata={"environment_status": report["status"]},
        )

    @rllm.evaluator
    def gem_causal_evaluator(task: Task, episode: Episode) -> EvalOutput:
        """Reward goal achievement and causal validity, not plan matching."""

        environment = GemTaskEnvironment(_bundle_path(task))
        report = episode.artifacts.get("environment_report", {})
        scored = environment.score(report)
        return EvalOutput(
            reward=scored["reward"],
            is_correct=scored["is_correct"],
            signals=[
                Signal(name=name, value=float(value))
                for name, value in scored["signals"].items()
            ],
        )

else:
    gem_tool_rollout = None
    gem_causal_evaluator = None


def require_rllm() -> None:
    if rllm is None:
        raise RuntimeError(
            "rLLM is not installed; install rllm[verl] in the training environment"
        )
