#!/usr/bin/env python3
"""Run a task-first bundle with an LLM that sees only public episode context."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_validation import (
    validate_adaptive_profile,
    validate_episode,
    validate_goal_alignment,
)
from causal_validation.intervention import evaluate_counterfactuals
from rollout import EpisodeRunner
from runtime.executor import RuntimeError as EpisodeRuntimeError
from task_factory import load_task_bundle
from scripts.llm_client import PROVIDERS, call_chat, parse_json_object, render_template


PROMPT = PROJECT_ROOT / "prompts" / "hidden_rollout_policy.txt"


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize_action(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("policy response must be an object")
    action = value.get("action")
    if action == "tool_call":
        name = value.get("name")
        arguments = value.get("arguments")
        if not isinstance(name, str) or not name:
            raise ValueError("tool_call requires a non-empty name")
        if not isinstance(arguments, dict):
            raise ValueError("tool_call requires object arguments")
        return {"action": action, "name": name, "arguments": arguments}
    if action == "final":
        content = value.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("final requires non-empty content")
        return {"action": action, "content": content.strip()}
    raise ValueError("action must be 'tool_call' or 'final'")


def public_prompt(runner: EpisodeRunner, last_error: str) -> str:
    context = runner.policy_context()
    return render_template(
        PROMPT.read_text(encoding="utf-8"),
        {"context_json": compact_json(context), "last_error_json": compact_json(last_error)},
    )


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=PROVIDERS, default=os.environ.get("GEM_LLM_PROVIDER", "openai"))
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    bundle = load_task_bundle(args.bundle)
    runner = EpisodeRunner(bundle=bundle, max_steps=args.max_steps)
    policy_trace: list[dict[str, Any]] = []
    last_error = ""
    final_report: dict[str, Any] | None = None

    for step in range(1, args.max_steps + 1):
        accepted = False
        for attempt in range(1, args.max_retries + 2):
            prompt = public_prompt(runner, last_error)
            trace_item: dict[str, Any] = {
                "step": step,
                "attempt": attempt,
            }
            try:
                raw, usage = call_chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    provider=args.provider,
                )
                trace_item["raw_response"] = raw
                trace_item["usage"] = usage
                action = normalize_action(parse_json_object(raw))
                trace_item["action"] = action
                if action["action"] == "final":
                    final_report = runner.finish(action["content"])
                    policy_trace.append(trace_item)
                    accepted = True
                    break
                response = runner.tool_call(action["name"], action["arguments"])
                trace_item["tool_response"] = response
                policy_trace.append(trace_item)
                last_error = ""
                accepted = True
                break
            except (json.JSONDecodeError, ValueError, EpisodeRuntimeError, RuntimeError) as exc:
                last_error = str(exc)
                trace_item["rejected"] = last_error
                policy_trace.append(trace_item)
                if attempt <= args.max_retries:
                    time.sleep(min(2.0 * attempt, 5.0))
        if final_report is not None or runner.status != "running":
            break
        if not accepted:
            runner.status = "agent_stopped_incomplete"
            runner.errors.append(f"step {step} failed after retries: {last_error}")
            break

    if final_report is None:
        final_report = runner.finish("Stopped without a validated completion.")
    validation = validate_episode(bundle, final_report)
    goal_alignment = validate_goal_alignment(bundle, final_report)
    counterfactual = evaluate_counterfactuals(bundle)
    adaptive_profile = validate_adaptive_profile(
        bundle,
        final_report,
        counterfactual,
        semantic_recovery_count=len(
            validation.get("metrics", {}).get("semantic_recoveries", [])
        ),
    )
    output = {
        "task_id": bundle.task_id,
        "generation_mode": "model_policy_hidden_environment_rollout",
        "policy_provider": args.provider,
        "episode": final_report,
        "policy_trace": policy_trace,
        "validation": validation,
        "goal_alignment": goal_alignment,
        "counterfactual_validation": counterfactual,
        "adaptive": adaptive_profile["valid"],
        "adaptive_profile": adaptive_profile,
    }
    write_json(args.output, output)
    print(
        json.dumps(
            {
                "task_id": bundle.task_id,
                "status": final_report["status"],
                "valid": validation["valid"]
                and goal_alignment["valid"]
                and counterfactual["valid"]
                and adaptive_profile["valid"],
                "causal_valid": validation["valid"],
                "goal_alignment_valid": goal_alignment["valid"],
                "counterfactual_valid": counterfactual["valid"],
                "adaptive": adaptive_profile["valid"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
