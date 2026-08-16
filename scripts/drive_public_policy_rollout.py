#!/usr/bin/env python3
"""Drive a public-context policy interactively against one hidden task bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


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


def write_result(path: Path, runner: EpisodeRunner, *, content: str) -> dict:
    episode = runner.finish(content)
    causal = validate_episode(runner.bundle, episode)
    alignment = validate_goal_alignment(runner.bundle, episode)
    counterfactual = evaluate_counterfactuals(runner.bundle)
    adaptive_profile = validate_adaptive_profile(
        runner.bundle,
        episode,
        counterfactual,
        semantic_recovery_count=len(
            causal.get("metrics", {}).get("semantic_recoveries", [])
        ),
    )
    result = {
        "task_id": runner.bundle.task_id,
        "generation_mode": "subagent_policy_hidden_environment_rollout",
        "episode": episode,
        "validation": causal,
        "goal_alignment": alignment,
        "counterfactual_validation": counterfactual,
        "adaptive": adaptive_profile["valid"],
        "adaptive_profile": adaptive_profile,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=32)
    args = parser.parse_args()
    runner = EpisodeRunner(load_task_bundle(args.bundle), max_steps=args.max_steps)
    print(
        json.dumps(
            {"ready": True, **runner.policy_context()},
            ensure_ascii=False,
        ),
        flush=True,
    )
    for line in sys.stdin:
        try:
            action = json.loads(line)
            if action.get("action") == "tool_call":
                response = runner.tool_call(
                    str(action.get("name", "")), action.get("arguments", {})
                )
                print(
                    json.dumps(
                        {
                            "ok": True,
                            "response": response,
                            "remaining_steps": runner.max_steps - runner.attempt_count,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            if action.get("action") == "final":
                result = write_result(
                    args.output,
                    runner,
                    content=str(action.get("content", "Task complete.")),
                )
                print(
                    json.dumps(
                        {
                            "ok": True,
                            "terminal": True,
                            "status": result["episode"]["status"],
                            "causal_valid": result["validation"]["valid"],
                            "goal_alignment_valid": result["goal_alignment"]["valid"],
                            "counterfactual_valid": result[
                                "counterfactual_validation"
                            ]["valid"],
                            "adaptive": result["adaptive"],
                        }
                    ),
                    flush=True,
                )
                return
            raise ValueError("action must be tool_call or final")
        except (json.JSONDecodeError, ValueError, EpisodeRuntimeError) as exc:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": str(exc),
                        "remaining_steps": runner.max_steps - runner.attempt_count,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
