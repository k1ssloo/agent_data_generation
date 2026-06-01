#!/usr/bin/env python3
"""Generate Stage 3 trajectories with step-by-step executable rollout.

Unlike the one-shot Stage 3 prompt, this script never shows the model the full
environment. The model proposes one assistant action at a time; tool calls are
executed locally through the text-exec DSL and the real tool response is appended
before the next model call.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import time
import urllib.error
from pathlib import Path
from typing import Any

from executable_environment import build_environment_for_row, execute_tool, validate_environment_spec
from llm_client import call_chat, parse_json_object, render_template


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = PROJECT_ROOT / "prompts"
STATE_CHANGING_PREFIXES = (
    "create_",
    "update_",
    "delete_",
    "attach_",
    "upload_",
    "download_",
    "export_",
    "send_",
    "set_",
    "start_",
    "cancel_",
)
VERIFICATION_PREFIXES = ("verify_", "read_", "get_", "list_", "search_", "poll_")


class ActionParseError(ValueError):
    """Carries raw model output when the next-action JSON cannot be parsed."""

    def __init__(self, message: str, raw_response: str | None = None, usage: dict[str, Any] | None = None):
        super().__init__(message)
        self.raw_response = raw_response
        self.usage = usage


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def expected_type_matches(expected: str, value: Any) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def tool_parameters(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {tool["function"]["name"]: tool["function"].get("parameters", {}) for tool in row.get("tools", [])}


def validate_tool_call(name: str, arguments: Any, parameters_by_name: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if name not in parameters_by_name:
        return [f"unknown tool {name!r}"]
    if not isinstance(arguments, dict):
        return [f"arguments for {name!r} must be an object"]
    schema = parameters_by_name[name]
    properties = schema.get("properties", {})
    for required in schema.get("required", []):
        if required not in arguments:
            errors.append(f"missing required argument {required!r} for {name!r}")
    for arg_name, value in arguments.items():
        if arg_name not in properties:
            errors.append(f"unexpected argument {arg_name!r} for {name!r}")
            continue
        expected = properties[arg_name].get("type", "")
        if not expected_type_matches(expected, value):
            errors.append(f"argument {arg_name!r} for {name!r} expected {expected}, got {type(value).__name__}")
    return errors


def action_signature(name: str, arguments: dict[str, Any]) -> str:
    return stable_json({"name": name, "arguments": arguments})


def normalize_action(parsed: Any) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(parsed, dict):
        return None, ["model output must be a JSON object"]
    if "tool_call" in parsed and "action" not in parsed:
        parsed = {"action": "tool_call", "tool_call": parsed["tool_call"]}
    action = parsed.get("action")
    if action == "tool_call":
        call = parsed.get("tool_call")
        if not isinstance(call, dict):
            return None, ["tool_call action requires a tool_call object"]
        name = call.get("name")
        arguments = call.get("arguments", {})
        if not isinstance(name, str) or not name:
            return None, ["tool_call.name must be a non-empty string"]
        return {"action": "tool_call", "name": name, "arguments": arguments}, []
    if action in {"message", "final"}:
        content = parsed.get("content")
        if not isinstance(content, str) or not content.strip():
            return None, [f"{action} action requires non-empty string content"]
        return {"action": action, "content": content.strip()}, []
    return None, ["action must be one of tool_call, message, or final"]


def is_verification_tool(name: str) -> bool:
    tokens = name.split("_")
    return name.startswith(VERIFICATION_PREFIXES) or "verify" in tokens or "status" in tokens


def last_tool_name(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if message.get("role") == "assistant" and "tool_call" in message:
            return message["tool_call"].get("name")
    return None


def workflow_tool_names(row: dict[str, Any]) -> list[str]:
    graph = row.get("workflow", {}).get("execution_graph", "")
    return re.findall(r"\(([a-zA-Z_][a-zA-Z0-9_]*)\)", graph)


def used_tool_names(messages: list[dict[str, Any]]) -> set[str]:
    return {
        message["tool_call"].get("name", "")
        for message in messages
        if message.get("role") == "assistant" and "tool_call" in message
    }


def recent_state_changing_without_verification(messages: list[dict[str, Any]]) -> bool:
    saw_state_change = False
    for message in reversed(messages):
        if message.get("role") != "assistant" or "tool_call" not in message:
            continue
        name = message["tool_call"].get("name", "")
        if is_verification_tool(name):
            return False
        if name.startswith(STATE_CHANGING_PREFIXES):
            saw_state_change = True
            continue
        if saw_state_change:
            continue
    return saw_state_change


def build_deterministic_seed(row: dict[str, Any]) -> list[dict[str, str]]:
    workflow = row.get("workflow", {})
    system = (
        "You are a tool-use assistant. Follow the task rules and workflow constraints. "
        "Use only provided tools. Tool results must come from actual tool responses; "
        "do not invent hidden IDs, available options, statuses, or state changes."
    )
    user = (
        "Please help me complete this task. Use the original instructions and workflow as constraints. "
        "If a requested target is unavailable, use a valid alternative only after a tool reveals it.\n\n"
        f"Task:\n{row.get('text', '')}\n\nWorkflow summary:\n{workflow.get('description', '')}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_seed_messages(
    row: dict[str, Any],
    provider: str,
    max_tokens: int,
    temperature: float,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    template = (PROMPT_DIR / "stage3_rollout_seed.txt").read_text(encoding="utf-8")
    prompt = render_template(
        template,
        {
            "text": row.get("text", ""),
            "workflow_json": compact_json(row.get("workflow", {})),
            "tools_json": compact_json(row.get("tools", [])),
        },
    )
    raw, usage = call_chat([{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=temperature, provider=provider)
    parsed = parse_json_object(raw)
    messages = parsed.get("messages") if isinstance(parsed, dict) else None
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("seed response must contain exactly two messages")
    expected_roles = ["system", "user"]
    seed: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError("seed messages must be objects")
        role = message.get("role")
        content = message.get("content")
        if role != expected_roles[index] or not isinstance(content, str) or not content.strip():
            raise ValueError("seed messages must be system then user with non-empty content")
        seed.append({"role": role, "content": content.strip()})
    return seed, {"raw_response": raw, "usage": usage}


def build_step_prompt(row: dict[str, Any], messages: list[dict[str, Any]], last_error: str) -> str:
    template = (PROMPT_DIR / "stage3_rollout_step.txt").read_text(encoding="utf-8")
    return render_template(
        template,
        {
            "text": row.get("text", ""),
            "workflow_json": compact_json(row.get("workflow", {})),
            "tools_json": compact_json(row.get("tools", [])),
            "messages_json": compact_json(messages),
            "last_error_json": compact_json(last_error),
        },
    )


def call_next_action(
    row: dict[str, Any],
    messages: list[dict[str, Any]],
    last_error: str,
    provider: str,
    max_tokens: int,
    temperature: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = build_step_prompt(row, messages, last_error)
    raw, usage = call_chat([{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=temperature, provider=provider)
    try:
        parsed = parse_json_object(raw)
    except json.JSONDecodeError as exc:
        raise ActionParseError(str(exc), raw_response=raw, usage=usage) from exc
    action, errors = normalize_action(parsed)
    if errors or action is None:
        raise ActionParseError("; ".join(errors), raw_response=raw, usage=usage)
    return action, {"raw_response": raw, "usage": usage}


def rollout_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    output = dict(row)
    output["stage3_generation_mode"] = "step_by_step_rollout"
    output["rollout_status"] = "failed"
    output["rollout_errors"] = []
    output["rollout_trace"] = []

    if row.get("missing_tool_requirements") or row.get("stage2_status") == "missing_tool":
        output["rollout_errors"].append("row has missing_tool_requirements")
        return output
    if not all(key in row for key in ("workflow", "tools", "environment")):
        output["rollout_errors"].append("row must contain workflow, tools, and environment")
        return output

    environment = build_environment_for_row(row)
    tool_names = {tool.get("function", {}).get("name", "") for tool in row.get("tools", [])}
    env_errors = validate_environment_spec(environment, tool_names=tool_names)
    if env_errors:
        output["rollout_errors"].extend(env_errors)
        return output

    if args.seed_mode == "llm":
        try:
            messages, seed_trace = generate_seed_messages(
                row,
                provider=args.provider,
                max_tokens=args.seed_max_tokens,
                temperature=args.seed_temperature,
            )
            output["rollout_trace"].append({"type": "seed", **seed_trace})
        except (RuntimeError, urllib.error.URLError, json.JSONDecodeError, ValueError, KeyError) as exc:
            output["rollout_errors"].append(f"seed generation failed: {exc}")
            return output
    else:
        messages = build_deterministic_seed(row)
        output["rollout_trace"].append({"type": "seed", "mode": "deterministic"})

    state = copy.deepcopy(environment.get("initial_state", {}))
    parameters_by_name = tool_parameters(row)
    last_error = ""
    tool_call_count = 0
    seen_tool_calls: set[str] = set()
    tool_call_name_counts: dict[str, int] = {}
    started = time.monotonic()

    for step_index in range(1, args.max_steps + 1):
        accepted = False
        for attempt_index in range(1, args.max_retries + 2):
            try:
                action, trace = call_next_action(
                    row,
                    messages,
                    last_error=last_error,
                    provider=args.provider,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
            except ActionParseError as exc:
                last_error = f"Invalid next action on step {step_index}, attempt {attempt_index}: {exc}"
                output["rollout_trace"].append(
                    {
                        "type": "step",
                        "step": step_index,
                        "attempt": attempt_index,
                        "error": last_error,
                        "raw_response": exc.raw_response,
                        "usage": exc.usage,
                    }
                )
                continue
            except (RuntimeError, urllib.error.URLError, json.JSONDecodeError, ValueError, KeyError) as exc:
                last_error = f"Invalid next action on step {step_index}, attempt {attempt_index}: {exc}"
                output["rollout_trace"].append({"type": "step", "step": step_index, "attempt": attempt_index, "error": last_error})
                continue

            trace.update({"type": "step", "step": step_index, "attempt": attempt_index, "action": action})

            if action["action"] == "tool_call":
                call_errors = validate_tool_call(action["name"], action["arguments"], parameters_by_name)
                if call_errors:
                    last_error = "; ".join(call_errors)
                    trace["rejected"] = last_error
                    output["rollout_trace"].append(trace)
                    continue

                signature = action_signature(action["name"], action["arguments"])
                if args.reject_duplicate_tool_calls and signature in seen_tool_calls:
                    last_error = (
                        f"Duplicate tool call rejected: {action['name']} with the same arguments was already executed. "
                        "Use new grounded information, choose a different tool, or finish if the task is impossible."
                    )
                    trace["rejected"] = last_error
                    output["rollout_trace"].append(trace)
                    continue
                if args.max_same_tool_calls and tool_call_name_counts.get(action["name"], 0) >= args.max_same_tool_calls:
                    last_error = (
                        f"Tool {action['name']!r} has already been called {tool_call_name_counts[action['name']]} times. "
                        "Stop repeating it; use a different grounded step or finish with the available evidence."
                    )
                    trace["rejected"] = last_error
                    output["rollout_trace"].append(trace)
                    continue

                response, execution_errors = execute_tool(action["name"], action["arguments"], state, environment)
                if execution_errors:
                    last_error = "; ".join(execution_errors)
                    trace["rejected"] = last_error
                    output["rollout_trace"].append(trace)
                    continue

                messages.append({"role": "assistant", "tool_call": {"name": action["name"], "arguments": action["arguments"]}})
                messages.append({"role": "tool", "name": action["name"], "content": response})
                tool_call_count += 1
                seen_tool_calls.add(signature)
                tool_call_name_counts[action["name"]] = tool_call_name_counts.get(action["name"], 0) + 1
                last_error = ""
                trace["tool_response"] = response
                output["rollout_trace"].append(trace)
                accepted = True
                break

            if action["action"] == "final":
                if args.min_tool_calls and tool_call_count < args.min_tool_calls:
                    last_error = f"Need at least {args.min_tool_calls} tool calls before finalizing; only {tool_call_count} observed."
                    trace["rejected"] = last_error
                    output["rollout_trace"].append(trace)
                    continue
                if args.require_workflow_tools:
                    missing = sorted(set(workflow_tool_names(row)) - used_tool_names(messages))
                    if missing:
                        last_error = f"Finalization rejected because workflow tools have not been used: {missing}"
                        trace["rejected"] = last_error
                        output["rollout_trace"].append(trace)
                        continue
                if args.require_final_verification and recent_state_changing_without_verification(messages):
                    last = last_tool_name(messages)
                    last_error = f"Finalization rejected because last relevant tool {last!r} was not a verification/read/get/list/search/poll tool."
                    trace["rejected"] = last_error
                    output["rollout_trace"].append(trace)
                    continue
                messages.append({"role": "assistant", "content": action["content"]})
                output["messages"] = messages
                output["rollout_status"] = "completed"
                output["rollout_tool_calls"] = tool_call_count
                output["rollout_elapsed_sec"] = round(time.monotonic() - started, 3)
                output["rollout_trace"].append(trace)
                return output

            if "?" in action["content"]:
                last_error = "Do not ask the user a new question during rollout; inspect with tools or finish with the available evidence."
                trace["rejected"] = last_error
                output["rollout_trace"].append(trace)
                continue

            messages.append({"role": "assistant", "content": action["content"]})
            last_error = ""
            output["rollout_trace"].append(trace)
            accepted = True
            break

        if not accepted:
            output["rollout_errors"].append(f"step {step_index} failed after retries: {last_error}")
            output["messages"] = messages
            output["rollout_tool_calls"] = tool_call_count
            output["rollout_elapsed_sec"] = round(time.monotonic() - started, 3)
            return output

    output["messages"] = messages
    output["rollout_status"] = "max_steps"
    output["rollout_tool_calls"] = tool_call_count
    output["rollout_elapsed_sec"] = round(time.monotonic() - started, 3)
    output["rollout_errors"].append(f"exceeded max_steps={args.max_steps}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Stage 2 artifacts JSONL with workflow, tools, and environment.")
    parser.add_argument("--output", type=Path, required=True, help="Stage 3 rollout artifacts JSONL.")
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--provider", choices=["openai", "gemini"], default=os.environ.get("GEM_LLM_PROVIDER", "openai"))
    parser.add_argument("--seed-mode", choices=["llm", "deterministic"], default="llm")
    parser.add_argument("--seed-max-tokens", type=int, default=2000)
    parser.add_argument("--seed-temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--min-tool-calls", type=int, default=0)
    parser.add_argument("--require-workflow-tools", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-final-verification", action="store_true")
    parser.add_argument("--reject-duplicate-tool-calls", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-same-tool-calls", type=int, default=6, help="0 disables the per-tool repetition guard.")
    parser.add_argument("--completed-only", action="store_true", help="Write only completed rollout rows.")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]

    outputs = []
    for index, row in enumerate(rows, start=1):
        result = rollout_row(row, args)
        if not args.completed_only or result.get("rollout_status") == "completed":
            outputs.append(result)
        print(
            json.dumps(
                {
                    "done": index,
                    "id": row.get("id"),
                    "status": result.get("rollout_status"),
                    "tool_calls": result.get("rollout_tool_calls", 0),
                    "errors": result.get("rollout_errors", [])[:2],
                },
                ensure_ascii=False,
            )
        )

    write_jsonl(args.output, outputs)
    print(
        json.dumps(
            {
                "checked": len(rows),
                "written": len(outputs),
                "completed": sum(row.get("rollout_status") == "completed" for row in outputs),
                "output": str(args.output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
