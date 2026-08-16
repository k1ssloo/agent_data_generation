"""Render alternate public APIs over stable internal capabilities."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import re
from typing import Any

from task_factory.bundle import TaskBundle


def _alias(kind: str, seed: str, value: str, index: int) -> str:
    digest = hashlib.sha256(f"{seed}:{kind}:{value}".encode("utf-8")).hexdigest()[:6]
    return f"{kind}_{index}_{digest}"


def _words(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("_", " ").strip())


def _semantic_tool_description(name: str, existing: Any) -> str:
    """Preserve public affordance semantics after removing the lexical name."""
    words = _words(name)
    verb, _, target = words.partition(" ")
    target = target or "current object"
    templates = {
        "inspect": f"Inspect the observable state of {target}.",
        "observe": f"Observe the current public state of {target}.",
        "open": f"Open {target} using the supplied public inputs.",
        "find": f"Find {target} from the supplied public criteria.",
        "resolve": f"Resolve {target} from the supplied public criteria.",
        "choose": f"Choose {target} from the currently available options.",
        "select": f"Select {target} from the currently available options.",
        "start": f"Start {target} using the supplied public inputs.",
        "create": f"Create {target} from the supplied public inputs.",
        "set": f"Set {target} to the supplied public value.",
        "login": f"Access {target} using supplied account or session evidence.",
        "logout": f"End the current {target} session.",
        "send": f"Send {target} using the supplied public object reference.",
        "place": f"Place {target} according to the supplied public settings.",
        "poll": f"Observe the current lifecycle state of {target}.",
        "diagnose": f"Diagnose the observed failure affecting {target}.",
        "repair": f"Repair the diagnosed state of {target}.",
        "reserve": f"Reserve {target} using current public evidence.",
    }
    semantic = templates.get(verb, f"Perform the public operation: {words}.")
    if isinstance(existing, str) and existing.strip():
        return semantic + " " + existing.strip()
    return semantic


def _semantic_parameter_description(name: str, existing: Any) -> str:
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    return f"Public input for {_words(name)}."


def render_alternate_api(bundle: TaskBundle, *, seed: str) -> TaskBundle:
    """Return an equivalent bundle with deterministic public name/schema changes.

    Tool descriptions and per-argument descriptions retain semantics. Internal
    capability IDs, state transitions, and goal predicates are unchanged.
    """
    bindings = copy.deepcopy(bundle.bindings)
    reference = copy.deepcopy(bundle.reference_plan)
    tool_name_map: dict[str, str] = {}
    arg_name_maps: dict[str, dict[str, str]] = {}

    for tool_index, tool in enumerate(bindings["tools"], start=1):
        old_name = tool["name"]
        new_name = _alias("operation", seed, old_name, tool_index)
        tool_name_map[old_name] = new_name
        tool["name"] = new_name
        tool["description"] = _semantic_tool_description(
            old_name, tool.get("description")
        )
        schema = tool["parameters"]
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        old_input_map = tool.get("input_map", {})
        old_provenance = set(tool.get("provenance_required", []))
        new_properties: dict[str, Any] = {}
        new_required: list[str] = []
        new_input_map: dict[str, str] = {}
        new_provenance: list[str] = []
        arg_map: dict[str, str] = {}
        for arg_index, (old_arg, definition) in enumerate(properties.items(), start=1):
            new_arg = _alias("parameter", seed + old_name, old_arg, arg_index)
            arg_map[old_arg] = new_arg
            updated_definition = copy.deepcopy(definition)
            updated_definition["description"] = _semantic_parameter_description(
                old_arg, updated_definition.get("description")
            )
            new_properties[new_arg] = updated_definition
            new_input_map[new_arg] = old_input_map.get(old_arg, old_arg)
            if old_arg in required:
                new_required.append(new_arg)
            if old_arg in old_provenance:
                new_provenance.append(new_arg)
        schema["properties"] = new_properties
        schema["required"] = new_required
        tool["input_map"] = new_input_map
        tool["provenance_required"] = new_provenance
        arg_name_maps[old_name] = arg_map

    action_lists = [reference["actions"]]
    action_lists.extend(
        variant["actions"] for variant in reference.get("counterfactuals", [])
    )
    for actions in action_lists:
        for action in actions:
            old_name = action["tool"]
            action["tool"] = tool_name_map[old_name]
            arg_map = arg_name_maps[old_name]
            action["arguments"] = {
                arg_map.get(name, name): value
                for name, value in action.get("arguments", {}).items()
            }

    manifest = copy.deepcopy(bundle.manifest)
    manifest["task_id"] = f"{bundle.task_id}__rendered_{seed}"
    manifest["renderer"] = {"kind": "semantic_alias_v2", "seed": seed}
    return replace(bundle, manifest=manifest, bindings=bindings, reference_plan=reference)
