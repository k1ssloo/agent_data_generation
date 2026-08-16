"""Execute episode-specific tools against hidden causal state."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import re
from typing import Any

from task_factory.bundle import TaskBundle
from .predicates import EvaluationError, evaluate_predicate, evaluate_value, resolve_path


class RuntimeError(ValueError):
    pass


@dataclass
class ExecutionResult:
    response: dict[str, Any]
    trace: dict[str, Any]


def _schema_matches(expected: str, value: Any) -> bool:
    mapping = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    python_type = mapping.get(expected)
    return python_type is not None and isinstance(value, python_type) and not (
        expected in {"integer", "number"} and isinstance(value, bool)
    )


def _validate_schema_value(value: Any, schema: dict[str, Any], path: str) -> None:
    expected = schema.get("type")
    if not _schema_matches(expected, value):
        raise RuntimeError(
            f"argument {path!r} expected {expected}, got {type(value).__name__}"
        )
    if "const" in schema and value != schema["const"]:
        raise RuntimeError(f"argument {path!r} must equal public const {schema['const']!r}")
    choices = schema.get("enum")
    if isinstance(choices, list) and value not in choices:
        raise RuntimeError(
            f"argument {path!r} must be one of the public enum values {choices!r}"
        )
    if expected == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for name in required:
            if name not in value:
                raise RuntimeError(f"missing required argument {path + '.' + name!r}")
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(properties))
            if unknown:
                raise RuntimeError(
                    f"argument {path!r} has unexpected properties {unknown!r}"
                )
        for name, item in value.items():
            definition = properties.get(name)
            if isinstance(definition, dict):
                _validate_schema_value(item, definition, f"{path}.{name}")
    elif expected == "array" and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            _validate_schema_value(item, schema["items"], f"{path}[{index}]")


def _set_path(path: str, context: dict[str, Any], value: Any) -> None:
    if not path.startswith("$state."):
        raise RuntimeError(f"effect target must be under $state: {path!r}")
    parts = path[len("$state.") :].split(".")
    current = context["state"]
    for raw in parts[:-1]:
        current = current.setdefault(raw, {})
        if not isinstance(current, dict):
            raise RuntimeError(f"cannot traverse effect path {path!r}")
    current[parts[-1]] = copy.deepcopy(value)


def _delete_path(path: str, context: dict[str, Any]) -> None:
    if not path.startswith("$state."):
        raise RuntimeError(f"effect target must be under $state: {path!r}")
    parts = path[len("$state.") :].split(".")
    current = context["state"]
    for raw in parts[:-1]:
        current = current[raw]
    current.pop(parts[-1], None)


def _collect_handles(value: Any, prefix: str = "$") -> list[dict[str, str]]:
    handles: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}"
            if isinstance(item, str) and (key.endswith("_handle") or key.endswith("_id")):
                handles.append({"value": item, "path": child})
            handles.extend(_collect_handles(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            handles.extend(_collect_handles(item, f"{prefix}[{index}]"))
    return handles


def _collect_observed_scalars(value: Any, prefix: str = "$") -> list[dict[str, Any]]:
    """Collect values that can legitimately be reused as later tool evidence.

    APIs commonly expose paths, URLs, tokens, names, and references without an
    ``_id`` suffix. Provenance therefore tracks every scalar observation while
    structural chain metrics continue to use canonical handle fields only.
    """
    observations: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            observations.extend(_collect_observed_scalars(item, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            observations.extend(_collect_observed_scalars(item, f"{prefix}[{index}]"))
    elif isinstance(value, (str, int, float, bool)) or value is None:
        observations.append({"value": value, "path": prefix})
    return observations


def _value_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _instruction_contains(instruction: str, value: Any) -> bool:
    if not isinstance(value, (str, int, float, bool)) or value is None:
        return False
    needle = _normalized_text(str(value))
    if not needle:
        return False
    haystack = _normalized_text(instruction)
    if isinstance(value, str) and re.fullmatch(r"[a-z0-9_]+", needle):
        return re.search(rf"(?<![a-z0-9_]){re.escape(needle)}(?![a-z0-9_])", haystack) is not None
    return needle in haystack


def _schema_evidence(definition: dict[str, Any], value: Any) -> list[dict[str, Any]]:
    evidence = []
    if "const" in definition and value == definition["const"]:
        evidence.append({"kind": "schema", "constraint": "const"})
    if isinstance(definition.get("enum"), list) and value in definition["enum"]:
        evidence.append({"kind": "schema", "constraint": "enum"})
    if "default" in definition and value == definition["default"]:
        evidence.append({"kind": "schema", "constraint": "default"})
    if definition.get("type") == "boolean" and isinstance(value, bool):
        evidence.append({"kind": "schema", "constraint": "boolean_domain"})
    return evidence


def _derived_evidence(
    definition: dict[str, Any], value: Any, observations: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    operation = definition.get("x-derivation")
    if operation not in {"lowercase", "uppercase", "snake_case", "slug"} or not isinstance(value, str):
        return []
    for entries in observations.values():
        for entry in entries:
            source_value = entry.get("value")
            if not isinstance(source_value, str):
                continue
            if operation == "lowercase":
                derived = source_value.lower()
            elif operation == "uppercase":
                derived = source_value.upper()
            elif operation == "snake_case":
                derived = re.sub(r"[^a-z0-9]+", "_", source_value.casefold()).strip("_")
            else:
                derived = re.sub(r"[^a-z0-9]+", "-", source_value.casefold()).strip("-")
            if value == derived:
                return [{"kind": "derivation", "operation": operation, "source": entry}]
    return []


def _leaf_values(value: Any, prefix: str = "$") -> list[tuple[str, Any]]:
    """Return scalar leaves so structured arguments can be grounded precisely."""
    if isinstance(value, dict):
        return [
            leaf
            for key, item in value.items()
            for leaf in _leaf_values(item, f"{prefix}.{key}")
        ]
    if isinstance(value, list):
        return [
            leaf
            for index, item in enumerate(value)
            for leaf in _leaf_values(item, f"{prefix}[{index}]")
        ]
    return [(prefix, value)]


def _sensitive_argument(name: str) -> bool:
    lowered = name.casefold()
    return any(token in lowered for token in ("password", "secret", "credential", "api_key", "private_key"))


def _state_paths(value: Any) -> set[str]:
    paths: set[str] = set()
    if isinstance(value, str) and value.startswith("$state"):
        paths.add(value)
    elif isinstance(value, list):
        for item in value:
            paths |= _state_paths(item)
    elif isinstance(value, dict):
        for item in value.values():
            paths |= _state_paths(item)
    return paths


class CausalRuntime:
    def __init__(self, bundle: TaskBundle):
        self.bundle = bundle
        self.state = copy.deepcopy(bundle.environment["initial_state"])
        self.step = 0
        self.trace: list[dict[str, Any]] = []
        self._bindings = {tool["name"]: tool for tool in bundle.tools}
        self._observations: dict[str, list[dict[str, Any]]] = {}

    def public_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool["parameters"],
                },
            }
            for tool in self.bundle.tools
        ]

    def _validate_args(self, binding: dict[str, Any], args: dict[str, Any]) -> None:
        schema = binding["parameters"]
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in args:
                raise RuntimeError(f"missing required argument {name!r}")
        for name, value in args.items():
            if name not in properties:
                raise RuntimeError(f"unexpected argument {name!r}")
            _validate_schema_value(value, properties[name], name)

    def _adapt_args(self, binding: dict[str, Any], public_args: dict[str, Any]) -> dict[str, Any]:
        mapping = binding.get("input_map", {})
        effective_args = copy.deepcopy(public_args)
        properties = binding["parameters"].get("properties", {})
        for name, definition in properties.items():
            if name not in effective_args and isinstance(definition, dict) and "default" in definition:
                effective_args[name] = copy.deepcopy(definition["default"])
        return {
            mapping.get(name, name): copy.deepcopy(value)
            for name, value in effective_args.items()
        }

    def _adapt_response(self, binding: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
        mapping = binding.get("output_map", {})
        return {mapping.get(name, name): copy.deepcopy(value) for name, value in response.items()}

    def _argument_provenance(self, binding: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
        required = set(binding.get("provenance_required", []))
        properties = binding["parameters"].get("properties", {})
        result: dict[str, Any] = {}
        for name, value in args.items():
            candidates = self._observations.get(_value_key(value), [])
            source = candidates[-1] if candidates else None
            classes = []
            evidence: list[dict[str, Any]] = []
            if source is not None:
                classes.append("tool_observation_grounded")
                evidence.append({"kind": "tool_observation", **source})
            if _instruction_contains(self.bundle.instruction, value):
                classes.append("user_grounded")
                evidence.append({"kind": "user_instruction"})
            schema_evidence = _schema_evidence(properties.get(name, {}), value)
            if schema_evidence:
                classes.append("schema_grounded")
                evidence.extend(schema_evidence)
                if any(item["constraint"] in {"enum", "boolean_domain"} for item in schema_evidence):
                    classes.append("agent_choice")
            derived_evidence = _derived_evidence(
                properties.get(name, {}), value, self._observations
            )
            if derived_evidence:
                classes.append("derived")
                evidence.extend(derived_evidence)
            if isinstance(value, (dict, list)):
                leaf_evidence = []
                leaf_classes: set[str] = set()
                all_grounded = True
                for leaf_path, leaf_value in _leaf_values(value):
                    leaf_candidates = self._observations.get(
                        _value_key(leaf_value), []
                    )
                    leaf_source = leaf_candidates[-1] if leaf_candidates else None
                    leaf_kind = None
                    if leaf_source is not None:
                        leaf_kind = "tool_observation_grounded"
                    elif _instruction_contains(self.bundle.instruction, leaf_value):
                        leaf_kind = "user_grounded"
                    elif leaf_value in (None, "", {}, []):
                        leaf_kind = "schema_grounded"
                    if leaf_kind is None:
                        all_grounded = False
                    else:
                        leaf_classes.add(leaf_kind)
                    leaf_evidence.append(
                        {
                            "kind": "structured_leaf",
                            "path": leaf_path,
                            "value": copy.deepcopy(leaf_value),
                            "provenance_kind": leaf_kind or "unexplained",
                            "source": leaf_source,
                        }
                    )
                if all_grounded and leaf_evidence:
                    classes.append("structured_grounded")
                    classes.extend(sorted(leaf_classes))
                    evidence.extend(leaf_evidence)
            if _sensitive_argument(name) and "user_grounded" not in classes:
                classes = []
                evidence = [{"kind": "rejected_sensitive_literal"}]
                source = None
            if source is not None:
                primary = "tool_observation_grounded"
            elif "user_grounded" in classes:
                primary = "user_grounded"
            elif "derived" in classes:
                primary = "derived"
            elif "schema_grounded" in classes:
                primary = "schema_grounded"
            elif "structured_grounded" in classes:
                primary = "structured_grounded"
            else:
                primary = "unexplained"
                classes = ["unexplained"]
            result[name] = {
                "value": copy.deepcopy(value),
                "source": source,
                "required": name in required,
                "provenance_kind": primary,
                "provenance_classes": classes,
                "evidence": evidence,
            }
        return result

    def execute(self, public_name: str, public_args: dict[str, Any]) -> ExecutionResult:
        if public_name not in self._bindings:
            raise RuntimeError(f"unknown public tool {public_name!r}")
        if not isinstance(public_args, dict):
            raise RuntimeError("tool arguments must be an object")
        binding = self._bindings[public_name]
        self._validate_args(binding, public_args)
        internal_args = self._adapt_args(binding, public_args)
        capability_id = binding["capability_id"]
        capability = self.bundle.environment["capabilities"][capability_id]
        context = {"state": self.state, "args": internal_args, "response": {}}
        selected: dict[str, Any] | None = None
        for branch in capability.get("branches", []):
            try:
                if evaluate_predicate(branch.get("when", True), context):
                    selected = branch
                    break
            except EvaluationError as exc:
                raise RuntimeError(f"cannot evaluate {capability_id}: {exc}") from exc
        if selected is None:
            raise RuntimeError(f"no branch matched for capability {capability_id!r}")
        try:
            internal_response = evaluate_value(selected.get("response", {}), context)
        except EvaluationError as exc:
            raise RuntimeError(f"cannot build response for {capability_id}: {exc}") from exc
        if not isinstance(internal_response, dict):
            raise RuntimeError(f"capability {capability_id!r} response must be an object")
        context["response"] = internal_response
        action_writes: list[str] = []
        environment_writes: list[str] = []

        def apply_effects(
            effects: list[dict[str, Any]], writes: list[str]
        ) -> None:
            for effect in effects:
                if "set" in effect:
                    value = evaluate_value(effect.get("value"), context)
                    _set_path(effect["set"], context, value)
                    writes.append(effect["set"])
                elif "increment" in effect:
                    current = resolve_path(effect["increment"], context)
                    amount = evaluate_value(effect.get("by", 1), context)
                    _set_path(effect["increment"], context, current + amount)
                    writes.append(effect["increment"])
                elif "delete" in effect:
                    _delete_path(effect["delete"], context)
                    writes.append(effect["delete"])
                else:
                    raise RuntimeError(f"unsupported effect {effect!r}")

        apply_effects(selected.get("effects", []), action_writes)
        # These model environment transitions that happen after the public
        # response snapshot, such as a concurrent revision advancing.
        transition_paths = sorted(
            {
                effect.get("set") or effect.get("increment") or effect.get("delete")
                for effect in selected.get("after_response_effects", [])
                if isinstance(effect, dict)
                and isinstance(
                    effect.get("set") or effect.get("increment") or effect.get("delete"),
                    str,
                )
            }
        )
        transition_before = {
            path: copy.deepcopy(resolve_path(path, context)) for path in transition_paths
        }
        apply_effects(
            selected.get("after_response_effects", []), environment_writes
        )
        environment_transitions = [
            {
                "path": path,
                "before": transition_before[path],
                "after": copy.deepcopy(resolve_path(path, context)),
            }
            for path in transition_paths
        ]
        response = self._adapt_response(binding, internal_response)
        self.step += 1
        produced = _collect_handles(response)
        provenance = self._argument_provenance(binding, public_args)
        consumed = [
            item["value"]
            for item in provenance.values()
            if item.get("source") is not None and isinstance(item.get("value"), str)
        ]
        raw_error_code = response.get("error_code")
        normalized_error_code = (
            None
            if isinstance(raw_error_code, str)
            and raw_error_code.strip().upper() in {"", "NO_ERROR", "NONE", "OK"}
            else raw_error_code
        )
        trace = {
            "step": self.step,
            "public_tool": public_name,
            "capability_id": capability_id,
            "arguments": provenance,
            "selected_branch": selected.get("id", "unnamed"),
            "read_set": sorted(set(selected.get("reads", []))),
            "write_set": sorted(
                set(action_writes + environment_writes + selected.get("writes", []))
            ),
            "action_write_set": sorted(set(action_writes)),
            "environment_write_set": sorted(set(environment_writes)),
            "environment_transitions": environment_transitions,
            "produced_handles": produced,
            "consumed_handles": consumed,
            "error_code": normalized_error_code,
            "resolves_errors": selected.get("resolves_errors", []),
            "response": response,
            "observed_state_paths": sorted(
                _state_paths(selected.get("response", {}))
                | set(selected.get("observes", []))
            ),
        }
        for observation in _collect_observed_scalars(response):
            self._observations.setdefault(_value_key(observation["value"]), []).append(
                {
                    "step": self.step,
                    "output_path": observation["path"],
                    "tool": public_name,
                    "value": copy.deepcopy(observation["value"]),
                }
            )
        self.trace.append(trace)
        return ExecutionResult(response=response, trace=trace)

    def evaluate_goals(self) -> list[dict[str, Any]]:
        results = []
        for item in self.bundle.contract.get("goal_predicates", []):
            predicate = item.get("predicate", item)
            try:
                valid = evaluate_predicate(predicate, {"state": self.state, "args": {}, "response": {}})
                error = None
            except EvaluationError as exc:
                valid = False
                error = str(exc)
            results.append({"id": item.get("id", "goal"), "valid": valid, "error": error})
        return results

    def evaluate_invariants(self) -> list[dict[str, Any]]:
        results = []
        for item in self.bundle.contract.get("invariants", []):
            predicate = item.get("predicate", item)
            try:
                valid = evaluate_predicate(predicate, {"state": self.state, "args": {}, "response": {}})
                error = None
            except EvaluationError as exc:
                valid = False
                error = str(exc)
            results.append({"id": item.get("id", "invariant"), "valid": valid, "error": error})
        return results
