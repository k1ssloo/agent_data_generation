"""Static validation for causal-runtime-v1 environment specifications."""

from __future__ import annotations

from typing import Any

from .predicates import EvaluationError, evaluate_predicate, evaluate_value


def validate_runtime_spec(environment: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if environment.get("runtime_version") != "causal-runtime-v1":
        errors.append("runtime_version must be 'causal-runtime-v1'")
    initial_state = environment.get("initial_state")
    if not isinstance(initial_state, dict):
        errors.append("initial_state must be an object")
        initial_state = {}
    capabilities = environment.get("capabilities")
    if not isinstance(capabilities, dict) or not capabilities:
        errors.append("capabilities must be a non-empty object")
        return errors
    for capability_id, capability in capabilities.items():
        prefix = f"capabilities.{capability_id}"
        if not isinstance(capability, dict):
            errors.append(f"{prefix} must be an object")
            continue
        branches = capability.get("branches")
        if not isinstance(branches, list) or not branches:
            errors.append(f"{prefix}.branches must be a non-empty list")
            continue
        branch_ids: set[str] = set()
        for index, branch in enumerate(branches):
            branch_prefix = f"{prefix}.branches[{index}]"
            if not isinstance(branch, dict):
                errors.append(f"{branch_prefix} must be an object")
                continue
            branch_id = branch.get("id")
            if not isinstance(branch_id, str) or not branch_id:
                errors.append(f"{branch_prefix}.id must be a non-empty string")
            elif branch_id in branch_ids:
                errors.append(f"{prefix} has duplicate branch id {branch_id!r}")
            else:
                branch_ids.add(branch_id)
            if "response" not in branch or not isinstance(branch.get("response"), dict):
                errors.append(f"{branch_prefix}.response must be an object")
            for field in ("reads", "writes", "resolves_errors"):
                value = branch.get(field, [])
                if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                    errors.append(f"{branch_prefix}.{field} must be a string list")
            observes = branch.get("observes", [])
            if not isinstance(observes, list) or any(
                not isinstance(item, str) or not item.startswith("$state.")
                for item in observes
            ):
                errors.append(f"{branch_prefix}.observes must be a $state path list")
                observes = []
            reads = branch.get("reads", [])
            if isinstance(reads, list):
                undeclared = sorted(set(observes) - set(reads))
                if undeclared:
                    errors.append(
                        f"{branch_prefix}.observes paths must also be declared in reads: "
                        f"{undeclared}"
                    )
            effects = branch.get("effects", [])
            after_response_effects = branch.get("after_response_effects", [])
            if not isinstance(effects, list):
                errors.append(f"{branch_prefix}.effects must be a list")
                effects = []
            if not isinstance(after_response_effects, list):
                errors.append(f"{branch_prefix}.after_response_effects must be a list")
                after_response_effects = []
            for effect_index, effect in enumerate(effects + after_response_effects):
                field = "effects" if effect_index < len(effects) else "after_response_effects"
                local_index = effect_index if field == "effects" else effect_index - len(effects)
                effect_prefix = f"{branch_prefix}.{field}[{local_index}]"
                if not isinstance(effect, dict):
                    errors.append(f"{effect_prefix} must be an object")
                    continue
                operations = {"set", "increment", "delete"} & set(effect)
                if len(operations) != 1:
                    errors.append(f"{effect_prefix} must contain exactly one effect operation")
                    continue
                target = effect[next(iter(operations))]
                if not isinstance(target, str) or not target.startswith("$state."):
                    errors.append(f"{effect_prefix} target must start with '$state.'")
            # Expressions depending on args cannot be fully evaluated statically,
            # but literal/state-only expressions should still be checked.
            context = {"state": initial_state, "args": {}, "response": {}}
            try:
                when = branch.get("when", True)
                condition_matches: bool | None = None
                if "$args" not in str(when):
                    condition_matches = evaluate_predicate(when, context)
                response = branch.get("response", {})
                if (
                    condition_matches is True
                    and "$args" not in str(response)
                    and "$response" not in str(response)
                ):
                    evaluate_value(response, context)
            except EvaluationError as exc:
                errors.append(f"{branch_prefix} expression error: {exc}")
    return errors
