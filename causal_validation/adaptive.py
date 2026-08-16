"""Validate evidence-backed adaptive behavior profiles."""

from __future__ import annotations

from typing import Any

from task_factory.bundle import TaskBundle


def _covers(path: str, observed: list[str]) -> bool:
    return any(path.startswith(item) or item.startswith(path) for item in observed)


def validate_closed_loop_control(
    bundle: TaskBundle, episode: dict[str, Any]
) -> dict[str, Any]:
    """Prove that a public measurement causally grounds a later control action."""

    spec = bundle.contract.get("requirements", {}).get("closed_loop_control")
    if not isinstance(spec, dict):
        return {
            "required": False,
            "valid": False,
            "errors": ["closed-loop control is not declared"],
        }
    required_strings = (
        "measurement_tool",
        "control_tool",
        "evidence_argument",
        "final_observation_tool",
    )
    errors = [
        f"closed_loop_control.{key} must be a non-empty string"
        for key in required_strings
        if not isinstance(spec.get(key), str) or not spec[key]
    ]
    path_fields = ("measurement_paths", "controlled_paths", "settled_paths")
    for key in path_fields:
        value = spec.get(key)
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(path, str) or not path.startswith("$state.") for path in value)
        ):
            errors.append(f"closed_loop_control.{key} must be a non-empty $state path list")
    if errors:
        return {"required": True, "valid": False, "errors": errors}

    trace = episode.get("trace", [])
    controls = [
        step for step in trace if step.get("public_tool") == spec["control_tool"]
    ]
    if not controls:
        errors.append("declared control tool was not executed")
        return {"required": True, "valid": False, "errors": errors}

    witnesses = []
    for control in controls:
        provenance = control.get("arguments", {}).get(spec["evidence_argument"], {})
        source = provenance.get("source") if isinstance(provenance, dict) else None
        if not isinstance(source, dict) or source.get("tool") != spec["measurement_tool"]:
            continue
        measurement_step = int(source.get("step", 0))
        if measurement_step <= 0 or measurement_step >= int(control.get("step", 0)):
            continue
        measurement = next(
            (step for step in trace if int(step.get("step", 0)) == measurement_step),
            None,
        )
        if not isinstance(measurement, dict):
            continue
        missing_measurements = [
            path
            for path in spec["measurement_paths"]
            if not _covers(path, measurement.get("observed_state_paths", []))
        ]
        missing_writes = [
            path
            for path in spec["controlled_paths"]
            if not _covers(path, control.get("write_set", []))
        ]
        final = next(
            (
                step
                for step in trace
                if int(step.get("step", 0)) > int(control.get("step", 0))
                and step.get("public_tool") == spec["final_observation_tool"]
                and not step.get("write_set")
                and all(
                    _covers(path, step.get("observed_state_paths", []))
                    for path in spec["settled_paths"]
                )
            ),
            None,
        )
        witnesses.append(
            {
                "measurement_step": measurement_step,
                "control_step": control.get("step"),
                "final_observation_step": final.get("step") if final else None,
                "missing_measurement_paths": missing_measurements,
                "missing_controlled_paths": missing_writes,
                "valid": not missing_measurements and not missing_writes and final is not None,
            }
        )
    if not any(item["valid"] for item in witnesses):
        errors.append(
            "no control action consumed public measurement evidence, changed all "
            "declared controlled paths, and received a final settle observation"
        )
    return {
        "required": True,
        "valid": not errors,
        "errors": errors,
        "witnesses": witnesses,
    }


def validate_temporal_provenance(
    bundle: TaskBundle, episode: dict[str, Any]
) -> dict[str, Any]:
    """Prove that observed temporal identities flow into the final commit."""

    spec = bundle.contract.get("requirements", {}).get("temporal_provenance")
    if not isinstance(spec, dict):
        return {
            "required": False,
            "valid": False,
            "errors": ["temporal provenance is not declared"],
        }
    links = spec.get("links")
    final_tool = spec.get("final_observation_tool")
    final_paths = spec.get("final_paths")
    errors = []
    if not isinstance(links, list) or not links:
        errors.append("temporal_provenance.links must be a non-empty list")
        links = []
    if not isinstance(final_tool, str) or not final_tool:
        errors.append("temporal_provenance.final_observation_tool must be a string")
    if (
        not isinstance(final_paths, list)
        or not final_paths
        or any(
            not isinstance(path, str) or not path.startswith("$state.")
            for path in final_paths
        )
    ):
        errors.append("temporal_provenance.final_paths must be a non-empty $state path list")
        final_paths = []

    trace = episode.get("trace", [])
    witnesses = []
    consumer_steps = []
    for index, link in enumerate(links):
        if not isinstance(link, dict) or any(
            not isinstance(link.get(key), str) or not link[key]
            for key in ("consumer_tool", "argument", "producer_tool")
        ):
            errors.append(f"temporal_provenance.links[{index}] is invalid")
            continue
        witness = None
        for step in trace:
            if step.get("public_tool") != link["consumer_tool"]:
                continue
            detail = step.get("arguments", {}).get(link["argument"], {})
            source = detail.get("source") if isinstance(detail, dict) else None
            if (
                isinstance(source, dict)
                and source.get("tool") == link["producer_tool"]
                and 0 < int(source.get("step", 0)) < int(step.get("step", 0))
            ):
                witness = {
                    "producer_step": int(source["step"]),
                    "producer_tool": source["tool"],
                    "consumer_step": int(step["step"]),
                    "consumer_tool": step["public_tool"],
                    "argument": link["argument"],
                }
                consumer_steps.append(int(step["step"]))
                break
        witnesses.append(witness)
        if witness is None:
            errors.append(
                f"no {link['consumer_tool']}.{link['argument']} consumed "
                f"evidence from {link['producer_tool']}"
            )

    last_consumer = max(consumer_steps, default=0)
    final = next(
        (
            step
            for step in trace
            if int(step.get("step", 0)) > last_consumer
            and step.get("public_tool") == final_tool
            and not step.get("write_set")
            and all(
                _covers(path, step.get("observed_state_paths", []))
                for path in final_paths
            )
        ),
        None,
    )
    if final is None:
        errors.append(
            "no post-commit read-only observation covers temporal provenance paths"
        )
    return {
        "required": True,
        "valid": not errors,
        "errors": errors,
        "witnesses": witnesses,
        "final_observation_step": final.get("step") if final else None,
    }


def validate_adaptive_profile(
    bundle: TaskBundle,
    episode: dict[str, Any],
    counterfactual: dict[str, Any],
    *,
    semantic_recovery_count: int,
) -> dict[str, Any]:
    decision_metrics = counterfactual.get("decision_metrics", {})
    has_planning = (
        int(decision_metrics.get("meaningful_planning_decision_count", 0)) > 0
        and float(decision_metrics.get("decision_entropy_bits", 0.0)) > 0.0
    )
    closed_loop = validate_closed_loop_control(bundle, episode)
    temporal = validate_temporal_provenance(bundle, episode)
    profiles = []
    if has_planning and semantic_recovery_count > 0:
        profiles.append("planning_with_semantic_recovery")
    if has_planning and closed_loop["valid"]:
        profiles.append("planning_with_closed_loop_control")
    if has_planning and temporal["valid"]:
        profiles.append("planning_with_temporal_provenance")
    return {
        "valid": bool(profiles),
        "profiles": profiles,
        "has_planning": has_planning,
        "semantic_recovery_count": semantic_recovery_count,
        "closed_loop_control": closed_loop,
        "temporal_provenance": temporal,
    }


__all__ = [
    "validate_adaptive_profile",
    "validate_closed_loop_control",
    "validate_temporal_provenance",
]
