"""Validate model-compiled WikiHow seed specifications against source text."""

from __future__ import annotations

import hashlib
from typing import Any


SEED_VERSION = "wikihow-seed-v1"


def source_sha256(source_text: str) -> str:
    return hashlib.sha256(source_text.encode("utf-8")).hexdigest()


def _spans(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def validate_wikihow_seed(
    seed: dict[str, Any],
    source_text: str,
    *,
    assigned_operator: str | None = None,
    source_id: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if seed.get("seed_version") != SEED_VERSION:
        errors.append(f"seed_version must be {SEED_VERSION!r}")
    if source_id is not None and seed.get("source_id") != source_id:
        errors.append(f"source_id must be {source_id!r}")
    expected_hash = source_sha256(source_text)
    if seed.get("source_sha256") != expected_hash:
        errors.append("source_sha256 does not match the current source text")
    if not isinstance(seed.get("objective"), str) or not seed.get("objective", "").strip():
        errors.append("objective must be a non-empty string")
    facts = seed.get("source_supported_facts")
    if not isinstance(facts, list) or len(facts) < 2:
        errors.append("source_supported_facts must contain at least two items")
        facts = []
    steps = seed.get("normalized_steps")
    if not isinstance(steps, list) or len(steps) < 4:
        errors.append("normalized_steps must contain at least four steps")
        steps = []
    seen_ids: set[str] = set()
    for kind, items in (("fact", facts), ("step", steps)):
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append(f"{kind}[{index}] must be an object")
                continue
            spans = _spans(item.get("evidence_spans"))
            if not spans:
                errors.append(f"{kind}[{index}] requires evidence_spans")
            for span in spans:
                if span not in source_text:
                    errors.append(f"{kind}[{index}] evidence span is not verbatim source text")
            if kind == "step":
                step_id = item.get("id")
                if not isinstance(step_id, str) or not step_id:
                    errors.append(f"step[{index}].id must be non-empty")
                elif step_id in seen_ids:
                    errors.append(f"duplicate step id {step_id!r}")
                else:
                    seen_ids.add(step_id)
                if not isinstance(item.get("action"), str) or not item.get("action", "").strip():
                    errors.append(f"step[{index}].action must be non-empty")
    affordances = seed.get("observable_affordances")
    if not isinstance(affordances, list) or not affordances:
        errors.append("observable_affordances must be a non-empty list")
        affordances = []
    for index, affordance in enumerate(affordances):
        if not isinstance(affordance, dict):
            errors.append(f"observable_affordances[{index}] must be an object")
            continue
        if not affordance.get("system"):
            errors.append(f"observable_affordances[{index}].system is required")
        for field in ("observations", "actions"):
            value = affordance.get(field)
            if not isinstance(value, list) or not value:
                errors.append(f"observable_affordances[{index}].{field} must be non-empty")
    extension = seed.get("synthetic_extension")
    if not isinstance(extension, dict):
        errors.append("synthetic_extension must be an object")
    else:
        operator = extension.get("operator")
        if assigned_operator and operator != assigned_operator:
            errors.append(
                f"synthetic_extension.operator must match assigned operator {assigned_operator!r}"
            )
        if not isinstance(extension.get("requirement"), str) or not extension.get(
            "requirement", ""
        ).strip():
            errors.append("synthetic_extension.requirement must be non-empty")
        if extension.get("claimed_as_source") is not False:
            errors.append("synthetic_extension.claimed_as_source must be false")
    feasibility = seed.get("operator_feasibility")
    if not isinstance(feasibility, dict) or feasibility.get("supported") is not True:
        errors.append("operator_feasibility.supported must be true")
    elif not isinstance(feasibility.get("reason"), str) or not feasibility.get("reason", "").strip():
        errors.append("operator_feasibility.reason must be non-empty")
    limits = seed.get("environment_design_limits")
    if not isinstance(limits, list) or not limits:
        errors.append("environment_design_limits must be a non-empty list")
    return errors


__all__ = ["SEED_VERSION", "source_sha256", "validate_wikihow_seed"]
