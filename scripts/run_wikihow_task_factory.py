#!/usr/bin/env python3
"""Generate, repair, validate, and recursively evolve task-first WikiHow data."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import tomllib
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_validation import (
    evaluate_action_ablation,
    validate_adaptive_profile,
    validate_episode,
    validate_goal_alignment,
    validate_tool_identifiability,
    validate_vnext_adaptive_profile,
)
from causal_validation.intervention import evaluate_counterfactuals
from rollout import run_reference_plan
from runtime.tool_renderer import render_alternate_api
from scripts.llm_client import PROVIDERS, call_chat, parse_json_object, render_template
from task_factory import totalize_public_capabilities, validate_public_executability
from task_factory.bundle import TaskBundle, load_task_bundle, validate_bundle
from task_factory.contracts import normalize_contract, validate_contract
from task_factory.control_repair import repair_immediate_ordinal_provenance
from task_factory.evidence_repair import repair_final_goal_evidence
from task_factory.materialize import materialize_candidate
from task_factory.json_patch import JsonPatchError, apply_json_patch
from task_factory.hooks import infer_audit_checkpoint_hook
from task_factory.goal_alignment import alignment_context, compile_alignment_plan
from task_factory.operators.base import manifest_metadata
from task_factory.prepare import admit_valid_counterfactuals, prepare_recursive_parent
from task_factory.search import candidate_metadata, generate_candidates, select_candidates
from task_factory.state_schema import complete_initial_state_schema
from task_factory.wikihow_seed import source_sha256, validate_wikihow_seed


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def template(name: str, values: dict[str, str]) -> str:
    source = (PROJECT_ROOT / "prompts" / name).read_text(encoding="utf-8")
    return render_template(source, values)


def _file_sha256(path_value: str) -> str:
    if not path_value:
        return ""
    path = Path(path_value).expanduser()
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_cache_key(
    prompt: str,
    *,
    provider: str,
    reasoning_effort: str | None,
    temperature: float,
    max_tokens: int,
    response_schema: dict[str, Any] | None = None,
) -> str:
    """Hash the exact request and non-secret provider configuration."""
    provider_environment = {
        "codex": {
            "model": os.environ.get("GEM_CODEX_MODEL", ""),
            "provider": os.environ.get("GEM_CODEX_PROVIDER", ""),
            "reasoning_effort": os.environ.get("GEM_CODEX_REASONING_EFFORT", ""),
            "config_sha256": _file_sha256(os.environ.get("GEM_CODEX_CONFIG", "")),
        },
        "responses": {
            "base_url": os.environ.get("GEM_RESPONSES_BASE_URL", ""),
            "model": os.environ.get("GEM_RESPONSES_MODEL", ""),
            "reasoning_effort": os.environ.get(
                "GEM_RESPONSES_REASONING_EFFORT", ""
            ),
            "service_tier": os.environ.get("GEM_RESPONSES_SERVICE_TIER", ""),
        },
        "openai": {
            "base_url": os.environ.get("GEM_LLM_BASE_URL", ""),
            "model": os.environ.get("GEM_LLM_MODEL", ""),
        },
        "gemini": {
            "base_url": os.environ.get("GEMINI_BASE_URL", ""),
            "model": os.environ.get("GEMINI_MODEL", ""),
            "thinking_budget": os.environ.get("GEMINI_THINKING_BUDGET", ""),
        },
    }.get(provider, {})
    payload = compact(
        {
            "cache_version": "model-json-v2",
            "prompt": prompt,
            "provider": provider,
            "provider_environment": provider_environment,
            "reasoning_effort": reasoning_effort,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_schema": response_schema,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_model_cache(cache_dir: Path, key: str) -> dict[str, Any] | None:
    path = cache_dir / key[:2] / f"{key}.json"
    if not path.is_file():
        return None
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(entry, dict) or entry.get("cache_version") != "model-json-v2":
        return None
    value = entry.get("value")
    return value if isinstance(value, dict) else None


def _write_model_cache(cache_dir: Path, key: str, value: dict[str, Any]) -> None:
    directory = cache_dir / key[:2]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.json"
    temporary = directory / f".{key}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    temporary.write_text(
        json.dumps(
            {"cache_version": "model-json-v2", "value": value},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def evict_audit_cache_entries(
    audit: dict[str, Any], cache_dir: Path | None
) -> int:
    """Evict rejected bundle generations while retaining valid seed/contract work."""
    if cache_dir is None:
        return 0
    removed = 0
    keys = {
        usage.get("cache_key")
        for stage in audit.get("stages", [])
        if isinstance(stage, dict)
        and str(stage.get("stage", "")).startswith("bundle")
        for usage in [stage.get("usage")]
        if isinstance(usage, dict) and isinstance(usage.get("cache_key"), str)
    }
    for key in keys:
        path = cache_dir / key[:2] / f"{key}.json"
        try:
            path.unlink()
            removed += 1
        except FileNotFoundError:
            pass
    return removed


def evict_usage_cache_entry(usage: dict[str, Any], cache_dir: Path | None) -> bool:
    if cache_dir is None or not isinstance(usage.get("cache_key"), str):
        return False
    key = usage["cache_key"]
    try:
        (cache_dir / key[:2] / f"{key}.json").unlink()
        return True
    except FileNotFoundError:
        return False


class ModelJsonError(RuntimeError):
    """Model request or JSON decoding failure with billable-attempt metadata."""

    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def call_json(
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    provider: str = "codex",
    request_retries: int = 1,
    retry_backoff: float = 2.0,
    reasoning_effort: str | None = None,
    cache_dir: Path | None = None,
    response_schema: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    cache_key = model_cache_key(
        prompt,
        provider=provider,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        max_tokens=max_tokens,
        response_schema=response_schema,
    )
    if cache_dir is not None:
        cached = _read_model_cache(cache_dir, cache_key)
        if cached is not None:
            return cached, {
                "provider": provider,
                "cache_hit": True,
                "cache_key": cache_key,
                "latency_sec": round(time.monotonic() - started, 3),
                "prompt_chars": len(prompt),
                "response_chars": len(compact(cached)),
                "physical_prompt_chars": 0,
                "physical_response_chars": 0,
                "request_count": 0,
                "transport_attempts": 0,
                "attempts": [],
            }
    last_error: Exception | None = None
    attempts: list[dict[str, Any]] = []
    current_max_tokens = max_tokens
    for attempt in range(request_retries + 1):
        attempt_started = time.monotonic()
        try:
            raw, usage = call_chat(
                [{"role": "user", "content": prompt}],
                max_tokens=current_max_tokens,
                temperature=temperature,
                provider=provider,
                reasoning_effort=reasoning_effort,
                response_schema=response_schema if provider == "codex" else None,
            )
        except RuntimeError as exc:
            last_error = exc
            attempts.append(
                {
                    "latency_sec": round(time.monotonic() - attempt_started, 3),
                    "response_chars": 0,
                    "max_tokens_requested": current_max_tokens,
                    "error": str(exc),
                }
            )
            if attempt >= request_retries:
                profile = {
                    "latency_sec": round(time.monotonic() - started, 3),
                    "prompt_chars": len(prompt),
                    "response_chars": sum(
                        int(item.get("response_chars", 0)) for item in attempts
                    ),
                    "physical_prompt_chars": len(prompt) * len(attempts),
                    "physical_response_chars": sum(
                        int(item.get("response_chars", 0)) for item in attempts
                    ),
                    "request_count": len(attempts),
                    "transport_attempts": attempt + 1,
                    "attempts": attempts,
                }
                raise ModelJsonError(str(exc), profile) from exc
            time.sleep(retry_backoff * (attempt + 1))
            continue

        attempt_record = {
            **usage,
            "latency_sec": round(time.monotonic() - attempt_started, 3),
            "response_chars": len(raw),
            "max_tokens_requested": current_max_tokens,
        }
        attempts.append(attempt_record)
        try:
            value = parse_json_object(raw)
            if not isinstance(value, dict):
                raise ValueError("model response must be a JSON object")
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            attempt_record["error"] = str(exc)
            if attempt >= request_retries:
                profile = {
                    "latency_sec": round(time.monotonic() - started, 3),
                    "prompt_chars": len(prompt),
                    "response_chars": sum(
                        int(item.get("response_chars", 0)) for item in attempts
                    ),
                    "physical_prompt_chars": len(prompt) * len(attempts),
                    "physical_response_chars": sum(
                        int(item.get("response_chars", 0)) for item in attempts
                    ),
                    "request_count": len(attempts),
                    "transport_attempts": attempt + 1,
                    "attempts": attempts,
                }
                raise ModelJsonError(str(exc), profile) from exc
            current_max_tokens = min(current_max_tokens * 2, 65536)
            time.sleep(retry_backoff * (attempt + 1))
            continue
        profiled_usage = {
            **usage,
            "latency_sec": round(time.monotonic() - started, 3),
            "prompt_chars": len(prompt),
            "response_chars": sum(
                int(item.get("response_chars", 0)) for item in attempts
            ),
            "physical_prompt_chars": len(prompt) * len(attempts),
            "physical_response_chars": sum(
                int(item.get("response_chars", 0)) for item in attempts
            ),
            "request_count": len(attempts),
            "transport_attempts": attempt + 1,
            "attempts": attempts,
        }
        if cache_dir is not None:
            _write_model_cache(cache_dir, cache_key, value)
            profiled_usage["cache_key"] = cache_key
        return value, profiled_usage
    raise ModelJsonError(
        str(last_error or "model request failed"),
        {
            "latency_sec": round(time.monotonic() - started, 3),
            "prompt_chars": len(prompt),
            "physical_prompt_chars": len(prompt) * len(attempts),
            "physical_response_chars": sum(
                int(item.get("response_chars", 0)) for item in attempts
            ),
            "request_count": len(attempts),
            "attempts": attempts,
        },
    )


OPERATOR_HINTS = {
    "discovery_and_evidence": ("find", "select", "search", "choose"),
    "configuration_consistency": ("setting", "link", "format", "configure", "revision"),
    "permission_authorization": ("login", "password", "access", "share", "permission"),
    "asynchronous_lifecycle": ("scan", "send", "upload", "export", "wait"),
    "failure_diagnosis_recovery": ("error", "problem", "retry", "alternative"),
    "resource_budget_constraints": ("multiple", "pages", "size", "limit"),
    "artifact_provenance": ("file", "document", "track", "csv", "attachment"),
    "rollback_idempotency": ("save", "replace", "again", "logout"),
    "multi_resource_coordination": ("forward", "recipient", "class", "assignment"),
    "alternative_plan_affordance": ("alternative", "option", "if", "or"),
    "closed_loop_feedback": (
        "check",
        "monitor",
        "test",
        "cpu",
        "memory",
        "power",
        "diagnostic",
    ),
    "temporal_revision_provenance": (
        "file",
        "version",
        "backup",
        "restore",
        "download",
        "attachment",
    ),
}

STRICT_ADAPTIVE_OPERATORS = frozenset(
    {
        "failure_diagnosis_recovery",
        "alternative_recovery_affordance",
        "closed_loop_feedback",
        "temporal_revision_provenance",
    }
)


def assign_operators(
    rows: list[dict[str, Any]], catalog: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    assignments = []
    for row in rows:
        text = str(row.get("text", "")).lower()
        ranked = []
        for card in catalog:
            operator_id = card["id"]
            family = card["family"]
            lexical = sum(text.count(hint) for hint in OPERATOR_HINTS.get(operator_id, ()))
            unused_operator = int(counts.get(operator_id, 0) == 0)
            unused_family = int(family_counts.get(family, 0) == 0)
            score = lexical * 5 - counts.get(operator_id, 0) * 4 - family_counts.get(family, 0) * 2
            ranked.append((unused_operator, unused_family, score, operator_id, card))
        _unused_operator, _unused_family, _score, operator_id, card = max(
            ranked, key=lambda item: (item[0], item[1], item[2], item[3])
        )
        counts[operator_id] = counts.get(operator_id, 0) + 1
        family = card["family"]
        family_counts[family] = family_counts.get(family, 0) + 1
        assignments.append(card)
    return assignments


def memory_bundle(
    task_id: str,
    contract: dict[str, Any],
    candidate: dict[str, Any],
    metadata: dict[str, Any],
) -> TaskBundle:
    manifest = {
        "bundle_version": "task-bundle-v1",
        "task_id": task_id,
        "instruction_file": "instruction.md",
        "contract_file": "contract.json",
        "environment_file": "environment/environment.json",
        "bindings_file": "capabilities/bindings.json",
        "reference_plan_file": "solution/reference_plan.json",
        "lineage": {"generation": 0, "operators": []},
        **metadata,
    }
    return TaskBundle(
        root=Path("<memory>"),
        manifest=manifest,
        instruction=str(candidate.get("instruction", "")),
        contract=contract,
        environment=candidate.get("environment", {}),
        bindings=candidate.get("bindings", {}),
        reference_plan=candidate.get("reference_plan", {}),
    )


def quality_profile(report: dict[str, Any]) -> dict[str, float | int]:
    """Extract component-wise causal quality dimensions from a full report."""
    metrics = report.get("causal_validation", {}).get("metrics", {})
    counterfactual = report.get("counterfactual_validation", {})
    decision = counterfactual.get("decision_metrics", {})
    return {
        "steps": int(metrics.get("steps", 0)),
        "max_delayed_handle_distance": int(
            metrics.get("max_delayed_handle_distance", 0)
        ),
        "handle_chain_depth": int(metrics.get("handle_chain_depth", 0)),
        "semantic_recovery_count": len(metrics.get("semantic_recoveries", [])),
        "observation_dependent_branch_count": int(
            metrics.get("observation_dependent_branch_count", 0)
        ),
        "counterfactual_count": int(counterfactual.get("counterfactual_count", 0)),
        "decision_entropy_bits": float(decision.get("decision_entropy_bits", 0.0)),
        "necessary_action_ratio": float(
            report.get("ablation", {}).get("necessary_action_ratio", 0.0)
        ),
    }


def quality_floor_errors(
    report: dict[str, Any], quality_floor: dict[str, float | int] | None
) -> list[str]:
    if not quality_floor or report.get("phase") != "execution":
        return []
    current = quality_profile(report)
    return [
        f"quality floor regression: {name}={current.get(name, 0)} < {minimum}"
        for name, minimum in quality_floor.items()
        if name in current and float(current[name]) + 1e-9 < float(minimum)
    ]


def has_quality_floor_regression(report: dict[str, Any]) -> bool:
    return any(
        str(error).startswith("quality floor regression:")
        for error in report.get("errors", [])
    )


def evaluate(
    bundle: TaskBundle,
    min_ratio: float,
    quality_floor: dict[str, float | int] | None = None,
    *,
    require_counterfactual: bool = False,
    strict_adaptive: bool = False,
    strict_vnext: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    errors = validate_bundle(bundle)
    if errors:
        return {
            "valid": False,
            "errors": errors,
            "phase": "static",
            "profiling": {"latency_sec": round(time.monotonic() - started, 3)},
        }
    episode = run_reference_plan(bundle)
    causal = validate_episode(bundle, episode)
    rendered = render_alternate_api(bundle, seed="heldout_factory_api")
    rendered_causal = validate_episode(rendered, run_reference_plan(rendered))
    rendered_identifiability = validate_tool_identifiability(rendered)
    ablation = evaluate_action_ablation(bundle)
    counterfactual = evaluate_counterfactuals(bundle)
    goal_alignment = validate_goal_alignment(bundle, episode)
    public_executability = validate_public_executability(bundle)
    adaptive_profile = validate_adaptive_profile(
        bundle,
        episode,
        counterfactual,
        semantic_recovery_count=len(
            causal.get("metrics", {}).get("semantic_recoveries", [])
        ),
    )
    vnext_profile = validate_vnext_adaptive_profile(
        bundle,
        episode,
        causal,
        counterfactual,
        ablation=ablation,
    )
    evolution_hook: dict[str, Any] | None = None
    errors = list(causal["errors"])
    if not rendered_causal["valid"]:
        errors.append("alternate API rendering changed task validity")
    if not rendered_identifiability["valid"]:
        errors.append("alternate API rendering is not publicly identifiable")
    if ablation["necessary_action_ratio"] < min_ratio:
        errors.append("necessary action ratio is below threshold")
    if bundle.reference_plan.get("counterfactuals") and not counterfactual["valid"]:
        errors.append("counterfactual strategy adaptation failed")
    if require_counterfactual and (
        counterfactual["counterfactual_count"] < 1 or not counterfactual["valid"]
    ):
        errors.append(
            "adaptive generation requires at least one valid counterfactual "
            "policy change"
        )
    if strict_adaptive:
        if not public_executability["valid"]:
            errors.append("strict adaptive root requires a runtime-total public interface")
            errors.extend(public_executability["errors"])
        if not goal_alignment["valid"]:
            errors.append("strict adaptive root requires complete instruction-goal alignment")
            errors.extend(goal_alignment["errors"])
        if not counterfactual["valid"]:
            errors.append("strict adaptive root requires a valid counterfactual policy change")
        if not adaptive_profile["valid"]:
            errors.append(
                "strict adaptive root requires planning plus semantic recovery, "
                "closed-loop control, or temporal provenance"
            )
    if strict_vnext and not vnext_profile["valid"]:
        errors.append("strict vNext adaptive quality gate failed")
        errors.extend(vnext_profile["errors"])
    if causal["valid"]:
        try:
            evolution_hook = infer_audit_checkpoint_hook(bundle, episode)
        except ValueError as exc:
            errors.append(f"recursive evolution hook unavailable: {exc}")
    assigned_operator = bundle.manifest.get("assigned_operator")
    if assigned_operator == "alternative_plan_affordance":
        if counterfactual["counterfactual_count"] < 1 or not counterfactual["valid"]:
            errors.append(
                "assigned alternative_plan_affordance requires at least one valid "
                "state intervention where the adapted strategy succeeds and the stale strategy fails"
            )
    elif assigned_operator == "asynchronous_lifecycle":
        if causal["metrics"]["observation_dependent_branch_count"] < 1:
            errors.append(
                "assigned asynchronous_lifecycle requires a within-rollout "
                "observation-dependent branch"
            )
    elif assigned_operator == "failure_diagnosis_recovery":
        if not causal["metrics"]["semantic_recoveries"]:
            errors.append(
                "assigned failure_diagnosis_recovery requires an observed error and "
                "a later action resolving the same error code"
            )
    result = {
        "valid": not errors,
        "errors": errors,
        "phase": "execution",
        "episode": episode,
        "causal_validation": causal,
        "rendered_validation": rendered_causal,
        "rendered_identifiability": rendered_identifiability,
        "ablation": ablation,
        "counterfactual_validation": counterfactual,
        "goal_alignment": goal_alignment,
        "public_executability": public_executability,
        "adaptive_profile": adaptive_profile,
        "vnext_adaptive_profile": vnext_profile,
        "require_counterfactual": require_counterfactual,
        "strict_adaptive": strict_adaptive,
        "strict_vnext": strict_vnext,
        "evolution_hook": evolution_hook,
        "profiling": {"latency_sec": round(time.monotonic() - started, 3)},
    }
    floor_errors = quality_floor_errors(result, quality_floor)
    if floor_errors:
        result["errors"].extend(floor_errors)
        result["valid"] = False
        result["quality_floor"] = copy.deepcopy(quality_floor)
        result["quality_profile"] = quality_profile(result)
    return result


def compile_deterministic_candidate_repairs(
    *,
    task_id: str,
    contract: dict[str, Any],
    candidate: dict[str, Any],
    report: dict[str, Any],
    metadata: dict[str, Any],
    assigned_operator: str,
    require_counterfactual: bool,
    evaluate_candidate: Any,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Apply composable local repairs while monotonically shrinking errors."""
    current = candidate
    current_report = report
    audits: list[dict[str, Any]] = []

    def adopt(
        stage: str,
        proposed: dict[str, Any] | None,
        details: dict[str, Any],
        pass_index: int,
    ) -> bool:
        nonlocal current, current_report
        if proposed is None:
            return False
        proposed_report = evaluate_candidate(proposed)
        before = set(map(str, current_report.get("errors", [])))
        after = set(map(str, proposed_report.get("errors", [])))
        improved = proposed_report.get("valid", False) or after < before
        audits.append(
            {
                "stage": stage,
                "pass": pass_index,
                "accepted": bool(proposed_report.get("valid", False)),
                "improved": improved,
                "errors": proposed_report.get("errors", []),
                **details,
            }
        )
        if improved:
            current = proposed
            current_report = proposed_report
        return improved

    for pass_index in range(1, 4):
        changed = False
        evidence_candidate, evidence_details = repair_final_goal_evidence(
            contract, current, current_report
        )
        changed |= adopt(
            "deterministic_final_evidence_compile",
            evidence_candidate,
            evidence_details,
            pass_index,
        )
        if current_report.get("valid"):
            break

        ordinal_candidate, ordinal_details = repair_immediate_ordinal_provenance(
            current, current_report
        )
        changed |= adopt(
            "deterministic_ordinal_provenance_compile",
            ordinal_candidate,
            ordinal_details,
            pass_index,
        )
        if current_report.get("valid"):
            break

        counterfactuals = current.get("reference_plan", {}).get(
            "counterfactuals", []
        )
        counterfactual_evaluation = current_report.get(
            "counterfactual_validation"
        )
        if (
            isinstance(counterfactuals, list)
            and counterfactuals
            and isinstance(counterfactual_evaluation, dict)
            and not counterfactual_evaluation.get("valid", False)
        ):
            admitted_bundle, admission = admit_valid_counterfactuals(
                memory_bundle(task_id, contract, current, metadata),
                evaluation=counterfactual_evaluation,
            )
            optional = (
                assigned_operator != "alternative_plan_affordance"
                and not require_counterfactual
            )
            if (
                admission["accepted_count"] < admission["input_count"]
                and (admission["accepted_count"] > 0 or optional)
            ):
                admitted_candidate = {
                    "instruction": admitted_bundle.instruction,
                    "environment": admitted_bundle.environment,
                    "bindings": admitted_bundle.bindings,
                    "reference_plan": admitted_bundle.reference_plan,
                }
                changed |= adopt(
                    "deterministic_counterfactual_compile",
                    admitted_candidate,
                    {
                        "input_count": admission["input_count"],
                        "accepted_count": admission["accepted_count"],
                        "rejected_count": admission["rejected_count"],
                        "variants": admission["variants"],
                    },
                    pass_index,
                )
        if current_report.get("valid") or not changed:
            break
    return current, current_report, audits


def feedback(report: dict[str, Any]) -> list[str]:
    result = list(report.get("errors", []))
    causal = report.get("causal_validation", {})
    if isinstance(causal, dict) and isinstance(causal.get("metrics"), dict):
        metrics = causal["metrics"]
        result.append(
            "causal_diagnostic="
            + compact(
                {
                    key: metrics.get(key)
                    for key in (
                        "steps",
                        "max_delayed_handle_distance",
                        "handle_chain_depth",
                        "goal_evidence_coverage",
                        "missing_provenance",
                        "unexplained_arguments",
                        "overconcentrated_argument_sources",
                        "invariant_violations",
                        "semantic_recoveries",
                        "observation_dependent_branch_count",
                    )
                    if key in metrics
                }
            )
        )
    counterfactual = report.get("counterfactual_validation")
    if isinstance(counterfactual, dict):
        result.append(
            "counterfactual_diagnostic="
            + compact(
                {
                    "valid": counterfactual.get("valid"),
                    "counterfactual_count": counterfactual.get(
                        "counterfactual_count"
                    ),
                    "variants": [
                        {
                            key: item.get(key)
                            for key in (
                                "id",
                                "valid",
                                "strategy_changed",
                                "adapted_valid",
                                "stale_strategy_valid",
                                "adapted_errors",
                                "stale_strategy_errors",
                                "decision_grounding",
                            )
                            if key in item
                        }
                        for item in counterfactual.get("variants", [])
                        if isinstance(item, dict) and not item.get("valid")
                    ],
                }
            )
        )
    episode = report.get("episode")
    if isinstance(episode, dict):
        trace = episode.get("trace", [])
        result.append(
            "episode_diagnostic="
            + compact(
                {
                    "status": episode.get("status"),
                    "runtime_errors": episode.get("errors", []),
                    "executed_steps": len(trace),
                    "last_trace_step": trace[-1] if trace else None,
                    "goal_results": episode.get("goal_results", []),
                }
            )
        )
    return result


def contract_candidate_context(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return only the executable evidence needed to repair a contract."""
    environment = candidate.get("environment", {})
    bindings = candidate.get("bindings", {})
    reference_plan = candidate.get("reference_plan", {})
    capabilities = environment.get("capabilities", {})
    return {
        "instruction": candidate.get("instruction", ""),
        "initial_state": environment.get("initial_state", {}),
        "capability_topology": {
            capability_id: [
                {
                    "id": branch.get("id"),
                    "error_code": branch.get("response", {}).get("error_code"),
                    "resolves_errors": branch.get("resolves_errors", []),
                }
                for branch in definition.get("branches", [])
                if isinstance(branch, dict)
            ]
            for capability_id, definition in capabilities.items()
            if isinstance(definition, dict)
        },
        "public_tools": [
            {
                "name": tool.get("name"),
                "description": tool.get("description"),
                "capability_id": tool.get("capability_id"),
            }
            for tool in bindings.get("tools", [])
            if isinstance(tool, dict)
        ],
        "reference_action_tools": [
            action.get("tool")
            for action in reference_plan.get("actions", [])
            if isinstance(action, dict)
        ],
        "counterfactuals": [
            {
                "name": variant.get("name") or variant.get("id"),
                "state_overrides": variant.get("state_overrides", {}),
                "action_tools": [
                    action.get("tool")
                    for action in variant.get("actions", [])
                    if isinstance(action, dict)
                ],
            }
            for variant in reference_plan.get("counterfactuals", [])
            if isinstance(variant, dict)
        ],
    }


def downstream_seed_context(seed: dict[str, Any]) -> dict[str, Any]:
    """Project a validated seed to semantics needed by later model stages."""
    facts = [
        item.get("fact")
        for item in seed.get("source_supported_facts", [])
        if isinstance(item, dict) and isinstance(item.get("fact"), str)
    ]
    steps = [
        {
            key: copy.deepcopy(item.get(key))
            for key in ("id", "action", "inputs", "outputs")
            if key in item
        }
        for item in seed.get("normalized_steps", [])
        if isinstance(item, dict)
    ]
    return {
        key: copy.deepcopy(seed.get(key))
        for key in (
            "seed_version",
            "source_id",
            "source_sha256",
            "title",
            "objective",
            "observable_affordances",
            "uncertainties",
            "environment_design_limits",
            "operator_feasibility",
            "synthetic_extension",
        )
        if key in seed
    } | {
        "source_supported_facts": facts,
        "normalized_steps": steps,
        "grounding_note": (
            "Verbatim evidence spans were validated against the source and remain "
            "stored in seed.json; this downstream view omits their duplicate text."
        ),
    }


def _json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def provenance_patch_context(
    candidate: dict[str, Any], report: dict[str, Any]
) -> dict[str, Any] | None:
    """Slice the parent around unexplained arguments while preserving edit paths."""
    causal = report.get("causal_validation", {})
    metrics = causal.get("metrics", {}) if isinstance(causal, dict) else {}
    unexplained = metrics.get("unexplained_arguments", [])
    if not isinstance(unexplained, list) or not unexplained:
        return None
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("step"), int)
        or not isinstance(item.get("tool"), str)
        for item in unexplained
    ):
        return None

    environment = candidate.get("environment", {})
    capabilities = environment.get("capabilities", {})
    bindings = candidate.get("bindings", {}).get("tools", [])
    reference_plan = candidate.get("reference_plan", {})
    actions = reference_plan.get("actions", [])
    trace = report.get("episode", {}).get("trace", [])
    if not isinstance(capabilities, dict) or not isinstance(bindings, list):
        return None
    if not isinstance(actions, list) or not isinstance(trace, list):
        return None

    relevant_steps: set[int] = set()
    relevant_tools: set[str] = set()
    for item in unexplained:
        step = int(item["step"])
        relevant_steps.update(value for value in (step - 1, step) if value >= 1)
        relevant_tools.add(str(item["tool"]))
    for step in relevant_steps:
        if step <= len(trace) and isinstance(trace[step - 1], dict):
            tool = trace[step - 1].get("public_tool")
            if isinstance(tool, str):
                relevant_tools.add(tool)
        if step <= len(actions) and isinstance(actions[step - 1], dict):
            tool = actions[step - 1].get("tool")
            if isinstance(tool, str):
                relevant_tools.add(tool)

    binding_fragments = []
    capability_ids: set[str] = set()
    for index, binding in enumerate(bindings):
        if not isinstance(binding, dict) or binding.get("name") not in relevant_tools:
            continue
        capability_id = binding.get("capability_id")
        if isinstance(capability_id, str):
            capability_ids.add(capability_id)
        binding_fragments.append(
            {"path": f"/bindings/tools/{index}", "value": binding}
        )
    branch_fragments = []
    for step in sorted(relevant_steps):
        if step > len(trace) or not isinstance(trace[step - 1], dict):
            continue
        trace_step = trace[step - 1]
        capability_id = trace_step.get("capability_id")
        selected_branch = trace_step.get("selected_branch")
        capability = capabilities.get(capability_id)
        if not isinstance(capability_id, str) or not isinstance(capability, dict):
            continue
        branches = capability.get("branches", [])
        if not isinstance(branches, list):
            continue
        branch_index = next(
            (
                index
                for index, branch in enumerate(branches)
                if isinstance(branch, dict) and branch.get("id") == selected_branch
            ),
            None,
        )
        if branch_index is None:
            continue
        branch_fragments.append(
            {
                "path": (
                    "/environment/capabilities/"
                    + _json_pointer_token(capability_id)
                    + f"/branches/{branch_index}"
                ),
                "value": branches[branch_index],
            }
        )
    unique_branch_fragments = {
        fragment["path"]: fragment for fragment in branch_fragments
    }
    branch_fragments = [
        unique_branch_fragments[path] for path in sorted(unique_branch_fragments)
    ]
    if not branch_fragments:
        return None

    action_fragments = [
        {"path": f"/reference_plan/actions/{step - 1}", "value": actions[step - 1]}
        for step in sorted(relevant_steps)
        if step <= len(actions) and isinstance(actions[step - 1], dict)
    ]
    counterfactual_fragments = []
    for variant_index, variant in enumerate(reference_plan.get("counterfactuals", [])):
        if not isinstance(variant, dict):
            continue
        variant_actions = variant.get("actions", [])
        if not isinstance(variant_actions, list):
            continue
        matches = [
            {
                "path": (
                    f"/reference_plan/counterfactuals/{variant_index}/actions/{index}"
                ),
                "value": action,
            }
            for index, action in enumerate(variant_actions)
            if isinstance(action, dict) and action.get("tool") in relevant_tools
        ]
        if matches:
            counterfactual_fragments.append(
                {
                    "variant_index": variant_index,
                    "name": variant.get("name") or variant.get("id"),
                    "state_overrides": variant.get("state_overrides", {}),
                    "matching_actions": matches,
                }
            )

    return {
        "context_mode": "provenance_local_slice",
        "instruction": candidate.get("instruction", ""),
        "unexplained_arguments": unexplained,
        "initial_state": environment.get("initial_state", {}),
        "selected_branch_fragments": branch_fragments,
        "binding_fragments": binding_fragments,
        "reference_action_fragments": action_fragments,
        "counterfactual_fragments": counterfactual_fragments,
        "trace_window": [
            {
                key: trace[step - 1].get(key)
                for key in (
                    "step",
                    "public_tool",
                    "capability_id",
                    "arguments",
                    "selected_branch",
                    "response",
                    "error_code",
                )
            }
            for step in sorted(relevant_steps)
            if step <= len(trace) and isinstance(trace[step - 1], dict)
        ],
    }


def patch_strictly_improves(
    previous_errors: set[str], patched_report: dict[str, Any]
) -> bool:
    """Accept a partial repair only when it introduces no new error class."""
    if patched_report.get("valid"):
        return True
    patched_errors = set(patched_report.get("errors", []))
    return bool(patched_errors and patched_errors < previous_errors)


def repair_quality_key(report: dict[str, Any]) -> tuple[int, ...]:
    """Rank candidates without allowing later repairs to degrade the best one."""
    if report.get("valid"):
        return (3, 0, 0, 0, 0, 0)
    phase = report.get("phase")
    if phase == "static":
        return (0, -len(report.get("errors", [])), 0, 0, 0, 0)
    causal = report.get("causal_validation", {}).get("metrics", {})
    ablation = report.get("ablation", {})
    counterfactual = report.get("counterfactual_validation", {})
    valid_counterfactuals = sum(
        bool(variant.get("valid"))
        for variant in counterfactual.get("variants", [])
        if isinstance(variant, dict)
    )
    return (
        2 if phase == "execution" else 1,
        valid_counterfactuals,
        -len(report.get("errors", [])),
        int(round(float(causal.get("goal_evidence_coverage", 0.0)) * 1000)),
        -len(causal.get("unexplained_arguments", [])),
        int(round(float(ablation.get("necessary_action_ratio", 0.0)) * 1000)),
        int(bool(counterfactual.get("valid"))),
    )


def select_best_bundle_sample(
    samples: list[tuple[int, dict[str, Any], dict[str, Any], list[str], dict[str, Any]]],
) -> tuple[int, dict[str, Any], dict[str, Any], list[str], dict[str, Any]]:
    if not samples:
        raise ValueError("cannot select from an empty bundle sample list")
    return max(samples, key=lambda item: (repair_quality_key(item[2]), -item[0]))


def contract_error_ownership(
    contract: dict[str, Any], candidate: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Separate malformed contracts from candidate/contract state mismatches."""
    structural = validate_contract(contract)
    initial_state = candidate.get("environment", {}).get("initial_state")
    compatibility = validate_contract(
        contract, initial_state if isinstance(initial_state, dict) else None
    )
    structural_set = set(structural)
    return structural, [error for error in compatibility if error not in structural_set]


def progress(task_id: str, stage: str, **values: Any) -> None:
    print(json.dumps({"task_id": task_id, "stage": stage, **values}, ensure_ascii=False), flush=True)


def next_repair_number(path: Path) -> int:
    numbers = []
    for candidate in path.glob("candidate.repair_*.json"):
        try:
            numbers.append(int(candidate.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(numbers, default=0) + 1


def configure_responses_provider(config_path: Path) -> None:
    """Load a Responses-compatible provider without persisting its credential."""
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    project_matches: list[tuple[int, dict[str, Any]]] = []
    for raw_path, settings in config.get("projects", {}).items():
        if not isinstance(raw_path, str) or not isinstance(settings, dict):
            continue
        try:
            PROJECT_ROOT.resolve().relative_to(Path(raw_path).expanduser().resolve())
        except ValueError:
            continue
        project_matches.append((len(Path(raw_path).parts), settings))
    project = max(project_matches, key=lambda item: item[0])[1] if project_matches else {}
    provider_name = str(
        project.get("model_provider") or config.get("model_provider") or ""
    )
    provider = config.get("model_providers", {}).get(provider_name, {})
    if not isinstance(provider, dict) or provider.get("wire_api") != "responses":
        raise ValueError(
            f"configured provider {provider_name!r} is not a Responses API provider"
        )
    env_key = provider.get("env_key")
    embedded_key = config.get(env_key) if isinstance(env_key, str) else None
    api_key = (
        os.environ.get("GEM_RESPONSES_API_KEY", "")
        or (os.environ.get(env_key, "") if isinstance(env_key, str) else "")
        or (embedded_key if isinstance(embedded_key, str) else "")
    )
    if not api_key:
        raise ValueError("Responses provider credential is unavailable")
    base_url = str(provider.get("base_url") or "").rstrip("/")
    model = str(project.get("model") or config.get("model") or "")
    if not base_url or not model:
        raise ValueError("Responses provider requires base_url and model")
    header_names = list((provider.get("http_headers") or {}).keys())
    api_key_header = next(
        (name for name in header_names if name.casefold() in {"api-key", "authorization"}),
        "Authorization",
    )
    os.environ["GEM_RESPONSES_BASE_URL"] = base_url
    os.environ["GEM_RESPONSES_API_KEY"] = api_key
    os.environ["GEM_RESPONSES_API_KEY_HEADER"] = api_key_header
    os.environ["GEM_RESPONSES_MODEL"] = model
    effort = str(
        project.get("model_reasoning_effort")
        or config.get("model_reasoning_effort")
        or ""
    )
    if effort:
        os.environ["GEM_RESPONSES_REASONING_EFFORT"] = effort
    tier = str(project.get("service_tier") or config.get("service_tier") or "")
    if tier:
        os.environ["GEM_RESPONSES_SERVICE_TIER"] = {
            "fast": "priority",
        }.get(tier, tier)


def refresh_materialized_root(path: Path, desired: TaskBundle) -> TaskBundle:
    """Refresh audit metadata only when an existing root has identical task content."""
    existing = load_task_bundle(path)
    content_fields = ("instruction", "contract", "environment", "bindings", "reference_plan")
    changed = []
    for field in content_fields:
        existing_value = getattr(existing, field)
        desired_value = getattr(desired, field)
        if field == "instruction":
            existing_value = existing_value.rstrip()
            desired_value = desired_value.rstrip()
        elif field == "contract":
            existing_value = normalize_contract(existing_value)
            desired_value = normalize_contract(desired_value)
        if existing_value != desired_value:
            changed.append(field)
    if changed:
        raise ValueError(
            "refusing to refresh an existing root with different task content: "
            + ", ".join(changed)
        )
    manifest = {
        **existing.manifest,
        **manifest_metadata(desired),
        "lineage": copy.deepcopy(desired.manifest.get("lineage", {})),
    }
    write_json(path / "manifest.json", manifest)
    write_json(path / "contract.json", normalize_contract(desired.contract))
    return TaskBundle(
        root=path,
        manifest=manifest,
        instruction=desired.instruction,
        contract=normalize_contract(desired.contract),
        environment=desired.environment,
        bindings=desired.bindings,
        reference_plan=desired.reference_plan,
    )


def resume_operator_card(
    row: dict[str, Any],
    assigned: dict[str, Any],
    catalog: list[dict[str, Any]],
    *,
    output_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    """Keep the original operator stable when resuming a partially built task."""
    if not resume:
        return assigned
    seed_path = output_dir / "tasks" / str(row["id"]) / "seed.json"
    if not seed_path.is_file():
        return assigned
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    operator_id = seed.get("synthetic_extension", {}).get("operator")
    for card in catalog:
        if card["id"] == operator_id:
            return card
    raise ValueError(f"resume seed references unknown operator {operator_id!r}")


def generate_one(
    row: dict[str, Any],
    card: dict[str, Any],
    *,
    output_dir: Path,
    repair_rounds: int,
    min_ratio: float,
    temperature: float,
    provider: str,
    reasoning_effort: str | None,
    seed_reasoning_effort: str | None,
    contract_reasoning_effort: str | None,
    bundle_reasoning_effort: str | None,
    patch_reasoning_effort: str | None,
    bundle_candidates: int,
    request_retries: int,
    retry_backoff: float,
    model_cache_dir: Path | None,
    always_repair_contract: bool,
    patch_repair: bool,
    resume: bool,
    regenerate_seed: bool,
    quality_floor: dict[str, float | int] | None,
    repair_quality_regressions: bool,
    quality_resample_candidates: int,
    quality_resample_reasoning_effort: str | None,
    strict_adaptive: bool,
    strict_vnext: bool,
) -> tuple[TaskBundle | None, dict[str, Any]]:
    task_started = time.monotonic()
    task_id = str(row["id"])
    task_dir = output_dir / "tasks" / task_id
    source_text = str(row.get("text", ""))
    audit: dict[str, Any] = {
        "task_id": task_id,
        "assigned_operator": card["id"],
        "operator_family": card["family"],
        "stages": [],
    }

    def model_json(
        prompt: str,
        *,
        stage: str,
        attempt: int,
        max_tokens: int,
        temperature: float,
        effort: str | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if provider == "responses":
            max_tokens = max_tokens * 2
        try:
            return call_json(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                provider=provider,
                reasoning_effort=effort or reasoning_effort,
                request_retries=request_retries,
                retry_backoff=retry_backoff,
                cache_dir=model_cache_dir,
                response_schema=response_schema,
            )
        except ModelJsonError as exc:
            audit["stages"].append(
                {
                    "stage": stage,
                    "attempt": attempt,
                    "errors": [str(exc)],
                    "usage": exc.usage,
                    "request_failed": True,
                }
            )
            save_audit()
            raise

    def save_audit() -> None:
        usages = [
            stage["usage"]
            for stage in audit["stages"]
            if isinstance(stage.get("usage"), dict)
        ]
        audit["profiling"] = {
            "elapsed_sec": round(time.monotonic() - task_started, 3),
            "llm_calls": sum(
                int(usage.get("request_count", 1)) for usage in usages
            ),
            "llm_stage_calls": len(usages),
            "llm_latency_sec": round(
                sum(float(usage.get("latency_sec", 0.0)) for usage in usages),
                3,
            ),
            "cache_hits": sum(bool(usage.get("cache_hit")) for usage in usages),
        }
        if audit.get("final_errors"):
            audit["profiling"]["evicted_cache_entries"] = evict_audit_cache_entries(
                audit, model_cache_dir
            )
        write_json(task_dir / "audit.json", audit)

    def evaluate_candidate(value: dict[str, Any]) -> dict[str, Any]:
        return evaluate(
            memory_bundle(task_id, contract, value, metadata),
            min_ratio,
            quality_floor,
            require_counterfactual=strict_adaptive,
            strict_vnext=strict_vnext,
        )

    progress(task_id, "start", operator=card["id"], family=card["family"])
    seed_prompt = template(
        "wikihow_seed_compile.txt",
        {
            "source_id": task_id,
            "source_sha256": source_sha256(source_text),
            "source_text": source_text,
            "assigned_operator": card["id"],
            "assigned_operator_card_json": compact(card),
        },
    )
    seed_path = task_dir / "seed.json"
    seed: dict[str, Any] | None = (
        json.loads(seed_path.read_text(encoding="utf-8"))
        if resume and not regenerate_seed and seed_path.is_file()
        else None
    )
    seed_errors: list[str] = []
    if seed is not None:
        seed_errors = validate_wikihow_seed(
            seed,
            source_text,
            assigned_operator=card["id"],
            source_id=task_id,
        )
        progress(task_id, "seed_resume", valid=not seed_errors)
        if seed_errors:
            audit["final_errors"] = [
                "resume seed does not match the current source; use --regenerate-seed "
                "only when replacing the task source intentionally",
                *seed_errors,
            ]
            save_audit()
            return None, audit
    for attempt in range(repair_rounds + 1 if seed is None or seed_errors else 0):
        prompt = seed_prompt
        if seed_errors:
            prompt += "\n\nPrevious output failed grounding validation. Return a corrected full seed.\nErrors:\n" + compact(seed_errors)
        try:
            seed, usage = model_json(
                prompt,
                stage="seed",
                attempt=attempt + 1,
                max_tokens=6144,
                temperature=temperature,
                effort=seed_reasoning_effort,
            )
            seed_errors = validate_wikihow_seed(
                seed,
                source_text,
                assigned_operator=card["id"],
                source_id=task_id,
            )
            audit["stages"].append({"stage": "seed", "attempt": attempt + 1, "errors": seed_errors, "usage": usage})
            if not seed_errors:
                break
        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            seed_errors = [str(exc)]
            if not isinstance(exc, ModelJsonError):
                audit["stages"].append(
                    {"stage": "seed", "attempt": attempt + 1, "errors": seed_errors}
                )
    if seed is None or seed_errors:
        audit["final_errors"] = seed_errors
        save_audit()
        return None, audit
        progress(task_id, "seed", attempt=attempt + 1, valid=not seed_errors, errors=seed_errors)
    write_json(seed_path, seed)

    contract_prompt = template(
        "task_contract_generate.txt",
        {
            "seed_json": compact(downstream_seed_context(seed)),
            "operators_json": compact([card["id"]]),
        },
    )
    contract_path = task_dir / "contract.json"
    contract: dict[str, Any] | None = (
        normalize_contract(json.loads(contract_path.read_text(encoding="utf-8")))
        if resume and contract_path.is_file()
        else None
    )
    contract_errors: list[str] = []
    if contract is not None:
        contract_errors = validate_contract(contract)
        if contract.get("selected_operator") != card["id"]:
            contract_errors.append(f"selected_operator must be {card['id']!r}")
        progress(task_id, "contract_resume", valid=not contract_errors)
    for attempt in range(repair_rounds + 1 if contract is None or contract_errors else 0):
        prompt = contract_prompt
        if contract_errors:
            prompt += "\n\nPrevious contract failed validation. Return a corrected full contract.\nErrors:\n" + compact(contract_errors)
        contract_value, usage = model_json(
            prompt,
            stage="contract",
            attempt=attempt + 1,
            max_tokens=6144,
            temperature=temperature,
            effort=contract_reasoning_effort,
        )
        removed_fields = sorted(
            set(contract_value) - set(normalize_contract(contract_value))
        )
        contract = normalize_contract(contract_value)
        contract_errors = validate_contract(contract)
        if contract.get("selected_operator") != card["id"]:
            contract_errors.append(f"selected_operator must be {card['id']!r}")
        audit["stages"].append(
            {
                "stage": "contract",
                "attempt": attempt + 1,
                "errors": contract_errors,
                "removed_non_contract_fields": removed_fields,
                "usage": usage,
            }
        )
        if not contract_errors:
            break
    if contract is None or contract_errors:
        audit["final_errors"] = contract_errors
        save_audit()
        return None, audit
        progress(task_id, "contract", attempt=attempt + 1, valid=not contract_errors, errors=contract_errors)
    write_json(contract_path, contract)

    metadata = {
        "domain": "wikihow_generated",
        "seed_family": "wikihow_text_compiled_v1",
        "source_id": task_id,
        "assigned_operator": card["id"],
        "operator_family": card["family"],
    }
    bundle_base_prompt = template(
        "task_bundle_generate.txt",
        {
            "contract_json": compact(contract),
            "seed_json": compact(downstream_seed_context(seed)),
            "quality_floor_json": compact(quality_floor or {}),
        },
    )

    def request_bundle_candidate(
        sample_index: int,
        *,
        effort: str | None,
        prompt_suffix: str = "",
    ) -> tuple[int, dict[str, Any] | None, dict[str, Any], str | None]:
        sample_prompt = bundle_base_prompt
        if sample_index or prompt_suffix:
            sample_prompt += (
                "\n\nIndependent candidate sample: "
                f"{sample_index + 1}. Produce a fresh valid design."
                + prompt_suffix
            )
        max_tokens = 32768 if provider == "responses" else 16384
        try:
            value, usage = call_json(
                sample_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                provider=provider,
                reasoning_effort=effort,
                request_retries=request_retries,
                retry_backoff=retry_backoff,
                cache_dir=model_cache_dir,
            )
            return sample_index, value, usage, None
        except ModelJsonError as exc:
            return sample_index, None, exc.usage, str(exc)

    candidate_path = task_dir / "candidate.latest.json"
    if resume and candidate_path.is_file():
        resume_candidates: list[
            tuple[int, dict[str, Any], dict[str, Any], list[str], dict[str, Any]]
        ] = []
        resume_paths = [candidate_path, *sorted(task_dir.glob("candidate.sample_*.json"))]
        for resume_index, resume_path in enumerate(resume_paths):
            resume_candidate = json.loads(resume_path.read_text(encoding="utf-8"))
            resume_candidate.pop("verifier", None)
            resume_candidate, resume_completed = complete_initial_state_schema(
                contract, resume_candidate
            )
            resume_report = evaluate_candidate(resume_candidate)
            resume_candidates.append(
                (
                    resume_index,
                    resume_candidate,
                    resume_report,
                    resume_completed,
                    {"resume_path": str(resume_path)},
                )
            )
        (
            _resume_index,
            candidate,
            report,
            completed_paths,
            resume_metadata,
        ) = select_best_bundle_sample(resume_candidates)
        progress(
            task_id,
            "bundle_resume",
            selected=resume_metadata["resume_path"],
            candidates=len(resume_candidates),
        )
    else:
        def generate_bundle_sample(
            sample_index: int,
        ) -> tuple[int, dict[str, Any] | None, dict[str, Any], str | None]:
            return request_bundle_candidate(
                sample_index,
                effort=bundle_reasoning_effort or reasoning_effort,
            )

        if bundle_candidates == 1:
            sample_results = [generate_bundle_sample(0)]
        else:
            with ThreadPoolExecutor(max_workers=bundle_candidates) as executor:
                sample_results = list(
                    executor.map(generate_bundle_sample, range(bundle_candidates))
                )
        evaluated_samples: list[
            tuple[int, dict[str, Any], dict[str, Any], list[str], dict[str, Any]]
        ] = []
        for sample_index, sample, usage, error in sample_results:
            stage: dict[str, Any] = {
                "stage": "bundle",
                "attempt": sample_index + 1,
                "sample_index": sample_index,
                "usage": usage,
            }
            if error or sample is None:
                stage["errors"] = [error or "bundle sample returned no object"]
                stage["request_failed"] = True
                audit["stages"].append(stage)
                continue
            sample, sample_completed = complete_initial_state_schema(contract, sample)
            sample_report = evaluate_candidate(sample)
            stage.update(
                {
                    "errors": sample_report.get("errors", []),
                    "completed_state_paths": sample_completed,
                    "quality_key": list(repair_quality_key(sample_report)),
                }
            )
            if not sample_report["valid"]:
                stage["cache_entry_evicted"] = evict_usage_cache_entry(
                    usage, model_cache_dir
                )
            audit["stages"].append(stage)
            evaluated_samples.append(
                (sample_index, sample, sample_report, sample_completed, usage)
            )
            write_json(task_dir / f"candidate.sample_{sample_index + 1:02d}.json", sample)
        if not evaluated_samples:
            audit["final_errors"] = ["all bundle candidate requests failed"]
            save_audit()
            return None, audit
        selected = select_best_bundle_sample(evaluated_samples)
        selected_index, candidate, report, completed_paths, _usage = selected
        for stage in audit["stages"]:
            if stage.get("stage") == "bundle":
                stage["selected"] = stage.get("sample_index") == selected_index
        audit["bundle_candidate_selection"] = {
            "candidate_count": bundle_candidates,
            "selected_sample": selected_index + 1,
            "selected_valid": report["valid"],
        }
        progress(
            task_id,
            "bundle",
            candidates=bundle_candidates,
            selected=selected_index + 1,
        )
    write_json(candidate_path, candidate)
    write_json(task_dir / "evaluation.latest.json", report)
    progress(task_id, "evaluation", valid=report["valid"], errors=report["errors"])
    candidate, report, deterministic_compile_audit = (
        compile_deterministic_candidate_repairs(
            task_id=task_id,
            contract=contract,
            candidate=candidate,
            report=report,
            metadata=metadata,
            assigned_operator=card["id"],
            require_counterfactual=strict_adaptive,
            evaluate_candidate=evaluate_candidate,
        )
    )
    if deterministic_compile_audit:
        audit["stages"].extend(deterministic_compile_audit)
        write_json(candidate_path, candidate)
        write_json(task_dir / "evaluation.latest.json", report)
        progress(
            task_id,
            "deterministic_compile",
            valid=report["valid"],
            errors=report["errors"],
        )
    if has_quality_floor_regression(report) and quality_resample_candidates > 0:
        suffix = (
            " The previous independent design was below one or more causal quality "
            "floors. Redesign from the approved contract; do not copy or shorten "
            "the previous plan. Meet every floor component independently."
        )
        start_index = bundle_candidates
        if quality_resample_candidates == 1:
            resample_results = [
                request_bundle_candidate(
                    start_index,
                    effort=quality_resample_reasoning_effort or reasoning_effort,
                    prompt_suffix=suffix,
                )
            ]
        else:
            with ThreadPoolExecutor(
                max_workers=quality_resample_candidates
            ) as executor:
                resample_results = list(
                    executor.map(
                        lambda sample_index: request_bundle_candidate(
                            sample_index,
                            effort=(
                                quality_resample_reasoning_effort or reasoning_effort
                            ),
                            prompt_suffix=suffix,
                        ),
                        range(
                            start_index,
                            start_index + quality_resample_candidates,
                        ),
                    )
                )
        resampled: list[
            tuple[int, dict[str, Any], dict[str, Any], list[str], dict[str, Any]]
        ] = []
        for sample_index, sample, usage, error in resample_results:
            stage: dict[str, Any] = {
                "stage": "bundle_quality_resample",
                "attempt": sample_index + 1,
                "sample_index": sample_index,
                "usage": usage,
            }
            if error or sample is None:
                stage["errors"] = [error or "quality resample returned no object"]
                stage["request_failed"] = True
                audit["stages"].append(stage)
                continue
            sample, sample_completed = complete_initial_state_schema(contract, sample)
            sample_report = evaluate_candidate(sample)
            stage.update(
                {
                    "errors": sample_report.get("errors", []),
                    "completed_state_paths": sample_completed,
                    "quality_key": list(repair_quality_key(sample_report)),
                }
            )
            if not sample_report["valid"]:
                stage["cache_entry_evicted"] = evict_usage_cache_entry(
                    usage, model_cache_dir
                )
            audit["stages"].append(stage)
            resampled.append(
                (sample_index, sample, sample_report, sample_completed, usage)
            )
            write_json(
                task_dir / f"candidate.sample_{sample_index + 1:02d}.json", sample
            )
        if resampled:
            best_resample = select_best_bundle_sample(resampled)
            if repair_quality_key(best_resample[2]) > repair_quality_key(report):
                _index, candidate, report, _completed, _usage = best_resample
                write_json(candidate_path, candidate)
                write_json(task_dir / "evaluation.latest.json", report)
        progress(
            task_id,
            "bundle_quality_resample",
            candidates=quality_resample_candidates,
            valid=report["valid"],
            errors=report["errors"],
        )
    repair_start = next_repair_number(task_dir) if resume else 1
    for repair_offset in range(repair_rounds):
        if report["valid"]:
            break
        repair_round = repair_start + repair_offset
        errors = feedback(report)
        previous_error_set = set(report.get("errors", []))
        contract_structure_errors, contract_candidate_errors = (
            contract_error_ownership(contract, candidate)
        )
        if always_repair_contract or contract_structure_errors:
            contract_value, contract_usage = model_json(
                template(
                    "task_contract_repair.txt",
                    {
                        "validation_errors_json": compact(
                            errors
                            + contract_structure_errors
                            + contract_candidate_errors
                        ),
                        "contract_json": compact(contract),
                        "seed_json": compact(downstream_seed_context(seed)),
                        "candidate_context_json": compact(
                            contract_candidate_context(candidate)
                        ),
                    },
                ),
                stage="contract_repair",
                attempt=repair_round,
                max_tokens=6144,
                temperature=0.0,
                effort=contract_reasoning_effort,
            )
            removed_fields = sorted(
                set(contract_value) - set(normalize_contract(contract_value))
            )
            contract = normalize_contract(contract_value)
            contract_structure_errors, contract_candidate_errors = (
                contract_error_ownership(contract, candidate)
            )
            audit["stages"].append(
                {
                    "stage": "contract_repair",
                    "attempt": repair_round,
                    "errors": contract_structure_errors,
                    "candidate_compatibility_errors": contract_candidate_errors,
                    "removed_non_contract_fields": removed_fields,
                    "usage": contract_usage,
                }
            )
            write_json(task_dir / f"contract.repair_{repair_round:02d}.json", contract)
            write_json(contract_path, contract)
            report = evaluate_candidate(candidate)
            errors = feedback(report)
            previous_error_set = set(report.get("errors", []))
            progress(
                task_id,
                "contract_repair",
                attempt=repair_round,
                errors=contract_structure_errors,
            )
        else:
            audit["stages"].append(
                {
                    "stage": "contract_repair_skipped",
                    "attempt": repair_round,
                    "reason": "approved contract remains statically valid",
                    "candidate_compatibility_errors": contract_candidate_errors,
                }
            )
        repaired_with_patch = False
        deterministic_candidate, deterministic_details = repair_final_goal_evidence(
            contract, candidate, report
        )
        if deterministic_candidate is not None:
            deterministic_report = evaluate_candidate(deterministic_candidate)
            repaired_with_patch = deterministic_report["valid"]
            deterministic_improved = (
                not repaired_with_patch
                and set(deterministic_report.get("errors", []))
                < set(report.get("errors", []))
            )
            audit["stages"].append(
                {
                    "stage": "deterministic_final_evidence_repair",
                    "attempt": repair_round,
                    "accepted": repaired_with_patch,
                    "improved": deterministic_improved,
                    "errors": deterministic_report.get("errors", []),
                    **deterministic_details,
                }
            )
            if repaired_with_patch or deterministic_improved:
                candidate = deterministic_candidate
                report = deterministic_report
        if not repaired_with_patch:
            ordinal_candidate, ordinal_details = repair_immediate_ordinal_provenance(
                candidate, report
            )
            if ordinal_candidate is not None:
                ordinal_report = evaluate_candidate(ordinal_candidate)
                residual_candidate, residual_details = repair_final_goal_evidence(
                    contract, ordinal_candidate, ordinal_report
                )
                if residual_candidate is not None:
                    residual_report = evaluate_candidate(residual_candidate)
                    if residual_report["valid"]:
                        ordinal_candidate = residual_candidate
                        ordinal_report = residual_report
                        ordinal_details["final_evidence_paths"] = (
                            residual_details.get("added_goal_paths", [])
                        )
                repaired_with_patch = ordinal_report["valid"]
                audit["stages"].append(
                    {
                        "stage": "deterministic_ordinal_provenance_repair",
                        "attempt": repair_round,
                        "accepted": repaired_with_patch,
                        "errors": ordinal_report.get("errors", []),
                        **ordinal_details,
                    }
                )
                if repaired_with_patch:
                    candidate = ordinal_candidate
                    report = ordinal_report
        if not repaired_with_patch:
            source_counterfactuals = candidate.get("reference_plan", {}).get(
                "counterfactuals", []
            )
            counterfactual_evaluation = report.get("counterfactual_validation")
            if (
                isinstance(source_counterfactuals, list)
                and source_counterfactuals
                and isinstance(counterfactual_evaluation, dict)
                and not counterfactual_evaluation.get("valid", False)
            ):
                admitted_bundle, admission = admit_valid_counterfactuals(
                    memory_bundle(task_id, contract, candidate, metadata),
                    evaluation=counterfactual_evaluation,
                )
                optional_counterfactuals = (
                    card["id"] != "alternative_plan_affordance"
                    and not strict_adaptive
                )
                if (
                    admission["accepted_count"] < admission["input_count"]
                    and (
                        admission["accepted_count"] > 0
                        or optional_counterfactuals
                    )
                ):
                    admitted_candidate = {
                        "instruction": admitted_bundle.instruction,
                        "environment": admitted_bundle.environment,
                        "bindings": admitted_bundle.bindings,
                        "reference_plan": admitted_bundle.reference_plan,
                    }
                    admitted_report = evaluate_candidate(admitted_candidate)
                    residual_candidate, residual_details = repair_final_goal_evidence(
                        contract, admitted_candidate, admitted_report
                    )
                    if residual_candidate is not None:
                        residual_report = evaluate_candidate(residual_candidate)
                        if residual_report["valid"]:
                            admitted_candidate = residual_candidate
                            admitted_report = residual_report
                            admission["final_evidence_paths"] = (
                                residual_details.get("added_goal_paths", [])
                            )
                    admitted = admitted_report["valid"]
                    audit["stages"].append(
                        {
                            "stage": "deterministic_counterfactual_admission",
                            "attempt": repair_round,
                            "accepted": admitted,
                            "errors": admitted_report.get("errors", []),
                            "input_count": admission["input_count"],
                            "accepted_count": admission["accepted_count"],
                            "rejected_count": admission["rejected_count"],
                            "final_evidence_paths": admission.get(
                                "final_evidence_paths", []
                            ),
                            "variants": admission["variants"],
                        }
                    )
                    if admitted:
                        candidate = admitted_candidate
                        report = admitted_report
                        repaired_with_patch = True
        if (
            not repaired_with_patch
            and has_quality_floor_regression(report)
            and not repair_quality_regressions
        ):
            audit["stages"].append(
                {
                    "stage": "quality_floor_repair_skipped",
                    "attempt": repair_round,
                    "reason": (
                        "candidate complexity is below the accepted baseline; "
                        "resample instead of rewriting a weaker parent"
                    ),
                    "errors": [
                        error
                        for error in report.get("errors", [])
                        if str(error).startswith("quality floor regression:")
                    ],
                }
            )
            progress(
                task_id,
                "quality_floor_repair_skipped",
                attempt=repair_round,
                errors=audit["stages"][-1]["errors"],
            )
            break
        if patch_repair and not repaired_with_patch:
            local_context = provenance_patch_context(candidate, report)
            patch_contexts = []
            if local_context is not None:
                patch_contexts.append(("bundle_patch_repair_local", local_context))
            patch_contexts.append(("bundle_patch_repair", candidate))
            for patch_stage, patch_context in patch_contexts:
                try:
                    patch_value, patch_usage = model_json(
                        template(
                            "task_bundle_patch_repair.txt",
                            {
                                "validation_errors_json": compact(
                                    errors + contract_candidate_errors
                                ),
                                "contract_json": compact(contract),
                                "quality_floor_json": compact(quality_floor or {}),
                                "candidate_context_json": compact(patch_context),
                            },
                        ),
                        stage=patch_stage,
                        attempt=repair_round,
                        max_tokens=4096,
                        temperature=0.0,
                        effort=patch_reasoning_effort or bundle_reasoning_effort,
                    )
                    patched_candidate = apply_json_patch(
                        candidate, patch_value.get("operations")
                    )
                    patched_candidate, completed_paths = complete_initial_state_schema(
                        contract, patched_candidate
                    )
                    patched_report = evaluate_candidate(patched_candidate)
                    residual_candidate, residual_details = repair_final_goal_evidence(
                        contract, patched_candidate, patched_report
                    )
                    if residual_candidate is not None:
                        residual_report = evaluate_candidate(residual_candidate)
                        if residual_report["valid"]:
                            patched_candidate = residual_candidate
                            patched_report = residual_report
                            completed_paths = [
                                *completed_paths,
                                *[
                                    f"final-evidence:{path}"
                                    for path in residual_details.get(
                                        "added_goal_paths", []
                                    )
                                ],
                            ]
                    repaired_with_patch = patch_strictly_improves(
                        previous_error_set, patched_report
                    )
                    audit["stages"].append(
                        {
                            "stage": patch_stage,
                            "attempt": repair_round,
                            "accepted": repaired_with_patch,
                            "errors": patched_report.get("errors", []),
                            "operation_count": len(patch_value.get("operations", [])),
                            "completed_state_paths": completed_paths,
                            "usage": patch_usage,
                        }
                    )
                    if repaired_with_patch:
                        candidate = patched_candidate
                        report = patched_report
                        break
                    audit["stages"][-1]["cache_entry_evicted"] = (
                        evict_usage_cache_entry(patch_usage, model_cache_dir)
                    )
                except (JsonPatchError, KeyError, TypeError, ValueError) as exc:
                    audit["stages"].append(
                        {
                            "stage": patch_stage,
                            "attempt": repair_round,
                            "accepted": False,
                            "errors": [str(exc)],
                        }
                    )
        if not repaired_with_patch:
            proposed_candidate, bundle_usage = model_json(
                template(
                    "task_bundle_repair.txt",
                    {
                        "validation_errors_json": compact(
                            errors + contract_candidate_errors
                        ),
                        "contract_json": compact(contract),
                        "seed_json": compact(downstream_seed_context(seed)),
                        "quality_floor_json": compact(quality_floor or {}),
                        "candidate_json": compact(candidate),
                    },
                ),
                stage="bundle_repair",
                attempt=repair_round,
                max_tokens=16384,
                temperature=0.0,
                effort=bundle_reasoning_effort,
            )
            audit["stages"].append(
                {
                    "stage": "bundle_repair",
                    "attempt": repair_round,
                    "usage": bundle_usage,
                }
            )
            proposed_candidate, completed_paths = complete_initial_state_schema(
                contract, proposed_candidate
            )
            audit["stages"][-1]["completed_state_paths"] = completed_paths
            proposed_report = evaluate_candidate(proposed_candidate)
            repair_improved = repair_quality_key(proposed_report) > repair_quality_key(
                report
            )
            audit["stages"][-1].update(
                {
                    "accepted": repair_improved,
                    "errors": proposed_report.get("errors", []),
                    "quality_key_before": list(repair_quality_key(report)),
                    "quality_key_after": list(repair_quality_key(proposed_report)),
                }
            )
            if repair_improved:
                candidate = proposed_candidate
                report = proposed_report
            else:
                audit["stages"][-1]["cache_entry_evicted"] = (
                    evict_usage_cache_entry(bundle_usage, model_cache_dir)
                )
        write_json(task_dir / f"candidate.repair_{repair_round:02d}.json", candidate)
        write_json(candidate_path, candidate)
        write_json(task_dir / f"evaluation.repair_{repair_round:02d}.json", report)
        write_json(task_dir / "evaluation.latest.json", report)
        progress(task_id, "bundle_repair", attempt=repair_round, valid=report["valid"], errors=report["errors"])
    write_json(task_dir / "candidate.json", candidate)
    write_json(task_dir / "evaluation.json", report)
    if not report["valid"]:
        audit["final_errors"] = report["errors"]
        save_audit()
        return None, audit

    source_bundle = memory_bundle(task_id, contract, candidate, metadata)
    if strict_adaptive:
        source_bundle = totalize_public_capabilities(source_bundle)
        progress(task_id, "goal_alignment")
        alignment_prompt = template(
            "goal_alignment_plan.txt",
            {
                "alignment_context_json": compact(
                    alignment_context(source_bundle)
                )
            },
        )
        alignment_schema = json.loads(
            (PROJECT_ROOT / "schemas" / "goal_alignment_plan_v1.json").read_text(
                encoding="utf-8"
            )
        )
        alignment_plan_path = task_dir / "alignment.plan.json"
        alignment_plan: dict[str, Any] | None = None
        alignment_usage: dict[str, Any] = {}
        if resume and alignment_plan_path.is_file():
            alignment_plan = json.loads(
                alignment_plan_path.read_text(encoding="utf-8")
            )
            alignment_usage = {"artifact_resume": True, "request_count": 0}
        elif resume and model_cache_dir is not None:
            previous_audit_path = task_dir / "audit.json"
            if previous_audit_path.is_file():
                previous_audit = json.loads(
                    previous_audit_path.read_text(encoding="utf-8")
                )
                previous_keys = [
                    stage.get("usage", {}).get("cache_key")
                    for stage in previous_audit.get("stages", [])
                    if stage.get("stage") == "goal_alignment_compile"
                    and isinstance(stage.get("usage"), dict)
                ]
                for cache_key in reversed(previous_keys):
                    if not isinstance(cache_key, str):
                        continue
                    alignment_plan = _read_model_cache(model_cache_dir, cache_key)
                    if alignment_plan is not None:
                        alignment_usage = {
                            "legacy_cache_resume": True,
                            "cache_key": cache_key,
                            "request_count": 0,
                        }
                        break
        try:
            if alignment_plan is None:
                alignment_plan, alignment_usage = model_json(
                    alignment_prompt,
                    stage="goal_alignment",
                    attempt=1,
                    max_tokens=6000,
                    temperature=0.0,
                    effort=contract_reasoning_effort,
                    response_schema=alignment_schema,
                )
            write_json(alignment_plan_path, alignment_plan)
            aligned_bundle, alignment_compile = compile_alignment_plan(
                source_bundle, alignment_plan
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            alignment_usage = {}
            aligned_bundle = None
            alignment_compile = {"valid": False, "errors": [str(exc)]}
        audit["stages"].append(
            {
                "stage": "goal_alignment_compile",
                "attempt": 1,
                "accepted": aligned_bundle is not None,
                "errors": alignment_compile.get("errors", []),
                "usage": alignment_usage,
            }
        )
        if aligned_bundle is None:
            audit["final_errors"] = alignment_compile.get("errors", [])
            save_audit()
            return None, audit
        strict_report = evaluate(
            aligned_bundle,
            min_ratio,
            quality_floor,
            strict_adaptive=True,
            strict_vnext=strict_vnext,
        )
        audit["stages"].append(
            {
                "stage": "strict_adaptive_gate",
                "accepted": strict_report["valid"],
                "errors": strict_report["errors"],
                "adaptive_profile": strict_report.get("adaptive_profile", {}),
                "vnext_adaptive_profile": strict_report.get(
                    "vnext_adaptive_profile", {}
                ),
                "goal_alignment": strict_report.get("goal_alignment", {}),
                "public_executability": strict_report.get(
                    "public_executability", {}
                ),
            }
        )
        if not strict_report["valid"]:
            audit["final_errors"] = strict_report["errors"]
            save_audit()
            return None, audit
        source_bundle = aligned_bundle
        contract = source_bundle.contract
        report = strict_report
    progress(task_id, "prepare_recursive_parent")
    try:
        prepared, preparation = prepare_recursive_parent(
            source_bundle,
            counterfactual_evaluation=report["counterfactual_validation"],
            episode_report=report["episode"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        audit["final_errors"] = [str(exc)]
        save_audit()
        return None, audit
    semantic_content_unchanged = (
        prepared.instruction == source_bundle.instruction
        and prepared.contract == source_bundle.contract
        and prepared.environment == source_bundle.environment
        and prepared.bindings == source_bundle.bindings
        and prepared.reference_plan == source_bundle.reference_plan
    )
    prepared_report = (
        report
        if semantic_content_unchanged
        else evaluate(
            prepared,
            min_ratio,
            quality_floor,
            strict_adaptive=strict_adaptive,
            strict_vnext=strict_vnext,
        )
    )
    if not prepared_report["valid"]:
        audit["final_errors"] = prepared_report["errors"]
        audit["preparation"] = preparation
        save_audit()
        return None, audit
    prepared = TaskBundle(
        root=prepared.root,
        manifest={
            **prepared.manifest,
            "recursive_preparation": preparation,
            "source_grounding": {
                "source_id": task_id,
                "source_sha256": source_sha256(source_text),
                "seed_file": str(task_dir / "seed.json"),
                "synthetic_extension": seed["synthetic_extension"],
            },
        },
        instruction=prepared.instruction,
        contract=prepared.contract,
        environment=prepared.environment,
        bindings=prepared.bindings,
        reference_plan=prepared.reference_plan,
    )
    audit["preparation"] = preparation
    audit["accepted"] = True
    audit["metrics"] = prepared_report["causal_validation"]["metrics"]
    audit["necessary_action_ratio"] = prepared_report["ablation"]["necessary_action_ratio"]
    save_audit()
    progress(task_id, "accepted", metrics=audit["metrics"])
    return prepared, audit


def recursive_search(
    roots: list[TaskBundle],
    output_dir: Path,
    generations: int,
    beam_size: int,
    *,
    resume: bool = False,
    objective: str = "decision_nodes",
) -> dict[str, Any]:
    parents = roots
    fingerprints: set[str] = set()
    reports = []
    for generation in range(1, generations + 1):
        candidates, rejected = generate_candidates(
            parents,
            [
                "audit_checkpoint_v1",
                "execution_route_branch_v1",
                "async_readiness_retry_v1",
            ],
            objective=objective,
        )
        selected, selection_rejected = select_candidates(
            candidates,
            existing_fingerprints=fingerprints,
            max_per_cell=1,
            max_per_parent=2,
            max_per_operator=max(1, beam_size // 2),
        )
        selected = selected[:beam_size]
        next_parents = []
        selected_rows = []
        for item in selected:
            child = item.evaluation.product.bundle
            recursive_root = output_dir / "recursive" / "bundles"
            path = recursive_root / child.task_id
            if resume and path.is_dir():
                materialized = refresh_materialized_root(path, child)
                path = materialized.root
            else:
                path = materialize_candidate(
                    recursive_root,
                    task_id=child.task_id,
                    contract=child.contract,
                    candidate={
                        "instruction": child.instruction,
                        "environment": child.environment,
                        "bindings": child.bindings,
                        "reference_plan": child.reference_plan,
                    },
                    lineage=child.manifest["lineage"],
                    manifest_metadata=manifest_metadata(child),
                )
            metadata = candidate_metadata(item)
            metadata.update({"bundle": str(path), "generation": generation})
            write_json(output_dir / "recursive" / "audits" / child.task_id / "evaluation.json", item.evaluation.report)
            fingerprints.add(item.fingerprint)
            next_parents.append(TaskBundle(
                root=path,
                manifest=child.manifest,
                instruction=child.instruction,
                contract=child.contract,
                environment=child.environment,
                bindings=child.bindings,
                reference_plan=child.reference_plan,
            ))
            selected_rows.append(metadata)
        reports.append(
            {
                "generation": generation,
                "parents": len(parents),
                "candidates": len(candidates),
                "selected": selected_rows,
                "generation_rejections": rejected,
                "selection_rejections": selection_rejected,
            }
        )
        if not next_parents:
            break
        parents = next_parents
    return {"generations": reports, "distinct_semantic_fingerprints": len(fingerprints)}


def summarize_stage_profiles(audits: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate physical model work by semantic stage for bottleneck profiling."""
    profiles: dict[str, dict[str, Any]] = {}
    for audit in audits:
        for stage in audit.get("stages", []):
            name = str(stage.get("stage", "unknown"))
            usage = stage.get("usage")
            if not isinstance(usage, dict):
                continue
            profile = profiles.setdefault(
                name,
                {
                    "stage_calls": 0,
                    "physical_requests": 0,
                    "latency_sec": 0.0,
                    "logical_prompt_chars": 0,
                    "physical_prompt_chars": 0,
                    "physical_response_chars": 0,
                    "cache_hits": 0,
                },
            )
            request_count = int(usage.get("request_count", 0))
            prompt_chars = int(usage.get("prompt_chars", 0))
            physical_prompt_chars = int(
                usage.get(
                    "physical_prompt_chars",
                    prompt_chars if request_count else 0,
                )
            )
            physical_response_chars = int(
                usage.get(
                    "physical_response_chars",
                    usage.get("response_chars", 0) if request_count else 0,
                )
            )
            profile["stage_calls"] += 1
            profile["physical_requests"] += request_count
            profile["latency_sec"] += float(usage.get("latency_sec", 0.0))
            profile["logical_prompt_chars"] += prompt_chars
            profile["physical_prompt_chars"] += physical_prompt_chars
            profile["physical_response_chars"] += physical_response_chars
            profile["cache_hits"] += int(bool(usage.get("cache_hit", False)))
    for profile in profiles.values():
        profile["latency_sec"] = round(profile["latency_sec"], 3)
        requests = profile["physical_requests"]
        profile["avg_request_latency_sec"] = (
            round(profile["latency_sec"] / requests, 3) if requests else 0.0
        )
    return dict(sorted(profiles.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--roots-subdir",
        default="roots",
        help="Materialize accepted roots under this output-directory child.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Provider TOML configuration. Required for codex; optional for responses.",
    )
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        default="codex",
        help="LLM provider used for root semantic generation and repair.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent independent root tasks. Increase within endpoint limits.",
    )
    parser.add_argument("--request-retries", type=int, default=1)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument(
        "--llm-cache-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / ".llm_cache",
        help="Content-addressed cache for successfully parsed model JSON.",
    )
    parser.add_argument(
        "--cache-namespace",
        default="default",
        help="Reuse only within this sampling namespace; change it to resample.",
    )
    parser.add_argument(
        "--no-llm-cache",
        action="store_true",
        help="Disable content-addressed model response reuse.",
    )
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Only process the selected source ID. Repeat to select multiple IDs.",
    )
    parser.add_argument(
        "--operator",
        help=(
            "Force one rewrite operator for all selected tasks. Useful for "
            "controlled topology experiments; default assignment remains balanced."
        ),
    )
    parser.add_argument("--repair-rounds", type=int, default=3)
    parser.add_argument(
        "--bundle-candidates",
        type=int,
        default=1,
        help="Independent bundle samples generated concurrently per root.",
    )
    parser.add_argument("--recursive-generations", type=int, default=2)
    parser.add_argument("--beam-size", type=int, default=6)
    parser.add_argument("--min-necessary-action-ratio", type=float, default=0.6)
    parser.add_argument(
        "--quality-baseline-dir",
        type=Path,
        help=(
            "Accepted root directory keyed by task ID. New roots must meet or "
            "exceed every baseline causal-quality component."
        ),
    )
    parser.add_argument(
        "--repair-quality-regressions",
        action="store_true",
        help=(
            "Allow LLM repair of a candidate below the component-wise quality "
            "baseline. Disabled by default; fresh resampling is usually cheaper."
        ),
    )
    parser.add_argument(
        "--quality-resample-candidates",
        type=int,
        default=0,
        help=(
            "Fresh bundle candidates requested when the best initial design is "
            "below --quality-baseline-dir. Experimental and disabled by default; "
            "set a positive value to opt in."
        ),
    )
    parser.add_argument(
        "--quality-resample-reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        default="xhigh",
        help="Reasoning tier for fresh quality-floor resampling.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh"])
    parser.add_argument(
        "--seed-reasoning-effort", choices=["low", "medium", "high", "xhigh"]
    )
    parser.add_argument(
        "--contract-reasoning-effort", choices=["low", "medium", "high", "xhigh"]
    )
    parser.add_argument(
        "--bundle-reasoning-effort", choices=["low", "medium", "high", "xhigh"]
    )
    parser.add_argument(
        "--patch-reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        default="medium",
        help="Optional cheaper reasoning tier for guarded patch generation.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--always-repair-contract",
        action="store_true",
        help="Use the legacy behavior of regenerating a valid contract on every bundle repair.",
    )
    parser.add_argument(
        "--patch-repair",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Try a small JSON Patch repair before falling back to complete bundle repair.",
    )
    parser.add_argument(
        "--regenerate-seed",
        action="store_true",
        help="Explicitly replace a cached seed using the selected current source text.",
    )
    parser.add_argument(
        "--strict-adaptive",
        action="store_true",
        help=(
            "Admit roots only after executable instruction alignment, public "
            "totality, counterfactual planning, and an adaptive evidence profile."
        ),
    )
    parser.add_argument(
        "--strict-vnext",
        action="store_true",
        help=(
            "Require 15-25 necessary calls, 3-5 grounded decisions, 1-2 "
            "semantic failures, alternative recovery, route hiding, strict "
            "provenance, and final state evidence. Implies --strict-adaptive."
        ),
    )
    args = parser.parse_args()
    if args.strict_vnext:
        args.strict_adaptive = True
    if (
        args.limit < 1
        or args.repair_rounds < 0
        or args.recursive_generations < 0
        or args.workers < 1
        or args.bundle_candidates < 1
        or args.request_retries < 0
        or args.retry_backoff < 0
        or args.quality_resample_candidates < 0
    ):
        raise SystemExit("limit must be >= 1 and round counts must be >= 0")
    if args.provider == "codex":
        if args.config is None:
            raise SystemExit("--config is required for --provider codex")
        os.environ["GEM_CODEX_CONFIG"] = str(args.config.resolve())
        os.environ["GEM_CODEX_PROJECT_DIR"] = str(PROJECT_ROOT)
        os.environ.setdefault("GEM_CODEX_DISABLE_MCP", "1")
        os.environ.setdefault("GEM_CODEX_TIMEOUT", "900")
    elif args.provider == "responses" and args.config is not None:
        configure_responses_provider(args.config.resolve())
    if args.reasoning_effort:
        os.environ["GEM_CODEX_REASONING_EFFORT"] = args.reasoning_effort
    cache_namespace = hashlib.sha256(
        args.cache_namespace.encode("utf-8")
    ).hexdigest()[:16]
    model_cache_dir = (
        None
        if args.no_llm_cache
        else args.llm_cache_dir.resolve() / cache_namespace
    )

    all_rows = load_jsonl(args.input)
    if args.task_id:
        selected_ids = set(args.task_id)
        rows = [row for row in all_rows if str(row.get("id")) in selected_ids]
        missing_ids = sorted(selected_ids - {str(row.get("id")) for row in rows})
        if missing_ids:
            raise SystemExit(f"unknown task IDs: {missing_ids}")
    else:
        rows = all_rows[: args.limit]
    catalog_value = json.loads((PROJECT_ROOT / "config" / "task_rewrite_operators.json").read_text(encoding="utf-8"))
    catalog = catalog_value["operators"]
    if args.strict_adaptive:
        catalog = [
            card for card in catalog if card.get("id") in STRICT_ADAPTIVE_OPERATORS
        ]
        if not catalog:
            raise SystemExit("strict adaptive operator catalog is empty")
    if args.operator:
        forced = next(
            (card for card in catalog if card.get("id") == args.operator), None
        )
        if forced is None:
            raise SystemExit(
                f"operator {args.operator!r} is unavailable in the selected mode"
            )
        assigned_cards = [forced for _row in rows]
    else:
        assigned_cards = assign_operators(rows, catalog)
    cards = [
        resume_operator_card(
            row,
            assigned,
            catalog,
            output_dir=args.output_dir,
            resume=args.resume,
        )
        for row, assigned in zip(rows, assigned_cards)
    ]
    quality_floors: dict[str, dict[str, float | int]] = {}
    explicit_quality_baseline = args.quality_baseline_dir is not None
    effective_quality_baseline_dir = args.quality_baseline_dir
    if effective_quality_baseline_dir is None and args.resume:
        resume_roots = args.output_dir / args.roots_subdir
        if resume_roots.is_dir():
            effective_quality_baseline_dir = resume_roots
    if effective_quality_baseline_dir is not None:
        for row in rows:
            task_id = str(row["id"])
            baseline_path = effective_quality_baseline_dir / task_id
            if not baseline_path.is_dir():
                if explicit_quality_baseline:
                    raise SystemExit(
                        f"quality baseline is missing task {task_id!r}: {baseline_path}"
                    )
                continue
            baseline_bundle = load_task_bundle(baseline_path)
            baseline_report = evaluate(
                baseline_bundle, args.min_necessary_action_ratio
            )
            if not baseline_report["valid"]:
                raise SystemExit(
                    f"quality baseline {task_id!r} is invalid: "
                    + "; ".join(baseline_report["errors"])
                )
            quality_floors[task_id] = quality_profile(baseline_report)
    roots_by_index: dict[int, TaskBundle] = {}
    audits_by_index: dict[int, dict[str, Any]] = {}
    started = time.monotonic()

    def generate_indexed(
        index: int, row: dict[str, Any], card: dict[str, Any]
    ) -> tuple[int, TaskBundle | None, dict[str, Any]]:
        try:
            root, audit = generate_one(
                row,
                card,
                output_dir=args.output_dir,
                repair_rounds=args.repair_rounds,
                min_ratio=args.min_necessary_action_ratio,
                temperature=args.temperature,
                provider=args.provider,
                reasoning_effort=args.reasoning_effort,
                seed_reasoning_effort=args.seed_reasoning_effort,
                contract_reasoning_effort=args.contract_reasoning_effort,
                bundle_reasoning_effort=args.bundle_reasoning_effort,
                patch_reasoning_effort=args.patch_reasoning_effort,
                bundle_candidates=args.bundle_candidates,
                request_retries=args.request_retries,
                retry_backoff=args.retry_backoff,
                model_cache_dir=model_cache_dir,
                always_repair_contract=args.always_repair_contract,
                patch_repair=args.patch_repair,
                resume=args.resume,
                regenerate_seed=args.regenerate_seed,
                quality_floor=quality_floors.get(str(row["id"])),
                repair_quality_regressions=args.repair_quality_regressions,
                quality_resample_candidates=args.quality_resample_candidates,
                quality_resample_reasoning_effort=(
                    args.quality_resample_reasoning_effort
                ),
                strict_adaptive=args.strict_adaptive,
                strict_vnext=args.strict_vnext,
            )
        except Exception as exc:
            root = None
            audit_path = (
                args.output_dir / "tasks" / str(row.get("id")) / "audit.json"
            )
            if audit_path.is_file():
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                audit["final_errors"] = [str(exc)]
            else:
                audit = {
                    "task_id": row.get("id"),
                    "assigned_operator": card["id"],
                    "final_errors": [str(exc)],
                }
            removed = evict_audit_cache_entries(audit, model_cache_dir)
            profiling = audit.setdefault("profiling", {})
            profiling["evicted_cache_entries"] = (
                int(profiling.get("evicted_cache_entries", 0)) + removed
            )
        return index, root, audit

    work = list(enumerate(zip(rows, cards)))
    if args.workers == 1:
        completed = (
            generate_indexed(index, row, card)
            for index, (row, card) in work
        )
    else:
        executor = ThreadPoolExecutor(max_workers=args.workers)
        futures = [
            executor.submit(generate_indexed, index, row, card)
            for index, (row, card) in work
        ]
        completed = (future.result() for future in as_completed(futures))

    for index, root, audit in completed:
        audits_by_index[index] = audit
        write_jsonl(
            args.output_dir / "task_audits.checkpoint.jsonl",
            [audits_by_index[item] for item in sorted(audits_by_index)],
        )
        if root is None:
            continue
        path = args.output_dir / args.roots_subdir / root.task_id
        if args.resume and path.is_dir():
            roots_by_index[index] = refresh_materialized_root(path, root)
        else:
            path = materialize_candidate(
                args.output_dir / args.roots_subdir,
                task_id=root.task_id,
                contract=root.contract,
                candidate={
                    "instruction": root.instruction,
                    "environment": root.environment,
                    "bindings": root.bindings,
                    "reference_plan": root.reference_plan,
                },
                lineage=root.manifest["lineage"],
                manifest_metadata=manifest_metadata(root),
            )
            roots_by_index[index] = TaskBundle(
                root=path,
                manifest=root.manifest,
                instruction=root.instruction,
                contract=root.contract,
                environment=root.environment,
                bindings=root.bindings,
                reference_plan=root.reference_plan,
            )
    if args.workers > 1:
        executor.shutdown(wait=True)
    roots = [roots_by_index[index] for index in sorted(roots_by_index)]
    audits = [audits_by_index[index] for index in sorted(audits_by_index)]
    recursive = (
        recursive_search(
            roots,
            args.output_dir,
            args.recursive_generations,
            args.beam_size,
            resume=args.resume,
            objective="decision_nodes",
        )
        if roots and args.recursive_generations
        else {"generations": [], "distinct_semantic_fingerprints": 0}
    )
    elapsed_sec = round(time.monotonic() - started, 3)
    accepted_recursive = sum(
        len(generation.get("selected", []))
        for generation in recursive.get("generations", [])
    )
    accepted_semantic_episodes = len(roots) + accepted_recursive
    stage_counts: dict[str, int] = {}
    for audit in audits:
        for stage in audit.get("stages", []):
            name = str(stage.get("stage", "unknown"))
            stage_counts[name] = stage_counts.get(name, 0) + 1
    summary = {
        "input": len(rows),
        "accepted_roots": len(roots),
        "rejected_roots": len(rows) - len(roots),
        "provider": args.provider,
        "workers": args.workers,
        "bundle_candidates": args.bundle_candidates,
        "max_generation_concurrency": args.workers
        * max(args.bundle_candidates, args.quality_resample_candidates, 1),
        "llm_cache_enabled": model_cache_dir is not None,
        "cache_namespace": args.cache_namespace if model_cache_dir is not None else None,
        "quality_baseline_dir": (
            str(effective_quality_baseline_dir.resolve())
            if effective_quality_baseline_dir is not None
            else None
        ),
        "quality_baseline_mode": (
            "explicit"
            if explicit_quality_baseline
            else "resume_auto"
            if effective_quality_baseline_dir is not None
            else "none"
        ),
        "quality_floors": quality_floors,
        "quality_resample_candidates": args.quality_resample_candidates,
        "quality_resample_reasoning_effort": (
            args.quality_resample_reasoning_effort
        ),
        "strict_adaptive": args.strict_adaptive,
        "strict_vnext": args.strict_vnext,
        "operator_assignments": [
            {"task_id": row["id"], "operator": card["id"], "family": card["family"]}
            for row, card in zip(rows, cards)
        ],
        "forced_operator": args.operator,
        "root_metrics": [
            {
                "task_id": audit.get("task_id"),
                "accepted": audit.get("accepted", False),
                "metrics": audit.get("metrics"),
                "necessary_action_ratio": audit.get("necessary_action_ratio"),
                "profiling": audit.get("profiling", {}),
                "errors": audit.get("final_errors", []),
            }
            for audit in audits
        ],
        "recursive": recursive,
        "elapsed_sec": elapsed_sec,
        "accepted_recursive_episodes": accepted_recursive,
        "accepted_semantic_episodes": accepted_semantic_episodes,
        "accepted_semantic_episodes_per_hour": (
            round(accepted_semantic_episodes * 3600 / elapsed_sec, 3)
            if elapsed_sec
            else 0.0
        ),
        "accepted_roots_per_hour": (
            round(len(roots) * 3600 / elapsed_sec, 3) if elapsed_sec else 0.0
        ),
        "stage_counts": stage_counts,
        "stage_profiles": summarize_stage_profiles(audits),
        "llm_calls": sum(
            int(audit.get("profiling", {}).get("llm_calls", 0)) for audit in audits
        ),
        "llm_latency_sec": round(
            sum(
                float(audit.get("profiling", {}).get("llm_latency_sec", 0.0))
                for audit in audits
            ),
            3,
        ),
        "cache_hits": sum(
            int(audit.get("profiling", {}).get("cache_hits", 0))
            for audit in audits
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    write_jsonl(args.output_dir / "task_audits.jsonl", audits)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not roots:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
