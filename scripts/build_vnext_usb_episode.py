#!/usr/bin/env python3
"""Compile the model-planned USB alternative-recovery contract onto a parent."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_validation import (  # noqa: E402
    evaluate_action_ablation,
    validate_episode,
    validate_vnext_adaptive_profile,
)
from causal_validation.intervention import evaluate_counterfactuals  # noqa: E402
from rollout import run_reference_plan  # noqa: E402
from task_factory import load_task_bundle  # noqa: E402
from task_factory.bundle import TaskBundle, validate_bundle  # noqa: E402
from task_factory.materialize import materialize_candidate  # noqa: E402


def _profile_capability() -> dict:
    def success(branch_id: str, profile: str, allocation: str, format_type: str) -> dict:
        return {
            "id": branch_id,
            "when": {
                "all": [
                    {"eq": ["$state.format.dialog_open", True]},
                    {"eq": ["$args.format_dialog_handle", "$state.format.dialog_handle"]},
                    {"eq": ["$state.format_dialog.supported_profile", profile]},
                ]
            },
            "response": {
                "ok": True,
                "error_code": "NO_ERROR",
                "supported_profile": profile,
                "profile_evidence_ref": profile,
                "allocation_size": allocation,
                "format_type": format_type,
            },
            "reads": [
                "$state.format.dialog_open",
                "$state.format.dialog_handle",
                "$state.format_dialog.supported_profile",
            ],
            "observes": ["$state.format_dialog.supported_profile"],
            "writes": [],
            "effects": [],
            "resolves_errors": [],
        }

    return {
        "branches": [
            success(
                "default_profile_visible",
                "visible_defaults_supported",
                "default_allocation_size",
                "quick_format",
            ),
            success(
                "alternative_profile_visible",
                "visible_defaults_rejected_with_supported_alternative",
                "large_allocation_size",
                "full_format",
            ),
            {
                "id": "format_profile_unavailable",
                "when": True,
                "response": {
                    "ok": False,
                    "recoverable": True,
                    "error_code": "FORMAT_PROFILE_NOT_VISIBLE",
                    "message": "Open the target drive format dialog before inspecting its profiles.",
                },
                "reads": ["$state.format.dialog_open", "$state.format.dialog_handle"],
                "writes": [],
                "effects": [],
                "resolves_errors": [],
            },
        ]
    }


def _configure_capability(parent_capability: dict) -> dict:
    original = copy.deepcopy(parent_capability["branches"])
    default_branch = original[0]
    default_branch["id"] = "visible_default_profile_applied"
    default_branch["when"]["all"].append(
        {
            "eq": [
                "$state.format_dialog.supported_profile",
                "visible_defaults_supported",
            ]
        }
    )
    default_branch["when"]["all"].append(
        {
            "eq": [
                "$args.profile_evidence_ref",
                "$state.format_dialog.supported_profile",
            ]
        }
    )
    default_branch["reads"].append("$state.format_dialog.supported_profile")

    return {"branches": [default_branch, *original[1:]]}


def _load_profile_capability() -> dict:
    return {
        "branches": [
            {
                "id": "visible_profile_loaded",
                "when": {
                    "all": [
                        {"eq": ["$state.format.dialog_open", True]},
                        {"eq": ["$args.format_dialog_handle", "$state.format.dialog_handle"]},
                        {
                            "eq": [
                                "$state.format_dialog.supported_profile",
                                "visible_defaults_rejected_with_supported_alternative",
                            ]
                        },
                        {
                            "eq": [
                                "$args.profile_evidence_ref",
                                "$state.format_dialog.supported_profile",
                            ]
                        },
                    ]
                },
                "response": {
                    "ok": True,
                    "error_code": "NO_ERROR",
                    "profile_loaded": "$args.profile_evidence_ref",
                    "start_button_available": True,
                },
                "reads": [
                    "$state.format.dialog_open",
                    "$state.format.dialog_handle",
                    "$state.format_dialog.supported_profile",
                ],
                "writes": [
                    "$state.format_dialog.allocation_size",
                    "$state.format_dialog.format_type",
                    "$state.format_dialog.options_configured",
                ],
                "effects": [
                    {"set": "$state.format_dialog.allocation_size", "value": "large_allocation_size"},
                    {"set": "$state.format_dialog.format_type", "value": "full_format"},
                    {"set": "$state.format_dialog.options_configured", "value": True},
                ],
                "resolves_errors": [],
            },
            {
                "id": "format_profile_loader_unavailable",
                "when": True,
                "response": {
                    "ok": False,
                    "recoverable": True,
                    "error_code": "FORMAT_PROFILE_LOADER_UNAVAILABLE",
                    "message": "The visible dialog does not currently expose a loadable profile.",
                },
                "reads": [
                    "$state.format.dialog_open",
                    "$state.format_dialog.supported_profile",
                ],
                "writes": [],
                "effects": [],
                "resolves_errors": [],
            },
        ]
    }


def _activation_capabilities(restart_capability: dict) -> tuple[dict, dict, dict]:
    restart = copy.deepcopy(restart_capability)
    success = restart["branches"][0]
    success["when"]["all"].append(
        {"eq": ["$state.system.policy_activation_mode", "restart_required"]}
    )
    success["when"]["all"].append(
        {
            "eq": [
                "$args.activation_evidence_ref",
                "$state.system.policy_activation_mode",
            ]
        }
    )
    success["reads"].append("$state.system.policy_activation_mode")
    success["writes"].append("$state.system.policy_activation_completed")
    success["effects"].append(
        {"set": "$state.system.policy_activation_completed", "value": True}
    )
    success["response"]["activation_method"] = "computer_restart"
    restart["branches"].insert(
        1,
        {
            "id": "restart_not_supported_for_current_activation_mode",
            "when": {
                "all": [
                    {"eq": ["$state.system.restart_pending", True]},
                    {"eq": ["$state.system.policy_activation_mode", "session_refresh_supported"]},
                ]
            },
            "response": {
                "ok": False,
                "recoverable": True,
                "error_code": "ACTIVATION_RESTART_NOT_AVAILABLE",
                "message": "The visible policy status offers session refresh instead of restart.",
            },
            "reads": [
                "$state.system.restart_pending",
                "$state.system.policy_activation_mode",
            ],
            "writes": [],
            "effects": [],
            "resolves_errors": [],
        },
    )
    inspect = {
        "branches": [
            {
                "id": "activation_options_visible",
                "when": {"eq": ["$state.system.restart_pending", True]},
                "response": {
                    "ok": True,
                    "error_code": "NO_ERROR",
                    "activation_mode": "$state.system.policy_activation_mode",
                },
                "reads": [
                    "$state.system.restart_pending",
                    "$state.system.policy_activation_mode",
                ],
                "writes": [],
                "effects": [],
                "resolves_errors": [],
            },
            {
                "id": "activation_options_not_ready",
                "when": True,
                "response": {
                    "ok": False,
                    "recoverable": True,
                    "error_code": "ACTIVATION_OPTIONS_NOT_READY",
                },
                "reads": ["$state.system.restart_pending"],
                "writes": [],
                "effects": [],
                "resolves_errors": [],
            },
        ]
    }
    refresh_success = {
        "id": "removable_storage_session_refreshed",
        "when": {
            "all": [
                {"eq": ["$state.ui.desktop_available", True]},
                {"eq": ["$state.system.restart_pending", True]},
                {
                    "eq": [
                        "$state.system.policy_activation_mode",
                        "session_refresh_supported",
                    ]
                },
                {
                    "eq": [
                        "$args.activation_evidence_ref",
                        "$state.system.policy_activation_mode",
                    ]
                },
            ]
        },
        "response": {
            "ok": True,
            "error_code": "NO_ERROR",
            "activation_method": "session_refresh",
            "explorer_session_preserved": "$state.usb.explorer_open",
            "format_dialog_preserved": "$state.format.dialog_open",
        },
        "reads": [
            "$state.ui.desktop_available",
            "$state.system.restart_pending",
            "$state.system.policy_activation_mode",
            "$state.usb.explorer_open",
            "$state.format.dialog_open",
        ],
        "writes": [
            "$state.system.restart_pending",
            "$state.system.policy_activation_completed",
            "$state.usb.connected_drive_status",
        ],
        "effects": [
            {"set": "$state.system.restart_pending", "value": False},
            {"set": "$state.system.policy_activation_completed", "value": True},
            {"set": "$state.usb.connected_drive_status", "value": "not_write_protected"},
        ],
        "resolves_errors": ["RECOVERABLE_RESTART_REQUIRED"],
    }
    refresh = {
        "branches": [
            refresh_success,
            {
                "id": "session_refresh_unavailable",
                "when": True,
                "response": {
                    "ok": False,
                    "recoverable": True,
                    "error_code": "SESSION_REFRESH_UNAVAILABLE",
                },
                "reads": [
                    "$state.system.restart_pending",
                    "$state.system.policy_activation_mode",
                ],
                "writes": [],
                "effects": [],
                "resolves_errors": [],
            },
        ]
    }
    return restart, inspect, refresh


def _create_policy_capability() -> dict:
    return {
        "branches": [
            {
                "id": "missing_storage_policy_structure_created",
                "when": {
                    "all": [
                        {"eq": ["$state.registry.editor_open", True]},
                        {"eq": ["$args.registry_window_handle", "$state.registry.window_handle"]},
                        {"eq": ["$state.registry.policy_status.storage_device_policies_exists", False]},
                        {"eq": ["$args.policy_lookup_error_code", "RECOVERABLE_STORAGE_DEVICE_POLICIES_MISSING"]},
                        {"eq": ["$args.format_error_code", "WRITE_PROTECT_ACTIVE"]},
                        {"eq": ["$state.format.last_result.error_code", "$args.format_error_code"]},
                        {"eq": ["$state.format.write_protect_failure_observed", True]},
                    ]
                },
                "response": {
                    "ok": True,
                    "error_code": "NO_ERROR",
                    "created_scope": "StorageDevicePolicies/WriteProtect",
                    "write_protect_effective": "inactive",
                    "restart_required": True,
                },
                "reads": [
                    "$state.registry.editor_open",
                    "$state.registry.window_handle",
                    "$state.registry.policy_status.storage_device_policies_exists",
                    "$state.format.last_result.error_code",
                    "$state.format.write_protect_failure_observed",
                ],
                "writes": [
                    "$state.registry.policy_status.storage_device_policies_exists",
                    "$state.registry.policy_status.write_protect_effective",
                    "$state.registry.policy_status.write_protect_value_state",
                    "$state.registry.policy_status.write_protect_data",
                    "$state.recovery.visible_error_code_used",
                    "$state.recovery.policy_repair_after_write_protect_error_observed",
                    "$state.system.restart_pending",
                ],
                "effects": [
                    {"set": "$state.registry.policy_status.storage_device_policies_exists", "value": True},
                    {"set": "$state.registry.policy_status.write_protect_effective", "value": "inactive"},
                    {"set": "$state.registry.policy_status.write_protect_value_state", "value": "inactive"},
                    {"set": "$state.registry.policy_status.write_protect_data", "value": 0},
                    {"set": "$state.recovery.visible_error_code_used", "value": "$args.format_error_code"},
                    {"set": "$state.recovery.policy_repair_after_write_protect_error_observed", "value": True},
                    {"set": "$state.system.restart_pending", "value": True},
                ],
                "resolves_errors": [
                    "WRITE_PROTECT_ACTIVE",
                    "RECOVERABLE_STORAGE_DEVICE_POLICIES_MISSING",
                ],
            },
            {
                "id": "policy_structure_creation_not_applicable",
                "when": True,
                "response": {
                    "ok": False,
                    "recoverable": True,
                    "error_code": "POLICY_STRUCTURE_CREATION_NOT_APPLICABLE",
                    "message": "The visible registry state does not expose a missing policy structure.",
                },
                "reads": [
                    "$state.registry.editor_open",
                    "$state.registry.policy_status.storage_device_policies_exists",
                    "$state.format.last_result.error_code",
                ],
                "writes": [],
                "effects": [],
                "resolves_errors": [],
            },
        ]
    }


def _profile_binding() -> dict:
    return {
        "name": "inspect_format_profile_options",
        "description": "Inspect the open Format dialog and report its currently supported profile choices.",
        "capability_id": "inspect_format_profiles",
        "parameters": {
            "type": "object",
            "properties": {
                "format_dialog_handle": {
                    "type": "string",
                    "description": "Opaque handle for the visible Format dialog to inspect.",
                }
            },
            "required": ["format_dialog_handle"],
            "additionalProperties": False,
        },
        "provenance_required": ["format_dialog_handle"],
    }


def _create_policy_binding() -> dict:
    return {
        "name": "create_storage_policy_value",
        "description": "Create a missing StorageDevicePolicies WriteProtect DWORD with the supplied visible value data.",
        "capability_id": "create_storage_policy_structure",
        "parameters": {
            "type": "object",
            "properties": {
                "registry_window_handle": {
                    "type": "string",
                    "description": "Opaque handle for the open Registry Editor window.",
                },
                "policy_lookup_error_code": {
                    "type": "string",
                    "description": "Visible code returned when the requested policy view was absent.",
                },
                "format_error_code": {
                    "type": "string",
                    "description": "Visible code from the blocked Format operation.",
                },
            },
            "required": [
                "registry_window_handle",
                "policy_lookup_error_code",
                "format_error_code",
            ],
            "additionalProperties": False,
        },
        "provenance_required": [
            "registry_window_handle",
            "policy_lookup_error_code",
            "format_error_code",
        ],
    }


def _load_profile_binding() -> dict:
    return {
        "name": "load_format_profile",
        "description": "Load a profile that is visibly offered by the open Format dialog.",
        "capability_id": "load_visible_format_profile",
        "parameters": {
            "type": "object",
            "properties": {
                "format_dialog_handle": {"type": "string"},
                "profile_evidence_ref": {"type": "string"},
            },
            "required": ["format_dialog_handle", "profile_evidence_ref"],
            "additionalProperties": False,
        },
        "provenance_required": ["format_dialog_handle", "profile_evidence_ref"],
    }


def _activation_bindings() -> list[dict]:
    return [
        {
            "name": "inspect_policy_activation_options",
            "description": "Inspect the visible activation options for a pending policy change.",
            "capability_id": "inspect_policy_activation_options",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            "provenance_required": [],
        },
        {
            "name": "refresh_removable_storage_session",
            "description": "Refresh the removable-storage session when that activation option is visible.",
            "capability_id": "refresh_removable_storage_session",
            "parameters": {
                "type": "object",
                "properties": {
                    "activation_evidence_ref": {
                        "type": "string",
                        "description": "Activation mode returned by the current policy status observation.",
                    }
                },
                "required": ["activation_evidence_ref"],
                "additionalProperties": False,
            },
            "provenance_required": ["activation_evidence_ref"],
        },
    ]


def _insert_profile_observations(actions: list[dict]) -> list[dict]:
    result = []
    observed = False
    for action in actions:
        if action["tool"] == "set_format_dialog_options" and not observed:
            result.append(
                {
                    "tool": "inspect_format_profile_options",
                    "arguments": {
                        "format_dialog_handle": action["arguments"]["format_dialog_handle"]
                    },
                }
            )
            observed = True
        copied = copy.deepcopy(action)
        if copied["tool"] == "set_format_dialog_options":
            copied["arguments"]["profile_evidence_ref"] = (
                "visible_defaults_supported"
            )
        result.append(copied)
    return result


def compile_child(parent: TaskBundle) -> TaskBundle:
    contract = copy.deepcopy(parent.contract)
    environment = copy.deepcopy(parent.environment)
    bindings = copy.deepcopy(parent.bindings)
    reference = copy.deepcopy(parent.reference_plan)

    instruction = (
        "Using the simulated Windows UI, discover the target pen drive through ordinary "
        "USB and Explorer surfaces, format that same drive, and leave it no longer "
        "write-protected. Base every choice on visible application state, never format an "
        "undiscovered or non-target drive, do not start formatting while a restart is "
        "pending, keep any system change within a visibly authorized scope, and finish "
        "by checking an ordinary status view."
    )
    contract["selected_operator"] = "alternative_recovery_affordance"
    contract["new_requirements"] = [
        "A visible environment condition selects among genuinely different successful strategies."
    ]
    contract["expected_reasoning_features"] = [
        "delayed_handle_use",
        "observation_dependent_decision",
        "semantic_recovery",
        "alternative_plan",
        "goal_grounded_verification",
    ]
    contract["counterfactual_axes"] = [
        {
            "state_path": "$state.registry.policy_status.storage_device_policies_exists",
            "variants": [True, False],
        },
        {
            "state_path": "$state.usb.actual_target_drive_handle",
            "variants": ["visible_target_a", "visible_target_b"],
        },
        {
            "state_path": "$state.format_dialog.supported_profile",
            "variants": [
                "visible_defaults_supported",
                "visible_defaults_rejected_with_supported_alternative",
            ],
        },
        {
            "state_path": "$state.system.policy_activation_mode",
            "variants": ["restart_required", "session_refresh_supported"],
        },
    ]
    requirements = contract.setdefault("requirements", {})
    requirements["semantic_recovery"] = True
    contract["instruction_claims"] = [
        {
            "evidence_span": (
                "Using the simulated Windows UI, discover the target pen drive "
                "through ordinary USB and Explorer surfaces, format that same drive, "
                "and leave it no longer write-protected."
            ),
            "clause_ids": ["c1", "c2"],
            "kind": "goal",
        },
        {
            "evidence_span": (
                "Base every choice on visible application state, never format an "
                "undiscovered or non-target drive, do not start formatting while a "
                "restart is pending, keep any system change within a visibly authorized "
                "scope, and finish by checking an ordinary status view."
            ),
            "clause_ids": ["c3", "c4", "c5", "c6", "c7"],
            "kind": "constraint",
        },
    ]

    state = environment["initial_state"]
    state["format_dialog"]["supported_profile"] = "visible_defaults_supported"
    state["system"]["policy_activation_mode"] = "restart_required"
    state["system"]["policy_activation_completed"] = False
    capabilities = environment["capabilities"]
    capabilities["inspect_format_profiles"] = _profile_capability()
    capabilities["configure_format_dialog"] = _configure_capability(
        capabilities["configure_format_dialog"]
    )
    capabilities["load_visible_format_profile"] = _load_profile_capability()
    capabilities["create_storage_policy_structure"] = _create_policy_capability()
    restart, inspect_activation, refresh = _activation_capabilities(
        capabilities["restart_computer"]
    )
    capabilities["restart_computer"] = restart
    capabilities["inspect_policy_activation_options"] = inspect_activation
    capabilities["refresh_removable_storage_session"] = refresh
    for branch in capabilities["start_format"]["branches"]:
        if branch.get("id") == "format_reports_active_write_protect":
            branch.get("response", {}).pop("drive_handle", None)
    for branch in capabilities["open_storage_policy_location"]["branches"][:2]:
        branch["observes"] = [
            "$state.registry.policy_status.storage_device_policies_exists"
        ]

    for tool in bindings["tools"]:
        if tool["name"] == "set_write_protect_policy_inactive":
            tool["description"] = (
                "Set the visible WriteProtect DWORD data in the currently open "
                "StorageDevicePolicies value view."
            )
        elif tool["name"] == "set_format_dialog_options":
            properties = tool["parameters"]["properties"]
            properties["allocation_size"]["enum"] = [
                "default_allocation_size",
                "large_allocation_size",
            ]
            properties["format_type"]["enum"] = ["quick_format", "full_format"]
            properties["profile_evidence_ref"] = {
                "type": "string",
                "description": (
                    "Evidence reference returned by the current Format dialog profile observation."
                ),
            }
            tool["parameters"]["required"].append("profile_evidence_ref")
            tool["provenance_required"].append("profile_evidence_ref")
        elif tool["name"] == "restart_computer_to_apply_policy":
            tool["parameters"]["properties"]["activation_evidence_ref"] = {
                "type": "string",
                "description": "Activation mode returned by the current policy status observation.",
            }
            tool["parameters"]["required"].append("activation_evidence_ref")
            tool["provenance_required"].append("activation_evidence_ref")
    bindings["tools"].extend(
        [
            _profile_binding(),
            _create_policy_binding(),
            _load_profile_binding(),
            *_activation_bindings(),
        ]
    )

    parent_actions = [
        action
        for action in reference["actions"]
        if action["tool"] not in {"observe_restart_status", "close_registry_editor"}
    ]
    discovery_tools = {
        "open_start_search",
        "search_start_for_registry_editor",
        "launch_registry_editor_from_result",
    }
    delayed_system_discovery = [
        action for action in parent_actions if action["tool"] in discovery_tools
    ]
    parent_actions = [
        action for action in parent_actions if action["tool"] not in discovery_tools
    ]
    first_attempt = next(
        index
        for index, action in enumerate(parent_actions)
        if action["tool"] == "start_format_from_dialog"
    )
    parent_actions[first_attempt + 1 : first_attempt + 1] = delayed_system_discovery
    baseline = _insert_profile_observations(parent_actions)
    restart_index = next(
        index
        for index, action in enumerate(baseline)
        if action["tool"] == "restart_computer_to_apply_policy"
    )
    baseline.insert(
        restart_index,
        {"tool": "inspect_policy_activation_options", "arguments": {}},
    )
    baseline[restart_index + 1]["arguments"]["activation_evidence_ref"] = (
        "restart_required"
    )
    reference["actions"] = baseline

    target_b = copy.deepcopy(baseline)
    for action in target_b:
        if action["tool"] == "open_drive_context_menu":
            action["arguments"]["drive_handle"] = "visible_target_b"

    alternative_profile = copy.deepcopy(baseline)
    for index, action in enumerate(alternative_profile):
        if action["tool"] == "set_format_dialog_options":
            alternative_profile[index] = {
                "tool": "load_format_profile",
                "arguments": {
                    "format_dialog_handle": action["arguments"]["format_dialog_handle"],
                    "profile_evidence_ref": "visible_defaults_rejected_with_supported_alternative",
                },
            }

    session_refresh = copy.deepcopy(baseline)
    for action in session_refresh:
        if action["tool"] == "restart_computer_to_apply_policy":
            action["tool"] = "refresh_removable_storage_session"
            action["arguments"]["activation_evidence_ref"] = (
                "session_refresh_supported"
            )
    refresh_index = next(
        index
        for index, action in enumerate(session_refresh)
        if action["tool"] == "refresh_removable_storage_session"
    )
    preserved_session_skips = {
        "open_windows_explorer",
        "open_drive_context_menu",
        "choose_format_menu_item",
        "set_format_dialog_options",
    }
    session_refresh = session_refresh[: refresh_index + 1] + [
        action
        for action in session_refresh[refresh_index + 1 :]
        if action["tool"] not in preserved_session_skips
    ]

    missing_policy = copy.deepcopy(baseline)
    open_index = next(
        index
        for index, action in enumerate(missing_policy)
        if action["tool"] == "open_storage_device_policy_view"
    )
    missing_policy[open_index + 1 : open_index + 3] = [
        {
            "tool": "create_storage_policy_value",
            "arguments": {
                "registry_window_handle": "registry_editor_main",
                "policy_lookup_error_code": "RECOVERABLE_STORAGE_DEVICE_POLICIES_MISSING",
                "format_error_code": "WRITE_PROTECT_ACTIVE",
            },
        }
    ]
    reference["counterfactuals"] = [
        {
            "name": "missing_policy_structure_uses_creation_affordance",
            "state_overrides": {
                "$state.registry.policy_status.storage_device_policies_exists": False,
                "$state.registry.policy_status.write_protect_effective": "missing",
                "$state.registry.policy_status.write_protect_value_state": "missing",
                "$state.registry.policy_status.write_protect_data": -1,
            },
            "actions": missing_policy,
        },
        {
            "name": "alternate_visible_target_is_selected_from_listing",
            "state_overrides": {
                "$state.usb.actual_target_drive_handle": "visible_target_b"
            },
            "actions": target_b,
        },
        {
            "name": "visible_supported_format_profile_changes_configuration",
            "state_overrides": {
                "$state.format_dialog.supported_profile": (
                    "visible_defaults_rejected_with_supported_alternative"
                )
            },
            "actions": alternative_profile,
        },
        {
            "name": "visible_activation_mode_uses_session_refresh",
            "state_overrides": {
                "$state.system.policy_activation_mode": "session_refresh_supported"
            },
            "actions": session_refresh,
        },
    ]

    manifest = copy.deepcopy(parent.manifest)
    manifest["task_id"] = parent.task_id + "_vnext"
    manifest["assigned_operator"] = "alternative_recovery_affordance"
    manifest["lineage"] = {
        "parent_task_id": parent.task_id,
        "generation": int(parent.manifest.get("lineage", {}).get("generation", 0)) + 1,
        "operators": [
            *parent.manifest.get("lineage", {}).get("operators", []),
            "alternative_recovery_affordance",
        ],
    }
    return TaskBundle(
        root=Path("<memory>"),
        manifest=manifest,
        instruction=instruction,
        contract=contract,
        environment=environment,
        bindings=bindings,
        reference_plan=reference,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    child = compile_child(load_task_bundle(args.parent))
    static_errors = validate_bundle(child)
    episode = run_reference_plan(child) if not static_errors else {"trace": []}
    causal = validate_episode(child, episode) if not static_errors else {"metrics": {}}
    counterfactual = evaluate_counterfactuals(child) if not static_errors else {}
    ablation = evaluate_action_ablation(child) if not static_errors else {}
    vnext = (
        validate_vnext_adaptive_profile(
            child, episode, causal, counterfactual, ablation=ablation
        )
        if not static_errors
        else {"valid": False, "errors": static_errors}
    )
    audit = {
        "task_id": child.task_id,
        "static_errors": static_errors,
        "episode_status": episode.get("status"),
        "causal": causal,
        "counterfactual": counterfactual,
        "ablation": ablation,
        "vnext": vnext,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if vnext["valid"]:
        path = materialize_candidate(
            args.output_dir / "bundles",
            task_id=child.task_id,
            contract=child.contract,
            candidate={
                "instruction": child.instruction,
                "environment": child.environment,
                "bindings": child.bindings,
                "reference_plan": child.reference_plan,
            },
            lineage=child.manifest["lineage"],
            manifest_metadata={
                "assigned_operator": "alternative_recovery_affordance",
                "compiler": "model_planned_code_compiled_v2",
            },
        )
        audit["bundle"] = str(path)
        (args.output_dir / "evaluation.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps({"valid": vnext["valid"], "errors": vnext["errors"]}, indent=2))
    if not vnext["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
