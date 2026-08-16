#!/usr/bin/env python3
"""Build a WikiHow-derived temporal provenance recovery task."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_validation import validate_goal_alignment
from task_factory import load_task_bundle, totalize_public_capabilities
from task_factory.bundle import TaskBundle
from task_factory.materialize import materialize_candidate
from task_factory.operators.base import manifest_metadata


DEFAULT_SOURCE = PROJECT_ROOT / (
    "outputs/task_first/wikihow_compile200_context_v5/parents/"
    "wikihow_computer_002787__task_first"
)


def build_bundle(source: Path) -> TaskBundle:
    parent = load_task_bundle(source)
    instruction = parent.instruction
    session = "$state.session"
    context = "$state.workflow_context"
    selected = "$state.local_downloads.selected"
    restored = "$state.files.restored_tax_return_2023"

    initial_state = {
        "ui": {"bitcasa_open": False},
        "session": {
            "authenticated": False,
            "current_view": "login",
            "target_date": "today",
            "selected_snapshot_handle": "",
            "workspace_revision": 1,
        },
        "revision_catalog": {
            "rev_tax_2023_morning": {
                "revision_id": "rev_tax_2023_morning",
                "name": "tax_return_2023.pdf",
                "source_date": "2023-10-15",
                "capture_sequence": 1,
                "completeness": "complete",
            },
            "rev_tax_2023_evening": {
                "revision_id": "rev_tax_2023_evening",
                "name": "tax_return_2023.pdf",
                "source_date": "2023-10-15",
                "capture_sequence": 2,
                "completeness": "complete",
            },
            "rev_tax_2023_partial": {
                "revision_id": "rev_tax_2023_partial",
                "name": "tax_return_2023.pdf",
                "source_date": "2023-10-15",
                "capture_sequence": 3,
                "completeness": "partial",
            },
        },
        "local_downloads": {
            "selected": {
                "artifact_id": "",
                "name": "",
                "source_date": "",
                "source_revision_id": "",
                "source_snapshot_handle": "",
                "downloaded": False,
            }
        },
        "upload_sessions": {
            "current": {
                "session_id": "",
                "workspace_ref": "",
                "workspace_revision": 0,
            }
        },
        "files": {
            "restored_tax_return_2023": {
                "record_id": "restored_tax_return_2023",
                "name": "",
                "status": "absent",
                "source_date": "",
                "source_revision_id": "",
                "restored_from_artifact_id": "",
                "version_date": "",
            }
        },
        "selection_violation": False,
        "workflow_context": {
            "handle": "workflow_context_wikihow_computer_002787",
            "source_task_id": "wikihow_computer_002787",
            "requested_name": "tax_return_2023.pdf",
            "requested_date": "2023-10-15",
            "recovery_policy": "latest_complete_on_date",
        },
    }

    candidates = [
        initial_state["revision_catalog"]["rev_tax_2023_morning"],
        initial_state["revision_catalog"]["rev_tax_2023_evening"],
        initial_state["revision_catalog"]["rev_tax_2023_partial"],
    ]

    def download_success(
        branch_id: str, policy: str, revision_id: str, artifact_id: str
    ) -> dict:
        return {
            "id": branch_id,
            "when": {
                "all": [
                    {"eq": ["$args.destination", "local_downloads"]},
                    {"eq": ["$args.resource_id", revision_id]},
                    {
                        "eq": [
                            "$args.snapshot_handle",
                            f"{session}.selected_snapshot_handle",
                        ]
                    },
                    {"eq": [f"{context}.recovery_policy", policy]},
                    {"eq": [f"{session}.target_date", "2023-10-15"]},
                ]
            },
            "response": {
                "status": "success",
                "artifact_id": artifact_id,
                "name": "tax_return_2023.pdf",
                "source_date": "2023-10-15",
                "source_revision_id": revision_id,
                "source_snapshot_handle": f"{session}.selected_snapshot_handle",
            },
            "effects": [
                {
                    "set": selected,
                    "value": {
                        "artifact_id": artifact_id,
                        "name": "tax_return_2023.pdf",
                        "source_date": "2023-10-15",
                        "source_revision_id": revision_id,
                        "source_snapshot_handle": f"{session}.selected_snapshot_handle",
                        "downloaded": True,
                    },
                }
            ],
            "reads": [
                f"$state.revision_catalog.{revision_id}",
                f"{session}.selected_snapshot_handle",
                f"{session}.target_date",
                f"{context}.recovery_policy",
            ],
            "writes": [selected],
        }

    def upload_success(
        branch_id: str, artifact_id: str, revision_id: str
    ) -> dict:
        return {
            "id": branch_id,
            "when": {
                "all": [
                    {
                        "eq": [
                            "$args.session_id",
                            "$state.upload_sessions.current.session_id",
                        ]
                    },
                    {"eq": ["$args.resource_id", artifact_id]},
                    {"eq": [f"{selected}.artifact_id", artifact_id]},
                    {"eq": [f"{selected}.downloaded", True]},
                    {"eq": [f"{session}.target_date", "today"]},
                    {
                        "eq": [
                            "$state.upload_sessions.current.workspace_revision",
                            f"{session}.workspace_revision",
                        ]
                    },
                ]
            },
            "response": {
                "status": "success",
                "restored_record_id": "restored_tax_return_2023",
                "restored_name": "tax_return_2023.pdf",
                "source_revision_id": revision_id,
            },
            "effects": [
                {
                    "set": restored,
                    "value": {
                        "record_id": "restored_tax_return_2023",
                        "name": "tax_return_2023.pdf",
                        "status": "active",
                        "source_date": "2023-10-15",
                        "source_revision_id": revision_id,
                        "restored_from_artifact_id": artifact_id,
                        "version_date": "today",
                    },
                }
            ],
            "reads": [selected, session, "$state.upload_sessions.current"],
            "writes": [restored],
        }

    capabilities = {
        "wikihow.workflow_context.observe.v1": {
            "branches": [
                {
                    "id": "latest_complete_policy_visible",
                    "when": {
                        "eq": [
                            f"{context}.recovery_policy",
                            "latest_complete_on_date",
                        ]
                    },
                    "response": {
                        "workflow_context_handle": f"{context}.handle",
                        "requested_name": f"{context}.requested_name",
                        "requested_date": f"{context}.requested_date",
                        "recovery_policy": f"{context}.recovery_policy",
                        "policy_description": (
                            "When multiple complete captures exist on the requested "
                            "date, restore the latest complete capture."
                        ),
                    },
                    "effects": [],
                    "reads": [context],
                    "writes": [],
                },
                {
                    "id": "earliest_complete_policy_visible",
                    "when": {
                        "eq": [
                            f"{context}.recovery_policy",
                            "earliest_complete_on_date",
                        ]
                    },
                    "response": {
                        "workflow_context_handle": f"{context}.handle",
                        "requested_name": f"{context}.requested_name",
                        "requested_date": f"{context}.requested_date",
                        "recovery_policy": f"{context}.recovery_policy",
                        "policy_description": (
                            "When multiple complete captures exist on the requested "
                            "date, restore the earliest complete capture."
                        ),
                    },
                    "effects": [],
                    "reads": [context],
                    "writes": [],
                },
            ]
        },
        "wikihow.open_resource.v1": {
            "branches": [
                {
                    "id": "bitcasa_opened",
                    "when": {
                        "all": [
                            {"eq": ["$args.resource_type", "website"]},
                            {"eq": ["$args.target", "https://my.bitcasa.com"]},
                        ]
                    },
                    "response": {"status": "success", "page": "Bitcasa login"},
                    "effects": [{"set": "$state.ui.bitcasa_open", "value": True}],
                    "reads": [],
                    "writes": ["$state.ui.bitcasa_open"],
                }
            ]
        },
        "wikihow.authenticate.v1": {
            "branches": [
                {
                    "id": "bitcasa_authenticated",
                    "when": {
                        "all": [
                            {"eq": ["$args.service", "bitcasa"]},
                            {"eq": ["$args.account_id", "user_99"]},
                            {"eq": ["$args.credential_type", "password"]},
                            {"eq": ["$args.credential_value", "secure_pass_123"]},
                            {"eq": ["$state.ui.bitcasa_open", True]},
                        ]
                    },
                    "response": {
                        "status": "success",
                        "authenticated_session_ref": "bitcasa_user_99_session",
                    },
                    "effects": [
                        {"set": f"{session}.authenticated", "value": True},
                        {"set": f"{session}.current_view", "value": "dashboard"},
                    ],
                    "reads": ["$state.ui.bitcasa_open"],
                    "writes": [
                        f"{session}.authenticated",
                        f"{session}.current_view",
                    ],
                }
            ]
        },
        "wikihow.read_state.v1": {
            "branches": [
                {
                    "id": "versions_view_opened",
                    "when": {
                        "all": [
                            {"eq": ["$args.view", "versions"]},
                            {"eq": ["$args.target_id", ""]},
                            {"eq": [f"{session}.authenticated", True]},
                        ]
                    },
                    "response": {
                        "status": "success",
                        "view_ref": "versions_view_user_99",
                        "session_record_id": "current",
                        "current_target_date": f"{session}.target_date",
                        "available_range": "historical",
                    },
                    "effects": [
                        {"set": f"{session}.current_view", "value": "versions"}
                    ],
                    "reads": [f"{session}.authenticated", f"{session}.target_date"],
                    "writes": [f"{session}.current_view"],
                }
            ]
        },
        "wikihow.update_record.v1": {
            "branches": [
                {
                    "id": "historical_snapshot_selected",
                    "when": {
                        "all": [
                            {"eq": ["$args.collection", "session"]},
                            {"eq": ["$args.record_id", "current"]},
                            {"eq": ["$args.patch", {"target_date": "2023-10-15"}]},
                            {"eq": ["$args.navigation_handle", "versions_view_user_99"]},
                            {"eq": [f"{session}.current_view", "versions"]},
                        ]
                    },
                    "response": {
                        "status": "success",
                        "snapshot_handle": "snapshot_2023_10_15_user_99_r7",
                        "target_date": "2023-10-15",
                        "recovery_policy": f"{context}.recovery_policy",
                    },
                    "effects": [
                        {"set": f"{session}.target_date", "value": "2023-10-15"},
                        {
                            "set": f"{session}.selected_snapshot_handle",
                            "value": "snapshot_2023_10_15_user_99_r7",
                        },
                    ],
                    "reads": [f"{session}.current_view", f"{context}.recovery_policy"],
                    "writes": [
                        f"{session}.target_date",
                        f"{session}.selected_snapshot_handle",
                    ],
                },
                {
                    "id": "current_workspace_selected",
                    "when": {
                        "all": [
                            {"eq": ["$args.collection", "session"]},
                            {"eq": ["$args.record_id", "current"]},
                            {"eq": ["$args.patch", {"target_date": "today"}]},
                            {
                                "eq": [
                                    "$args.navigation_handle",
                                    f"{selected}.artifact_id",
                                ]
                            },
                            {"eq": [f"{selected}.downloaded", True]},
                        ]
                    },
                    "response": {
                        "status": "success",
                        "current_workspace_ref": "workspace_today_user_99_r2",
                        "target_date": "today",
                        "workspace_revision": 2,
                    },
                    "effects": [
                        {"set": f"{session}.target_date", "value": "today"},
                        {"set": f"{session}.workspace_revision", "value": 2},
                        {"set": f"{session}.current_view", "value": "current_files"},
                    ],
                    "reads": [selected],
                    "writes": [
                        f"{session}.target_date",
                        f"{session}.workspace_revision",
                        f"{session}.current_view",
                    ],
                },
            ]
        },
        "wikihow.list_records.v1": {
            "branches": [
                {
                    "id": "historical_candidates_listed",
                    "when": {
                        "all": [
                            {"eq": ["$args.collection", "files"]},
                            {"eq": ["$args.query", "tax_return_2023.pdf"]},
                            {
                                "eq": [
                                    "$args.snapshot_handle",
                                    f"{session}.selected_snapshot_handle",
                                ]
                            },
                            {"eq": [f"{session}.target_date", "2023-10-15"]},
                        ]
                    },
                    "response": {
                        "status": "success",
                        "snapshot_handle": f"{session}.selected_snapshot_handle",
                        "recovery_policy": f"{context}.recovery_policy",
                        "candidates": candidates,
                    },
                    "effects": [],
                    "reads": [
                        "$state.revision_catalog",
                        f"{session}.selected_snapshot_handle",
                        f"{session}.target_date",
                        f"{context}.recovery_policy",
                    ],
                    "writes": [],
                }
            ]
        },
        "wikihow.download_resource.v1": {
            "branches": [
                download_success(
                    "latest_complete_revision_downloaded",
                    "latest_complete_on_date",
                    "rev_tax_2023_evening",
                    "artifact_tax_return_evening",
                ),
                download_success(
                    "earliest_complete_revision_downloaded",
                    "earliest_complete_on_date",
                    "rev_tax_2023_morning",
                    "artifact_tax_return_morning",
                ),
                {
                    "id": "revision_policy_mismatch",
                    "when": {
                        "all": [
                            {"eq": ["$args.destination", "local_downloads"]},
                            {
                                "eq": [
                                    "$args.snapshot_handle",
                                    f"{session}.selected_snapshot_handle",
                                ]
                            },
                            {
                                "any": [
                                    {
                                        "all": [
                                            {
                                                "eq": [
                                                    f"{context}.recovery_policy",
                                                    "latest_complete_on_date",
                                                ]
                                            },
                                            {
                                                "ne": [
                                                    "$args.resource_id",
                                                    "rev_tax_2023_evening",
                                                ]
                                            },
                                        ]
                                    },
                                    {
                                        "all": [
                                            {
                                                "eq": [
                                                    f"{context}.recovery_policy",
                                                    "earliest_complete_on_date",
                                                ]
                                            },
                                            {
                                                "ne": [
                                                    "$args.resource_id",
                                                    "rev_tax_2023_morning",
                                                ]
                                            },
                                        ]
                                    },
                                ]
                            },
                        ]
                    },
                    "response": {
                        "ok": False,
                        "error_code": "REVISION_POLICY_MISMATCH",
                        "message": (
                            "The selected revision does not satisfy the visible "
                            "recovery policy for this date."
                        ),
                    },
                    "effects": [{"set": "$state.selection_violation", "value": True}],
                    "reads": [
                        f"{session}.selected_snapshot_handle",
                        f"{context}.recovery_policy",
                    ],
                    "writes": ["$state.selection_violation"],
                },
            ]
        },
        "wikihow.create_upload_session.v1": {
            "branches": [
                {
                    "id": "current_upload_session_created",
                    "when": {
                        "all": [
                            {"eq": ["$args.provider", "bitcasa"]},
                            {"eq": ["$args.destination", "root"]},
                            {"eq": ["$args.workspace_handle", "workspace_today_user_99_r2"]},
                            {"eq": [f"{session}.target_date", "today"]},
                            {"eq": [f"{session}.workspace_revision", 2]},
                        ]
                    },
                    "response": {
                        "status": "success",
                        "session_id": "upload_session_user_99_r2",
                        "workspace_revision": 2,
                    },
                    "effects": [
                        {
                            "set": "$state.upload_sessions.current",
                            "value": {
                                "session_id": "upload_session_user_99_r2",
                                "workspace_ref": "workspace_today_user_99_r2",
                                "workspace_revision": 2,
                            },
                        }
                    ],
                    "reads": [session],
                    "writes": ["$state.upload_sessions.current"],
                }
            ]
        },
        "wikihow.upload_resource.v1": {
            "branches": [
                upload_success(
                    "evening_revision_restored",
                    "artifact_tax_return_evening",
                    "rev_tax_2023_evening",
                ),
                upload_success(
                    "morning_revision_restored",
                    "artifact_tax_return_morning",
                    "rev_tax_2023_morning",
                ),
            ]
        },
        "wikihow.workflow_outcome.observe.v1": {
            "branches": [
                {
                    "id": "recovery_outcome_visible",
                    "when": {
                        "eq": ["$args.workflow_context_handle", f"{context}.handle"]
                    },
                    "response": {
                        "outcome_state": {
                            "ui": "$state.ui",
                            "session": session,
                            "local_downloads": "$state.local_downloads",
                            "files": "$state.files",
                            "upload_sessions": "$state.upload_sessions",
                            "selection_violation": "$state.selection_violation",
                            "workflow_context": context,
                        }
                    },
                    "effects": [],
                    "reads": [
                        "$state.ui",
                        session,
                        "$state.local_downloads",
                        "$state.files",
                        "$state.upload_sessions",
                        "$state.selection_violation",
                        context,
                    ],
                    "writes": [],
                }
            ]
        },
    }

    bindings = copy.deepcopy(parent.bindings)
    by_name = {tool["name"]: tool for tool in bindings["tools"]}
    by_name["open_resource"]["parameters"]["properties"]["target"]["enum"] = [
        "https://my.bitcasa.com"
    ]
    authenticate = by_name["authenticate"]
    authenticate["parameters"]["properties"]["service"]["enum"] = ["bitcasa"]
    authenticate["parameters"]["properties"]["credential_type"]["enum"] = [
        "password"
    ]
    read = by_name["read_state"]
    read["parameters"]["properties"]["target_id"] = {
        "type": "string",
        "description": (
            "This versions view is global; omit this argument. The only accepted "
            "value is the empty global target."
        ),
        "enum": [""],
        "default": "",
    }
    read["parameters"]["required"] = ["view"]
    update = by_name["update_record"]
    update["parameters"]["properties"]["record_id"]["enum"] = ["current"]
    update["parameters"]["properties"]["navigation_ref"] = {
        "type": "string",
        "description": (
            "Visible view, snapshot, or downloaded artifact that grounds the "
            "requested navigation change."
        ),
    }
    update["parameters"]["required"].append("navigation_ref")
    update.setdefault("input_map", {})["navigation_ref"] = "navigation_handle"
    update.setdefault("provenance_required", []).append("navigation_ref")
    listing = by_name["list_records"]
    listing["parameters"]["properties"]["snapshot_ref"] = {
        "type": "string",
        "description": "Historical snapshot whose file revisions should be listed.",
    }
    listing["parameters"]["required"].append("snapshot_ref")
    listing.setdefault("input_map", {})["snapshot_ref"] = "snapshot_handle"
    listing.setdefault("provenance_required", []).append("snapshot_ref")
    download = by_name["download_resource"]
    download["parameters"]["properties"]["destination"]["enum"] = [
        "local_downloads"
    ]
    download["parameters"]["properties"]["snapshot_ref"] = {
        "type": "string",
        "description": "Snapshot that proves the selected revision's historical context.",
    }
    download["parameters"]["required"].append("snapshot_ref")
    download.setdefault("input_map", {})["snapshot_ref"] = "snapshot_handle"
    download.setdefault("provenance_required", []).append("snapshot_ref")
    create = by_name["create_upload_session"]
    create["parameters"]["properties"]["destination"]["enum"] = ["root"]
    create["parameters"]["properties"]["workspace_ref"] = {
        "type": "string",
        "description": "Current workspace identity returned after leaving history view.",
    }
    create["parameters"]["required"].append("workspace_ref")
    create.setdefault("input_map", {})["workspace_ref"] = "workspace_handle"
    create.setdefault("provenance_required", []).append("workspace_ref")
    verify = by_name["verify_resource"]
    bindings["tools"].remove(verify)

    selection_matches_policy = {
        "any": [
            {
                "all": [
                    {"eq": [f"{context}.recovery_policy", "latest_complete_on_date"]},
                    {"eq": [f"{selected}.source_revision_id", "rev_tax_2023_evening"]},
                    {
                        "eq": [
                            f"{restored}.source_revision_id",
                            "rev_tax_2023_evening",
                        ]
                    },
                    {
                        "eq": [
                            f"{restored}.restored_from_artifact_id",
                            "artifact_tax_return_evening",
                        ]
                    },
                ]
            },
            {
                "all": [
                    {"eq": [f"{context}.recovery_policy", "earliest_complete_on_date"]},
                    {"eq": [f"{selected}.source_revision_id", "rev_tax_2023_morning"]},
                    {
                        "eq": [
                            f"{restored}.source_revision_id",
                            "rev_tax_2023_morning",
                        ]
                    },
                    {
                        "eq": [
                            f"{restored}.restored_from_artifact_id",
                            "artifact_tax_return_morning",
                        ]
                    },
                ]
            },
        ]
    }
    authenticated = {
        "all": [
            {"eq": ["$state.ui.bitcasa_open", True]},
            {"eq": [f"{session}.authenticated", True]},
        ]
    }
    downloaded = {
        "all": [
            {"eq": [f"{selected}.downloaded", True]},
            {"eq": [f"{selected}.name", "tax_return_2023.pdf"]},
            {"eq": [f"{selected}.source_date", "2023-10-15"]},
            {"eq": ["$state.selection_violation", False]},
            selection_matches_policy,
        ]
    }
    restored_current = {
        "all": [
            {"eq": [f"{session}.target_date", "today"]},
            {"eq": [f"{session}.workspace_revision", 2]},
            {"eq": [f"{session}.current_view", "current_files"]},
            {"eq": [f"{restored}.name", "tax_return_2023.pdf"]},
            {"eq": [f"{restored}.status", "active"]},
            {"eq": [f"{restored}.source_date", "2023-10-15"]},
            {"eq": [f"{restored}.version_date", "today"]},
            selection_matches_policy,
        ]
    }
    goal = {"all": [authenticated, downloaded, restored_current]}
    contract = copy.deepcopy(parent.contract)
    contract["goal_predicates"] = [{"id": "recover_exact_revision", "predicate": goal}]
    contract["instruction_claims"] = [
        {
            "evidence_span": (
                "I need to recover a deleted file named 'tax_return_2023.pdf' from "
                "my Bitcasa account."
            ),
            "kind": "goal",
            "clause_ids": ["restored_to_current_files"],
        },
        {
            "evidence_span": "I know it was available on October 15, 2023.",
            "kind": "goal",
            "clause_ids": ["historical_revision_downloaded"],
        },
        {
            "evidence_span": (
                "Please log in using my account ID 'user_99' and password "
                "'secure_pass_123', find the file from that date, download it, and "
                "then restore it to my current active files."
            ),
            "kind": "goal",
            "clause_ids": [
                "authenticated_to_bitcasa",
                "historical_revision_downloaded",
                "restored_to_current_files",
            ],
        },
    ]
    contract["goal_clauses"] = [
        {
            "id": "authenticated_to_bitcasa",
            "predicate": authenticated,
            "transition_paths": [
                "$state.ui.bitcasa_open",
                f"{session}.authenticated",
            ],
            "evidence_paths": [
                "$state.ui.bitcasa_open",
                f"{session}.authenticated",
            ],
            "witness_tools": ["open_resource", "authenticate"],
        },
        {
            "id": "historical_revision_downloaded",
            "predicate": downloaded,
            "transition_paths": [selected],
            "evidence_paths": [selected, "$state.selection_violation", context],
            "witness_tools": ["download_resource"],
        },
        {
            "id": "restored_to_current_files",
            "predicate": restored_current,
            "transition_paths": [restored, session],
            "evidence_paths": [restored, session, selected, context],
            "witness_tools": ["update_record", "upload_resource"],
        },
    ]
    contract["invariants"] = [
        {
            "id": "revision_selection_policy_is_respected",
            "predicate": {"eq": ["$state.selection_violation", False]},
        },
        {
            "id": "session_remains_authenticated_after_login",
            "predicate": {
                "any": [
                    {"eq": [f"{session}.current_view", "login"]},
                    {"eq": [f"{session}.authenticated", True]},
                ]
            },
        },
    ]
    contract["requirements"] = {
        "semantic_recovery": False,
        "async_decision": False,
        "goal_grounded_verification": True,
        "temporal_provenance": {
            "links": [
                {
                    "consumer_tool": "list_records",
                    "argument": "snapshot_ref",
                    "producer_tool": "update_record",
                },
                {
                    "consumer_tool": "download_resource",
                    "argument": "resource_id",
                    "producer_tool": "list_records",
                },
                {
                    "consumer_tool": "download_resource",
                    "argument": "snapshot_ref",
                    "producer_tool": "list_records",
                },
                {
                    "consumer_tool": "update_record",
                    "argument": "navigation_ref",
                    "producer_tool": "download_resource",
                },
                {
                    "consumer_tool": "create_upload_session",
                    "argument": "workspace_ref",
                    "producer_tool": "update_record",
                },
                {
                    "consumer_tool": "upload_resource",
                    "argument": "session_id",
                    "producer_tool": "create_upload_session",
                },
                {
                    "consumer_tool": "upload_resource",
                    "argument": "resource_id",
                    "producer_tool": "download_resource",
                },
            ],
            "final_observation_tool": "observe_workflow_outcome",
            "final_paths": [
                session,
                selected,
                restored,
                context,
                "$state.selection_violation",
            ],
        },
    }
    contract["forbidden_shortcuts"] = [
        "select a same-name revision without applying the visible recovery policy",
        "download a revision without its historical snapshot identity",
        "reuse a historical workspace as the current upload destination",
        "claim restoration without observing source date and revision provenance",
    ]
    contract["expected_reasoning_features"] = [
        "delayed_handle_use",
        "observation_dependent_decision",
        "derived_object_dependency",
        "goal_grounded_verification",
        "alternative_plan",
        "temporal_provenance",
    ]
    contract["counterfactual_axes"] = [
        {
            "state_path": f"{context}.recovery_policy",
            "variants": [
                "latest_complete_on_date",
                "earliest_complete_on_date",
            ],
        }
    ]

    def action(tool: str, **arguments: object) -> dict:
        return {"tool": tool, "arguments": arguments}

    prefix = [
        action("observe_workflow_context"),
        action(
            "open_resource",
            resource_type="website",
            target="https://my.bitcasa.com",
        ),
        action(
            "authenticate",
            service="bitcasa",
            account_id="user_99",
            credential_type="password",
            credential_value="secure_pass_123",
        ),
        action("read_state", view="versions"),
        action(
            "update_record",
            collection="session",
            record_id="current",
            patch={"target_date": "2023-10-15"},
            navigation_ref="versions_view_user_99",
        ),
        action(
            "list_records",
            collection="files",
            query="tax_return_2023.pdf",
            snapshot_ref="snapshot_2023_10_15_user_99_r7",
        ),
    ]

    def finish(revision_id: str, artifact_id: str) -> list[dict]:
        return [
            action(
                "download_resource",
                resource_id=revision_id,
                destination="local_downloads",
                snapshot_ref="snapshot_2023_10_15_user_99_r7",
            ),
            action(
                "update_record",
                collection="session",
                record_id="current",
                patch={"target_date": "today"},
                navigation_ref=artifact_id,
            ),
            action(
                "create_upload_session",
                provider="bitcasa",
                destination="root",
                workspace_ref="workspace_today_user_99_r2",
            ),
            action(
                "upload_resource",
                session_id="upload_session_user_99_r2",
                resource_id=artifact_id,
            ),
            action(
                "observe_workflow_outcome",
                workflow_context_handle="workflow_context_wikihow_computer_002787",
            ),
        ]

    reference_plan = {
        "actions": prefix
        + finish("rev_tax_2023_evening", "artifact_tax_return_evening"),
        "counterfactuals": [
            {
                "id": "earliest_complete_policy_changes_revision_choice",
                "state_overrides": {
                    f"{context}.recovery_policy": "earliest_complete_on_date"
                },
                "actions": prefix
                + finish("rev_tax_2023_morning", "artifact_tax_return_morning"),
            }
        ],
    }

    bundle = TaskBundle(
        root=parent.root,
        manifest={
            **copy.deepcopy(parent.manifest),
            "task_id": "wikihow_computer_002787__strict_temporal_provenance",
            "seed_family": "wikihow_strict_temporal_provenance_v1",
            "lineage": {
                "root_task_id": parent.task_id,
                "parent_task_id": parent.task_id,
                "generation": 1,
                "operators": ["temporal_snapshot_provenance_v1"],
            },
        },
        instruction=instruction,
        contract=contract,
        environment={
            "runtime_version": "causal-runtime-v1",
            "initial_state": initial_state,
            "capabilities": capabilities,
        },
        bindings=bindings,
        reference_plan=reference_plan,
    )
    return totalize_public_capabilities(bundle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    bundle = build_bundle(args.source)
    alignment = validate_goal_alignment(bundle)
    if not alignment["valid"]:
        raise SystemExit("goal alignment failed: " + "; ".join(alignment["errors"]))
    path = materialize_candidate(
        args.output_dir,
        task_id=bundle.task_id,
        contract=bundle.contract,
        candidate={
            "instruction": bundle.instruction,
            "environment": bundle.environment,
            "bindings": bundle.bindings,
            "reference_plan": bundle.reference_plan,
        },
        lineage=bundle.manifest["lineage"],
        manifest_metadata=manifest_metadata(bundle),
    )
    print(path)


if __name__ == "__main__":
    main()
