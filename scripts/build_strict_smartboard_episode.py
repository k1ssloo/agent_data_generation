#!/usr/bin/env python3
"""Build one fully aligned, adaptive WikiHow smartboard task lineage."""

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
from task_factory.evolve import evolve_once
from task_factory.materialize import materialize_candidate
from task_factory.operators.base import manifest_metadata


DEFAULT_SOURCE = PROJECT_ROOT / (
    "outputs/task_first/wikihow_compile200_context_v5/children/"
    "wikihow_computer_001881__task_first__g1_workflow_commit_audit"
)


def binding(bundle: object, name: str) -> dict:
    for item in bundle.bindings["tools"]:
        if item["name"] == name:
            return item
    raise ValueError(f"missing tool binding {name!r}")


def strict_parent(source: Path):
    parent = load_task_bundle(source)
    contract = copy.deepcopy(parent.contract)
    environment = copy.deepcopy(parent.environment)
    reference_plan = copy.deepcopy(parent.reference_plan)

    calibration_path = "$state.smartboard_status.calibration"
    session_path = "$state.notebook_sessions"
    audit_path = "$state.workflow_commit_audit.committed"
    calibration_goal = {
        "eq": [
            calibration_path,
            {"status": "calibrated", "last_calibrated": "just now"},
        ]
    }
    session_goal = {
        "all": [
            {
                "eq": [
                    f"{session_path}.session_001.title",
                    "Morning Brainstorm",
                ]
            },
            {
                "eq": [
                    f"{session_path}.session_001.content",
                    "Discussing project milestones",
                ]
            },
            {"eq": [f"{session_path}.session_001.status", "saved"]},
        ]
    }
    audit_goal = {
        "all": [
            {"eq": [audit_path, True]},
            {
                "eq": [
                    "$state.workflow_commit_audit.approved_policy",
                    "$state.workflow_commit_audit.current_policy",
                ]
            },
            {
                "eq": [
                    "$state.workflow_commit_audit.committed_policy",
                    "$state.workflow_commit_audit.current_policy",
                ]
            },
            {
                "eq": [
                    "$state.workflow_commit_audit.target",
                    "session_001",
                ]
            },
            {
                "eq": [
                    "$state.workflow_commit_audit.committed_target",
                    "session_001",
                ]
            },
        ]
    }
    contract["goal_predicates"] = [
        {
            "id": "complete_user_goal",
            "predicate": {
                "all": [calibration_goal, session_goal, audit_goal]
            },
        }
    ]
    contract["instruction_claims"] = [
        {
            "evidence_span": (
                "I just turned on the smartboard smartboard_01 and opened the "
                "Smart Notebook workspace."
            ),
            "kind": "context",
            "clause_ids": [],
        },
        {
            "evidence_span": (
                "I need to check if it needs calibration, calibrate it if necessary, "
                "and then create a new notes session titled 'Morning Brainstorm' with "
                "the content 'Discussing project milestones'."
            ),
            "kind": "goal",
            "clause_ids": ["calibrated_if_needed", "notes_created"],
        },
        {
            "evidence_span": "Please make sure it is saved successfully.",
            "kind": "goal",
            "clause_ids": ["notes_created"],
        },
        {
            "evidence_span": (
                "The consequential workflow_commit action must satisfy the current "
                "target-specific authorization policy."
            ),
            "kind": "synthetic_constraint",
            "clause_ids": ["authorized_commit"],
        },
    ]
    contract["goal_clauses"] = [
        {
            "id": "calibrated_if_needed",
            "predicate": calibration_goal,
            "transition_paths": [calibration_path],
            "evidence_paths": [calibration_path],
            "witness_tools": ["poll_job"],
        },
        {
            "id": "notes_created",
            "predicate": session_goal,
            "transition_paths": [session_path],
            "evidence_paths": [
                f"{session_path}.session_001.title",
                f"{session_path}.session_001.content",
                f"{session_path}.session_001.status",
            ],
            "witness_tools": ["create_record", "update_record"],
        },
        {
            "id": "authorized_commit",
            "predicate": audit_goal,
            "transition_paths": [
                audit_path,
                "$state.workflow_commit_audit.current_policy",
                "$state.workflow_commit_audit.approved_policy",
                "$state.workflow_commit_audit.committed_policy",
                "$state.workflow_commit_audit.target",
                "$state.workflow_commit_audit.committed_target",
            ],
            "evidence_paths": [
                audit_path,
                "$state.workflow_commit_audit.current_policy",
                "$state.workflow_commit_audit.approved_policy",
                "$state.workflow_commit_audit.committed_policy",
                "$state.workflow_commit_audit.target",
                "$state.workflow_commit_audit.committed_target",
            ],
            "witness_tools": [
                "inspect_workflow_commit_policy",
                "approve_workflow_commit_action",
                "update_record",
            ],
        },
    ]
    contract["counterfactual_axes"] = [
        {
            "state_path": calibration_path,
            "variants": [
                {"status": "needs_calibration", "last_calibrated": "never"},
                {"status": "calibrated", "last_calibrated": "just now"},
            ],
        }
    ]
    contract["expected_reasoning_features"] = sorted(
        set(contract["expected_reasoning_features"])
        | {"observation_dependent_decision", "alternative_plan"}
    )
    contract["requirements"]["async_decision"] = False
    contract["invariants"] = [
        {
            "id": "calibration_timestamp_consistent",
            "predicate": {
                "any": [
                    {"ne": [f"{calibration_path}.status", "calibrated"]},
                    {"eq": [f"{calibration_path}.last_calibrated", "just now"]},
                ]
            },
        },
        {
            "id": "saved_session_has_requested_content",
            "predicate": {
                "any": [
                    {"not_exists": "$state.notebook_sessions.session_001"},
                    {
                        "all": [
                            {
                                "eq": [
                                    "$state.notebook_sessions.session_001.title",
                                    "Morning Brainstorm",
                                ]
                            },
                            {
                                "any": [
                                    {
                                        "ne": [
                                            "$state.notebook_sessions.session_001.status",
                                            "saved",
                                        ]
                                    },
                                    {
                                        "eq": [
                                            "$state.notebook_sessions.session_001.content",
                                            "Discussing project milestones",
                                        ]
                                    },
                                ]
                            },
                        ]
                    },
                ]
            },
        },
    ]

    environment["initial_state"]["workflow_commit_audit"].setdefault(
        "committed_target", ""
    )
    audit_state = environment["initial_state"]["workflow_commit_audit"]
    audit_state.update(
        {
            "current_policy": "policy_workflow_commit_audit_1",
            "approved_policy": "",
            "committed_policy": "",
        }
    )
    approve_capability = environment["capabilities"][
        "audit.approve.workflow_commit.v1"
    ]
    approve_branch = approve_capability["branches"][0]
    approve_branch["when"] = {
        "all": [
            copy.deepcopy(approve_branch.get("when", True)),
            {
                "eq": [
                    "$args.policy_handle",
                    "$state.workflow_commit_audit.current_policy",
                ]
            },
        ]
    }
    approve_branch.setdefault("effects", []).append(
        {
            "set": "$state.workflow_commit_audit.approved_policy",
            "value": "$args.policy_handle",
        }
    )
    approve_branch.setdefault("reads", []).append(
        "$state.workflow_commit_audit.current_policy"
    )
    approve_branch.setdefault("writes", []).append(
        "$state.workflow_commit_audit.approved_policy"
    )
    parent.manifest.setdefault("evolution_hooks", {}).setdefault(
        "audit_checkpoint", {}
    )["target_revision_path"] = "$state.notebook_sessions.session_001.revision"
    outcome_capability = environment["capabilities"][
        "wikihow.workflow_outcome.observe.v1"
    ]
    outcome_branch = next(
        branch
        for branch in outcome_capability["branches"]
        if branch.get("id") == "current_workflow_outcome_visible"
    )
    outcome_branch.setdefault("response", {})["audit_approved_target"] = (
        "$state.workflow_commit_audit.target"
    )
    outcome_branch.setdefault("response", {})["audit_committed_target"] = (
        "$state.workflow_commit_audit.committed_target"
    )
    outcome_branch.setdefault("response", {})["audit_current_policy"] = (
        "$state.workflow_commit_audit.current_policy"
    )
    outcome_branch.setdefault("response", {})["audit_approved_policy"] = (
        "$state.workflow_commit_audit.approved_policy"
    )
    outcome_branch.setdefault("response", {})["audit_committed_policy"] = (
        "$state.workflow_commit_audit.committed_policy"
    )
    outcome_branch.setdefault("reads", []).extend(
        [
            "$state.workflow_commit_audit.target",
            "$state.workflow_commit_audit.committed_target",
            "$state.workflow_commit_audit.current_policy",
            "$state.workflow_commit_audit.approved_policy",
            "$state.workflow_commit_audit.committed_policy",
        ]
    )

    read_capability = environment["capabilities"]["wikihow.read_state.v1"]
    read_branch = read_capability["branches"][0]
    read_branch["when"] = {
        "all": [
            {"eq": ["$args.view", "calibration_status"]},
            {
                "any": [
                    {"eq": ["$args.target_id", ""]},
                    {"eq": ["$args.target_id", "smartboard_01"]},
                ]
            },
        ]
    }
    read_branch["response"]["calibration"] = calibration_path
    read_branch["response"]["calibration_evidence_handle"] = (
        "calibration_evidence_smartboard_01"
    )
    read_branch["reads"] = [calibration_path]
    read_binding = next(
        item for item in parent.bindings["tools"] if item["name"] == "read_state"
    )
    target_schema = read_binding["parameters"]["properties"]["target_id"]
    target_schema["default"] = ""
    read_binding["parameters"]["required"] = [
        name
        for name in read_binding["parameters"].get("required", [])
        if name != "target_id"
    ]

    start_capability = environment["capabilities"]["wikihow.start_job.v1"]
    start_branch = start_capability["branches"][0]
    original_when = copy.deepcopy(start_branch["when"])
    start_branch["when"] = {
        "all": [
            original_when,
            {"eq": [f"{calibration_path}.status", "needs_calibration"]},
            {
                "eq": [
                    "$args.calibration_evidence_handle",
                    "calibration_evidence_smartboard_01",
                ]
            },
        ]
    }
    start_branch["reads"] = sorted(
        set(start_branch.get("reads", [])) | {f"{calibration_path}.status"}
    )
    start_binding = next(
        item for item in parent.bindings["tools"] if item["name"] == "start_job"
    )
    start_binding["parameters"]["required"] = [
        name
        for name in start_binding["parameters"].get("required", [])
        if name != "options"
    ]
    start_binding["parameters"]["properties"]["calibration_evidence_ref"] = {
        "type": "string",
        "description": "Evidence from the observed calibration-status view.",
    }
    start_binding["parameters"]["required"].append("calibration_evidence_ref")
    start_binding.setdefault("input_map", {})["calibration_evidence_ref"] = (
        "calibration_evidence_handle"
    )
    start_binding.setdefault("provenance_required", []).append(
        "calibration_evidence_ref"
    )
    start_branch["effects"] = [
        {
            "set": "$state.jobs.job_cal_992",
            "value": {"status": "running", "type": "calibration"},
        }
    ]
    start_branch["writes"] = ["$state.jobs.job_cal_992"]

    poll_capability = environment["capabilities"]["wikihow.poll_job.v1"]
    poll_branch = poll_capability["branches"][0]
    poll_branch["when"] = {
        "all": [
            copy.deepcopy(poll_branch["when"]),
            {"eq": ["$state.jobs[$args.job_id].status", "running"]},
        ]
    }
    poll_branch["response"] = {
        "status": "success",
        "job_status": "completed",
        "calibration": {
            "status": "calibrated",
            "last_calibrated": "just now",
        },
    }
    poll_branch["effects"] = [
        {
            "set": "$state.jobs.job_cal_992.status",
            "value": "completed",
        },
        {
            "set": calibration_path,
            "value": {
                "status": "calibrated",
                "last_calibrated": "just now",
            },
        },
    ]
    poll_branch["reads"] = ["$state.jobs[$args.job_id].status"]
    poll_branch["writes"] = [
        "$state.jobs.job_cal_992.status",
        calibration_path,
    ]

    create_binding = next(
        item for item in parent.bindings["tools"] if item["name"] == "create_record"
    )
    create_binding["parameters"]["properties"]["fields"] = {
        "type": "object",
        "description": "Initial note fields. Content may be supplied when the draft is created.",
        "properties": {
            "title": {
                "type": "string",
                "description": "User-requested notes session title.",
            },
            "content": {
                "type": "string",
                "description": "Optional initial notes content.",
            },
        },
        "required": ["title"],
        "additionalProperties": False,
    }
    create_capability = environment["capabilities"]["wikihow.create_record.v1"]
    title_only_branch = create_capability["branches"][0]
    title_only_branch.setdefault("response", {}).setdefault("fields", {})[
        "revision"
    ] = 1
    title_only_value = title_only_branch["effects"][0].get("value", {})
    if isinstance(title_only_value, dict):
        title_only_value.setdefault("session_001", {})["revision"] = 1
    create_with_content = copy.deepcopy(title_only_branch)
    create_with_content["id"] = "create_notes_draft_with_content"
    create_with_content["when"] = {
        "all": [
            {"eq": ["$args.collection", "notebook_sessions"]},
            {
                "eq": [
                    "$args.fields",
                    {
                        "title": "Morning Brainstorm",
                        "content": "Discussing project milestones",
                    },
                ]
            },
        ]
    }
    create_with_content["response"] = {
        "status": "success",
        "record_id": "session_001",
        "fields": {
            "title": "Morning Brainstorm",
            "content": "Discussing project milestones",
            "status": "draft",
            "revision": 1,
        },
    }
    create_with_content["effects"] = [
        {
            "set": "$state.notebook_sessions.session_001",
            "value": {
                "title": "Morning Brainstorm",
                "content": "Discussing project milestones",
                "status": "draft",
                "revision": 1,
            },
        }
    ]
    create_capability["branches"].insert(0, create_with_content)

    update_binding = next(
        item for item in parent.bindings["tools"] if item["name"] == "update_record"
    )
    update_binding["parameters"]["properties"]["patch"] = {
        "type": "object",
        "description": "Exact requested content and final saved status for the draft.",
        "properties": {
            "content": {
                "type": "string",
                "description": "User-requested notes content.",
            },
            "status": {
                "type": "string",
                "description": "Final record status.",
                "enum": ["saved"],
            },
        },
        "required": ["content", "status"],
        "additionalProperties": False,
    }
    update_capability = environment["capabilities"]["wikihow.update_record.v1"]
    for branch in update_capability["branches"]:
        if branch.get("id") != "observed_06":
            continue
        branch["effects"] = [
            {
                "set": "$state.notebook_sessions.session_001.content",
                "value": "Discussing project milestones",
            },
            {
                "set": "$state.notebook_sessions.session_001.status",
                "value": "saved",
            },
            {"set": "$state.workflow_commit_audit.committed", "value": True},
            {
                "set": "$state.workflow_commit_audit.committed_policy",
                "value": "$state.workflow_commit_audit.current_policy",
            },
            {
                "set": "$state.workflow_commit_audit.committed_target",
                "value": "$args.record_id",
            },
        ]
        branch["writes"] = [
            "$state.notebook_sessions.session_001.content",
            "$state.notebook_sessions.session_001.status",
            "$state.workflow_commit_audit.committed",
            "$state.workflow_commit_audit.committed_policy",
            "$state.workflow_commit_audit.committed_target",
        ]

    verify_capability = environment["capabilities"]["wikihow.verify_resource.v1"]
    verify_branch = verify_capability["branches"][0]
    verify_branch["when"] = {
        "all": [
            copy.deepcopy(verify_branch["when"]),
            {
                "eq": [
                    "$state.notebook_sessions[$args.resource_id].title",
                    "Morning Brainstorm",
                ]
            },
            {
                "eq": [
                    "$state.notebook_sessions[$args.resource_id].content",
                    "Discussing project milestones",
                ]
            },
            {
                "eq": [
                    "$state.notebook_sessions[$args.resource_id].status",
                    "saved",
                ]
            },
        ]
    }
    verify_branch["reads"] = sorted(
        set(verify_branch.get("reads", []))
        | {
            "$state.notebook_sessions[$args.resource_id].title",
            "$state.notebook_sessions[$args.resource_id].content",
            "$state.notebook_sessions[$args.resource_id].status",
        }
    )

    baseline_actions = reference_plan["actions"]
    baseline_actions = [
        copy.deepcopy(action)
        for action in baseline_actions
        if action["tool"] not in {"open_resource", "verify_resource"}
    ]
    for action in baseline_actions:
        if action["tool"] == "start_job":
            action.setdefault("arguments", {})["calibration_evidence_ref"] = (
                "calibration_evidence_smartboard_01"
            )
    reference_plan["actions"] = baseline_actions
    adapted_actions = [
        copy.deepcopy(action)
        for action in baseline_actions
        if not (
            action["tool"] == "start_job"
            and action.get("arguments", {}).get("job_type") == "calibration"
        )
        and action["tool"] != "poll_job"
    ]
    reference_plan["counterfactuals"] = [
        {
            "id": "already_calibrated_skip_calibration",
            "state_overrides": {
                calibration_path: {
                    "status": "calibrated",
                    "last_calibrated": "just now",
                }
            },
            "actions": adapted_actions,
        }
    ]

    return type(parent)(
        root=parent.root,
        manifest={
            **copy.deepcopy(parent.manifest),
            "task_id": "wikihow_smartboard_strict_parent",
            "seed_family": "wikihow_goal_aligned_v1",
            "lineage": {
                "root_task_id": "wikihow_smartboard_strict_parent",
                "generation": 0,
                "operators": ["audit_checkpoint_v1"],
            },
        },
        instruction=parent.instruction,
        contract=contract,
        environment=environment,
        bindings=copy.deepcopy(parent.bindings),
        reference_plan=reference_plan,
    )


def write_bundle(bundle, output: Path) -> Path:
    return materialize_candidate(
        output,
        task_id=bundle.task_id,
        contract=bundle.contract,
        candidate={
            "instruction": bundle.instruction,
            "environment": bundle.environment,
            "bindings": bundle.bindings,
            "reference_plan": bundle.reference_plan,
        },
        lineage=bundle.manifest.get("lineage", {}),
        manifest_metadata=manifest_metadata(bundle),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    parent = strict_parent(args.source)
    parent_alignment = validate_goal_alignment(parent)
    if not parent_alignment["valid"]:
        raise SystemExit("strict parent alignment failed: " + "; ".join(parent_alignment["errors"]))
    parent_path = write_bundle(parent, args.output_dir)

    routed = evolve_once(parent, "execution_route_branch_v1")
    if not routed.report["accepted"]:
        raise SystemExit("route child rejected: " + "; ".join(routed.report["errors"]))
    routed_alignment = validate_goal_alignment(routed.product.bundle)
    if not routed_alignment["valid"]:
        raise SystemExit("route child alignment failed: " + "; ".join(routed_alignment["errors"]))
    routed_path = write_bundle(routed.product.bundle, args.output_dir)

    recovered = evolve_once(
        routed.product.bundle, "semantic_failure_recovery_v1"
    )
    if not recovered.report["accepted"]:
        raise SystemExit(
            "recovery child rejected: " + "; ".join(recovered.report["errors"])
        )
    recovered_bundle = totalize_public_capabilities(recovered.product.bundle)
    recovered_alignment = validate_goal_alignment(recovered_bundle)
    if not recovered_alignment["valid"]:
        raise SystemExit(
            "recovery child alignment failed: "
            + "; ".join(recovered_alignment["errors"])
        )
    recovered_path = write_bundle(recovered_bundle, args.output_dir)
    print(parent_path)
    print(routed_path)
    print(recovered_path)


if __name__ == "__main__":
    main()
