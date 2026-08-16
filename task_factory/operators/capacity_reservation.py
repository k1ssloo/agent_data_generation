"""Add observation-dependent distribution-capacity reservation."""

from __future__ import annotations

import copy
from dataclasses import replace

from .base import EvolutionProduct, action_index, append_goal_condition, capability, clone_bundle
from task_factory.bundle import TaskBundle, validate_bundle


class CapacityReservationOperator:
    operator_id = "capacity_reservation_branch_v1"

    def apply(self, parent: TaskBundle, *, generation: int) -> EvolutionProduct:
        if self.operator_id in parent.manifest.get("lineage", {}).get("operators", []):
            raise ValueError(f"{self.operator_id} cannot be applied twice in one lineage")
        child = clone_bundle(
            parent,
            task_id=f"{parent.task_id}__g{generation}_capacity",
            operator_id=self.operator_id,
            generation=generation,
        )
        state = child.environment["initial_state"]
        state["capacity"] = {
            "mode": "staged_only",
            "available_slots": 1,
            "reserved": False,
            "cohort": "",
        }
        state["release"]["cohort"] = ""
        capabilities = child.environment["capabilities"]
        capabilities["capacity.inspect.v1"] = {
            "branches": [
                {
                    "id": "full_beta_available",
                    "when": {"eq": ["$state.capacity.mode", "full_beta"]},
                    "response": {
                        "capacity_report_handle": "capacity_full_1",
                        "recommended_cohort": "full_beta",
                        "available_slots": "$state.capacity.available_slots",
                    },
                    "reads": ["$state.capacity.mode", "$state.capacity.available_slots"],
                    "observes": ["$state.capacity.mode"],
                },
                {
                    "id": "staged_beta_only",
                    "when": {"eq": ["$state.capacity.mode", "staged_only"]},
                    "response": {
                        "capacity_report_handle": "capacity_staged_1",
                        "recommended_cohort": "staged_beta",
                        "available_slots": "$state.capacity.available_slots",
                    },
                    "reads": ["$state.capacity.mode", "$state.capacity.available_slots"],
                    "observes": ["$state.capacity.mode"],
                },
            ]
        }
        capabilities["capacity.reserve.full.v1"] = {
            "branches": [
                {
                    "id": "reserve_full_beta",
                    "when": {"all": [
                        {"eq": ["$state.capacity.mode", "full_beta"]},
                        {"eq": ["$args.capacity_report_handle", "capacity_full_1"]},
                        {"eq": ["$args.candidate_handle", "candidate_340b"]},
                        {"gt": ["$state.capacity.available_slots", 0]},
                    ]},
                    "response": {"reservation_handle": "reservation_full_1", "cohort": "full_beta"},
                    "effects": [
                        {"set": "$state.capacity.reserved", "value": True},
                        {"set": "$state.capacity.cohort", "value": "full_beta"},
                        {"increment": "$state.capacity.available_slots", "by": -1},
                    ],
                    "reads": ["$state.capacity.mode", "$state.capacity.available_slots"],
                }
            ]
        }
        capabilities["capacity.reserve.staged.v1"] = {
            "branches": [
                {
                    "id": "reserve_staged_beta",
                    "when": {"all": [
                        {"eq": ["$state.capacity.mode", "staged_only"]},
                        {"eq": ["$args.capacity_report_handle", "capacity_staged_1"]},
                        {"eq": ["$args.candidate_handle", "candidate_340b"]},
                        {"gt": ["$state.capacity.available_slots", 0]},
                    ]},
                    "response": {"reservation_handle": "reservation_staged_1", "cohort": "staged_beta"},
                    "effects": [
                        {"set": "$state.capacity.reserved", "value": True},
                        {"set": "$state.capacity.cohort", "value": "staged_beta"},
                        {"increment": "$state.capacity.available_slots", "by": -1},
                    ],
                    "reads": ["$state.capacity.mode", "$state.capacity.available_slots"],
                }
            ]
        }
        child.bindings["tools"].extend(
            [
                {
                    "name": "inspect_release_capacity",
                    "description": "Inspect current beta distribution capacity and the permitted cohort.",
                    "capability_id": "capacity.inspect.v1",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
                {
                    "name": "reserve_staged_beta_slot",
                    "description": "Reserve observed staged-beta capacity for a prepared candidate.",
                    "capability_id": "capacity.reserve.staged.v1",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "candidate_ref": {"type": "string"},
                            "capacity_report_ref": {"type": "string"},
                        },
                        "required": ["candidate_ref", "capacity_report_ref"],
                    },
                    "input_map": {
                        "candidate_ref": "candidate_handle",
                        "capacity_report_ref": "capacity_report_handle",
                    },
                    "provenance_required": ["candidate_ref", "capacity_report_ref"],
                },
                {
                    "name": "reserve_full_beta_slot",
                    "description": "Reserve observed full-beta capacity for a prepared candidate.",
                    "capability_id": "capacity.reserve.full.v1",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "candidate_ref": {"type": "string"},
                            "capacity_report_ref": {"type": "string"},
                        },
                        "required": ["candidate_ref", "capacity_report_ref"],
                    },
                    "input_map": {
                        "candidate_ref": "candidate_handle",
                        "capacity_report_ref": "capacity_report_handle",
                    },
                    "provenance_required": ["candidate_ref", "capacity_report_ref"],
                },
            ]
        )

        publish_tool = next(tool for tool in child.bindings["tools"] if tool["name"] == "publish_candidate")
        publish_tool["parameters"]["properties"]["reservation_ref"] = {"type": "string"}
        publish_tool["parameters"]["required"].append("reservation_ref")
        publish_tool.setdefault("input_map", {})["reservation_ref"] = "reservation_handle"
        publish_tool.setdefault("provenance_required", []).append("reservation_ref")
        publish = capability(child, "release.publish.v1")["branches"][0]
        publish["when"] = {"all": [
            publish["when"],
            {"in": ["$args.reservation_handle", ["reservation_full_1", "reservation_staged_1"]]},
            {"eq": ["$state.capacity.reserved", True]},
        ]}
        publish["effects"].append(
            {"set": "$state.release.cohort", "value": "$state.capacity.cohort"}
        )
        publish.setdefault("reads", []).extend(
            ["$state.capacity.reserved", "$state.capacity.cohort"]
        )
        inspect = capability(child, "release.inspect.v1")["branches"][0]
        inspect["response"]["cohort"] = "$state.release.cohort"
        inspect["reads"].append("$state.release.cohort")
        append_goal_condition(
            child,
            {"in": ["$state.release.cohort", ["staged_beta", "full_beta"]]},
        )
        child.contract["forbidden_shortcuts"].append(
            "publish without observing and reserving current distribution capacity"
        )
        child.contract["counterfactual_axes"].append(
            {"state_path": "$state.capacity.mode", "variants": ["staged_only", "full_beta"]}
        )
        child = replace(
            child,
            instruction=child.instruction.rstrip()
            + " Distribution must remain within current capacity and target a policy-compliant beta cohort.\n",
        )

        actions = child.reference_plan["actions"]
        publish_index = action_index(child, "publish_candidate")
        actions.insert(publish_index, {"tool": "inspect_release_capacity", "arguments": {}})
        actions.insert(
            publish_index + 1,
            {
                "tool": "reserve_staged_beta_slot",
                "arguments": {
                    "candidate_ref": "candidate_340b",
                    "capacity_report_ref": "capacity_staged_1",
                },
            },
        )
        actions[publish_index + 2]["arguments"]["reservation_ref"] = "reservation_staged_1"
        full_actions = copy.deepcopy(actions)
        for action in full_actions:
            if action["tool"] == "reserve_staged_beta_slot":
                action["tool"] = "reserve_full_beta_slot"
                action["arguments"]["capacity_report_ref"] = "capacity_full_1"
            elif action["tool"] == "publish_candidate":
                action["arguments"]["reservation_ref"] = "reservation_full_1"
        child.reference_plan.setdefault("counterfactuals", []).append(
            {
                "id": "full_beta_capacity",
                "state_overrides": {"$state.capacity.mode": "full_beta"},
                "actions": full_actions,
            }
        )

        errors = validate_bundle(child)
        if errors:
            raise ValueError("capacity operator produced invalid bundle: " + "; ".join(errors))
        return EvolutionProduct(
            bundle=child,
            patch={
                "operator_id": self.operator_id,
                "semantic_changes": [
                    "distribution capacity becomes a hidden observation",
                    "capacity observation selects full or staged beta strategy",
                    "publication consumes a reservation derived from the selected branch",
                    "the final goal verifies the chosen cohort",
                ],
                "added_goal_paths": ["$state.release.cohort"],
            },
        )
