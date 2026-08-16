"""Couple artifact construction to a policy revision change."""

from __future__ import annotations

from dataclasses import replace

from .base import (
    EvolutionProduct,
    action_index,
    append_goal_condition,
    capability,
    clone_bundle,
    tool_binding,
)
from task_factory.bundle import TaskBundle, validate_bundle


class PolicyFreshnessOperator:
    operator_id = "policy_freshness_coupling_v1"

    def apply(self, parent: TaskBundle, *, generation: int) -> EvolutionProduct:
        if self.operator_id in parent.manifest.get("lineage", {}).get("operators", []):
            raise ValueError(f"{self.operator_id} cannot be applied twice in one lineage")
        child = clone_bundle(
            parent,
            task_id=f"{parent.task_id}__g{generation}_policy_freshness",
            operator_id=self.operator_id,
            generation=generation,
        )
        state = child.environment["initial_state"]
        state["policy"].update(
            {"revision": 1, "signing_profile": "sign_profile_7", "rotation_on_build": True}
        )
        state["artifact"]["signing_profile"] = ""
        state["release"]["policy_revision"] = 0

        policy = capability(child, "policy.read.v1")
        original_when = policy["branches"][0]["when"]
        policy["branches"] = [
            {
                "id": "policy_revision_1",
                "when": {"all": [original_when, {"eq": ["$state.policy.revision", 1]}]},
                "response": {
                    "policy_handle": "policy_beta_7",
                    "signing_profile_handle": "sign_profile_7",
                    "policy_revision": 1,
                    "minimum_coverage": 85,
                    "maximum_critical": 0,
                    "signature_required": True,
                    "maximum_cost": 50,
                },
                "reads": ["$state.policy"],
            },
            {
                "id": "policy_revision_2",
                "when": {"all": [original_when, {"eq": ["$state.policy.revision", 2]}]},
                "response": {
                    "policy_handle": "policy_beta_8",
                    "signing_profile_handle": "sign_profile_8",
                    "policy_revision": 2,
                    "minimum_coverage": 85,
                    "maximum_critical": 0,
                    "signature_required": True,
                    "maximum_cost": 50,
                },
                "reads": ["$state.policy"],
            },
        ]

        build = capability(child, "build.signed.v1")
        original_build_when = build["branches"][0]["when"]
        build["branches"] = [
            {
                "id": "policy_rotated_after_build",
                "when": {"all": [
                    original_build_when,
                    {"eq": ["$state.policy.revision", 1]},
                    {"eq": ["$args.signing_profile_handle", "sign_profile_7"]},
                ]},
                "response": {
                    "artifact_handle": "artifact_stale_91",
                    "signed": True,
                    "error_code": "POLICY_ROTATED_AFTER_BUILD",
                },
                "effects": [
                    {"set": "$state.artifact.handle", "value": "artifact_stale_91"},
                    {"set": "$state.artifact.signed", "value": True},
                    {"set": "$state.artifact.signing_profile", "value": "sign_profile_7"},
                    {"increment": "$state.cost.total", "by": 8},
                    {"set": "$state.policy.revision", "value": 2},
                    {"set": "$state.policy.handle", "value": "policy_beta_8"},
                    {"set": "$state.policy.signing_profile", "value": "sign_profile_8"},
                ],
                "reads": ["$state.quality.coverage", "$state.policy.revision"],
                "writes": [
                    "$state.artifact.handle",
                    "$state.artifact.signed",
                    "$state.artifact.signing_profile",
                    "$state.policy.revision",
                    "$state.policy.handle",
                    "$state.policy.signing_profile",
                ],
            },
            {
                "id": "rebuilt_with_current_policy",
                "when": {"all": [
                    original_build_when,
                    {"eq": ["$state.policy.revision", 2]},
                    {"eq": ["$args.signing_profile_handle", "sign_profile_8"]},
                ]},
                "response": {"artifact_handle": "artifact_92", "signed": True},
                "effects": [
                    {"set": "$state.artifact.handle", "value": "artifact_92"},
                    {"set": "$state.artifact.signed", "value": True},
                    {"set": "$state.artifact.signing_profile", "value": "sign_profile_8"},
                    {"increment": "$state.cost.total", "by": 8},
                ],
                "reads": ["$state.quality.coverage", "$state.policy.revision"],
                "writes": [
                    "$state.artifact.handle",
                    "$state.artifact.signed",
                    "$state.artifact.signing_profile",
                ],
                "resolves_errors": ["POLICY_ROTATED_AFTER_BUILD"],
            },
        ]

        build_tool = tool_binding(child, "build_signed_package")
        build_tool["parameters"]["properties"]["signing_profile_ref"] = {
            "type": "string",
            "description": "Signing profile discovered from current policy evidence.",
        }
        build_tool["parameters"]["required"].append("signing_profile_ref")
        build_tool.setdefault("input_map", {})["signing_profile_ref"] = "signing_profile_handle"
        build_tool.setdefault("provenance_required", []).append("signing_profile_ref")

        candidate = capability(child, "candidate.prepare.v1")["branches"][0]
        candidate["when"] = {
            "all": [
                candidate["when"],
                {"eq": ["$args.policy_handle", "policy_beta_8"]},
                {"eq": ["$state.policy.revision", 2]},
            ]
        }
        # Remove the parent's stale-handle equality from the nested condition.
        parent_all = candidate["when"]["all"][0].get("all", [])
        parent_all[:] = [
            item
            for item in parent_all
            if item
            not in (
                {"eq": ["$args.policy_handle", "policy_beta_7"]},
                {"eq": ["$args.artifact_handle", "artifact_91"]},
            )
        ]
        candidate["when"]["all"].append({"eq": ["$args.artifact_handle", "artifact_92"]})

        publish = capability(child, "release.publish.v1")["branches"][0]
        publish["effects"].append({"set": "$state.release.policy_revision", "value": 2})
        inspect = capability(child, "release.inspect.v1")["branches"][0]
        inspect["response"]["policy_revision"] = "$state.release.policy_revision"
        inspect["response"]["signing_profile"] = "$state.artifact.signing_profile"
        inspect["reads"].extend(
            ["$state.release.policy_revision", "$state.artifact.signing_profile"]
        )
        append_goal_condition(child, {"eq": ["$state.release.policy_revision", 2]})
        append_goal_condition(child, {"eq": ["$state.artifact.signing_profile", "sign_profile_8"]})

        child.contract["forbidden_shortcuts"].append(
            "reuse an artifact signed with stale policy evidence after policy rotation"
        )
        child.contract["counterfactual_axes"].append(
            {"state_path": "$state.policy.revision", "variants": [1, 2]}
        )
        child = replace(
            child,
            instruction=child.instruction.rstrip()
            + " Policy and signing evidence may change while preparing the artifact; "
            + "publish only with evidence that is current at candidate preparation time.\n",
        )

        actions = child.reference_plan["actions"]
        build_index = action_index(child, "build_signed_package")
        actions[build_index]["arguments"]["signing_profile_ref"] = "sign_profile_7"
        actions.insert(
            build_index + 1,
            {"tool": "load_channel_rules", "arguments": {"space_ref": "ws_42", "channel": "beta"}},
        )
        actions.insert(
            build_index + 2,
            {
                "tool": "build_signed_package",
                "arguments": {
                    "source_ref": "src_128",
                    "signing_profile_ref": "sign_profile_8",
                },
            },
        )
        for action in actions[build_index + 3 :]:
            arguments = action.get("arguments", {})
            if arguments.get("artifact_ref") == "artifact_91":
                arguments["artifact_ref"] = "artifact_92"
        candidate_index = action_index(child, "prepare_candidate")
        actions[candidate_index]["arguments"]["policy_ref"] = "policy_beta_8"

        errors = validate_bundle(child)
        if errors:
            raise ValueError("policy freshness operator produced invalid bundle: " + "; ".join(errors))
        return EvolutionProduct(
            bundle=child,
            patch={
                "operator_id": self.operator_id,
                "semantic_changes": [
                    "artifact construction rotates policy revision",
                    "signing consumes an early discovered profile handle",
                    "policy reread produces a new profile used to rebuild the artifact",
                    "security and candidate preparation consume only the rebuilt artifact",
                    "final goal records the policy revision used for publication",
                ],
                "added_goal_paths": [
                    "$state.release.policy_revision",
                    "$state.artifact.signing_profile",
                ],
            },
        )
