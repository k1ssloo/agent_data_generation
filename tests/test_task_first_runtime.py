from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import copy
import json
import unittest

from causal_validation import (
    evaluate_action_ablation,
    minimize_action_plan,
    validate_episode,
    validate_goal_alignment,
    validate_tool_identifiability,
    validate_adaptive_profile,
    validate_closed_loop_control,
    validate_temporal_provenance,
    alternative_recovery_metrics,
    validate_instruction_route_hiding,
    validate_tool_oracle_resistance,
    validate_vnext_adaptive_profile,
)
from causal_validation.intervention import evaluate_counterfactuals
from rollout import EpisodeRunner, run_reference_plan
from runtime.predicates import predicate_paths, resolve_path
from runtime.predicates import EvaluationError
from runtime.tool_renderer import render_alternate_api
from runtime.spec import validate_runtime_spec
from runtime.executor import CausalRuntime
from task_factory import load_task_bundle
from task_factory import totalize_public_capabilities, validate_public_executability
from task_factory.bundle import BundleError
from task_factory.contracts import normalize_contract, validate_contract
from task_factory.control_repair import repair_immediate_ordinal_provenance
from task_factory.evolve import evolve_once
from task_factory.fingerprint import semantic_fingerprint
from task_factory.hooks import attach_inferred_evolution_hooks
from task_factory.materialize import materialize_candidate
from task_factory.json_patch import JsonPatchError, apply_json_patch
from task_factory.prepare import admit_valid_counterfactuals
from task_factory.prepare import prepare_recursive_parent
from task_factory.wikihow_seed import validate_wikihow_seed
from task_factory.wikihow_seed import source_sha256
from task_factory.wikihow_compiler import _compile_public_argument_choices
from task_factory.goal_alignment import compile_alignment_plan, normalize_alignment_plan
from task_factory.search import generate_candidates, select_candidates
from task_factory.state_schema import complete_initial_state_schema
from task_factory.evidence_repair import repair_final_goal_evidence
from scripts.run_wikihow_task_factory import (
    compile_deterministic_candidate_repairs,
    evaluate as evaluate_factory_bundle,
)


SCRIPTS = Path(__file__).parents[1] / "scripts"
PROJECT_ROOT = Path(__file__).parents[1]


FIXTURE = Path(__file__).parent / "fixtures" / "release_task"


class TaskFirstRuntimeTests(unittest.TestCase):
    def test_internal_branch_conditions_are_not_public_observations(self) -> None:
        parent = load_task_bundle(FIXTURE)
        report = run_reference_plan(parent)
        first = None
        hidden_only = set()
        for step in report["trace"]:
            capability = parent.environment["capabilities"][step["capability_id"]]
            branch = next(
                item
                for item in capability["branches"]
                if item["id"] == step["selected_branch"]
            )
            hidden_only = (
                predicate_paths(branch["when"])
                - predicate_paths(branch["response"])
                - set(branch.get("observes", []))
            )
            if hidden_only:
                first = step
                break
        self.assertIsNotNone(first)
        self.assertTrue(hidden_only)
        self.assertTrue(hidden_only.isdisjoint(first["observed_state_paths"]))

    def test_explicit_observes_must_be_declared_reads(self) -> None:
        parent = load_task_bundle(FIXTURE)
        environment = copy.deepcopy(parent.environment)
        capability = next(iter(environment["capabilities"].values()))
        capability["branches"][0]["observes"] = ["$state.not_declared"]
        errors = validate_runtime_spec(environment)
        self.assertTrue(any("observes paths must also be declared" in item for item in errors))

    def test_argument_binding_is_not_counted_as_planning(self) -> None:
        parent = load_task_bundle(FIXTURE)
        evaluation = {
            "variants": [
                {
                    "id": "same_tool_new_argument",
                    "valid": True,
                    "decision_grounding": {
                        "valid": True,
                        "first_strategy_divergence": 0,
                        "changed_axes": ["$state.release.target"],
                    },
                }
            ]
        }
        reference_plan = copy.deepcopy(parent.reference_plan)
        baseline = copy.deepcopy(reference_plan["actions"])
        variant_actions = copy.deepcopy(baseline)
        first_argument = next(iter(variant_actions[0]["arguments"]))
        variant_actions[0]["arguments"][first_argument] = "different_visible_handle"
        reference_plan["counterfactuals"] = [
            {
                "name": "same_tool_new_argument",
                "state_overrides": {},
                "actions": variant_actions,
            }
        ]
        bundle = replace(parent, reference_plan=reference_plan)
        from causal_validation.intervention import counterfactual_decision_metrics

        metrics = counterfactual_decision_metrics(bundle, evaluation)
        self.assertEqual(metrics["observation_dependent_decision_count"], 1)
        self.assertEqual(metrics["meaningful_planning_decision_count"], 0)
        self.assertEqual(metrics["decisions"][0]["decision_type"], "argument_binding")

    def test_stale_policy_error_after_goal_does_not_validate_counterfactual(self) -> None:
        from causal_validation.intervention import _stale_policy_timing

        parent = load_task_bundle(FIXTURE)
        actions = copy.deepcopy(parent.reference_plan["actions"])
        actions.append({"tool": "nonexistent_post_goal_action", "arguments": {}})
        timing = _stale_policy_timing(
            parent,
            actions,
            divergence=len(parent.reference_plan["actions"]),
        )
        self.assertIsNotNone(timing["first_goal_satisfaction_step"])
        self.assertGreater(
            timing["first_post_divergence_failure_step"],
            timing["first_goal_satisfaction_step"],
        )
        self.assertFalse(timing["failed_before_goal_satisfaction"])
        self.assertTrue(timing["goal_reached_before_failure"])

    def test_stale_policy_that_recovers_after_error_is_not_rejected(self) -> None:
        from causal_validation.intervention import _stale_policy_timing

        parent = load_task_bundle(FIXTURE)
        timing = _stale_policy_timing(
            parent,
            copy.deepcopy(parent.reference_plan["actions"]),
            divergence=0,
        )
        self.assertIsNotNone(timing["first_post_divergence_failure_step"])
        self.assertIsNotNone(timing["first_goal_satisfaction_step"])
        self.assertFalse(timing["failed_before_goal_satisfaction"])

    def test_vnext_rejects_solution_role_tool_names(self) -> None:
        parent = load_task_bundle(FIXTURE)
        bindings = copy.deepcopy(parent.bindings)
        bindings["tools"][0]["name"] = "diagnose_distribution_failure"
        candidate = replace(parent, bindings=bindings)
        result = validate_tool_oracle_resistance(candidate)
        self.assertFalse(result["valid"])
        self.assertEqual(
            result["violations"][0]["reasons"], ["solution_role_in_name"]
        )

    def test_vnext_detects_recovery_route_spoiler(self) -> None:
        parent = load_task_bundle(FIXTURE)
        candidate = replace(
            parent,
            instruction=(
                "Publish the release. If DISTRIBUTION_CONFLICT appears, "
                "open the policy view and retry."
            ),
        )
        counterfactual = {
            "baseline_recovery_strategies": [
                {
                    "error_code": "DISTRIBUTION_CONFLICT",
                    "recovery_tool": "inspect_policy",
                }
            ],
            "variants": [],
        }
        result = validate_instruction_route_hiding(candidate, counterfactual)
        self.assertFalse(result["valid"])
        self.assertEqual(result["exposed_error_codes"], ["DISTRIBUTION_CONFLICT"])
        self.assertTrue(result["procedural_recovery_language"])

    def test_alternative_recovery_requires_same_failure_two_tools(self) -> None:
        counterfactual = {
            "baseline_recovery_strategies": [
                {"error_code": "CONFLICT", "recovery_tool": "update_setting"}
            ],
            "variants": [
                {
                    "id": "alternate_route",
                    "valid": True,
                    "adapted_valid": True,
                    "recovery_strategies": [
                        {"error_code": "CONFLICT", "recovery_tool": "switch_route"}
                    ],
                },
                {
                    "id": "failure_absent",
                    "valid": True,
                    "adapted_valid": True,
                    "recovery_strategies": [],
                },
            ],
        }
        result = alternative_recovery_metrics(counterfactual)
        self.assertTrue(result["valid"])
        self.assertEqual(result["failures"][0]["strategy_count"], 2)

    def test_vnext_profile_encodes_decision_dense_target(self) -> None:
        parent = load_task_bundle(FIXTURE)
        bindings = copy.deepcopy(parent.bindings)
        for tool in bindings["tools"]:
            if tool["name"] == "diagnose_quality":
                tool["name"] = "inspect_quality_report"
        parent = replace(parent, bindings=bindings)
        episode = {"trace": [{"step": index} for index in range(1, 21)]}
        causal = {
            "metrics": {
                "semantic_recoveries": [{"error_code": "CONFLICT"}],
                "missing_provenance": [],
                "invariant_violations": [],
                "goal_evidence_coverage": 1.0,
                "final_goal_observation_step": 20,
            }
        }
        counterfactual = {
            "valid": True,
            "decision_metrics": {
                "meaningful_planning_decision_count": 3,
                "decision_entropy_bits": 3.0,
            },
            "baseline_recovery_strategies": [
                {"error_code": "CONFLICT", "recovery_tool": "update_setting"}
            ],
            "variants": [
                {
                    "id": "alternate_route",
                    "valid": True,
                    "adapted_valid": True,
                    "recovery_strategies": [
                        {"error_code": "CONFLICT", "recovery_tool": "switch_route"}
                    ],
                }
            ],
        }
        result = validate_vnext_adaptive_profile(
            parent,
            episode,
            causal,
            counterfactual,
            ablation={
                "necessary_actions": 18,
                "necessary_action_ratio": 0.9,
            },
        )
        self.assertTrue(result["valid"], result["errors"])

    def test_success_error_code_sentinel_does_not_hide_final_evidence(self) -> None:
        parent = load_task_bundle(FIXTURE)
        environment = copy.deepcopy(parent.environment)
        final_action = parent.reference_plan["actions"][-1]
        binding = next(
            tool
            for tool in parent.tools
            if tool["name"] == final_action["tool"]
        )
        capability = environment["capabilities"][binding["capability_id"]]
        branch = capability["branches"][0]
        branch["response"]["error_code"] = "NO_ERROR"
        bundle = replace(parent, environment=environment)
        report = run_reference_plan(bundle)
        self.assertIsNone(report["trace"][-1]["error_code"])
        validation = validate_episode(bundle, report)
        self.assertTrue(validation["valid"], validation["errors"])

    def test_one_failure_is_counted_as_one_semantic_recovery(self) -> None:
        parent = load_task_bundle(FIXTURE)
        report = run_reference_plan(parent)
        failure_step = report["trace"][0]
        failure_step["error_code"] = "ONE_FAILURE"
        report["trace"][1]["resolves_errors"] = ["ONE_FAILURE"]
        report["trace"][2]["resolves_errors"] = ["ONE_FAILURE"]
        validation = validate_episode(parent, report)
        matching = [
            item
            for item in validation["metrics"]["semantic_recoveries"]
            if item["error_code"] == "ONE_FAILURE"
        ]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["recovery_step"], report["trace"][1]["step"])

    def test_delayed_handle_uses_first_public_producer(self) -> None:
        from causal_validation.validator import _delayed_handles

        trace = [
            {"step": 1, "produced_handles": [{"value": "stable_handle"}], "consumed_handles": []},
            {"step": 5, "produced_handles": [{"value": "stable_handle"}], "consumed_handles": []},
            {"step": 9, "produced_handles": [], "consumed_handles": ["stable_handle"]},
        ]
        self.assertEqual(_delayed_handles(trace)[0]["producer_step"], 1)
        self.assertEqual(_delayed_handles(trace)[0]["distance"], 8)

    def test_temporal_provenance_requires_every_declared_causal_link(self) -> None:
        parent = load_task_bundle(FIXTURE)
        contract = copy.deepcopy(parent.contract)
        contract.setdefault("requirements", {})["temporal_provenance"] = {
            "links": [
                {
                    "consumer_tool": "download_revision",
                    "argument": "snapshot_ref",
                    "producer_tool": "observe_snapshot",
                },
                {
                    "consumer_tool": "restore_revision",
                    "argument": "artifact_ref",
                    "producer_tool": "download_revision",
                },
            ],
            "final_observation_tool": "observe_outcome",
            "final_paths": ["$state.file.status", "$state.file.source_revision"],
        }
        bundle = replace(parent, contract=contract)
        episode = {
            "trace": [
                {"step": 1, "public_tool": "observe_snapshot"},
                {
                    "step": 2,
                    "public_tool": "download_revision",
                    "arguments": {
                        "snapshot_ref": {
                            "source": {"step": 1, "tool": "observe_snapshot"}
                        }
                    },
                },
                {
                    "step": 3,
                    "public_tool": "restore_revision",
                    "arguments": {
                        "artifact_ref": {
                            "source": {"step": 2, "tool": "download_revision"}
                        }
                    },
                },
                {
                    "step": 4,
                    "public_tool": "observe_outcome",
                    "write_set": [],
                    "observed_state_paths": [
                        "$state.file.status",
                        "$state.file.source_revision",
                    ],
                },
            ]
        }
        valid = validate_temporal_provenance(bundle, episode)
        self.assertTrue(valid["valid"], valid)
        broken = copy.deepcopy(episode)
        broken["trace"][2]["arguments"]["artifact_ref"]["source"]["tool"] = (
            "hidden_oracle"
        )
        self.assertFalse(validate_temporal_provenance(bundle, broken)["valid"])

    def test_closed_loop_control_requires_grounded_measurement_and_settle(self) -> None:
        parent = load_task_bundle(FIXTURE)
        contract = copy.deepcopy(parent.contract)
        contract.setdefault("requirements", {})["closed_loop_control"] = {
            "measurement_tool": "measure_load",
            "control_tool": "stop_load",
            "evidence_argument": "measurement_ref",
            "final_observation_tool": "observe_outcome",
            "measurement_paths": ["$state.load.percent"],
            "controlled_paths": ["$state.job.status"],
            "settled_paths": ["$state.load.percent", "$state.job.status"],
        }
        bundle = replace(parent, contract=contract)
        episode = {
            "trace": [
                {
                    "step": 1,
                    "public_tool": "measure_load",
                    "observed_state_paths": ["$state.load.percent"],
                    "write_set": [],
                },
                {
                    "step": 2,
                    "public_tool": "stop_load",
                    "arguments": {
                        "measurement_ref": {
                            "source": {"step": 1, "tool": "measure_load"}
                        }
                    },
                    "write_set": ["$state.job.status"],
                },
                {
                    "step": 3,
                    "public_tool": "observe_outcome",
                    "observed_state_paths": [
                        "$state.load.percent",
                        "$state.job.status",
                    ],
                    "write_set": [],
                },
            ]
        }
        control = validate_closed_loop_control(bundle, episode)
        self.assertTrue(control["valid"], control["errors"])
        counterfactual = {
            "decision_metrics": {
                "meaningful_planning_decision_count": 1,
                "decision_entropy_bits": 1.0,
            }
        }
        adaptive = validate_adaptive_profile(
            bundle, episode, counterfactual, semantic_recovery_count=0
        )
        self.assertEqual(
            adaptive["profiles"], ["planning_with_closed_loop_control"]
        )

        broken = copy.deepcopy(episode)
        broken["trace"][1]["arguments"]["measurement_ref"]["source"] = None
        self.assertFalse(validate_closed_loop_control(bundle, broken)["valid"])

    def test_alignment_plan_compiler_rejects_unprovable_clause(self) -> None:
        parent = load_task_bundle(FIXTURE)
        sentences = [
            "Release version 3.4.0 of the Atlas Android app to the beta channel.",
            (
                "Follow the workspace release policy, keep the run within its cost limit, "
                "and do not distribute an artifact that fails testing, signing, or security requirements."
            ),
        ]
        impossible = {
            "alignment_version": "goal-alignment-v1",
            "supported": True,
            "rejection_reasons": [],
            "instruction_claims": [
                {
                    "evidence_span": sentence,
                    "kind": "goal" if index == 0 else "constraint",
                    "clause_ids": ["impossible"],
                }
                for index, sentence in enumerate(sentences)
            ],
            "goal_clauses": [
                {
                    "id": "impossible",
                    "predicate": {"eq": ["$state.release.version", "9.9.9"]},
                    "transition_paths": ["$state.release.version"],
                    "evidence_paths": ["$state.release.version"],
                    "witness_tools": ["publish_candidate"],
                }
            ],
            "domain_invariants": [],
        }
        aligned, report = compile_alignment_plan(parent, impossible)
        self.assertIsNone(aligned)
        self.assertFalse(report["valid"])
        self.assertTrue(
            any("does not satisfy clause" in error for error in report["errors"])
        )

    def test_goal_alignment_proves_exhaustive_instruction_clause(self) -> None:
        parent = load_task_bundle(FIXTURE)
        predicate = copy.deepcopy(parent.contract["goal_predicates"][0]["predicate"])
        contract = {
            **copy.deepcopy(parent.contract),
            "instruction_claims": [
                {
                    "evidence_span": (
                        "Release version 3.4.0 of the Atlas Android app to the beta channel."
                    ),
                    "kind": "goal",
                    "clause_ids": ["complete_release"],
                },
                {
                    "evidence_span": (
                        "Follow the workspace release policy, keep the run within its cost limit, "
                        "and do not distribute an artifact that fails testing, signing, or security requirements."
                    ),
                    "kind": "constraint",
                    "clause_ids": ["complete_release"],
                },
            ],
            "goal_clauses": [
                {
                    "id": "complete_release",
                    "predicate": predicate,
                    "transition_paths": [
                        "$state.release",
                        "$state.quality.coverage",
                        "$state.security.critical",
                        "$state.artifact.signed",
                    ],
                    "evidence_paths": [
                        "$state.release",
                        "$state.quality.coverage",
                        "$state.security.critical",
                        "$state.artifact.signed",
                        "$state.cost.total",
                    ],
                    "witness_tools": [
                        "check_run",
                        "build_signed_package",
                        "publish_candidate",
                    ],
                }
            ],
        }
        aligned = replace(parent, contract=contract)
        report = validate_goal_alignment(aligned)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["metrics"]["instruction_sentence_count"], 2)
        self.assertEqual(report["metrics"]["instruction_goal_coverage"], 1.0)

        incomplete = replace(
            aligned,
            contract={
                **copy.deepcopy(contract),
                "instruction_claims": contract["instruction_claims"][:1],
            },
        )
        rejected = validate_goal_alignment(incomplete)
        self.assertFalse(rejected["valid"])
        self.assertTrue(
            any("exhaustively match" in error for error in rejected["errors"])
        )

    def test_alignment_compiler_preserves_executable_goal_predicates(self) -> None:
        parent = load_task_bundle(FIXTURE)
        predicate = copy.deepcopy(parent.contract["goal_predicates"][0]["predicate"])
        plan = {
            "alignment_version": "goal-alignment-v1",
            "supported": True,
            "rejection_reasons": [],
            "instruction_claims": [
                {
                    "evidence_span": sentence,
                    "kind": "goal" if index == 0 else "constraint",
                    "clause_ids": ["complete_release"],
                }
                for index, sentence in enumerate(
                    [
                        "Release version 3.4.0 of the Atlas Android app to the beta channel.",
                        (
                            "Follow the workspace release policy, keep the run within its cost limit, "
                            "and do not distribute an artifact that fails testing, signing, or security requirements."
                        ),
                    ]
                )
            ],
            "goal_clauses": [
                {
                    "id": "complete_release",
                    "predicate": predicate,
                    "transition_paths": sorted(predicate_paths(predicate)),
                    "evidence_paths": sorted(predicate_paths(predicate)),
                    "witness_tools": [
                        "check_run",
                        "build_signed_package",
                        "publish_candidate",
                    ],
                }
            ],
            "domain_invariants": [],
        }
        aligned, report = compile_alignment_plan(parent, plan)
        self.assertIsNotNone(aligned, report["errors"])
        self.assertEqual(
            aligned.contract["goal_predicates"],
            parent.contract["goal_predicates"],
        )

    def test_alignment_normalizer_restores_conditional_goal_semantics(self) -> None:
        parent = load_task_bundle(FIXTURE)
        contract = copy.deepcopy(parent.contract)
        conditional = {
            "any": [
                {"eq": ["$state.failure.observed", False]},
                {
                    "all": [
                        {"eq": ["$state.failure.observed", True]},
                        {"eq": ["$state.failure.recovered", True]},
                    ]
                },
            ]
        }
        contract["goal_predicates"] = [
            {"id": "conditional_recovery", "predicate": conditional}
        ]
        bundle = replace(parent, contract=contract)
        narrowed = {
            "supported": True,
            "instruction_claims": [],
            "clauses": [
                {
                    "id": "recovery",
                    "predicate": conditional["any"][1],
                    "transition_paths": sorted(predicate_paths(conditional)),
                    "evidence_paths": sorted(predicate_paths(conditional)),
                    "witness_tools": ["diagnose_quality"],
                }
            ],
            "domain_invariants": [],
        }
        normalized, changes = normalize_alignment_plan(bundle, narrowed)
        self.assertEqual(normalized["goal_clauses"][0]["predicate"], conditional)
        self.assertTrue(any("restored executable predicate" in item for item in changes))

    def test_plan_minimization_removes_redundant_dependency_chain(self) -> None:
        from unittest.mock import patch

        parent = load_task_bundle(FIXTURE)
        reference_plan = {
            "actions": [
                {"tool": "required", "arguments": {}},
                {"tool": "noise_start", "arguments": {}},
                {"tool": "noise_poll", "arguments": {}},
            ],
            "counterfactuals": [],
        }
        candidate = replace(parent, reference_plan=reference_plan)

        def fake_run(_bundle, *, actions, **_kwargs):
            return {
                "status": "goal_satisfied",
                "tools": [action["tool"] for action in actions],
            }

        def fake_validate(_bundle, report, **_kwargs):
            tools = report["tools"]
            valid = "required" in tools and (
                ("noise_start" in tools) == ("noise_poll" in tools)
            )
            return {"valid": valid, "errors": [] if valid else ["invalid"]}

        with patch("causal_validation.ablation.run_reference_plan", fake_run), patch(
            "causal_validation.ablation.validate_episode", fake_validate
        ):
            single = evaluate_action_ablation(candidate)
            minimized = minimize_action_plan(candidate)

        self.assertEqual(single["necessary_actions"], 3)
        self.assertEqual(minimized["retained_indices"], [0])
        self.assertEqual(minimized["removed_indices"], [1, 2])

    def test_wikihow_schema_exposes_options_without_leaking_identifiers(self) -> None:
        parameters = {
            "type": "object",
            "properties": {
                "collection": {"type": "string"},
                "record_id": {"type": "string"},
                "query": {"type": "string"},
                "filters": {"type": "object"},
            },
        }
        calls = [
            {
                "arguments": {
                    "collection": "archive",
                    "record_id": "private-123",
                    "query": "Jordan Lee",
                    "filters": {},
                }
            }
        ]

        compiled = _compile_public_argument_choices(parameters, calls)
        properties = compiled["properties"]
        self.assertEqual(properties["collection"]["enum"], ["archive"])
        self.assertNotIn("enum", properties["record_id"])
        self.assertNotIn("enum", properties["query"])
        self.assertEqual(properties["filters"]["default"], {})

    def test_wikihow_schema_makes_defaulted_arguments_optional(self) -> None:
        parameters = {
            "type": "object",
            "properties": {
                "view": {"type": "string"},
                "target_id": {"type": "string"},
            },
            "required": ["view", "target_id"],
        }
        compiled = _compile_public_argument_choices(
            parameters,
            [{"arguments": {"view": "calibration_status", "target_id": ""}}],
        )
        self.assertEqual(compiled["properties"]["view"]["enum"], ["calibration_status"])
        self.assertEqual(compiled["properties"]["target_id"]["default"], "")
        self.assertEqual(compiled["required"], ["view"])

    def test_runtime_enforces_public_enum_and_applies_default(self) -> None:
        bundle = load_task_bundle(FIXTURE)
        runtime = CausalRuntime(bundle)
        binding = {
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["safe"]},
                    "target_id": {"type": "string", "default": ""},
                },
                "required": ["mode"],
            }
        }
        with self.assertRaisesRegex(ValueError, "public enum values"):
            runtime._validate_args(binding, {"mode": "hidden"})
        runtime._validate_args(binding, {"mode": "safe"})
        self.assertEqual(
            runtime._adapt_args(binding, {"mode": "safe"}),
            {"mode": "safe", "target_id": ""},
        )

    def test_episode_runner_keeps_rejected_public_attempts(self) -> None:
        bundle = load_task_bundle(FIXTURE)
        runner = EpisodeRunner(bundle, max_steps=2)
        with self.assertRaisesRegex(ValueError, "unknown public tool"):
            runner.tool_call("not_a_tool", {})
        self.assertEqual(runner.attempt_count, 1)
        self.assertEqual(runner.runtime.trace, [])
        self.assertEqual(runner.messages[-2]["role"], "assistant")
        self.assertEqual(runner.messages[-1]["role"], "tool")
        self.assertIn("unknown public tool", runner.messages[-1]["content"])
        self.assertEqual(runner.policy_context()["remaining_steps"], 1)

    def test_runtime_validates_nested_object_schema(self) -> None:
        bundle = load_task_bundle(FIXTURE)
        runtime = CausalRuntime(bundle)
        binding = {
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "enum": ["saved"]}
                        },
                        "required": ["status"],
                        "additionalProperties": False,
                    }
                },
                "required": ["patch"],
            }
        }
        with self.assertRaisesRegex(ValueError, "missing required argument"):
            runtime._validate_args(binding, {"patch": {}})
        with self.assertRaisesRegex(ValueError, "unexpected properties"):
            runtime._validate_args(
                binding, {"patch": {"status": "saved", "hidden": True}}
            )
        runtime._validate_args(binding, {"patch": {"status": "saved"}})

    def test_public_interface_totalizer_adds_structured_fallbacks(self) -> None:
        bundle = load_task_bundle(FIXTURE)
        total = totalize_public_capabilities(bundle)
        report = validate_public_executability(total)
        self.assertTrue(report["valid"], report["errors"])
        for tool in total.tools:
            branches = total.environment["capabilities"][tool["capability_id"]]["branches"]
            self.assertIs(branches[-1].get("when", True), True)

    def test_public_interface_totalizer_defaults_optional_arguments(self) -> None:
        bundle = load_task_bundle(FIXTURE)
        bindings = copy.deepcopy(bundle.bindings)
        tool = bindings["tools"][0]
        tool["parameters"]["properties"]["optional_mode"] = {
            "type": "string",
            "default": "safe",
        }
        tool["parameters"].setdefault("required", []).append("optional_mode")
        candidate = replace(bundle, bindings=bindings)
        total = totalize_public_capabilities(candidate)
        normalized = total.tools[0]["parameters"]
        self.assertNotIn("optional_mode", normalized["required"])
        self.assertEqual(
            normalized["properties"]["optional_mode"]["default"], "safe"
        )
        self.assertTrue(validate_public_executability(total)["valid"])

    def test_ordinal_provenance_repair_only_changes_prior_response(self) -> None:
        candidate = {
            "environment": {
                "capabilities": {
                    "inspect": {
                        "branches": [
                            {"id": "visible", "response": {"ok": True}}
                        ]
                    },
                    "load": {"branches": [{"id": "loaded", "response": {}}]},
                }
            },
            "bindings": {
                "tools": [
                    {
                        "name": "load_page",
                        "parameters": {
                            "properties": {"page_number": {"type": "integer"}}
                        },
                    }
                ]
            },
            "reference_plan": {
                "actions": [
                    {"tool": "inspect", "arguments": {}},
                    {"tool": "load_page", "arguments": {"page_number": 2}},
                ]
            },
        }
        report = {
            "phase": "execution",
            "causal_validation": {
                "metrics": {
                    "unexplained_arguments": [
                        {
                            "step": 2,
                            "tool": "load_page",
                            "argument": "page_number",
                            "value": 2,
                        }
                    ]
                }
            },
            "episode": {
                "trace": [
                    {
                        "capability_id": "inspect",
                        "selected_branch": "visible",
                        "response": {"ok": True},
                        "error_code": None,
                    },
                    {
                        "capability_id": "load",
                        "selected_branch": "loaded",
                        "response": {},
                        "error_code": None,
                    },
                ]
            },
        }
        repaired, details = repair_immediate_ordinal_provenance(candidate, report)
        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertEqual(
            repaired["environment"]["capabilities"]["inspect"]["branches"][0][
                "response"
            ]["page_number"],
            2,
        )
        self.assertEqual(repaired["reference_plan"], candidate["reference_plan"])
        self.assertNotIn(
            "page_number",
            candidate["environment"]["capabilities"]["inspect"]["branches"][0][
                "response"
            ],
        )
        self.assertEqual(details["repairs"][0]["producer_branch"], "visible")

        invalid = copy.deepcopy(report)
        invalid["causal_validation"]["metrics"]["unexplained_arguments"][0].update(
            {"argument": "record_id", "value": "secret-record"}
        )
        rejected, rejection = repair_immediate_ordinal_provenance(candidate, invalid)
        self.assertIsNone(rejected)
        self.assertEqual(rejection["reason"], "argument_is_not_an_immediate_ordinal")

    def test_deterministic_final_evidence_repair_is_narrow_and_gate_valid(self) -> None:
        parent = load_task_bundle(FIXTURE)
        candidate = {
            "instruction": parent.instruction,
            "environment": copy.deepcopy(parent.environment),
            "bindings": copy.deepcopy(parent.bindings),
            "reference_plan": copy.deepcopy(parent.reference_plan),
        }
        branch = next(
            item
            for item in candidate["environment"]["capabilities"][
                "release.inspect.v1"
            ]["branches"]
            if item["id"] == "final_evidence"
        )
        branch["reads"].remove("$state.release.channel")
        branch["response"].pop("channel")
        partial_bundle = replace(
            parent,
            environment=candidate["environment"],
            bindings=candidate["bindings"],
            reference_plan=candidate["reference_plan"],
        )
        partial_episode = run_reference_plan(partial_bundle)
        repaired, details = repair_final_goal_evidence(
            parent.contract,
            candidate,
            {
                "phase": "execution",
                "errors": [
                    "recursive evolution hook unavailable: no final domain "
                    "observation covers every goal path"
                ],
                "episode": partial_episode,
                "causal_validation": {"valid": True},
            },
        )
        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertEqual(details["added_goal_paths"], ["$state.release.channel"])
        repaired_bundle = replace(
            parent,
            environment=repaired["environment"],
            bindings=repaired["bindings"],
            reference_plan=repaired["reference_plan"],
        )
        evaluation = evaluate_factory_bundle(repaired_bundle, 0.6)
        self.assertTrue(evaluation["valid"], evaluation["errors"])

    def test_deterministic_final_evidence_repair_rejects_mixed_failures(self) -> None:
        repaired, details = repair_final_goal_evidence(
            {"goal_predicates": []},
            {},
            {
                "phase": "execution",
                "errors": [
                    "recursive evolution hook unavailable: no final domain "
                    "observation covers every goal path",
                    "one or more tool arguments lack an admissible provenance source",
                ],
                "causal_validation": {"valid": True},
            },
        )
        self.assertIsNone(repaired)
        self.assertEqual(details["reason"], "not_a_final_evidence_only_failure")

    def test_deterministic_evidence_repair_appends_read_after_goal_mutation(self) -> None:
        parent = load_task_bundle(FIXTURE)
        candidate = {
            "instruction": parent.instruction,
            "environment": copy.deepcopy(parent.environment),
            "bindings": copy.deepcopy(parent.bindings),
            "reference_plan": copy.deepcopy(parent.reference_plan),
        }
        candidate["reference_plan"]["actions"] = candidate["reference_plan"][
            "actions"
        ][:-1]
        partial_bundle = replace(
            parent,
            environment=candidate["environment"],
            bindings=candidate["bindings"],
            reference_plan=candidate["reference_plan"],
        )
        report = evaluate_factory_bundle(partial_bundle, 0.6)
        self.assertEqual(
            report["causal_validation"]["errors"],
            ["final observations do not cover every goal predicate path"],
        )

        repaired, details = repair_final_goal_evidence(
            parent.contract, candidate, report
        )
        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertEqual(details["mode"], "append_post_mutation_observation")
        self.assertEqual(
            repaired["reference_plan"]["actions"][-1]["tool"],
            details["tool_name"],
        )
        branch = repaired["environment"]["capabilities"][details["capability_id"]][
            "branches"
        ][0]
        self.assertEqual(branch["writes"], [])
        self.assertEqual(branch["effects"], [])
        self.assertEqual(set(branch["reads"]), set(details["added_goal_paths"]))
        repaired_bundle = replace(
            parent,
            environment=repaired["environment"],
            bindings=repaired["bindings"],
            reference_plan=repaired["reference_plan"],
        )
        evaluation = evaluate_factory_bundle(repaired_bundle, 0.6)
        self.assertTrue(evaluation["valid"], evaluation["errors"])

    def test_restricted_json_patch_is_atomic_and_preserves_unrelated_fields(self) -> None:
        source = {
            "instruction": "Publish Atlas.",
            "environment": {"initial_state": {"ready": False}},
            "bindings": {"tools": []},
            "reference_plan": {"actions": []},
        }
        patched = apply_json_patch(
            source,
            [
                {
                    "op": "replace",
                    "path": "/environment/initial_state/ready",
                    "value": True,
                },
                {
                    "op": "add",
                    "path": "/reference_plan/actions/-",
                    "value": {"tool": "publish", "arguments": {}},
                },
            ],
        )
        self.assertFalse(source["environment"]["initial_state"]["ready"])
        self.assertTrue(patched["environment"]["initial_state"]["ready"])
        self.assertEqual(patched["bindings"], source["bindings"])
        self.assertEqual(len(patched["reference_plan"]["actions"]), 1)

    def test_restricted_json_patch_rejects_private_or_partial_mutation(self) -> None:
        source = {"instruction": "x", "environment": {"value": 1}}
        with self.assertRaises(JsonPatchError):
            apply_json_patch(
                source,
                [
                    {"op": "replace", "path": "/contract/goal", "value": "easy"},
                    {"op": "replace", "path": "/environment/missing", "value": 2},
                ],
            )
        self.assertEqual(source["environment"]["value"], 1)

    def test_provenance_accepts_observed_url_without_id_suffix(self) -> None:
        original = load_task_bundle(FIXTURE)
        environment = copy.deepcopy(original.environment)
        bindings = copy.deepcopy(original.bindings)
        reference_plan = copy.deepcopy(original.reference_plan)
        # A URL is still reusable evidence even when its key has no _id suffix.
        environment["capabilities"]["workspace.find.v1"]["branches"][0]["response"][
            "workspace_url"
        ] = "https://example.test/workspaces/atlas"
        policy_tool = next(tool for tool in bindings["tools"] if tool["name"] == "load_channel_rules")
        policy_tool["parameters"]["properties"]["workspace_url"] = {
            "type": "string"
        }
        policy_tool["parameters"]["required"].append("workspace_url")
        policy_tool.setdefault("provenance_required", []).append("workspace_url")
        reference_plan["actions"][1]["arguments"]["workspace_url"] = (
            "https://example.test/workspaces/atlas"
        )
        policy = environment["capabilities"]["policy.read.v1"]["branches"][0]
        policy["when"] = {
            "all": [
                policy["when"],
                {"eq": ["$args.workspace_url", "https://example.test/workspaces/atlas"]},
            ]
        }
        bundle = replace(
            original,
            environment=environment,
            bindings=bindings,
            reference_plan=reference_plan,
        )
        validation = validate_episode(bundle, run_reference_plan(bundle))
        self.assertFalse(validation["metrics"]["missing_provenance"])

    def test_path_resolution_distinguishes_fields_and_dynamic_keys(self) -> None:
        context = {
            "args": {"query": "Atlas Android", "item_id": "item_7"},
            "state": {"items": {"item_7": {"status": "ready"}}},
        }
        self.assertEqual(resolve_path("$args.query", context), "Atlas Android")
        self.assertEqual(resolve_path("$state.items[item_id].status", context), "ready")

    def test_path_resolution_reports_non_scalar_dynamic_key(self) -> None:
        context = {"args": {"item_id": ["bad"]}, "state": {"items": {}}}
        with self.assertRaisesRegex(EvaluationError, "non-scalar map key"):
            resolve_path("$state.items[item_id]", context)

    def test_runtime_static_check_skips_unreachable_response_paths(self) -> None:
        branch = {
            "id": "future_object",
            "when": {"exists": "$state.result"},
            "response": {"value": "$state.result.value"},
            "reads": ["$state.result"],
            "writes": [],
            "effects": [],
            "resolves_errors": [],
        }
        environment = {
            "runtime_version": "causal-runtime-v1",
            "initial_state": {},
            "capabilities": {"inspect": {"branches": [branch]}},
        }
        self.assertEqual(validate_runtime_spec(environment), [])
        environment["capabilities"]["inspect"]["branches"][0]["when"] = True
        errors = validate_runtime_spec(environment)
        self.assertTrue(any("expression error" in error for error in errors))

    def test_policy_context_does_not_expose_private_task_data(self) -> None:
        bundle = load_task_bundle(FIXTURE)
        context = EpisodeRunner(bundle).policy_context()
        serialized = str(context)
        self.assertNotIn("goal_predicates", serialized)
        self.assertNotIn("reference_plan", serialized)
        self.assertNotIn("capability_id", serialized)
        self.assertNotIn("initial_state", serialized)

    def test_bundle_loader_applies_contract_static_checks(self) -> None:
        bundle = load_task_bundle(FIXTURE)
        self.assertIn("delayed_handle_use", bundle.contract["expected_reasoning_features"])

    def test_reference_plan_is_causally_valid(self) -> None:
        bundle = load_task_bundle(FIXTURE)
        report = run_reference_plan(bundle)
        validation = validate_episode(bundle, report)
        self.assertEqual(report["status"], "goal_satisfied")
        self.assertTrue(validation["valid"], validation["errors"])
        self.assertGreaterEqual(validation["metrics"]["observation_dependent_branch_count"], 1)
        tool_call = next(message for message in report["messages"] if "tool_calls" in message)
        tool_response = next(message for message in report["messages"] if message.get("role") == "tool")
        self.assertIsInstance(tool_call["tool_calls"][0]["function"]["arguments"], str)
        self.assertIsInstance(json.loads(tool_response["content"]), dict)

    def test_goal_writing_action_cannot_impersonate_final_observation(self) -> None:
        parent = load_task_bundle(FIXTURE)
        environment = copy.deepcopy(parent.environment)
        reference_plan = copy.deepcopy(parent.reference_plan)
        reference_plan["actions"] = reference_plan["actions"][:-1]
        publish_action = reference_plan["actions"][-1]
        publish_binding = next(
            tool
            for tool in parent.tools
            if tool["name"] == publish_action["tool"]
        )
        capability = environment["capabilities"][publish_binding["capability_id"]]
        publish_branch = next(
            branch
            for branch in capability["branches"]
            if branch["id"] == "published"
        )
        goal_paths = {
            path
            for item in parent.contract["goal_predicates"]
            for path in predicate_paths(item["predicate"])
        }
        publish_branch["reads"] = sorted(
            set(publish_branch.get("reads", [])) | goal_paths
        )
        candidate = replace(
            parent,
            environment=environment,
            reference_plan=reference_plan,
        )
        result = validate_episode(candidate, run_reference_plan(candidate))
        self.assertFalse(result["valid"])
        self.assertIn(
            "final observations do not cover every goal predicate path",
            result["errors"],
        )
        self.assertIsNone(result["metrics"]["final_goal_observation_step"])

    def test_validator_rejects_oracle_calls_without_provenance(self) -> None:
        bundle = load_task_bundle(FIXTURE)
        report = run_reference_plan(bundle)
        report["trace"][12]["arguments"]["policy_ref"]["source"] = None
        validation = validate_episode(bundle, report)
        self.assertFalse(validation["valid"])
        self.assertIn("required arguments lack tool-output provenance", validation["errors"])

    def test_argument_provenance_classifies_literals_and_rejects_secrets(self) -> None:
        bundle = load_task_bundle(FIXTURE)
        report = run_reference_plan(bundle)
        first = report["trace"][0]["arguments"]["query"]
        self.assertEqual(first["provenance_kind"], "user_grounded")
        observed = report["trace"][1]["arguments"]["space_ref"]
        self.assertEqual(observed["provenance_kind"], "tool_observation_grounded")

        environment = copy.deepcopy(bundle.environment)
        bindings = copy.deepcopy(bundle.bindings)
        reference_plan = copy.deepcopy(bundle.reference_plan)
        tool = next(item for item in bindings["tools"] if item["name"] == "find_project_space")
        tool["parameters"]["properties"]["password"] = {
            "type": "string",
            "enum": ["hidden-secret"],
        }
        tool["parameters"]["required"].append("password")
        reference_plan["actions"][0]["arguments"]["password"] = "hidden-secret"
        candidate = replace(
            bundle,
            environment=environment,
            bindings=bindings,
            reference_plan=reference_plan,
        )
        validation = validate_episode(candidate, run_reference_plan(candidate))
        self.assertFalse(validation["valid"])
        missing = validation["metrics"]["unexplained_arguments"]
        self.assertEqual(missing[0]["argument"], "password")
        self.assertEqual(missing[0]["value"], "hidden-secret")

    def test_structured_argument_requires_grounding_for_every_leaf(self) -> None:
        bundle = load_task_bundle(FIXTURE)
        runtime = CausalRuntime(bundle)
        runtime.execute("find_project_space", {"query": "Atlas Android"})
        binding = {
            "parameters": {
                "type": "object",
                "properties": {"selection": {"type": "object"}},
                "required": ["selection"],
            }
        }
        grounded = runtime._argument_provenance(
            binding,
            {"selection": {"project": "Atlas Android", "workspace": "ws_42"}},
        )["selection"]
        self.assertEqual(grounded["provenance_kind"], "structured_grounded")
        self.assertTrue(
            all(
                item["provenance_kind"] != "unexplained"
                for item in grounded["evidence"]
                if item["kind"] == "structured_leaf"
            )
        )

        unexplained = runtime._argument_provenance(
            binding,
            {
                "selection": {
                    "project": "Atlas Android",
                    "workspace": "ws_42",
                    "private_route": "hidden-route",
                }
            },
        )["selection"]
        self.assertEqual(unexplained["provenance_kind"], "unexplained")

    def test_counterfactual_requires_early_observation_of_changed_axes(self) -> None:
        maps = PROJECT_ROOT / (
            "outputs/task_first/gpt55_wikihow_factory_maps/strict_roots/"
            "wikihow_computer_000092"
        )
        if not maps.exists():
            self.skipTest("generated maps audit fixture is unavailable")
        try:
            bundle = load_task_bundle(maps)
        except BundleError as exc:
            self.skipTest(f"generated maps fixture uses an older schema: {exc}")
        result = evaluate_counterfactuals(bundle)
        self.assertFalse(result["valid"])
        grounding = result["variants"][0]["decision_grounding"]
        self.assertEqual(
            grounding["missing_baseline_axes"],
            ["$state.task.coverage_scope", "$state.task.requested_area"],
        )

    def test_contract_requires_causal_features_and_shortcuts(self) -> None:
        invalid = {
            "contract_version": "task-contract-v1",
            "goal": "Write one value.",
            "goal_predicates": [
                {"id": "done", "predicate": {"eq": ["$state.done", True]}}
            ],
            "invariants": [],
            "forbidden_shortcuts": [],
            "expected_reasoning_features": [],
            "counterfactual_axes": [],
        }
        errors = validate_contract(invalid, {"done": False})
        self.assertTrue(any("forbidden_shortcuts" in error for error in errors))
        self.assertTrue(any("causal dependency" in error for error in errors))

    def test_contract_rejects_executable_bundle_payloads(self) -> None:
        bundle = load_task_bundle(FIXTURE)
        contract = copy.deepcopy(bundle.contract)
        contract["environment"] = copy.deepcopy(bundle.environment)
        contract["bindings"] = copy.deepcopy(bundle.bindings)
        contract["reference_plan"] = copy.deepcopy(bundle.reference_plan)
        errors = validate_contract(contract)
        self.assertTrue(any("unexpected contract fields" in error for error in errors))
        normalized = normalize_contract(contract)
        self.assertEqual(normalized, bundle.contract)
        baseline = evaluate_factory_bundle(bundle, 0.6)
        normalized_report = evaluate_factory_bundle(
            replace(bundle, contract=normalized), 0.6
        )
        self.assertTrue(baseline["valid"], baseline["errors"])
        self.assertEqual(normalized_report["valid"], baseline["valid"])
        self.assertEqual(normalized_report["errors"], baseline["errors"])
        self.assertEqual(
            normalized_report["causal_validation"]["metrics"],
            baseline["causal_validation"]["metrics"],
        )
        self.assertEqual(normalized_report["ablation"], baseline["ablation"])
        self.assertEqual(
            normalized_report["counterfactual_validation"],
            baseline["counterfactual_validation"],
        )

    def test_state_schema_completion_adds_only_later_written_paths(self) -> None:
        contract = {
            "goal_predicates": [
                {"id": "done", "predicate": {"eq": ["$state.result.done", True]}}
            ],
            "invariants": [],
        }
        candidate = {
            "environment": {
                "initial_state": {"result": {}},
                "capabilities": {
                    "finish": {
                        "branches": [
                            {
                                "when": True,
                                "response": {},
                                "effects": [
                                    {"set": "$state.result.done", "value": True}
                                ],
                            }
                        ]
                    }
                },
            }
        }
        completed, paths = complete_initial_state_schema(contract, candidate)
        self.assertEqual(paths, ["$state.result.done"])
        self.assertIsNone(completed["environment"]["initial_state"]["result"]["done"])
        self.assertNotIn("done", candidate["environment"]["initial_state"]["result"])

    def test_state_schema_completion_preserves_absent_exists_semantics(self) -> None:
        contract = {
            "goal_predicates": [
                {"id": "created", "predicate": {"exists": "$state.result"}}
            ],
            "invariants": [],
        }
        candidate = {
            "environment": {
                "initial_state": {"result": None},
                "capabilities": {
                    "finish": {
                        "branches": [
                            {
                                "when": True,
                                "response": {},
                                "effects": [
                                    {
                                        "set": "$state.result",
                                        "value": {"done": True},
                                    }
                                ],
                            }
                        ]
                    }
                },
            }
        }
        completed, paths = complete_initial_state_schema(contract, candidate)
        self.assertEqual(paths, ["delete-null:$state.result"])
        self.assertNotIn("result", completed["environment"]["initial_state"])

    def test_state_schema_completion_removes_null_absent_object(self) -> None:
        contract = {
            "goal_predicates": [
                {"id": "created", "predicate": {"exists": "$state.result"}}
            ],
            "invariants": [
                {
                    "id": "absent_before_create",
                    "predicate": {
                        "any": [
                            {"not_exists": "$state.result"},
                            {"eq": ["$state.result.done", True]},
                        ]
                    },
                }
            ],
        }
        candidate = {
            "environment": {
                "initial_state": {"result": None},
                "capabilities": {
                    "finish": {
                        "branches": [
                            {
                                "when": True,
                                "response": {},
                                "effects": [
                                    {
                                        "set": "$state.result",
                                        "value": {"done": True},
                                    }
                                ],
                            }
                        ]
                    }
                },
            }
        }
        completed, paths = complete_initial_state_schema(contract, candidate)
        self.assertEqual(paths, ["delete-null:$state.result"])
        self.assertNotIn("result", completed["environment"]["initial_state"])

    def test_contract_rejects_semantic_pseudo_predicates_without_state(self) -> None:
        invalid = {
            "contract_version": "task-contract-v1",
            "goal": "Publish current content.",
            "goal_predicates": [
                {
                    "id": "current",
                    "predicate": {
                        "equals": ["$state.published_revision", "$state.current_revision"]
                    },
                }
            ],
            "invariants": [],
            "forbidden_shortcuts": ["do not claim success"],
            "expected_reasoning_features": ["observation_dependent_decision"],
            "counterfactual_axes": [],
        }
        errors = validate_contract(invalid)
        self.assertTrue(any("unsupported predicate operator 'equals'" in error for error in errors))

    def test_contract_rejects_terminal_condition_as_initial_invariant(self) -> None:
        invalid = {
            "contract_version": "task-contract-v1",
            "goal": "Publish an assignment.",
            "goal_predicates": [
                {"id": "published", "predicate": {"eq": ["$state.published", True]}}
            ],
            "invariants": [
                {"id": "must_be_published", "predicate": {"eq": ["$state.published", True]}}
            ],
            "forbidden_shortcuts": ["do not claim success"],
            "expected_reasoning_features": ["delayed_handle_use"],
            "counterfactual_axes": [],
        }
        errors = validate_contract(invalid, {"published": False})
        self.assertTrue(any("must hold on the initial state" in error for error in errors))

    def test_alternate_api_preserves_hidden_task_semantics(self) -> None:
        bundle = load_task_bundle(FIXTURE)
        rendered = render_alternate_api(bundle, seed="test_api")
        self.assertNotEqual(bundle.tools[0]["name"], rendered.tools[0]["name"])
        report = run_reference_plan(rendered)
        validation = validate_episode(rendered, report)
        self.assertTrue(validation["valid"], validation["errors"])
        identifiability = validate_tool_identifiability(rendered)
        self.assertTrue(identifiability["valid"], identifiability["errors"])
        self.assertEqual(semantic_fingerprint(bundle), semantic_fingerprint(rendered))

    def test_opaque_api_without_descriptions_is_rejected(self) -> None:
        bundle = load_task_bundle(FIXTURE)
        rendered = render_alternate_api(bundle, seed="unidentifiable_api")
        bindings = copy.deepcopy(rendered.bindings)
        for tool in bindings["tools"]:
            tool["description"] = ""
        unidentifiable = replace(rendered, bindings=bindings)
        result = validate_tool_identifiability(unidentifiable)
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("no public semantic description" in error for error in result["errors"])
        )

    def test_scanner_opaque_api_has_distinguishable_public_affordances(self) -> None:
        scanner = PROJECT_ROOT / (
            "outputs/task_first/grounded_adaptive_v2/scanner/bundles/"
            "wikihow_computer_000068__g1_send_forwarded_email_recovery"
            "__g2_send_forwarded_email_route"
        )
        if not scanner.exists():
            self.skipTest("generated scanner audit fixture is unavailable")
        try:
            scanner_bundle = load_task_bundle(scanner)
        except BundleError as exc:
            self.skipTest(f"generated scanner fixture uses an older schema: {exc}")
        rendered = render_alternate_api(
            scanner_bundle, seed="scanner_identifiability_audit"
        )
        result = validate_tool_identifiability(rendered)
        self.assertTrue(result["valid"], result["errors"])
        metrics = result["metrics"]
        self.assertEqual(metrics["described_tool_count"], metrics["tool_count"])
        self.assertEqual(
            metrics["described_parameter_count"], metrics["parameter_count"]
        )
        self.assertFalse(metrics["indistinguishable_groups"])

    def test_alternate_api_renders_counterfactual_solution_actions(self) -> None:
        bundle = evolve_once(
            load_task_bundle(FIXTURE), "execution_route_branch_v1"
        ).product.bundle
        rendered = render_alternate_api(bundle, seed="counterfactual_api")
        public_names = {tool["name"] for tool in rendered.tools}
        for variant in rendered.reference_plan["counterfactuals"]:
            self.assertTrue(
                all(action["tool"] in public_names for action in variant["actions"])
            )
        result = evaluate_counterfactuals(rendered)
        self.assertTrue(result["valid"], result["variants"])

    def test_action_ablation_detects_necessary_steps(self) -> None:
        bundle = load_task_bundle(FIXTURE)
        result = evaluate_action_ablation(bundle)
        self.assertGreaterEqual(result["necessary_action_ratio"], 0.6)

    def test_training_export_metrics_do_not_include_verifier_internals(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "export_task_first_sft", SCRIPTS / "export_task_first_sft.py"
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        public = module.public_causal_metrics(
            {
                "steps": 4,
                "goal_paths": ["$state.private"],
                "observation_dependent_branches": [{"branch": "private_branch"}],
            },
            {
                "meaningful_planning_decision_count": 2,
                "decision_entropy_bits": 1.5,
                "decisions": [{"changed_axes": ["$state.private_axis"]}],
            },
        )
        serialized = json.dumps(public)
        self.assertNotIn("$state", serialized)
        self.assertNotIn("private_branch", serialized)
        self.assertEqual(public["meaningful_planning_decision_count"], 2)
        self.assertEqual(public["decision_entropy_bits"], 1.5)

        bundle = load_task_bundle(FIXTURE)
        report = run_reference_plan(bundle)
        row = module.build_sft_row(
            {"episode": report, "validation": validate_episode(bundle, report)}
        )
        self.assertIsNotNone(row)
        self.assertEqual(row["id"], bundle.task_id)

    def test_contract_output_feeds_bundle_request_with_seed_context(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "build_task_first_requests", SCRIPTS / "build_task_first_requests.py"
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        contract = {"contract_version": "task-contract-v1", "goal": "publish"}
        seed = {"objective": "release an artifact"}
        recovered_seed, recovered_contract = module.request_inputs(
            {"json_response": contract, "metadata": {"seed": seed}}, "bundle"
        )
        self.assertEqual(recovered_contract, contract)
        self.assertEqual(recovered_seed, seed)

    def test_task_first_repair_validation_errors_are_indexed(self) -> None:
        import importlib.util
        import tempfile

        spec = importlib.util.spec_from_file_location(
            "build_task_first_requests_repair", SCRIPTS / "build_task_first_requests.py"
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp_name:
            report = Path(temp_name) / "report.json"
            report.write_text(
                json.dumps(
                    {"rejected": [{"id": "task_1", "errors": ["invalid binding"]}]}
                ),
                encoding="utf-8",
            )
            indexed = module.validation_errors_by_id(report)
        self.assertEqual(indexed, {"task_1": ["invalid binding"]})

    def test_task_first_repair_reads_nested_rollout_validation(self) -> None:
        import importlib.util
        import tempfile

        spec = importlib.util.spec_from_file_location(
            "build_task_first_requests_nested", SCRIPTS / "build_task_first_requests.py"
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp_name:
            report = Path(temp_name) / "result.json"
            report.write_text(
                json.dumps(
                    {
                        "validation": {
                            "task_id": "task_2",
                            "errors": ["invariant failed"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            indexed = module.validation_errors_by_id(report)
        self.assertEqual(indexed["task_2"][0], "invariant failed")
        self.assertTrue(indexed["task_2"][1].startswith("metrics="))

    def test_materializer_rejects_invalid_binding_before_writing(self) -> None:
        fixture = load_task_bundle(FIXTURE)
        candidate = {
            "instruction": fixture.instruction,
            "environment": copy.deepcopy(fixture.environment),
            "bindings": copy.deepcopy(fixture.bindings),
            "reference_plan": copy.deepcopy(fixture.reference_plan),
        }
        first_tool = candidate["bindings"]["tools"][0]
        first_tool["capability"] = first_tool.pop("capability_id")
        import tempfile

        with tempfile.TemporaryDirectory() as temp_name:
            output_dir = Path(temp_name)
            with self.assertRaisesRegex(ValueError, "references unknown capability"):
                materialize_candidate(
                    output_dir,
                    task_id="invalid_binding",
                    contract=fixture.contract,
                    candidate=candidate,
                )
            self.assertFalse((output_dir / "invalid_binding").exists())

    def test_recursive_evolution_changes_task_semantics(self) -> None:
        parent = load_task_bundle(FIXTURE)
        first = evolve_once(parent, "policy_freshness_coupling_v1")
        self.assertTrue(first.report["accepted"], first.report["errors"])
        self.assertFalse(first.report["parent_plan_valid_on_child"])
        self.assertGreater(first.report["complexity_delta"]["goal_paths"], 0)

        second = evolve_once(first.product.bundle, "capacity_reservation_branch_v1")
        self.assertTrue(second.report["accepted"], second.report["errors"])
        self.assertFalse(second.report["parent_plan_valid_on_child"])
        self.assertTrue(second.report["counterfactual_gate_passed"])
        variant = second.report["counterfactual_validation"]["variants"][0]
        self.assertTrue(variant["adapted_valid"])
        self.assertFalse(variant["stale_strategy_valid"])
        self.assertTrue(
            any("episode did not satisfy" in error for error in variant["stale_strategy_errors"])
        )

    def test_counterfactual_adaptation_need_not_repeat_parent_only_recovery(self) -> None:
        parent = load_task_bundle(FIXTURE)
        child = evolve_once(parent, "capacity_reservation_branch_v1").product.bundle
        child.contract.setdefault("requirements", {})["semantic_recovery"] = True
        result = evaluate_counterfactuals(child)
        self.assertTrue(result["valid"], result["variants"])

    def test_counterfactual_admission_removes_invalid_optional_witness(self) -> None:
        parent = load_task_bundle(FIXTURE)
        reference_plan = copy.deepcopy(parent.reference_plan)
        reference_plan["counterfactuals"] = [
            {
                "name": "stale_still_works",
                "state_overrides": {"$state.cost.total": 1},
                "actions": copy.deepcopy(reference_plan["actions"]),
            }
        ]
        candidate = replace(parent, reference_plan=reference_plan)
        admitted, audit = admit_valid_counterfactuals(candidate)
        self.assertEqual(admitted.reference_plan["counterfactuals"], [])
        self.assertEqual(audit["rejected_count"], 1)
        self.assertTrue(audit["variants"][0]["stale_strategy_valid"])

    def test_adaptive_generation_requires_counterfactual_before_alignment(self) -> None:
        parent = load_task_bundle(FIXTURE)
        reference_plan = copy.deepcopy(parent.reference_plan)
        reference_plan["counterfactuals"] = []
        candidate = replace(parent, reference_plan=reference_plan)
        report = evaluate_factory_bundle(
            candidate,
            0.0,
            require_counterfactual=True,
        )
        self.assertFalse(report["valid"])
        self.assertIn(
            "adaptive generation requires at least one valid counterfactual policy change",
            report["errors"],
        )

    def test_strict_compile_does_not_delete_last_invalid_counterfactual(self) -> None:
        parent = load_task_bundle(FIXTURE)
        reference_plan = copy.deepcopy(parent.reference_plan)
        reference_plan["counterfactuals"] = [
            {
                "name": "stale_still_works",
                "state_overrides": {"$state.cost.total": 1},
                "actions": copy.deepcopy(reference_plan["actions"]),
            }
        ]
        candidate = {
            "instruction": parent.instruction,
            "environment": copy.deepcopy(parent.environment),
            "bindings": copy.deepcopy(parent.bindings),
            "reference_plan": reference_plan,
        }

        def evaluate_candidate(value: dict[str, object]) -> dict[str, object]:
            bundle = replace(
                parent,
                instruction=str(value["instruction"]),
                environment=copy.deepcopy(value["environment"]),
                bindings=copy.deepcopy(value["bindings"]),
                reference_plan=copy.deepcopy(value["reference_plan"]),
            )
            return evaluate_factory_bundle(
                bundle,
                0.0,
                require_counterfactual=True,
            )

        report = evaluate_candidate(candidate)
        repaired, repaired_report, _audit = compile_deterministic_candidate_repairs(
            task_id=parent.task_id,
            contract=parent.contract,
            candidate=candidate,
            report=report,
            metadata={"assigned_operator": "failure_diagnosis_recovery"},
            assigned_operator="failure_diagnosis_recovery",
            require_counterfactual=True,
            evaluate_candidate=evaluate_candidate,
        )
        self.assertFalse(repaired_report["valid"])
        self.assertEqual(len(repaired["reference_plan"]["counterfactuals"]), 1)

    def test_recursive_preparation_reuses_verified_execution_evidence(self) -> None:
        parent = evolve_once(
            load_task_bundle(FIXTURE), "capacity_reservation_branch_v1"
        ).product.bundle
        episode = run_reference_plan(parent)
        counterfactual = evaluate_counterfactuals(parent)
        baseline, baseline_audit = prepare_recursive_parent(parent)
        reused, reused_audit = prepare_recursive_parent(
            parent,
            counterfactual_evaluation=counterfactual,
            episode_report=episode,
        )
        self.assertEqual(baseline.contract, reused.contract)
        self.assertEqual(baseline.environment, reused.environment)
        self.assertEqual(baseline.bindings, reused.bindings)
        self.assertEqual(baseline.reference_plan, reused.reference_plan)
        self.assertEqual(
            baseline.manifest["evolution_hooks"], reused.manifest["evolution_hooks"]
        )
        self.assertFalse(baseline_audit["reused_execution_evidence"])
        self.assertTrue(reused_audit["reused_execution_evidence"])

    def test_counterfactual_bad_override_path_is_a_rejection_not_exception(self) -> None:
        parent = load_task_bundle(FIXTURE)
        reference_plan = copy.deepcopy(parent.reference_plan)
        reference_plan["counterfactuals"] = [
            {
                "name": "missing_path",
                "state_overrides": {"$state.unknown.path": True},
                "actions": copy.deepcopy(reference_plan["actions"]),
            }
        ]
        candidate = replace(parent, reference_plan=reference_plan)
        result = evaluate_counterfactuals(candidate)
        self.assertFalse(result["valid"])
        self.assertIn("does not exist", result["variants"][0]["adapted_errors"][0])

    def test_counterfactual_rejects_internally_inconsistent_initial_override(self) -> None:
        parent = load_task_bundle(FIXTURE)
        contract = copy.deepcopy(parent.contract)
        contract["invariants"] = [
            *contract.get("invariants", []),
            {
                "id": "published_channel_matches_release",
                "predicate": {
                    "any": [
                        {"eq": ["$state.release.distributed", False]},
                        {"eq": ["$state.release.channel", "beta"]},
                    ]
                },
            },
        ]
        reference_plan = copy.deepcopy(parent.reference_plan)
        reference_plan["counterfactuals"] = [
            {
                "name": "inconsistent_release_surface",
                "state_overrides": {
                    "$state.release.distributed": True,
                    "$state.release.channel": "stable",
                },
                "actions": copy.deepcopy(reference_plan["actions"]),
            }
        ]
        bundle = replace(parent, contract=contract, reference_plan=reference_plan)
        result = evaluate_counterfactuals(bundle)
        self.assertFalse(result["valid"])
        self.assertIn(
            "published_channel_matches_release",
            result["variants"][0]["adapted_errors"][0],
        )

    def test_wikihow_seed_requires_verbatim_grounding(self) -> None:
        source = "Title: export a track. Steps: open the trip list. select the track. tap export. save the file."
        seed = {
            "seed_version": "wikihow-seed-v1",
            "source_id": "track_source",
            "source_sha256": source_sha256(source),
            "objective": "Export a track file.",
            "source_supported_facts": [
                {"fact": "a", "evidence_spans": ["open the trip list"]},
                {"fact": "b", "evidence_spans": ["tap export"]},
            ],
            "normalized_steps": [
                {"id": "s1", "action": "open", "evidence_spans": ["open the trip list"]},
                {"id": "s2", "action": "select", "evidence_spans": ["select the track"]},
                {"id": "s3", "action": "export", "evidence_spans": ["tap export"]},
                {"id": "s4", "action": "save", "evidence_spans": ["invented span"]},
            ],
            "observable_affordances": [
                {"system": "trip app", "observations": ["tracks"], "actions": ["export"]}
            ],
            "environment_design_limits": ["discover identifiers"],
            "operator_feasibility": {"supported": True, "reason": "export produces an artifact"},
            "synthetic_extension": {
                "operator": "artifact_provenance",
                "requirement": "export the selected revision",
                "claimed_as_source": False,
            },
        }
        errors = validate_wikihow_seed(
            seed,
            source,
            assigned_operator="artifact_provenance",
            source_id="track_source",
        )
        self.assertTrue(any("not verbatim" in error for error in errors))

    def test_wikihow_seed_rejects_same_id_with_different_source_text(self) -> None:
        source = "Title: first source. Steps: one. two. three. four."
        seed = {
            "seed_version": "wikihow-seed-v1",
            "source_id": "colliding_id",
            "source_sha256": source_sha256(source),
        }
        errors = validate_wikihow_seed(
            seed,
            "Title: replacement source. Steps: a. b. c. d.",
            source_id="colliding_id",
        )
        self.assertIn("source_sha256 does not match the current source text", errors)

    def test_factory_operator_assignment_covers_distinct_patterns_first(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "run_wikihow_task_factory", SCRIPTS / "run_wikihow_task_factory.py"
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        catalog = json.loads(
            (SCRIPTS.parent / "config" / "task_rewrite_operators.json").read_text(
                encoding="utf-8"
            )
        )["operators"]
        rows = [{"text": "if there is an alternative option"}] * 3
        assigned = module.assign_operators(rows, catalog)
        self.assertEqual(len({item["id"] for item in assigned}), 3)

    def test_wikihow_extractor_id_uses_stable_source_index(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "extract_wikihow", SCRIPTS / "extract_wikihow_computer_use.py"
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        first = module.stable_row_id(
            "wikihow", source_index=1427, selected_index=2
        )
        second = module.stable_row_id(
            "wikihow", source_index=1427, selected_index=68
        )
        self.assertEqual(first, "wikihow_000001427")
        self.assertEqual(first, second)

    def test_factory_resume_preserves_seed_operator(self) -> None:
        import importlib.util
        import tempfile

        spec = importlib.util.spec_from_file_location(
            "run_wikihow_task_factory_resume", SCRIPTS / "run_wikihow_task_factory.py"
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        catalog = json.loads(
            (SCRIPTS.parent / "config" / "task_rewrite_operators.json").read_text(
                encoding="utf-8"
            )
        )["operators"]
        row = {"id": "resume_task", "text": "source"}
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            seed_path = output_dir / "tasks" / "resume_task" / "seed.json"
            seed_path.parent.mkdir(parents=True)
            seed_path.write_text(
                json.dumps({"synthetic_extension": {"operator": "artifact_provenance"}}),
                encoding="utf-8",
            )
            card = module.resume_operator_card(
                row,
                catalog[0],
                catalog,
                output_dir=output_dir,
                resume=True,
            )
        self.assertEqual(card["id"], "artifact_provenance")

    def test_factory_alternative_plan_requires_valid_counterfactual(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "run_wikihow_task_factory_gate", SCRIPTS / "run_wikihow_task_factory.py"
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        parent = load_task_bundle(FIXTURE)
        manifest = {
            **parent.manifest,
            "assigned_operator": "alternative_plan_affordance",
        }
        candidate = replace(parent, manifest=manifest)
        report = module.evaluate(candidate, 0.6)
        self.assertFalse(report["valid"])
        self.assertTrue(
            any("requires at least one valid state intervention" in error for error in report["errors"])
        )

    def test_archive_search_keeps_distinct_semantic_candidates(self) -> None:
        parent = load_task_bundle(FIXTURE)
        candidates, rejected = generate_candidates([parent])
        self.assertFalse(rejected)
        selected, selection_rejected = select_candidates(
            candidates, max_per_parent=len(candidates)
        )
        self.assertEqual(len(selected), len(candidates))
        self.assertFalse(selection_rejected)
        self.assertEqual(len({item.fingerprint for item in selected}), len(candidates))

    def test_semantic_hook_drives_portable_audit_evolution(self) -> None:
        parent = load_task_bundle(FIXTURE)
        evaluation = evolve_once(parent, "audit_checkpoint_v1")
        self.assertTrue(evaluation.report["accepted"], evaluation.report["errors"])
        self.assertFalse(evaluation.report["parent_plan_valid_on_child"])
        trace = evaluation.report["child_validation"]["metrics"]
        self.assertEqual(trace["goal_evidence_coverage"], 1.0)
        self.assertIn("software_release", evaluation.product.bundle.manifest["domain"])

        routed = evolve_once(evaluation.product.bundle, "execution_route_branch_v1")
        self.assertTrue(routed.report["accepted"], routed.report["errors"])
        self.assertFalse(routed.report["parent_plan_valid_on_child"])
        self.assertTrue(routed.report["counterfactual_gate_passed"])
        self.assertGreaterEqual(
            routed.report["action_ablation"]["necessary_action_ratio"], 0.6
        )

    def test_async_readiness_requires_pending_then_ready_observation(self) -> None:
        parent = load_task_bundle(FIXTURE)
        evaluation = evolve_once(parent, "async_readiness_retry_v1")
        self.assertTrue(evaluation.report["accepted"], evaluation.report["errors"])
        self.assertFalse(evaluation.report["parent_plan_valid_on_child"])
        metrics = evaluation.report["child_validation"]["metrics"]
        decisions = [
            item
            for item in metrics["observation_dependent_branches"]
            if item["public_tool"] == "poll_distribution_readiness"
        ]
        self.assertEqual(len(decisions), 1)
        decision = decisions[0]
        self.assertEqual(
            set(decision["branches"]), {"pending_retry", "ready_with_evidence"}
        )
        self.assertGreaterEqual(
            evaluation.report["action_ablation"]["necessary_action_ratio"], 0.6
        )
        self.assertNotIn("poll", evaluation.product.bundle.instruction.lower())
        self.assertEqual(
            evaluation.report["decision_metrics"][
                "meaningful_planning_decision_count"
            ],
            0,
        )

    def test_semantic_recovery_retries_same_action_after_state_repair(self) -> None:
        parent = load_task_bundle(FIXTURE)
        evaluation = evolve_once(parent, "semantic_failure_recovery_v1")
        self.assertTrue(evaluation.report["accepted"], evaluation.report["errors"])
        metrics = evaluation.report["child_validation"]["metrics"]
        conflict = "DISTRIBUTION_TARGET_STATE_CONFLICT"
        recovery = next(
            item for item in metrics["semantic_recoveries"] if item["error_code"] == conflict
        )
        trace = run_reference_plan(evaluation.product.bundle)["trace"]
        failed = trace[recovery["failure_step"] - 1]
        prepare_steps = [
            step
            for step in trace
            if step["public_tool"] == "prepare_distribution_action"
        ]
        self.assertEqual(len(prepare_steps), 2)
        stale_preparation = prepare_steps[0]["response"]
        current_preparation = prepare_steps[1]["response"]
        self.assertNotEqual(
            stale_preparation["preparation_handle"],
            current_preparation["preparation_handle"],
        )
        self.assertEqual(stale_preparation["prepared_revision"], 1)
        self.assertEqual(current_preparation["prepared_revision"], 2)
        self.assertEqual(
            prepare_steps[0]["environment_transitions"],
            [
                {
                    "path": "$state.distribution_recovery.current_revision",
                    "before": 1,
                    "after": 2,
                }
            ],
        )
        self.assertEqual(prepare_steps[1]["environment_transitions"], [])
        retried = next(
            step
            for step in trace[recovery["recovery_step"] :]
            if step["public_tool"] == failed["public_tool"]
        )
        self.assertEqual(
            failed["arguments"]["candidate_ref"]["value"],
            retried["arguments"]["candidate_ref"]["value"],
        )
        self.assertEqual(
            retried["arguments"]["repair_evidence_ref"]["provenance_kind"],
            "tool_observation_grounded",
        )
        self.assertEqual(
            failed["arguments"]["preparation_ref"]["value"],
            stale_preparation["preparation_handle"],
        )
        self.assertEqual(
            retried["arguments"]["preparation_ref"]["value"],
            current_preparation["preparation_handle"],
        )
        self.assertNotIn("diagnose", evaluation.product.bundle.instruction.lower())
        self.assertNotIn("repair_", evaluation.product.bundle.instruction.lower())

        routed = evolve_once(
            evaluation.product.bundle, "execution_route_branch_v1"
        )
        self.assertTrue(routed.report["accepted"], routed.report["errors"])
        self.assertGreaterEqual(
            routed.report["decision_metrics"]["decision_entropy_bits"], 1.0
        )

    def test_route_counterfactual_has_strategy_entropy(self) -> None:
        parent = load_task_bundle(FIXTURE)
        evaluation = evolve_once(parent, "execution_route_branch_v1")
        metrics = evaluation.report["decision_metrics"]
        self.assertEqual(metrics["meaningful_planning_decision_count"], 1)
        self.assertEqual(metrics["decision_entropy_bits"], 1.0)
        self.assertNotIn("inspect_", evaluation.product.bundle.instruction.lower())
        self.assertNotIn("reserve_", evaluation.product.bundle.instruction.lower())

    def test_recursive_operator_preserves_existing_counterfactual_solution(self) -> None:
        parent = load_task_bundle(FIXTURE)
        routed = evolve_once(parent, "execution_route_branch_v1")
        self.assertTrue(routed.report["accepted"], routed.report["errors"])
        existing_counterfactual = routed.product.bundle.reference_plan[
            "counterfactuals"
        ][0]
        reference_plan = copy.deepcopy(routed.product.bundle.reference_plan)
        reference_plan["counterfactuals"] = [copy.deepcopy(existing_counterfactual)]
        parent_with_counterfactual = replace(
            routed.product.bundle,
            manifest={
                **routed.product.bundle.manifest,
                "lineage": {
                    **routed.product.bundle.manifest["lineage"],
                    "operators": [],
                },
            },
            reference_plan=reference_plan,
        )
        evolved = evolve_once(parent_with_counterfactual, "async_readiness_retry_v1")
        self.assertTrue(evolved.report["accepted"], evolved.report["errors"])
        variants = evolved.report["counterfactual_validation"]["variants"]
        self.assertEqual(len(variants), 1)
        self.assertTrue(variants[0]["adapted_valid"])
        self.assertFalse(variants[0]["stale_strategy_valid"])

    def test_evolution_hook_is_inferred_from_runtime_provenance(self) -> None:
        parent = load_task_bundle(FIXTURE)
        report = run_reference_plan(parent)
        without_hook = replace(
            parent,
            manifest={
                key: copy.deepcopy(value)
                for key, value in parent.manifest.items()
                if key != "evolution_hooks"
            },
        )
        inferred = attach_inferred_evolution_hooks(without_hook, report)
        hook = inferred.manifest["evolution_hooks"]["audit_checkpoint"]
        self.assertEqual(hook["commit_tool"], "publish_candidate")
        self.assertFalse(hook["commit_last"])
        self.assertEqual(hook["verify_capability"], "release.inspect.v1")
        evaluation = evolve_once(inferred, "audit_checkpoint_v1")
        self.assertTrue(evaluation.report["accepted"], evaluation.report["errors"])

    def test_hook_prefers_broad_goal_mutation_over_later_redundant_write(self) -> None:
        parent = load_task_bundle(FIXTURE)
        report = run_reference_plan(parent)
        publish = next(step for step in report["trace"] if step["public_tool"] == "publish_candidate")
        verify = report["trace"][-1]
        redundant = copy.deepcopy(verify)
        redundant["step"] = verify["step"]
        redundant["public_tool"] = "redundant_observation"
        redundant["capability_id"] = "release.inspect.v1"
        redundant["selected_branch"] = verify["selected_branch"]
        redundant["write_set"] = [publish["write_set"][0]]
        verify["step"] += 1
        report["trace"].insert(-1, redundant)
        inferred = attach_inferred_evolution_hooks(parent, report)
        hook = inferred.manifest["evolution_hooks"]["audit_checkpoint"]
        self.assertEqual(hook["commit_tool"], "publish_candidate")


if __name__ == "__main__":
    unittest.main()
