from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.llm_client import (
    call_codex,
    call_responses_compatible,
    codex_prompt,
    extract_responses_text,
    parse_json_object,
)


def load_factory_module():
    import importlib.util

    path = Path(__file__).parents[1] / "scripts" / "run_wikihow_task_factory.py"
    spec = importlib.util.spec_from_file_location("run_wikihow_task_factory_perf", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CodexProviderTests(unittest.TestCase):
    def test_responses_text_extracts_nested_output(self) -> None:
        body = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": '{"ok":true}'}],
                }
            ]
        }
        self.assertEqual(extract_responses_text(body), '{"ok":true}')

    def test_factory_responses_config_inherits_nearest_project(self) -> None:
        module = load_factory_module()
        with tempfile.TemporaryDirectory() as temp_name:
            config_path = Path(temp_name) / "config.toml"
            config_path.write_text(
                f'''
AZURE_TEST_KEY = "private-test-value"

[projects."{Path.home()}"]
model = "gpt-test"
model_provider = "azure"
model_reasoning_effort = "high"
service_tier = "fast"

[model_providers.azure]
base_url = "https://example.test/openai/v1"
env_key = "AZURE_TEST_KEY"
wire_api = "responses"

[model_providers.azure.http_headers]
api-key = "AZURE_TEST_KEY"
'''.strip(),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=False):
                module.configure_responses_provider(config_path)
                self.assertEqual(
                    os.environ["GEM_RESPONSES_BASE_URL"],
                    "https://example.test/openai/v1",
                )
                self.assertEqual(os.environ["GEM_RESPONSES_MODEL"], "gpt-test")
                self.assertEqual(os.environ["GEM_RESPONSES_API_KEY_HEADER"], "api-key")
                self.assertEqual(os.environ["GEM_RESPONSES_SERVICE_TIER"], "priority")

    @patch("scripts.llm_client.urllib.request.urlopen")
    def test_responses_client_uses_json_mode_and_api_key_header(
        self, urlopen: unittest.mock.Mock
    ) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "output_text": '{"ok":true}',
                        "usage": {"input_tokens": 11, "output_tokens": 2},
                    }
                ).encode()

        import json

        urlopen.return_value = Response()
        env = {
            "GEM_RESPONSES_BASE_URL": "https://example.test/openai/v1",
            "GEM_RESPONSES_API_KEY": "test-key",
            "GEM_RESPONSES_API_KEY_HEADER": "api-key",
            "GEM_RESPONSES_MODEL": "gpt-test",
            "GEM_RESPONSES_REASONING_EFFORT": "high",
            "GEM_RESPONSES_SERVICE_TIER": "fast",
        }
        with patch.dict(os.environ, env, clear=False):
            raw, usage = call_responses_compatible(
                [{"role": "user", "content": "Return JSON."}], 128, 0.0
            )
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "https://example.test/openai/v1/responses")
        self.assertEqual(request.headers["Api-key"], "test-key")
        self.assertEqual(payload["text"], {"format": {"type": "json_object"}})
        self.assertEqual(payload["reasoning"], {"effort": "high"})
        self.assertEqual(payload["service_tier"], "fast")
        self.assertNotIn("temperature", payload)
        self.assertEqual(raw, '{"ok":true}')
        self.assertEqual(usage["input_tokens"], 11)

    @patch("time.sleep")
    def test_factory_json_call_retries_transport_failures_and_profiles(
        self, sleep: unittest.mock.Mock
    ) -> None:
        module = load_factory_module()
        with patch.object(
            module,
            "call_chat",
            side_effect=[RuntimeError("temporary"), ('{"ok":true}', {"tokens": 7})],
        ) as call:
            value, usage = module.call_json(
                "return json",
                max_tokens=64,
                temperature=0.0,
                provider="gemini",
                request_retries=1,
                retry_backoff=0.1,
            )
        self.assertEqual(value, {"ok": True})
        self.assertEqual(call.call_count, 2)
        self.assertEqual(usage["transport_attempts"], 2)
        self.assertEqual(usage["prompt_chars"], len("return json"))
        self.assertEqual(usage["response_chars"], len('{"ok":true}'))
        self.assertGreaterEqual(usage["latency_sec"], 0)
        sleep.assert_called_once_with(0.1)

    @patch("time.sleep")
    def test_factory_json_call_retries_malformed_output_with_larger_budget(
        self, sleep: unittest.mock.Mock
    ) -> None:
        module = load_factory_module()
        with patch.object(
            module,
            "call_chat",
            side_effect=[('{"truncated":', {"output_tokens": 64}), ('{"ok":true}', {})],
        ) as call:
            value, usage = module.call_json(
                "return json",
                max_tokens=64,
                temperature=0.0,
                provider="responses",
                request_retries=1,
                retry_backoff=0.1,
            )
        self.assertEqual(value, {"ok": True})
        self.assertEqual(call.call_count, 2)
        self.assertEqual(call.call_args_list[0].kwargs["max_tokens"], 64)
        self.assertEqual(call.call_args_list[1].kwargs["max_tokens"], 128)
        self.assertEqual(usage["request_count"], 2)
        self.assertEqual(usage["physical_prompt_chars"], len("return json") * 2)
        self.assertEqual(
            usage["physical_response_chars"],
            len('{"truncated":') + len('{"ok":true}'),
        )
        self.assertIn("error", usage["attempts"][0])
        sleep.assert_called_once_with(0.1)

    def test_factory_json_failure_retains_billable_attempt_profile(self) -> None:
        module = load_factory_module()
        with patch.object(
            module,
            "call_chat",
            return_value=('{"truncated":', {"output_tokens": 64}),
        ):
            with self.assertRaises(module.ModelJsonError) as context:
                module.call_json(
                    "return json",
                    max_tokens=64,
                    temperature=0.0,
                    provider="responses",
                    request_retries=0,
                )
        self.assertEqual(context.exception.usage["request_count"], 1)
        self.assertEqual(context.exception.usage["response_chars"], 13)
        self.assertEqual(context.exception.usage["attempts"][0]["output_tokens"], 64)

    def test_patch_improvement_rejects_new_error_classes(self) -> None:
        module = load_factory_module()
        previous = {"missing provenance", "invalid goal", "bad route"}
        self.assertTrue(
            module.patch_strictly_improves(
                previous, {"valid": False, "errors": ["missing provenance"]}
            )
        )
        self.assertFalse(
            module.patch_strictly_improves(
                previous, {"valid": False, "errors": ["new error"]}
            )
        )
        self.assertFalse(
            module.patch_strictly_improves(
                previous,
                {"valid": False, "errors": ["missing provenance", "new error"]},
            )
        )
        self.assertTrue(module.patch_strictly_improves(previous, {"valid": True}))

    def test_contract_state_path_mismatch_is_owned_by_candidate_repair(self) -> None:
        module = load_factory_module()
        contract = {
            "contract_version": "task-contract-v1",
            "goal": "Finish the task.",
            "goal_predicates": [
                {
                    "id": "done",
                    "predicate": {"eq": ["$state.result.done", True]},
                }
            ],
            "invariants": [],
            "forbidden_shortcuts": ["Do not claim success."],
            "expected_reasoning_features": ["delayed_handle_use"],
            "counterfactual_axes": [],
        }
        candidate = {"environment": {"initial_state": {"result": {}}}}
        structural, compatibility = module.contract_error_ownership(
            contract, candidate
        )
        self.assertEqual(structural, [])
        self.assertTrue(any("cannot be evaluated" in error for error in compatibility))

    def test_repair_quality_never_prefers_static_regression(self) -> None:
        module = load_factory_module()
        execution = {
            "valid": False,
            "phase": "execution",
            "errors": ["one path mismatch"],
            "causal_validation": {
                "metrics": {
                    "goal_evidence_coverage": 0.8,
                    "unexplained_arguments": [],
                }
            },
            "ablation": {"necessary_action_ratio": 0.7},
            "counterfactual_validation": {"valid": False},
        }
        static = {
            "valid": False,
            "phase": "static",
            "errors": ["missing bindings", "missing actions"],
        }
        self.assertGreater(
            module.repair_quality_key(execution), module.repair_quality_key(static)
        )

    def test_repair_quality_prefers_valid_and_stronger_execution_candidate(self) -> None:
        module = load_factory_module()
        weak = {
            "valid": False,
            "phase": "execution",
            "errors": ["a", "b"],
            "causal_validation": {
                "metrics": {
                    "goal_evidence_coverage": 0.5,
                    "unexplained_arguments": ["x"],
                }
            },
            "ablation": {"necessary_action_ratio": 0.5},
            "counterfactual_validation": {"valid": False},
        }
        strong = {
            **weak,
            "errors": ["a"],
            "causal_validation": {
                "metrics": {
                    "goal_evidence_coverage": 1.0,
                    "unexplained_arguments": [],
                }
            },
        }
        self.assertGreater(
            module.repair_quality_key(strong), module.repair_quality_key(weak)
        )
        self.assertGreater(
            module.repair_quality_key({"valid": True}),
            module.repair_quality_key(strong),
        )
        samples = [
            (1, {"sample": 2}, strong, [], {}),
            (0, {"sample": 1}, {"valid": True}, [], {}),
        ]
        self.assertEqual(module.select_best_bundle_sample(samples)[0], 0)
        tied = [
            (1, {"sample": 2}, strong, [], {}),
            (0, {"sample": 1}, strong, [], {}),
        ]
        self.assertEqual(module.select_best_bundle_sample(tied)[0], 0)

    def test_quality_floor_is_component_wise_and_cannot_be_offset(self) -> None:
        module = load_factory_module()
        report = {
            "phase": "execution",
            "causal_validation": {
                "metrics": {
                    "steps": 10,
                    "max_delayed_handle_distance": 20,
                    "handle_chain_depth": 3,
                    "semantic_recoveries": [],
                    "observation_dependent_branch_count": 0,
                }
            },
            "ablation": {"necessary_action_ratio": 1.0},
            "counterfactual_validation": {
                "counterfactual_count": 2,
                "decision_metrics": {"decision_entropy_bits": 2.0},
            },
        }
        floor = {
            "steps": 12,
            "max_delayed_handle_distance": 5,
            "handle_chain_depth": 3,
            "semantic_recovery_count": 0,
            "observation_dependent_branch_count": 1,
            "counterfactual_count": 2,
            "decision_entropy_bits": 2.0,
            "necessary_action_ratio": 0.7,
        }
        errors = module.quality_floor_errors(report, floor)
        self.assertEqual(
            errors,
            [
                "quality floor regression: steps=10 < 12",
                "quality floor regression: observation_dependent_branch_count=0 < 1",
            ],
        )

    def test_factory_model_cache_replays_only_exact_requests(self) -> None:
        module = load_factory_module()
        with tempfile.TemporaryDirectory() as temp_name:
            cache_dir = Path(temp_name)
            with patch.object(
                module,
                "call_chat",
                return_value=('{"ok":true}', {"output_tokens": 2}),
            ) as call:
                first, first_usage = module.call_json(
                    "return json",
                    max_tokens=64,
                    temperature=0.0,
                    provider="responses",
                    reasoning_effort="high",
                    cache_dir=cache_dir,
                )
                second, second_usage = module.call_json(
                    "return json",
                    max_tokens=64,
                    temperature=0.0,
                    provider="responses",
                    reasoning_effort="high",
                    cache_dir=cache_dir,
                )
                module.call_json(
                    "return json",
                    max_tokens=128,
                    temperature=0.0,
                    provider="responses",
                    reasoning_effort="high",
                    cache_dir=cache_dir,
                )
        self.assertEqual(first, second)
        self.assertEqual(call.call_count, 2)
        self.assertFalse(first_usage.get("cache_hit", False))
        self.assertTrue(second_usage["cache_hit"])
        self.assertEqual(second_usage["request_count"], 0)
        self.assertEqual(second_usage["physical_prompt_chars"], 0)
        self.assertEqual(second_usage["physical_response_chars"], 0)

    def test_rejected_audit_evicts_cached_model_chain(self) -> None:
        module = load_factory_module()
        with tempfile.TemporaryDirectory() as temp_name:
            cache_dir = Path(temp_name)
            seed_key = "a" * 64
            bundle_key = "b" * 64
            module._write_model_cache(cache_dir, seed_key, {"seed": True})
            module._write_model_cache(cache_dir, bundle_key, {"bundle": True})
            audit = {
                "stages": [
                    {"stage": "seed", "usage": {"cache_key": seed_key}},
                    {"stage": "bundle", "usage": {"cache_key": bundle_key}},
                ]
            }
            self.assertEqual(module.evict_audit_cache_entries(audit, cache_dir), 1)
            self.assertEqual(
                module._read_model_cache(cache_dir, seed_key), {"seed": True}
            )
            self.assertIsNone(module._read_model_cache(cache_dir, bundle_key))

    def test_stage_profiles_count_physical_work_and_cache_hits(self) -> None:
        module = load_factory_module()
        profiles = module.summarize_stage_profiles(
            [
                {
                    "stages": [
                        {
                            "stage": "bundle",
                            "usage": {
                                "request_count": 1,
                                "latency_sec": 2.25,
                                "prompt_chars": 100,
                                "response_chars": 20,
                            },
                        },
                        {
                            "stage": "bundle",
                            "usage": {
                                "request_count": 0,
                                "latency_sec": 0.0,
                                "prompt_chars": 100,
                                "response_chars": 20,
                                "cache_hit": True,
                            },
                        },
                    ]
                }
            ]
        )
        self.assertEqual(
            profiles["bundle"],
            {
                "stage_calls": 2,
                "physical_requests": 1,
                "latency_sec": 2.25,
                "logical_prompt_chars": 200,
                "physical_prompt_chars": 100,
                "physical_response_chars": 20,
                "cache_hits": 1,
                "avg_request_latency_sec": 2.25,
            },
        )

    def test_provenance_patch_context_keeps_original_edit_paths(self) -> None:
        module = load_factory_module()
        candidate = {
            "instruction": "Publish the observed item.",
            "environment": {
                "initial_state": {"item": {"number": 2}},
                "capabilities": {
                    "item/inspect": {
                        "branches": [
                            {
                                "id": "visible",
                                "when": True,
                                "response": {"number": "$state.item.number"},
                            },
                            {"id": "unused", "when": True, "response": {}},
                        ]
                    },
                    "item.publish": {
                        "branches": [
                            {"id": "published", "when": True, "response": {}}
                        ]
                    },
                },
            },
            "bindings": {
                "tools": [
                    {
                        "name": "inspect_item",
                        "capability_id": "item/inspect",
                        "parameters": {"type": "object", "properties": {}},
                    },
                    {
                        "name": "publish_item",
                        "capability_id": "item.publish",
                        "parameters": {"type": "object", "properties": {}},
                    },
                ]
            },
            "reference_plan": {
                "actions": [
                    {"tool": "inspect_item", "arguments": {}},
                    {"tool": "publish_item", "arguments": {"number": 2}},
                ],
                "counterfactuals": [],
            },
        }
        report = {
            "causal_validation": {
                "metrics": {
                    "unexplained_arguments": [
                        {
                            "step": 2,
                            "tool": "publish_item",
                            "argument": "number",
                            "value": 2,
                        }
                    ]
                }
            },
            "episode": {
                "trace": [
                    {
                        "step": 1,
                        "public_tool": "inspect_item",
                        "capability_id": "item/inspect",
                        "selected_branch": "visible",
                        "arguments": {},
                        "response": {},
                    },
                    {
                        "step": 2,
                        "public_tool": "publish_item",
                        "capability_id": "item.publish",
                        "selected_branch": "published",
                        "arguments": {},
                        "response": {},
                    },
                ]
            },
        }
        context = module.provenance_patch_context(candidate, report)
        self.assertIsNotNone(context)
        assert context is not None
        self.assertEqual(
            [item["path"] for item in context["selected_branch_fragments"]],
            [
                "/environment/capabilities/item.publish/branches/0",
                "/environment/capabilities/item~1inspect/branches/0",
            ],
        )
        self.assertNotIn(
            "unused",
            json.dumps(context["selected_branch_fragments"]),
        )
        self.assertEqual(
            [item["path"] for item in context["reference_action_fragments"]],
            ["/reference_plan/actions/0", "/reference_plan/actions/1"],
        )

    def test_downstream_seed_context_preserves_semantics_without_duplicate_spans(
        self,
    ) -> None:
        module = load_factory_module()
        seed = {
            "seed_version": "wikihow-seed-v1",
            "source_id": "source-1",
            "source_sha256": "abc",
            "title": "Scan",
            "objective": "Scan the document.",
            "source_supported_facts": [
                {"fact": "The document has pages.", "evidence_spans": ["pages"]}
            ],
            "normalized_steps": [
                {
                    "id": "step_01",
                    "action": "Inspect",
                    "inputs": ["document"],
                    "outputs": ["page count"],
                    "evidence_spans": ["inspect pages"],
                }
            ],
            "observable_affordances": [{"system": "scanner"}],
            "synthetic_extension": {"operator": "alternative_plan_affordance"},
        }
        projected = module.downstream_seed_context(seed)
        self.assertEqual(
            projected["source_supported_facts"], ["The document has pages."]
        )
        self.assertEqual(projected["normalized_steps"][0]["action"], "Inspect")
        self.assertNotIn("evidence_spans", json.dumps(projected))
        self.assertIn("seed.json", projected["grounding_note"])

    def test_parse_json_object_uses_first_complete_object(self) -> None:
        self.assertEqual(parse_json_object('prefix {"ok":true}\n{"extra":1}'), {"ok": True})
    def test_prompt_preserves_roles_and_requests_json_only(self) -> None:
        prompt = codex_prompt(
            [
                {"role": "system", "content": "Follow policy."},
                {"role": "user", "content": "Build a task."},
            ],
            2048,
        )
        self.assertIn('role="SYSTEM"', prompt)
        self.assertIn('role="USER"', prompt)
        self.assertIn("Return only the requested JSON object", prompt)
        self.assertIn("2048", prompt)

    @patch("scripts.llm_client.shutil.which", return_value="/usr/local/bin/codex")
    @patch("scripts.llm_client.subprocess.run")
    def test_call_codex_uses_ephemeral_read_only_execution(
        self, run: unittest.mock.Mock, _which: unittest.mock.Mock
    ) -> None:
        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            output_index = command.index("--output-last-message") + 1
            Path(command[output_index]).write_text('{"ok":true}', encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")

        run.side_effect = fake_run
        raw, usage = call_codex(
            [{"role": "user", "content": "Return an object."}],
            max_tokens=128,
            temperature=0.1,
        )
        command = run.call_args.args[0]
        self.assertEqual(raw, '{"ok":true}')
        self.assertEqual(usage["provider"], "codex")
        self.assertIn("--ephemeral", command)
        self.assertIn("read-only", command)
        self.assertNotIn("--output-schema", command)

    @patch("scripts.llm_client.shutil.which", return_value="/usr/local/bin/codex")
    @patch("scripts.llm_client.subprocess.run")
    def test_call_codex_writes_and_uses_response_schema(
        self, run: unittest.mock.Mock, _which: unittest.mock.Mock
    ) -> None:
        schema = {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }

        def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            output_path = Path(command[command.index("--output-last-message") + 1])
            output_path.write_text('{"ok":true}', encoding="utf-8")
            schema_path = Path(command[command.index("--output-schema") + 1])
            self.assertEqual(json.loads(schema_path.read_text()), schema)
            return subprocess.CompletedProcess(command, 0, "", "")

        import json

        run.side_effect = fake_run
        raw, usage = call_codex(
            [{"role": "user", "content": "Return an object."}],
            128,
            0.0,
            response_schema=schema,
        )
        self.assertEqual(raw, '{"ok":true}')
        self.assertTrue(usage["structured_output"])

    @patch.dict("scripts.llm_client.os.environ", {"GEM_CODEX_IGNORE_USER_CONFIG": "1"})
    @patch("scripts.llm_client.shutil.which", return_value="/usr/local/bin/codex")
    @patch("scripts.llm_client.subprocess.run")
    def test_call_codex_can_use_builtin_provider_defaults(
        self, run: unittest.mock.Mock, _which: unittest.mock.Mock
    ) -> None:
        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            output_index = command.index("--output-last-message") + 1
            Path(command[output_index]).write_text('{"ok":true}', encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")

        run.side_effect = fake_run
        _raw, usage = call_codex(
            [{"role": "user", "content": "Return an object."}], 128, 0.0
        )
        command = run.call_args.args[0]
        self.assertIn("--ignore-user-config", command)
        self.assertEqual(usage["config_source"], "built_in_defaults")

    @patch("scripts.llm_client.shutil.which", return_value="/usr/local/bin/codex")
    @patch("scripts.llm_client.subprocess.run")
    def test_call_codex_honors_stage_reasoning_override(
        self, run: unittest.mock.Mock, _which: unittest.mock.Mock
    ) -> None:
        def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            output_index = command.index("--output-last-message") + 1
            Path(command[output_index]).write_text('{"ok":true}', encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")

        run.side_effect = fake_run
        with patch.dict(
            os.environ, {"GEM_CODEX_REASONING_EFFORT": "xhigh"}, clear=False
        ):
            _raw, usage = call_codex(
                [{"role": "user", "content": "Return an object."}],
                128,
                0.0,
                reasoning_effort_override="medium",
            )
        command = run.call_args.args[0]
        self.assertIn('model_reasoning_effort="medium"', command)
        self.assertEqual(usage["reasoning_effort"], "medium")

    @patch("scripts.llm_client.shutil.which", return_value="/usr/local/bin/codex")
    @patch("scripts.llm_client.subprocess.run")
    def test_call_codex_forwards_project_service_tier(
        self, run: unittest.mock.Mock, _which: unittest.mock.Mock
    ) -> None:
        def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            output_index = command.index("--output-last-message") + 1
            Path(command[output_index]).write_text('{"ok":true}', encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")

        run.side_effect = fake_run
        with patch.dict(os.environ, {"GEM_CODEX_SERVICE_TIER": "fast"}, clear=False):
            _raw, usage = call_codex(
                [{"role": "user", "content": "Return an object."}], 128, 0.0
            )
        command = run.call_args.args[0]
        self.assertIn('service_tier="fast"', command)
        self.assertEqual(usage["service_tier"], "fast")

    @patch("scripts.llm_client.shutil.which", return_value="/usr/local/bin/codex")
    @patch("scripts.llm_client.subprocess.run")
    def test_call_codex_uses_explicit_toml_without_leaking_secret_to_command(
        self, run: unittest.mock.Mock, _which: unittest.mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            config_path = Path(temp_name) / "config.toml"
            config_path.write_text(
                """
AZURE_TEST_KEY = "private-test-value"
model_reasoning_effort = "low"

[projects."/workspace"]
model = "gpt-5.5"
model_provider = "azure"
model_reasoning_effort = "high"

[model_providers.azure]
name = "Azure test"
base_url = "https://example.invalid/openai/v1"
env_key = "AZURE_TEST_KEY"
wire_api = "responses"

[mcp_servers.github]
url = "https://example.invalid/mcp"
""".strip(),
                encoding="utf-8",
            )

            def fake_run(
                command: list[str], **kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                output_index = command.index("--output-last-message") + 1
                Path(command[output_index]).write_text('{"ok":true}', encoding="utf-8")
                child_env = kwargs["env"]
                self.assertIsInstance(child_env, dict)
                self.assertEqual(child_env["AZURE_TEST_KEY"], "private-test-value")
                self.assertNotIn("private-test-value", " ".join(command))
                self.assertEqual(
                    Path(child_env["CODEX_HOME"], "config.toml").resolve(),
                    config_path.resolve(),
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            run.side_effect = fake_run
            with patch.dict(
                os.environ,
                {
                    "GEM_CODEX_CONFIG": str(config_path),
                    "GEM_CODEX_PROJECT_DIR": "/workspace/project",
                },
                clear=False,
            ):
                _raw, usage = call_codex(
                    [{"role": "user", "content": "Return an object."}], 128, 0.0
                )

        command = run.call_args.args[0]
        self.assertIn("gpt-5.5", command)
        self.assertIn('model_provider="azure"', command)
        self.assertIn('model_reasoning_effort="high"', command)
        self.assertIn("mcp_servers={}", command)
        self.assertIn("mcp_servers.github.enabled=false", command)
        self.assertEqual(usage["config_source"], "explicit_toml")
        self.assertEqual(usage["model"], "gpt-5.5")
        self.assertEqual(usage["model_provider"], "azure")

    @patch("scripts.llm_client.shutil.which", return_value="/usr/local/bin/codex")
    @patch("scripts.llm_client.subprocess.run")
    def test_call_codex_redacts_embedded_secret_from_failure(
        self, run: unittest.mock.Mock, _which: unittest.mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            config_path = Path(temp_name) / "config.toml"
            config_path.write_text(
                """
AZURE_TEST_KEY = "private-test-value"
model = "gpt-test"
model_provider = "azure"

[model_providers.azure]
env_key = "AZURE_TEST_KEY"
""".strip(),
                encoding="utf-8",
            )
            run.return_value = subprocess.CompletedProcess(
                ["codex"], 1, "", "request failed: private-test-value"
            )
            with patch.dict(
                os.environ, {"GEM_CODEX_CONFIG": str(config_path)}, clear=False
            ):
                with self.assertRaisesRegex(RuntimeError, "<redacted>") as context:
                    call_codex(
                        [{"role": "user", "content": "Return an object."}], 128, 0.0
                    )
        self.assertNotIn("private-test-value", str(context.exception))


if __name__ == "__main__":
    unittest.main()
