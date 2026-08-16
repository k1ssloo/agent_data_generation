from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "release_task"

from training import GemTaskEnvironment
from training.package import (
    anchor_portable_rows,
    portable_relative_path,
    resolve_bundle_path,
)
from causal_validation import validate_episode
from rollout import run_reference_plan
from scripts.export_task_first_sft import build_sft_row
from task_factory import load_task_bundle


class GemTaskEnvironmentTests(unittest.TestCase):
    def test_policy_context_does_not_expose_private_bundle_fields(self) -> None:
        environment = GemTaskEnvironment(FIXTURE)
        context = environment.reset()
        self.assertEqual(set(context), {"task_id", "messages", "tools", "remaining_steps"})
        serialized = json.dumps(context)
        self.assertNotIn("initial_state", serialized)
        self.assertNotIn("reference_plan", serialized)
        self.assertNotIn("goal_predicates", serialized)

    def test_reference_actions_receive_full_outcome_reward(self) -> None:
        environment = GemTaskEnvironment(FIXTURE)
        environment.reset()
        for action in environment.bundle.reference_plan["actions"]:
            transition = environment.step(action["tool"], action.get("arguments", {}))
            self.assertIn("observation", transition)
        report = environment.finish()
        score = environment.score(report)
        self.assertTrue(score["is_correct"])
        self.assertEqual(score["reward"], 1.0)

    def test_reset_restores_initial_state(self) -> None:
        environment = GemTaskEnvironment(FIXTURE)
        first = environment.reset()
        action = environment.bundle.reference_plan["actions"][0]
        environment.step(action["tool"], action.get("arguments", {}))
        second = environment.reset()
        self.assertEqual(first, second)


class RllmExportTests(unittest.TestCase):
    def test_resolve_rollout_bundle_requires_exact_public_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = load_task_bundle(FIXTURE)
            report = run_reference_plan(bundle)
            row = build_sft_row(
                {"episode": report, "validation": validate_episode(bundle, report)}
            )
            self.assertIsNotNone(row)
            sft = root / "sft.jsonl"
            sft.write_text(json.dumps(row) + "\n", encoding="utf-8")
            rollout_dir = root / "rollouts"
            rollout_dir.mkdir()
            (rollout_dir / "release.json").write_text(
                json.dumps({"task_id": bundle.task_id, "bundle": str(FIXTURE)}) + "\n",
                encoding="utf-8",
            )
            output = root / "resolved.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "resolve_rollout_bundles.py"),
                    "--sft",
                    str(sft),
                    "--rollout-dir",
                    str(rollout_dir),
                    "--output",
                    str(output),
                ],
                check=True,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            resolved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(resolved["id"], bundle.task_id)
            self.assertEqual(resolved["source"], "exact_rollout_audit")

    def test_export_sft_and_rl_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            validation = root / "validation.jsonl"
            validation.write_text(
                json.dumps({"id": "release_task", "valid": True, "bundle": str(FIXTURE)}) + "\n",
                encoding="utf-8",
            )
            output = root / "export"
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "export_rllm_dataset.py"),
                    "--validation",
                    str(validation),
                    "--output-dir",
                    str(output),
                    "--project-root",
                    str(PROJECT_ROOT),
                ],
                check=True,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            sft = json.loads((output / "sft.jsonl").read_text(encoding="utf-8"))
            rl = json.loads((output / "rl_tasks.jsonl").read_text(encoding="utf-8"))
            self.assertIn("Available tools", sft["messages"][0]["content"])
            self.assertIn("tools", sft)
            self.assertNotIn("reference_plan", json.dumps(sft))
            self.assertEqual(
                rl["metadata"]["bundle_path"], "bundles/release_atlas_beta_v1"
            )
            self.assertEqual(
                rl["metadata"]["bundle_path_base"], "training_package"
            )
            self.assertTrue(
                (
                    output
                    / "bundles"
                    / "release_atlas_beta_v1"
                    / "manifest.json"
                ).is_file()
            )
            self.assertNotIn("tools", rl)
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["portable_bundle_count"], 1)
            self.assertIn("package_sha256", manifest)
            rows = [rl]
            anchor_portable_rows(rows, output / "rl_tasks.jsonl")
            resolved = resolve_bundle_path(
                rows[0]["metadata"], fallback_root=PROJECT_ROOT
            )
            self.assertEqual(
                resolved,
                (output / "bundles" / "release_atlas_beta_v1").resolve(),
            )
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "preflight_rllm_package.py"),
                    "--package-dir",
                    str(output),
                ],
                check=True,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )

    def test_portable_path_cannot_escape_package(self) -> None:
        metadata = {
            "bundle_path": "../outside",
            "bundle_path_base": "training_package",
            "training_package_root": "/tmp/package",
        }
        with self.assertRaisesRegex(ValueError, "escapes"):
            resolve_bundle_path(metadata, fallback_root=PROJECT_ROOT)

    def test_bundle_manifest_path_cannot_escape_package(self) -> None:
        with self.assertRaisesRegex(ValueError, "safe relative path"):
            portable_relative_path("../../secret.json", field="environment_file")

    def test_portable_locator_rejects_absolute_bundle_path(self) -> None:
        metadata = {
            "bundle_path": "/tmp/outside",
            "bundle_path_base": "training_package",
            "training_package_root": "/tmp/package",
        }
        with self.assertRaisesRegex(ValueError, "must be relative"):
            resolve_bundle_path(metadata, fallback_root=PROJECT_ROOT)


if __name__ == "__main__":
    unittest.main()
