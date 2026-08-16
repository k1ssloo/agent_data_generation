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
            self.assertEqual(rl["metadata"]["bundle_path"], "tests/fixtures/release_task")
            self.assertNotIn("tools", rl)


if __name__ == "__main__":
    unittest.main()
