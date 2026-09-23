import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mavis"))
sys.path.insert(0, str(ROOT / "mavis" / "tests"))

from mavis.storage import read_json, sha256_file, write_json
from test_e1 import E1RunnerTests
import trial_runtime
from launch_core import launch_trial


class TrialRuntimeTests(unittest.TestCase):
    def setUp(self):
        fixture = E1RunnerTests("test_changed_manifest_fails_closed")
        fixture.setUp()
        self.addCleanup(fixture.temp.cleanup)
        self.fixture = fixture
        fixture._freeze()
        fixture.runner.prepare("repair", "baseline")
        fixture.runner.prepare("repair", "candidate")
        self.production = fixture.root / "production-codex-home"
        self.production.mkdir()
        (self.production / "config.toml").write_text('model = "model-a"\n')
        self.pointer = fixture.home / "profiles" / "main" / "active.json"
        write_json(self.pointer, {"version": 1, "path": "/retained/active"})
        self.pointer_before = self.pointer.read_bytes()
        self.config_before = (self.production / "config.toml").read_bytes()
        self.share = fixture.root / "share"
        self.share.mkdir()
        (self.share / "base-instructions.md").write_text("Base instructions\n")
        (self.share / "persona.toml").write_text('name = "Mavis"\n')
        self.core = self.share / "local-codex-core"
        self.core.write_text("#!/bin/sh\nexit 0\n")
        self.core.chmod(0o755)
        self.profile = {
            "profile_id": "accepted-a",
            "_source_path": str(self.pointer),
            "_source_sha256": sha256_file(self.pointer),
            "prompts": {"system": "A"},
            "model_identity": {"model_id": "model-a"},
        }
        self.records = [{"id": "model-a", "model_type": "llm", "loaded": True}]

    def _binding(self, arm, case="regression"):
        with patch.object(
            trial_runtime, "accepted_main_profile", return_value=self.profile
        ):
            return trial_runtime.trial_binding(self.fixture.home, "repair", arm, case)

    def test_arms_render_distinct_snapshots_into_isolated_homes(self):
        receipts = []
        for arm in ("baseline", "candidate"):
            binding = self._binding(arm)
            path = trial_runtime.prepare_trial(
                binding,
                mavis_home=self.fixture.home,
                share=self.share,
                core_binary=self.core,
                base_url="http://127.0.0.1:8001/v1",
                records=self.records,
            )
            receipts.append(read_json(path))
        baseline, candidate = receipts
        self.assertEqual(baseline["selected_model"], candidate["selected_model"])
        self.assertEqual(baseline["core_sha256"], candidate["core_sha256"])
        self.assertNotEqual(baseline["runtime_home"], candidate["runtime_home"])
        self.assertNotEqual(
            baseline["instructions_sha256"], candidate["instructions_sha256"]
        )
        self.assertEqual(
            Path(baseline["instructions_path"]).read_text(), "Base instructions\n\nA\n"
        )
        self.assertEqual(
            Path(candidate["instructions_path"]).read_text(), "Base instructions\n\nB\n"
        )
        self.assertEqual(self.pointer.read_bytes(), self.pointer_before)
        self.assertEqual(
            (self.production / "config.toml").read_bytes(), self.config_before
        )
        self.assertFalse(
            (self.fixture.home / "e1/repair/runtime/baseline/held-a").exists()
        )

    def test_stale_active_baseline_blocks_trial_before_any_runtime_home(self):
        active_path = self.fixture.runner.store._active_path("main")
        active = read_json(active_path)
        active["configuration"] = self.fixture.runner.store._snapshot(
            {"prompts": {"system": "stale"}, "tool_settings": {}, "retrieval": {}}
        )
        write_json(active_path, active)
        with patch.object(
            trial_runtime, "accepted_main_profile", return_value=self.profile
        ):
            with self.assertRaises(ValueError):
                trial_runtime.trial_binding(
                    self.fixture.home, "repair", "candidate", "regression"
                )
        self.assertFalse((self.fixture.home / "e1/repair/runtime").exists())

    def test_runtime_admission_is_after_frozen_binding_check(self):
        active_path = self.fixture.runner.store._active_path("main")
        active = read_json(active_path)
        active["configuration"] = self.fixture.runner.store._snapshot(
            {"prompts": {"system": "stale"}, "tool_settings": {}, "retrieval": {}}
        )
        write_json(active_path, active)
        with (patch.dict(os.environ, {"MAVIS_HOME": str(self.fixture.home),
                                   "LOCAL_CODEX_SHARE_DIR": str(self.share),
                                   "LOCAL_CODEX_BIN": str(self.core)}),
              patch.object(trial_runtime, "ensure_runtime") as ensure):
            with self.assertRaisesRegex(ValueError, "baseline no longer matches"):
                trial_runtime.run_trial("repair", "candidate", "regression", "task")
        ensure.assert_not_called()

    def test_changed_snapshot_and_wrong_case_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown E1 case"):
            self._binding("candidate", "wrong-case")
        record = self.fixture.runner.store.load("repair")
        Path(record["candidate"]["path"]).write_text("{}")
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            self._binding("candidate")

    def test_candidate_kind_without_prompt_adapter_is_rejected(self):
        record = self.fixture.runner.store.load("repair")
        record["kind"] = "tool_settings"
        write_json(self.fixture.runner.store._record_path("repair"), record)
        with self.assertRaisesRegex(ValueError, "pending main prompt"):
            self._binding("candidate")

    def test_prepared_receipt_rechecks_files_and_state_before_spawn(self):
        binding = self._binding("candidate")
        path = trial_runtime.prepare_trial(
            binding,
            mavis_home=self.fixture.home,
            share=self.share,
            core_binary=self.core,
            base_url="http://127.0.0.1:8001/v1",
            records=self.records,
        )
        receipt = read_json(path)
        with patch.object(
            trial_runtime, "accepted_main_profile", return_value=self.profile
        ):
            trial_runtime.validate_trial_receipt(receipt)
            Path(receipt["instructions_path"]).write_text("changed")
            with self.assertRaisesRegex(ValueError, "instructions changed"):
                trial_runtime.validate_trial_receipt(receipt)
        self.assertEqual(read_json(path)["state"], "prepared")

    def test_stub_core_cannot_claim_effective_configuration(self):
        binding = self._binding("candidate")
        path = trial_runtime.prepare_trial(
            binding,
            mavis_home=self.fixture.home,
            share=self.share,
            core_binary=self.core,
            base_url="http://127.0.0.1:8001/v1",
            records=self.records,
        )
        checkout = str(binding["checkout"])
        with patch.object(
            trial_runtime, "accepted_main_profile", return_value=self.profile
        ):
            result = launch_trial(
                path, [str(self.core), "exec", "-C", checkout, "--", "task"]
            )
        receipt = read_json(path)
        self.assertEqual(result, 2)
        self.assertEqual(receipt["core_exit_code"], 0)
        self.assertEqual(receipt["observation_status"], "inconclusive")
        self.assertEqual(receipt["state"], "exited")
        self.assertEqual(self.pointer.read_bytes(), self.pointer_before)

    def test_installed_launcher_routes_trial_before_production_admission(self):
        environment = os.environ.copy()
        environment.update(
            {
                "LOCAL_CODEX_SHARE_DIR": str(ROOT),
                "LOCAL_CODEX_BIN": str(self.fixture.root / "missing-core"),
                "PYTHONPATH": str(ROOT / "mavis"),
            }
        )
        result = subprocess.run(
            ["zsh", str(ROOT / "bin/local-codex"), "e1", "trial"],
            env=environment,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("required", result.stderr)
        self.assertNotIn("core binary not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
