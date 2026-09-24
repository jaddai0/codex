# ruff: noqa: E402
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mavis"))
sys.path.insert(0, str(ROOT / "mavis" / "tests"))

from mavis.storage import read_json, sha256_file, write_json
from mavis.package_provenance import package_tree_sha256
from test_e1 import E1RunnerTests
import trial_runtime
from launch_core import launch_trial, matching_trial_transcript
from generation_lease import generation_lease


class TrialRuntimeTests(unittest.TestCase):
    def setUp(self):
        fixture = E1RunnerTests("test_changed_manifest_fails_closed")
        fixture.setUp()
        self.addCleanup(fixture.temp.cleanup)
        self.fixture = fixture
        manifest = read_json(fixture.manifest)
        for case in manifest["cases"]:
            case["task"] = "task"
        write_json(fixture.manifest, manifest)
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
        (self.share / "launch_core.py").write_bytes((ROOT / "launch_core.py").read_bytes())
        (self.share / "prepare_runtime.py").write_bytes((ROOT / "prepare_runtime.py").read_bytes())
        (self.share / "generation_lease.py").write_bytes((ROOT / "generation_lease.py").read_bytes())
        (self.share / "mavis").mkdir()
        (self.share / "mavis" / "__init__.py").write_text("# installed fixture\n")
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
            Path(baseline["instructions_path"]).read_text(), "Base instructions\n\nA"
        )
        self.assertEqual(
            Path(candidate["instructions_path"]).read_text(), "Base instructions\n\nB"
        )
        for receipt in receipts:
            self.assertEqual(receipt["core_argv"][1:4],
                             ["exec", "--sandbox", "workspace-write"])
        self.assertEqual(self.pointer.read_bytes(), self.pointer_before)
        self.assertEqual(
            (self.production / "config.toml").read_bytes(), self.config_before
        )
        self.assertFalse(
            (self.fixture.home / "e1/repair/runtime/baseline/held-a").exists()
        )

    def test_empty_baseline_has_no_prompt_bytes_for_core_to_strip(self):
        binding = self._binding("baseline")
        binding["configuration"]["prompts"]["system"] = ""
        path = trial_runtime.prepare_trial(
            binding, mavis_home=self.fixture.home, share=self.share,
            core_binary=self.core, base_url="http://127.0.0.1:8001/v1",
            records=self.records,
        )
        receipt = read_json(path)
        self.assertEqual(Path(receipt["instructions_path"]).read_text(), "Base instructions")

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
        with (
            patch.dict(
                os.environ,
                {
                    "MAVIS_HOME": str(self.fixture.home),
                    "LOCAL_CODEX_SHARE_DIR": str(self.share),
                    "LOCAL_CODEX_BIN": str(self.core),
                },
            ),
            patch.object(trial_runtime, "admitted_e1_model") as admission,
        ):
            with self.assertRaisesRegex(ValueError, "baseline no longer matches"):
                trial_runtime.run_trial("repair", "candidate", "regression", "task")
        admission.assert_not_called()

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
            wrong_task = dict(receipt, task="different task")
            with self.assertRaisesRegex(ValueError, "task or core command"):
                trial_runtime.validate_trial_receipt(wrong_task)
            with self.assertRaisesRegex(ValueError, "core command differs"):
                launch_trial(path, [*receipt["core_argv"][:-1], "different task"])
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
        argv = read_json(path)["core_argv"]
        with patch.object(
            trial_runtime, "accepted_main_profile", return_value=self.profile
        ):
            result = launch_trial(path, argv)
        receipt = read_json(path)
        self.assertEqual(result, 2)
        self.assertEqual(receipt["core_exit_code"], 0)
        self.assertEqual(receipt["observation_status"], "inconclusive")
        self.assertEqual(receipt["state"], "exited")
        self.assertEqual(self.pointer.read_bytes(), self.pointer_before)

    def test_lost_gpu_lease_heartbeat_reaps_native_core(self):
        self.core.write_text("#!/bin/sh\nsleep 30\n")
        binding = self._binding("candidate")
        path = trial_runtime.prepare_trial(
            binding, mavis_home=self.fixture.home, share=self.share,
            core_binary=self.core, base_url="http://127.0.0.1:8001/v1",
            records=self.records,
        )
        calls = 0

        def heartbeat():
            nonlocal calls
            calls += 1
            if calls >= 2:
                raise RuntimeError("GPU lease was lost")

        with patch.object(trial_runtime, "accepted_main_profile", return_value=self.profile):
            with self.assertRaisesRegex(RuntimeError, "GPU lease was lost"):
                launch_trial(path, read_json(path)["core_argv"], heartbeat=heartbeat)
        receipt = read_json(path)
        self.assertEqual(receipt["state"], "spawned")
        self.assertGreaterEqual(calls, 2)
        with self.assertRaises(ProcessLookupError):
            os.kill(receipt["core_pid"], 0)

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

    def test_frozen_task_and_trial_id_fail_before_admission(self):
        with self.assertRaisesRegex(ValueError, "trial IDs"):
            trial_runtime.trial_binding(
                self.fixture.home, "repair", "candidate", "held.a"
            )
        with (
            patch.dict(os.environ, {"MAVIS_HOME": str(self.fixture.home)}),
            patch.object(
                trial_runtime, "accepted_main_profile", return_value=self.profile
            ),
            patch.object(trial_runtime, "admitted_e1_model") as admission,
        ):
            with self.assertRaisesRegex(ValueError, "task differs"):
                trial_runtime.run_trial(
                    "repair", "candidate", "regression", "different task"
                )
        admission.assert_not_called()

    def test_failed_preparation_removes_only_incomplete_home(self):
        binding = self._binding("candidate")
        with patch.object(
            trial_runtime, "write_profile", side_effect=OSError("write failed")
        ):
            with self.assertRaisesRegex(OSError, "write failed"):
                trial_runtime.prepare_trial(
                    binding,
                    mavis_home=self.fixture.home,
                    share=self.share,
                    core_binary=self.core,
                    base_url="http://127.0.0.1:8001/v1",
                    records=self.records,
                )
        self.assertFalse((binding["root"] / "runtime/candidate/regression").exists())
        self.assertFalse(
            (binding["root"] / "trials/candidate/regression.json").exists()
        )
        path = trial_runtime.prepare_trial(
            binding,
            mavis_home=self.fixture.home,
            share=self.share,
            core_binary=self.core,
            base_url="http://127.0.0.1:8001/v1",
            records=self.records,
        )
        self.assertTrue(path.is_file())

    def test_crashed_preparation_recovers_only_owned_partial_files(self):
        binding = self._binding("candidate")
        home = binding["root"] / "runtime/candidate/regression"
        home.mkdir(parents=True)
        write_json(
            home / ".e1-preparing.json",
            {
                "schema_version": "mavis.e1-preparing/v1",
                "experiment_id": "repair",
                "arm": "candidate",
                "case_id": "regression",
            },
        )
        (home / "config.toml").write_text("partial")
        path = trial_runtime.prepare_trial(
            binding,
            mavis_home=self.fixture.home,
            share=self.share,
            core_binary=self.core,
            base_url="http://127.0.0.1:8001/v1",
            records=self.records,
        )
        self.assertTrue(path.is_file())
        self.assertFalse((home / ".e1-preparing.json").exists())

    def test_trial_registers_production_gateway_and_package_provenance(self):
        gateway = self.fixture.root / "gateway"
        (gateway / "bin").mkdir(parents=True)
        launcher = gateway / "bin/mcp-server.sh"
        launcher.write_text("#!/bin/sh\nexit 0\n")
        launcher.chmod(0o755)
        env_file = self.fixture.root / "gateway.env"
        env_file.write_text("TEST_ONLY=1\n")
        manifest = self.share / "install-manifest.json"
        write_json(
            manifest,
            {
                "schema_version": "mavis.installed-core/v1",
                "core_binary": str(self.core.resolve()),
                "core_sha256": sha256_file(self.core),
                "trial_runtime_sha256": sha256_file(Path(trial_runtime.__file__)),
                "launch_core_sha256": sha256_file(self.share / "launch_core.py"),
                "prepare_runtime_sha256": sha256_file(self.share / "prepare_runtime.py"),
                "generation_lease_sha256": sha256_file(self.share / "generation_lease.py"),
                "base_instructions_sha256": sha256_file(self.share / "base-instructions.md"),
                "persona_sha256": sha256_file(self.share / "persona.toml"),
                "mavis_package_sha256": package_tree_sha256(self.share / "mavis"),
                "launcher": str((ROOT / "bin/local-codex").resolve()),
                "launcher_sha256": sha256_file(ROOT / "bin/local-codex"),
            },
        )
        binding = self._binding("candidate")
        path = trial_runtime.prepare_trial(
            binding,
            mavis_home=self.fixture.home,
            share=self.share,
            core_binary=self.core,
            base_url="http://127.0.0.1:8001/v1",
            records=self.records,
            gateway_root=gateway,
            gateway_env_file=env_file,
            package_manifest=manifest,
        )
        receipt = read_json(path)
        config = Path(receipt["config_path"]).read_text()
        self.assertIn("[mcp_servers.model-gateway]", config)
        self.assertIn(str(launcher.resolve()), config)
        self.assertEqual(receipt["core_provenance"], "installed-package")
        with patch.object(
            trial_runtime, "accepted_main_profile", return_value=self.profile
        ):
            trial_runtime.validate_trial_receipt(receipt)
            installed_source = self.share / "mavis" / "__init__.py"
            installed_source.write_text("# changed after installation\n")
            with self.assertRaisesRegex(ValueError, "installed core provenance changed"):
                trial_runtime.validate_trial_receipt(receipt)
            installed_source.write_text("# installed fixture\n")
            launch_source = self.share / "launch_core.py"
            original_launch = launch_source.read_bytes()
            launch_source.write_bytes(original_launch + b"\n# changed\n")
            with self.assertRaisesRegex(ValueError, "installed core provenance changed"):
                trial_runtime.validate_trial_receipt(receipt)
            launch_source.write_bytes(original_launch)
            launcher.write_text("changed")
            with self.assertRaisesRegex(ValueError, "gateway source changed"):
                trial_runtime.validate_trial_receipt(receipt)

    def test_generation_lease_refuses_second_owner(self):
        with generation_lease(self.fixture.home, purpose="foreground"):
            with self.assertRaisesRegex(RuntimeError, "owns the host lease"):
                with generation_lease(self.fixture.home, purpose="e1-trial"):
                    pass
        with generation_lease(self.fixture.home, purpose="e1-trial"):
            pass

    def _installed_trial_environment(self):
        gateway = self.fixture.root / "gateway"
        (gateway / "bin").mkdir(parents=True)
        script = gateway / "bin/mcp-server.sh"
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)
        env_file = self.fixture.root / "gateway.env"
        env_file.write_text("TEST_ONLY=1\n")
        write_json(
            self.share / "install-manifest.json",
            {
                "schema_version": "mavis.installed-core/v1",
                "core_binary": str(self.core.resolve()),
                "core_sha256": sha256_file(self.core),
                "trial_runtime_sha256": sha256_file(Path(trial_runtime.__file__)),
                "launch_core_sha256": sha256_file(self.share / "launch_core.py"),
                "prepare_runtime_sha256": sha256_file(self.share / "prepare_runtime.py"),
                "generation_lease_sha256": sha256_file(self.share / "generation_lease.py"),
                "base_instructions_sha256": sha256_file(self.share / "base-instructions.md"),
                "persona_sha256": sha256_file(self.share / "persona.toml"),
                "mavis_package_sha256": package_tree_sha256(self.share / "mavis"),
                "launcher": str((ROOT / "bin/local-codex").resolve()),
                "launcher_sha256": sha256_file(ROOT / "bin/local-codex"),
            },
        )
        return {
            "MAVIS_HOME": str(self.fixture.home),
            "LOCAL_CODEX_SHARE_DIR": str(self.share),
            "LOCAL_CODEX_BIN": str(self.core),
            "MAVIS_GATEWAY_ROOT": str(gateway),
            "MAVIS_GATEWAY_ENV_FILE": str(env_file),
        }

    def test_contended_trial_stops_before_runtime_load(self):
        environment = self._installed_trial_environment()
        active_home = self.fixture.home / "e1/repair/runtime/candidate/regression"
        active_home.mkdir(parents=True)
        write_json(
            active_home / ".e1-preparing.json",
            {
                "schema_version": "mavis.e1-preparing/v1",
                "experiment_id": "repair",
                "arm": "candidate",
                "case_id": "regression",
            },
        )
        active_marker = active_home / "config.toml"
        active_marker.write_text("owned by first trial")
        with (
            patch.dict(os.environ, environment),
            patch.object(
                trial_runtime, "accepted_main_profile", return_value=self.profile
            ),
            patch.object(trial_runtime, "admitted_e1_model") as admission,
            generation_lease(self.fixture.home, purpose="foreground"),
        ):
            with self.assertRaisesRegex(RuntimeError, "owns the host lease"):
                trial_runtime.run_trial("repair", "candidate", "regression", "task")
        admission.assert_not_called()
        self.assertEqual(active_marker.read_text(), "owned by first trial")

    def test_installed_trial_holds_shared_admission_through_native_launch(self):
        environment = self._installed_trial_environment()
        events = []

        class Heartbeat:
            def __call__(self):
                events.append(("heartbeat",))

            def monitor_read_only(self, work):
                events.append(("monitor-start",))
                result = work()
                events.append(("monitor-end",))
                return result

        @contextmanager
        def admitted(config, purpose):
            events.append(("admitted", config.model, purpose))
            yield Heartbeat()
            events.append(("cleanup",))

        expected = self.fixture.home / "e1/repair/trials/candidate/regression.json"
        with (
            patch.dict(os.environ, environment),
            patch.object(trial_runtime, "accepted_main_profile", return_value=self.profile),
            patch.object(trial_runtime, "admitted_e1_model", side_effect=admitted),
            patch.object(trial_runtime, "_prepare_trial_locked", return_value=expected),
            patch.object(trial_runtime, "inventory", return_value=self.records),
            patch("launch_core.launch_trial", return_value=0) as launch,
        ):
            self.assertEqual(trial_runtime.run_trial("repair", "candidate", "regression", "task"), expected)
        self.assertEqual(events[0], ("admitted", "model-a", "e1:repair:candidate:regression"))
        self.assertEqual(events[-1], ("cleanup",))
        self.assertIn(("monitor-start",), events)
        self.assertIn(("monitor-end",), events)
        self.assertEqual(
            launch.call_args.args[1],
            [
                str(self.core), "exec", "--sandbox", "workspace-write", "-C",
                str(self._binding("candidate")["checkout"].resolve()), "--", "task",
            ],
        )
        self.assertTrue(launch.call_args.kwargs["lease_held"])
        self.assertTrue(callable(launch.call_args.kwargs["heartbeat"]))

    def test_second_process_cannot_recover_first_process_home(self):
        environment = self._installed_trial_environment()
        active_home = self.fixture.home / "e1/repair/runtime/candidate/regression"
        ready = self.fixture.root / "first-ready"
        release = self.fixture.root / "first-release"
        holder_code = """
import json
import sys
import time
from pathlib import Path
from generation_lease import generation_lease
home, active, ready, release = map(Path, sys.argv[1:])
with generation_lease(home, purpose='first-trial'):
    active.mkdir(parents=True)
    (active / '.e1-preparing.json').write_text(json.dumps({
        'schema_version': 'mavis.e1-preparing/v1',
        'experiment_id': 'repair', 'arm': 'candidate', 'case_id': 'regression'}))
    (active / 'config.toml').write_text('live first trial')
    ready.write_text('ready')
    while not release.exists():
        time.sleep(0.05)
"""
        child_env = os.environ.copy()
        child_env["PYTHONPATH"] = str(ROOT)
        holder = subprocess.Popen(
            [
                sys.executable,
                "-c",
                holder_code,
                str(self.fixture.home),
                str(active_home),
                str(ready),
                str(release),
            ],
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            deadline = time.monotonic() + 10
            while (
                not ready.exists()
                and holder.poll() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            self.assertTrue(
                ready.exists(),
                holder.stderr.read().decode() if holder.poll() is not None else "",
            )
            with self.assertRaisesRegex(RuntimeError, "owns the host lease"):
                trial_runtime.prepare_trial(
                    self._binding("candidate"),
                    mavis_home=self.fixture.home,
                    share=self.share,
                    core_binary=self.core,
                    base_url="http://127.0.0.1:8001/v1",
                    records=self.records,
                )
            with (
                patch.dict(os.environ, environment),
                patch.object(
                    trial_runtime, "accepted_main_profile", return_value=self.profile
                ),
                patch.object(trial_runtime, "admitted_e1_model") as admission,
            ):
                with self.assertRaisesRegex(RuntimeError, "owns the host lease"):
                    trial_runtime.run_trial("repair", "candidate", "regression", "task")
            admission.assert_not_called()
            self.assertEqual(
                (active_home / "config.toml").read_text(), "live first trial"
            )
            self.assertTrue((active_home / ".e1-preparing.json").is_file())
        finally:
            release.write_text("done")
            holder.communicate(timeout=10)

    def test_transcript_must_belong_to_observed_session_and_home(self):
        home = self.fixture.root / "trial-home"
        home.mkdir()
        transcript = home / "rollout.jsonl"
        transcript.write_text('{"type":"session_meta","payload":{"id":"session-a"}}\n')
        self.assertEqual(
            matching_trial_transcript(transcript, home, "session-a"),
            transcript.resolve(),
        )
        with self.assertRaisesRegex(ValueError, "session differs"):
            matching_trial_transcript(transcript, home, "session-b")
        outside = self.fixture.root / "other.jsonl"
        outside.write_text(transcript.read_text())
        with self.assertRaisesRegex(ValueError, "outside disposable home"):
            matching_trial_transcript(outside, home, "session-a")

    def test_term_between_spawn_and_receipt_write_reaps_core(self):
        self.core.write_text(
            '#!/bin/sh\nprintf "%s" "$$" > "$CODEX_HOME/core.pid"\nexec sleep 30\n'
        )
        binding = self._binding("candidate")
        path = trial_runtime.prepare_trial(
            binding,
            mavis_home=self.fixture.home,
            share=self.share,
            core_binary=self.core,
            base_url="http://127.0.0.1:8001/v1",
            records=self.records,
        )
        argv = read_json(path)["core_argv"]
        script = """
import sys
import time
from pathlib import Path
from unittest.mock import patch
import trial_runtime
import launch_core
patch.object(trial_runtime, 'validate_trial_receipt').start()
original = launch_core.subprocess.Popen
def delayed_spawn(*args, **kwargs):
    child = original(*args, **kwargs)
    time.sleep(1)
    return child
launch_core.subprocess.Popen = delayed_spawn
sys.exit(launch_core.launch_trial(Path(sys.argv[1]), sys.argv[2:]))
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(ROOT / "mavis")))
        wrapper = subprocess.Popen(
            [sys.executable, "-c", script, str(path), *argv],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        pid_path = Path(read_json(path)["runtime_home"]) / "core.pid"
        try:
            deadline = time.monotonic() + 10
            while (
                not pid_path.exists()
                and wrapper.poll() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            self.assertTrue(
                pid_path.is_file(),
                wrapper.stderr.read().decode() if wrapper.poll() is not None else "",
            )
            self.assertEqual(read_json(path)["state"], "prepared")
            child_pid = int(pid_path.read_text())
            wrapper.terminate()
            wrapper.communicate(timeout=10)
            self.assertFalse(_pid_alive(child_pid))
            receipt = read_json(path)
            self.assertEqual(receipt["state"], "terminated")
            self.assertEqual(receipt["termination_signal"], signal.SIGTERM)
            self.assertEqual(receipt["observation_status"], "inconclusive")
        finally:
            if wrapper.poll() is None:
                wrapper.kill()
                wrapper.wait()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


if __name__ == "__main__":
    unittest.main()
