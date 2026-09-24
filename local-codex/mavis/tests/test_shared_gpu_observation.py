"""No-GPU fault tests for the installed observation wrapper."""

from pathlib import Path
import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import ANY, Mock, patch
from urllib.error import HTTPError

from mavis.runtime import RuntimeConfig, ensure_isolated_settings
from mavis.storage import write_json


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "shared_gpu_observation.py"
spec = importlib.util.spec_from_file_location("shared_gpu_observation", SCRIPT)
assert spec and spec.loader
observer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = observer
spec.loader.exec_module(observer)


class Lease:
    def __init__(self):
        self.purpose = None
        self.calls = []
        self.other_purpose = False
        self.game_live = False
        self.status_timeout = False

    def __call__(self, *args):
        self.calls.append(args)
        if args[0] == "status":
            if self.status_timeout:
                raise subprocess.TimeoutExpired(["gpu-lease", "status"], 0.5)
            if self.purpose is None:
                return "the GPU is free\nIRIS: idle\n"
            purpose = "another-session" if self.other_purpose else self.purpose
            game = "a game is live" if self.game_live else "idle"
            return f"codex-mavis has the GPU: {purpose} (90 min left)\nIRIS: {game}\n"
        if args[0] == "acquire":
            self.purpose = args[2]
        if args[0] == "release":
            self.purpose = None
        return ""


class SharedGpuObservationTests(unittest.TestCase):
    def test_every_shared_model_script_creates_and_passes_a_private_trial_key(self):
        scripts = SCRIPT.parent
        expected = {
            "observe_small_repository.py", "observe_buried_failure.py",
            "observe_compaction_restart.py", "observe_heldout_e2.py",
            "observe_phase2_repeated_compaction.py", "run_final_e0_with_handoff.py",
        }
        callers = {}
        for path in scripts.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == "shared_mavis_model" for node in ast.walk(tree)):
                callers[path.name] = tree
        self.assertEqual(set(callers), expected)

        def own_key(node):
            return (isinstance(node, ast.Attribute) and node.attr == "api_key"
                    and isinstance(node.value, ast.Name) and node.value.id == "config")

        for name, tree in callers.items():
            with self.subTest(script=name):
                calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
                self.assertTrue(any(
                    isinstance(call.func, ast.Name) and call.func.id == "RuntimeConfig"
                    and any(keyword.arg == "api_key"
                            and isinstance(keyword.value, ast.Call)
                            and isinstance(keyword.value.func, ast.Attribute)
                            and isinstance(keyword.value.func.value, ast.Name)
                            and keyword.value.func.value.id == "secrets"
                            and keyword.value.func.attr == "token_urlsafe"
                            for keyword in call.keywords)
                    for call in calls
                ))
                env_values = [
                    value for node in ast.walk(tree) if isinstance(node, ast.Dict)
                    for key, value in zip(node.keys, node.values)
                    if isinstance(key, ast.Constant) and key.value == "MAVIS_E0_TRIAL_API_KEY"
                ]
                env_values.extend(
                    node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Subscript)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "env"
                            and isinstance(target.slice, ast.Constant)
                            and target.slice.value == "MAVIS_E0_TRIAL_API_KEY"
                            for target in node.targets)
                )
                self.assertTrue(any(own_key(value) for value in env_values))

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = RuntimeConfig(home=Path(self.directory.name), api_key="test-secret",
                                    allow_concurrent_local=True)
        self.config.base_path.mkdir(parents=True)
        (self.config.base_path / "settings.json").write_text(json.dumps({
            "auth": {"api_key": "test-secret", "skip_api_key_verification": False}
        }))
        self.real_prove_trial_auth = observer._prove_trial_auth
        proof = patch.object(observer, "_prove_trial_auth")
        proof.start()
        self.addCleanup(proof.stop)
        self.lease = Lease()

    def test_private_trial_key_is_required_before_gpu_lease(self):
        config = RuntimeConfig(home=self.config.home, allow_concurrent_local=True)
        with patch.object(observer, "_lease_command") as lease, \
                self.assertRaisesRegex(ValueError, "private trial key"):
            with observer.shared_mavis_model(config, "test"):
                pass
        lease.assert_not_called()

    def test_residual_trial_key_is_preserved_after_rejected_start(self):
        error = RuntimeError("residual Mavis trial authentication requires recovery")
        error.residual_trial_auth = True
        with (patch.object(observer, "_lease_command", side_effect=self.lease),
              patch.object(observer, "loaded_generation_models", return_value=["iris-model"]),
              patch.object(observer, "require_idle_iris_handoff"),
              patch.object(observer, "reserve_empty_mavis_port", return_value=Mock()),
              patch.object(observer, "start_server", side_effect=error),
              patch.object(observer, "clear_trial_auth_settings") as clear,
              self.assertRaisesRegex(RuntimeError, "residual Mavis trial")):
            with observer.shared_mavis_model(self.config, "test"):
                pass
        clear.assert_not_called()
        self.assertIn("test-secret", (self.config.base_path / "settings.json").read_text())
        self.assertIn(("release", "codex-mavis"), self.lease.calls)

    def test_emergency_abort_skips_aggregate_request_status_on_exact_trial(self):
        with (patch.object(observer, "endpoint_alive") as alive,
              patch.object(observer, "request_json") as request,
              patch.object(observer, "stop_trial_server") as stop,
              patch.object(observer, "reserve_empty_mavis_port", return_value=Mock()),
              patch.object(observer, "clear_trial_auth_settings") as clear):
            observer._abort_observation_server(
                self.config, {"pid": 1234}, emergency=True,
            )
        alive.assert_not_called()
        request.assert_not_called()
        stop.assert_called_once_with(self.config, {"pid": 1234})
        clear.assert_called_once_with(self.config)

    def test_emergency_abort_refuses_changed_private_auth(self):
        (self.config.base_path / "settings.json").write_text(json.dumps({
            "auth": {"api_key": "another-key", "skip_api_key_verification": False}
        }))
        with patch.object(observer, "stop_trial_server") as stop, \
                self.assertRaisesRegex(RuntimeError, "authentication proof changed"):
            observer._abort_observation_server(
                self.config, {"pid": 1234}, emergency=True,
            )
        stop.assert_not_called()

    def test_game_begins_while_request_drain_is_polling(self):
        self.lease.purpose = "our-e0"
        token = observer._PURPOSE.set("our-e0")
        def active(*_args, **_kwargs):
            self.lease.game_live = True
            return {"status": "ok", "active_requests": 1,
                    "waiting_requests": 0, "models_loading": 0}
        try:
            with patch.object(observer, "_lease_command", side_effect=self.lease), \
                    patch.object(observer, "request_json", side_effect=active), \
                    self.assertRaisesRegex(observer.UnsafeSharedGPU, "game state is unsafe"):
                observer._drained_request_status(self.config, wait_seconds=2.5)
        finally:
            observer._PURPOSE.reset(token)

    def test_game_during_cleanup_status_aborts_exact_server_before_park(self):
        state = {"pid": 1234}
        loaded = {"yes": False}
        statuses = {"count": 0}
        def generation(endpoint, **_kwargs):
            return (["iris-model"] if endpoint == self.config.iris_endpoint else
                    ([self.config.model] if loaded["yes"] else []))
        def status(*_args, **_kwargs):
            statuses["count"] += 1
            if statuses["count"] == 2:
                self.lease.game_live = True
            return {"status": "ok", "active_requests": 0,
                    "waiting_requests": 0, "models_loading": 0}
        with (patch.object(observer, "_lease_command", side_effect=self.lease),
              patch.object(observer, "loaded_generation_models", side_effect=generation),
              patch.object(observer, "require_idle_iris_handoff"),
              patch.object(observer, "reserve_empty_mavis_port", return_value=Mock()),
              patch.object(observer, "start_server", return_value=state),
              patch.object(observer, "read_json", return_value=state),
              patch.object(observer, "inventory", return_value=[{"id": self.config.model}]),
              patch.object(observer, "endpoint_alive", return_value=True),
              patch.object(observer, "load_model",
                           side_effect=lambda _: loaded.__setitem__("yes", True)),
              patch.object(observer, "request_json", side_effect=status),
              patch.object(observer, "stop_trial_server") as stop,
              patch.object(observer, "park_mavis_server") as park,
              self.assertRaisesRegex(RuntimeError, "game state is unsafe")):
            with observer.shared_mavis_model(self.config, "test"):
                pass
        self.assertGreaterEqual(statuses["count"], 2)
        stop.assert_called_once_with(self.config, state)
        park.assert_not_called()
        self.assertIn(("release", "codex-mavis"), self.lease.calls)

    def test_unsafe_child_aborts_server_before_child_group(self):
        child = Mock(pid=4321)
        child.returncode = None
        order: list[str] = []
        shared = {"iris_generation_models": ["iris-model"],
                  "observation_child_safe": True}
        token = observer._PURPOSE.set("our-e0")
        abort_token = observer._EMERGENCY_ABORT.set(lambda: order.append("server"))
        self.lease.purpose = "our-e0"
        try:
            def peek(_child):
                self.lease.game_live = True
                return None
            with (patch.object(observer, "_lease_command", side_effect=self.lease),
                  patch.object(observer, "loaded_generation_models", return_value=["iris-model"]),
                  patch.object(observer.subprocess, "Popen", return_value=child),
                  patch.object(observer, "_child_exit_unreaped", side_effect=peek),
                  patch.object(observer, "_stop_spawned_process_group",
                               side_effect=lambda _: order.append("child")),
                  self.assertRaisesRegex(observer.UnsafeSharedGPU, "game state is unsafe")):
                observer.run_monitored_observation(
                    self.config, ["e0"], cwd=Path("/tmp"), env={}, stdout=None,
                    shared=shared, timeout=30,
                )
            self.assertEqual(order, ["server", "child"])
        finally:
            observer._EMERGENCY_ABORT.reset(abort_token)
            observer._PURPOSE.reset(token)

    def test_trial_auth_requires_live_401_then_keyed_status(self):
        config = RuntimeConfig(home=Path("/tmp/mavis-auth-test"), api_key="test-secret")
        unauthorized = HTTPError(config.endpoint, 401, "unauthorized", {}, None)
        with patch.object(observer, "request_json", side_effect=[
            unauthorized, {"status": "ok"},
        ]) as request:
            self.real_prove_trial_auth(config)
        unauthorized.close()
        self.assertEqual(request.call_args_list[1].kwargs["headers"],
                         {"Authorization": "Bearer test-secret"})
        with patch.object(observer, "request_json", return_value={"status": "ok"}), \
                self.assertRaisesRegex(RuntimeError, "without its key"):
                self.real_prove_trial_auth(config)

    def test_authenticated_stopped_child_waits_for_request_to_drain(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory), api_key="test-secret")
            config.base_path.mkdir(parents=True)
            (config.base_path / "settings.json").write_text(json.dumps({
                "auth": {"api_key": "test-secret", "skip_api_key_verification": False}
            }))
            self.lease.purpose = "our-e0"
            token = observer._PURPOSE.set("our-e0")
            rows = [{"status": "ok", "active_requests": active,
                     "waiting_requests": 0, "models_loading": 0} for active in (1, 0, 0)]
            with (patch.object(observer, "endpoint_alive", return_value=True),
                  patch.object(observer, "_lease_command", side_effect=self.lease),
                  patch.object(observer, "request_json", side_effect=rows) as status,
                  patch.object(observer, "stop_trial_server") as stop,
                  patch.object(observer, "reserve_empty_mavis_port", return_value=Mock()),
                  patch.object(observer, "clear_trial_auth_settings") as clear):
                observer._abort_observation_server(
                    config, {"pid": 1234}, own_generation=True,
                )
            observer._PURPOSE.reset(token)
            self.assertEqual(status.call_count, 3)
            stop.assert_called_once()
            clear.assert_called_once_with(config)

    def test_authenticated_own_unload_waits_for_request_to_drain_without_child(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory), api_key="test-secret")
            config.base_path.mkdir(parents=True)
            (config.base_path / "settings.json").write_text(json.dumps({
                "auth": {"api_key": "test-secret", "skip_api_key_verification": False}
            }))
            self.lease.purpose = "our-e0"
            token = observer._PURPOSE.set("our-e0")
            rows = [{"status": "ok", "active_requests": active,
                     "waiting_requests": 0, "models_loading": 0} for active in (1, 0, 0)]
            with (patch.object(observer, "endpoint_alive", return_value=True),
                  patch.object(observer, "_lease_command", side_effect=self.lease),
                  patch.object(observer, "request_json", side_effect=rows) as status,
                  patch.object(observer, "stop_trial_server") as stop,
                  patch.object(observer, "reserve_empty_mavis_port", return_value=Mock()),
                  patch.object(observer, "clear_trial_auth_settings") as clear):
                observer._abort_observation_server(
                    config, {"pid": 1234}, own_unload=True,
                )
            observer._PURPOSE.reset(token)
            self.assertEqual(status.call_count, 3)
            stop.assert_called_once()
            clear.assert_called_once_with(config)

    def test_authenticated_own_unload_refuses_second_request(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory), api_key="test-secret")
            config.base_path.mkdir(parents=True)
            (config.base_path / "settings.json").write_text(json.dumps({
                "auth": {"api_key": "test-secret", "skip_api_key_verification": False}
            }))
            self.lease.purpose = "our-e0"
            token = observer._PURPOSE.set("our-e0")
            with (patch.object(observer, "endpoint_alive", return_value=True),
                  patch.object(observer, "_lease_command", side_effect=self.lease),
                  patch.object(observer, "request_json", return_value={
                      "status": "ok", "active_requests": 2,
                      "waiting_requests": 0, "models_loading": 0,
                  }), patch.object(observer, "stop_trial_server") as stop,
                  self.assertRaisesRegex(RuntimeError, "did not drain")):
                observer._abort_observation_server(
                    config, {"pid": 1234}, own_unload=True,
                )
            observer._PURPOSE.reset(token)
            stop.assert_not_called()

    def test_post_launch_auth_failure_parks_then_scrubs_trial_key(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory), api_key="test-secret",
                                   allow_concurrent_local=True)
            state = {"pid": 1234}
            def start(_config, *, require_new, heartbeat):
                self.assertTrue(require_new)
                self.assertTrue(callable(heartbeat))
                ensure_isolated_settings(_config)
                return state
            with (patch.object(observer, "_lease_command", side_effect=self.lease),
                  patch.object(observer, "loaded_generation_models", return_value=["iris-model"]),
                  patch.object(observer, "require_idle_iris_handoff"),
                  patch.object(observer, "reserve_empty_mavis_port", return_value=Mock()),
                  patch.object(observer, "start_server", side_effect=start),
                  patch.object(observer, "read_json", return_value=state),
                  patch.object(observer, "_prove_trial_auth",
                               side_effect=RuntimeError("auth probe failed")),
                  patch.object(observer, "park_mavis_server", return_value=Mock()) as park,
                  patch.object(observer, "endpoint_alive", return_value=False),
                  self.assertRaisesRegex(RuntimeError, "auth probe failed")):
                with observer.shared_mavis_model(config, "test"):
                    pass
            park.assert_called_once()
            self.assertNotIn("test-secret", (config.base_path / "settings.json").read_text())
            self.assertIn(("release", "codex-mavis"), self.lease.calls)

    def test_failed_start_after_auth_write_scrubs_only_after_empty_port(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory), api_key="test-secret",
                                   allow_concurrent_local=True)
            def start(_config, *, require_new, heartbeat):
                self.assertTrue(require_new)
                self.assertTrue(callable(heartbeat))
                ensure_isolated_settings(_config)
                raise RuntimeError("startup failed after owned group stopped")
            with (patch.object(observer, "_lease_command", side_effect=self.lease),
                  patch.object(observer, "loaded_generation_models", return_value=["iris-model"]),
                  patch.object(observer, "require_idle_iris_handoff"),
                  patch.object(observer, "reserve_empty_mavis_port", return_value=Mock()) as reserve,
                  patch.object(observer, "start_server", side_effect=start),
                  self.assertRaisesRegex(RuntimeError, "startup failed")):
                with observer.shared_mavis_model(config, "test"):
                    pass
            self.assertEqual(reserve.call_count, 2)
            self.assertNotIn("test-secret", (config.base_path / "settings.json").read_text())
            self.assertIn(("release", "codex-mavis"), self.lease.calls)

    def test_game_during_authenticated_generation_aborts_own_server(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory), api_key="test-secret",
                                   allow_concurrent_local=True)
            state = {"pid": 1234}
            def start(_config, *, require_new, heartbeat):
                self.assertTrue(require_new)
                self.assertTrue(callable(heartbeat))
                ensure_isolated_settings(_config)
                write_json(_config.state_path, state)
                return state
            def generation(endpoint, **_kwargs):
                return ["iris-model"] if endpoint == config.iris_endpoint else []
            draining = {"checks": 0}
            def status(*_args, **_kwargs):
                if self.lease.game_live:
                    draining["checks"] += 1
                active = 1 if self.lease.game_live and draining["checks"] == 1 else 0
                return {"status": "ok", "active_requests": active,
                        "waiting_requests": 0, "models_loading": 0}
            with (patch.object(observer, "_lease_command", side_effect=self.lease),
                  patch.object(observer, "loaded_generation_models", side_effect=generation),
                  patch.object(observer, "require_idle_iris_handoff"),
                  patch.object(observer, "reserve_empty_mavis_port", return_value=Mock()),
                  patch.object(observer, "start_server", side_effect=start),
                  patch.object(observer, "_prove_trial_auth"),
                  patch.object(observer, "inventory", return_value=[{"id": config.model}]),
                  patch.object(observer, "endpoint_alive", return_value=True),
                  patch.object(observer, "load_model"),
                  patch.object(observer, "request_json", side_effect=status),
                  patch.object(observer, "stop_trial_server") as stop,
                  patch.object(observer, "park_mavis_server") as park,
                  patch.object(observer, "_failure_receipt",
                               return_value=Path("/tmp/e0-auth-game.json"))):
                with self.assertRaisesRegex(RuntimeError, "game state is unsafe"):
                    with observer.shared_mavis_model(config, "test") as shared:
                        shared["observation_child_pid"] = 4321
                        self.lease.game_live = True
            stop.assert_called_once_with(config, state)
            self.assertEqual(draining["checks"], 0)
            park.assert_not_called()
            self.assertNotIn("test-secret", (config.base_path / "settings.json").read_text())
            self.assertIn(("release", "codex-mavis"), self.lease.calls)

    def test_persistent_request_after_child_disconnect_retains_server_and_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory), api_key="test-secret",
                                   allow_concurrent_local=True)
            state = {"pid": 1234}
            def start(_config, *, require_new, heartbeat):
                self.assertTrue(require_new)
                self.assertTrue(callable(heartbeat))
                ensure_isolated_settings(_config)
                write_json(_config.state_path, state)
                return state
            def generations(endpoint, **_kwargs):
                return ["iris-model"] if endpoint == config.iris_endpoint else []
            active = {"yes": False}
            def status(*_args, **_kwargs):
                return {"status": "ok", "active_requests": 1 if active["yes"] else 0,
                        "waiting_requests": 0, "models_loading": 0}
            with (patch.object(observer, "_lease_command", side_effect=self.lease),
                  patch.object(observer, "loaded_generation_models", side_effect=generations),
                  patch.object(observer, "require_idle_iris_handoff"),
                  patch.object(observer, "reserve_empty_mavis_port", return_value=Mock()),
                  patch.object(observer, "start_server", side_effect=start),
                  patch.object(observer, "_prove_trial_auth"),
                  patch.object(observer, "inventory", return_value=[{"id": config.model}]),
                  patch.object(observer, "endpoint_alive", return_value=True),
                  patch.object(observer, "load_model"),
                  patch.object(observer, "request_json", side_effect=status),
                  patch.object(observer, "stop_trial_server") as stop,
                  patch.object(observer, "park_mavis_server",
                               side_effect=RuntimeError("active request remained")) as park,
                  self.assertRaisesRegex(RuntimeError, "lease retained")):
                with observer.shared_mavis_model(config, "test") as shared:
                    shared["observation_child_pid"] = 4321
                    active["yes"] = True
                    raise RuntimeError("observation child disconnected unexpectedly")
            stop.assert_not_called()
            park.assert_called_once()
            self.assertNotIn(("release", "codex-mavis"), self.lease.calls)
            self.assertIn("test-secret", (config.base_path / "settings.json").read_text())
            receipts = list((config.home / "evaluations/shared-gpu-failures").glob("*.json"))
            self.assertEqual(len(receipts), 1)
            self.assertIn("did not drain", json.loads(receipts[0].read_text())["error"])

    def test_game_during_shielded_unload_aborts_exact_trial_server(self):
        for had_child in (False, True):
            with self.subTest(had_child=had_child):
                self.lease.purpose = "our-e0"
                self.lease.game_live = False
                blocked = threading.Event()
                entered = threading.Event()
                stopped = {"yes": False}
                def unload(*_args, **_kwargs):
                    entered.set()
                    self.lease.game_live = True
                    blocked.wait(10)
                    return {"status": "ok"}
                def abort(_config, state, *, emergency, stopped):
                    self.assertEqual(state, {"pid": 1234})
                    self.assertTrue(emergency)
                    stopped["yes"] = True
                    blocked.set()
                with (patch.object(observer, "_lease_command", side_effect=self.lease),
                      patch.object(observer, "request_json", side_effect=unload),
                      patch.object(observer, "loaded_generation_models", return_value=["iris-model"]),
                      patch.object(observer, "_abort_observation_server", side_effect=abort) as stop,
                      self.assertRaisesRegex(RuntimeError, "game state is unsafe")):
                    observer._monitored_unload(self.config, "our-e0", {"pid": 1234},
                                               stopped, ["iris-model"],
                                               own_generation=had_child)
                self.assertTrue(entered.is_set())
                self.assertTrue(stopped["yes"])
                stop.assert_called_once()

    def test_scrub_failure_after_proven_park_releases_lease_with_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory), api_key="test-secret",
                                   allow_concurrent_local=True)
            state = {"pid": 1234}
            write_json(config.state_path, state)
            with (patch.object(observer, "_lease_command", side_effect=self.lease),
                  patch.object(observer, "loaded_generation_models", side_effect=lambda endpoint, **_: (
                      ["iris-model"] if endpoint == config.iris_endpoint else [])),
                  patch.object(observer, "require_idle_iris_handoff"),
                  patch.object(observer, "reserve_empty_mavis_port", return_value=Mock()),
                  patch.object(observer, "start_server", return_value=state),
                  patch.object(observer, "_prove_trial_auth"),
                  patch.object(observer, "inventory", return_value=[{"id": config.model}]),
                  patch.object(observer, "endpoint_alive", return_value=False),
                  patch.object(observer, "request_json", return_value={
                      "status": "ok", "active_requests": 0,
                      "waiting_requests": 0, "models_loading": 0,
                  }), patch.object(observer, "load_model"),
                  patch.object(observer, "park_mavis_server", return_value=Mock()) as park,
                  patch.object(observer, "clear_trial_auth_settings",
                               side_effect=RuntimeError("scrub failed")),
                  patch.object(observer, "stop_trial_server") as abort,
                  patch.object(observer, "_failure_receipt",
                               return_value=Path("/tmp/e0-auth-scrub.json")) as receipt):
                with self.assertRaisesRegex(RuntimeError, "scrub failed"):
                    with observer.shared_mavis_model(config, "test"):
                        pass
            park.assert_called_once()
            abort.assert_not_called()
            receipt.assert_called_once()
            self.assertIn(("release", "codex-mavis"), self.lease.calls)

    def test_status_call_has_bounded_timeout(self):
        with patch.object(observer.subprocess, "run",
                          side_effect=subprocess.TimeoutExpired(["gpu-lease", "status"], 0.5)) as run:
            with self.assertRaises(subprocess.TimeoutExpired):
                observer._lease_command("status")
        self.assertEqual(run.call_args.kwargs["timeout"], 0.5)

    def test_slow_status_during_observation_aborts_exact_child(self):
        child = Mock(pid=4321)
        child.returncode = None
        def peek(_child):
            self.lease.status_timeout = True
            return None
        shared = {"iris_generation_models": ["iris-model"], "observation_child_safe": True}
        token = observer._PURPOSE.set("our-e0")
        self.lease.purpose = "our-e0"
        try:
            with (patch.object(observer, "_lease_command", side_effect=self.lease),
                  patch.object(observer, "loaded_generation_models", return_value=["iris-model"]),
                  patch.object(observer.subprocess, "Popen", return_value=child),
                  patch.object(observer, "_child_exit_unreaped", side_effect=peek),
                  patch.object(observer, "_stop_spawned_process_group") as stop,
                  self.assertRaisesRegex(observer.UnsafeSharedGPU, "timed out")):
                observer.run_monitored_observation(
                    self.config, ["e0"], cwd=Path("/tmp"), env={}, stdout=None,
                    shared=shared, timeout=30,
                )
            stop.assert_called_once_with(child)
        finally:
            observer._PURPOSE.reset(token)

    def test_observation_child_checks_game_and_stops_only_its_own_group(self):
        child = Mock(pid=4321)
        child.returncode = None
        def peek(_child):
            self.lease.game_live = True
            return None
        shared = {"iris_generation_models": ["iris-model"], "observation_child_safe": True}
        token = observer._PURPOSE.set("our-e0")
        self.lease.purpose = "our-e0"
        try:
            with (patch.object(observer, "_lease_command", side_effect=self.lease),
                  patch.object(observer, "loaded_generation_models", return_value=["iris-model"]),
                  patch.object(observer.subprocess, "Popen", return_value=child) as spawn,
                  patch.object(observer, "_child_exit_unreaped", side_effect=peek),
                  patch.object(observer, "_stop_spawned_process_group") as stop,
                  self.assertRaisesRegex(RuntimeError, "game state is unsafe")):
                observer.run_monitored_observation(
                    self.config, ["e0"], cwd=Path("/tmp"), env={}, stdout=None,
                    shared=shared, timeout=30,
                )
            self.assertTrue(spawn.call_args.kwargs["start_new_session"])
            child.wait.assert_not_called()
            stop.assert_called_once_with(child)
            self.assertTrue(shared["observation_child_safe"])
        finally:
            observer._PURPOSE.reset(token)

    def test_unstopped_observation_child_retains_failure_state(self):
        child = Mock(pid=4321)
        child.returncode = None
        def peek(_child):
            self.lease.game_live = True
            return None
        shared = {"iris_generation_models": ["iris-model"], "observation_child_safe": True}
        token = observer._PURPOSE.set("our-e0")
        self.lease.purpose = "our-e0"
        try:
            with (patch.object(observer, "_lease_command", side_effect=self.lease),
                  patch.object(observer, "loaded_generation_models", return_value=["iris-model"]),
                  patch.object(observer.subprocess, "Popen", return_value=child),
                  patch.object(observer, "_child_exit_unreaped", side_effect=peek),
                  patch.object(observer, "_stop_spawned_process_group",
                               side_effect=RuntimeError("owned child would not stop"))):
                with self.assertRaisesRegex(RuntimeError, "owned child would not stop"):
                    observer.run_monitored_observation(
                        self.config, ["e0"], cwd=Path("/tmp"), env={}, stdout=None,
                        shared=shared, timeout=30,
                    )
            self.assertFalse(shared["observation_child_safe"])
        finally:
            observer._PURPOSE.reset(token)

    def test_observation_child_success_checks_group_exit(self):
        child = Mock(pid=4321)
        child.returncode = None
        child.wait.return_value = 0
        shared = {"iris_generation_models": ["iris-model"], "observation_child_safe": True}
        token = observer._PURPOSE.set("our-e0")
        self.lease.purpose = "our-e0"
        try:
            with (patch.object(observer, "_lease_command", side_effect=self.lease),
                  patch.object(observer, "loaded_generation_models", return_value=["iris-model"]),
                  patch.object(observer.subprocess, "Popen", return_value=child),
                  patch.object(observer, "_child_exit_unreaped", return_value=Mock(si_status=0)),
                  patch.object(observer, "_process_group_workers", return_value=set())):
                self.assertEqual(observer.run_monitored_observation(
                    self.config, ["e0"], cwd=Path("/tmp"), env={}, stdout=None,
                    shared=shared, timeout=30,
                ), 0)
        finally:
            observer._PURPOSE.reset(token)

    def test_surviving_observation_child_group_is_recorded_and_fails_closed(self):
        child = Mock(pid=4321)
        child.returncode = None
        child.wait.return_value = 0
        shared = {"iris_generation_models": ["iris-model"], "observation_child_safe": True}
        token = observer._PURPOSE.set("our-e0")
        self.lease.purpose = "our-e0"
        try:
            with (patch.object(observer, "_lease_command", side_effect=self.lease),
                  patch.object(observer, "loaded_generation_models", return_value=["iris-model"]),
                  patch.object(observer.subprocess, "Popen", return_value=child),
                  patch.object(observer, "_child_exit_unreaped", return_value=Mock(si_status=0)),
                  patch.object(observer, "_process_group_workers", return_value={5555}),
                  patch.object(observer, "_stop_spawned_process_group",
                               side_effect=RuntimeError("child group remained")),
                  self.assertRaisesRegex(RuntimeError, "child group remained")):
                observer.run_monitored_observation(
                    self.config, ["e0"], cwd=Path("/tmp"), env={}, stdout=None,
                    shared=shared, timeout=30,
                )
            self.assertFalse(shared["observation_child_safe"])
            self.assertTrue(shared["observation_child_group_remained"])
            with tempfile.TemporaryDirectory() as directory, patch.object(
                observer, "inventory", return_value=[]
            ):
                receipt = observer._failure_receipt(
                    RuntimeConfig(home=Path(directory)), "our-e0",
                    RuntimeError("child group remained"), shared,
                )
                self.assertEqual(json.loads(receipt.read_text())["observation_child"], {
                    "pid": 4321, "safe": False, "group_remained": True,
                    "cleanup_error": "child group remained",
                })
        finally:
            observer._PURPOSE.reset(token)

    def test_abort_refuses_another_sessions_active_request(self):
        self.lease.purpose = "our-e0"
        token = observer._PURPOSE.set("our-e0")
        with (patch.object(observer, "endpoint_alive", return_value=True),
              patch.object(observer, "_lease_command", side_effect=self.lease),
              patch.object(observer, "request_json", return_value={
                  "status": "ok", "active_requests": 2,
                  "waiting_requests": 0, "models_loading": 1,
              }), patch.object(observer, "stop_trial_server") as stop,
              self.assertRaisesRegex(RuntimeError, "did not drain")):
            observer._abort_observation_server(self.config, {"pid": 1234},
                                               own_loading=True)
        observer._PURPOSE.reset(token)
        stop.assert_not_called()

    def test_preexisting_server_fails_without_stopping_it_and_releases_lease(self):
        with (patch.object(observer, "_lease_command", side_effect=self.lease),
              patch.object(observer, "loaded_generation_models", return_value=["iris-model"]),
              patch.object(observer, "require_idle_iris_handoff"),
              patch.object(observer, "reserve_empty_mavis_port",
                           side_effect=RuntimeError("port occupied")),
              patch.object(observer, "start_server") as start,
              patch.object(observer, "park_mavis_server") as park):
            with self.assertRaisesRegex(RuntimeError, "port occupied"):
                with observer.shared_mavis_model(self.config, "test"):
                    pass
        start.assert_not_called()
        park.assert_not_called()
        self.assertIn(("release", "codex-mavis"), self.lease.calls)

    def test_race_after_reservation_does_not_attach_or_park_other_server(self):
        reservation = Mock()
        with (patch.object(observer, "_lease_command", side_effect=self.lease),
              patch.object(observer, "loaded_generation_models", return_value=["iris-model"]),
              patch.object(observer, "require_idle_iris_handoff"),
              patch.object(observer, "reserve_empty_mavis_port", return_value=reservation),
              patch.object(observer, "start_server",
                           side_effect=RuntimeError("Mavis endpoint was already running")) as start,
              patch.object(observer, "park_mavis_server") as park):
            with self.assertRaisesRegex(RuntimeError, "already running"):
                with observer.shared_mavis_model(self.config, "test"):
                    pass
        self.assertEqual(reservation.close.call_count, 2)
        start.assert_called_once_with(self.config, require_new=True, heartbeat=ANY)
        park.assert_not_called()
        self.assertIn(("release", "codex-mavis"), self.lease.calls)

    def test_partial_load_failure_unloads_and_parks_own_server_then_releases(self):
        reservation = Mock()
        loaded = []
        def generation(endpoint, **_kwargs):
            return ["iris-model"] if endpoint == self.config.iris_endpoint else list(loaded)
        def request(*args, **kwargs):
            if args[1].endswith("/unload"):
                loaded.clear()
            return {"status": "ok", "active_requests": 0,
                    "waiting_requests": 0, "models_loading": 0}
        def failed_load(_config):
            loaded.append(self.config.model)
            raise RuntimeError("load failed")
        with (patch.object(observer, "_lease_command", side_effect=self.lease),
              patch.object(observer, "loaded_generation_models", side_effect=generation),
              patch.object(observer, "require_idle_iris_handoff"),
              patch.object(observer, "reserve_empty_mavis_port", return_value=reservation),
              patch.object(observer, "start_server", return_value={"pid": 1234}),
              patch.object(observer, "read_json", return_value={"pid": 1234}),
              patch.object(observer, "inventory", return_value=[{"id": self.config.model}]),
              patch.object(observer, "endpoint_alive", side_effect=[True, False]),
              patch.object(observer, "load_model", side_effect=failed_load),
              patch.object(observer, "request_json", side_effect=request) as api,
              patch.object(observer, "park_mavis_server", return_value=Mock()) as park):
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                with observer.shared_mavis_model(self.config, "test"):
                    pass
        self.assertTrue(any(call.args[1].endswith("/unload") for call in api.call_args_list))
        park.assert_called_once_with(self.config, expected_pid=1234,
                                     expected_state={"pid": 1234}, heartbeat=ANY)
        self.assertIn(("release", "codex-mavis"), self.lease.calls)

    def test_failed_park_still_releases_own_lease(self):
        reservation = Mock()
        def generation(endpoint, **_kwargs):
            return ["iris-model"] if endpoint == self.config.iris_endpoint else []
        with (patch.object(observer, "_lease_command", side_effect=self.lease),
              patch.object(observer, "loaded_generation_models", side_effect=generation),
              patch.object(observer, "require_idle_iris_handoff"),
              patch.object(observer, "reserve_empty_mavis_port", return_value=reservation),
              patch.object(observer, "start_server", return_value={"pid": 1234}),
              patch.object(observer, "read_json", return_value={"pid": 1234}),
              patch.object(observer, "inventory", return_value=[{"id": self.config.model}]),
              patch.object(observer, "endpoint_alive", return_value=True),
              patch.object(observer, "load_model"),
              patch.object(observer, "request_json", return_value={"status": "ok",
                  "active_requests": 0, "waiting_requests": 0, "models_loading": 0}),
              patch.object(observer, "park_mavis_server",
                           side_effect=RuntimeError("park failed")),
              patch.object(observer, "stop_trial_server") as abort,
              patch.object(observer, "_failure_receipt", return_value=Path("/tmp/e0-cleanup.json")) as receipt):
            with self.assertRaisesRegex(RuntimeError, "inventory receipt: /tmp/e0-cleanup.json"):
                with observer.shared_mavis_model(self.config, "test"):
                    pass
        abort.assert_called_once()
        receipt.assert_called_once()
        self.assertIn(("release", "codex-mavis"), self.lease.calls)

    def test_proven_park_stop_releases_lease_when_foreign_port_blocks_reservation(self):
        state = {"pid": 1234}
        error = RuntimeError("group stopped, but foreign port blocked reservation")
        error.gpu_work_stopped = True
        def generation(endpoint, **_kwargs):
            return ["iris-model"] if endpoint == self.config.iris_endpoint else []
        with (patch.object(observer, "_lease_command", side_effect=self.lease),
              patch.object(observer, "loaded_generation_models", side_effect=generation),
              patch.object(observer, "require_idle_iris_handoff"),
              patch.object(observer, "reserve_empty_mavis_port", return_value=Mock()),
              patch.object(observer, "start_server", return_value=state),
              patch.object(observer, "read_json", return_value=state),
              patch.object(observer, "inventory", return_value=[{"id": self.config.model}]),
              patch.object(observer, "endpoint_alive", return_value=True),
              patch.object(observer, "load_model"),
              patch.object(observer, "request_json", return_value={
                  "status": "ok", "active_requests": 0,
                  "waiting_requests": 0, "models_loading": 0,
              }), patch.object(observer, "park_mavis_server", side_effect=error),
              patch.object(observer, "stop_trial_server") as abort,
              patch.object(observer, "_failure_receipt",
                           return_value=Path("/tmp/e0-port-race.json"))):
            with self.assertRaisesRegex(RuntimeError, "foreign port blocked"):
                with observer.shared_mavis_model(self.config, "test"):
                    pass
        abort.assert_not_called()
        self.assertIn(("release", "codex-mavis"), self.lease.calls)

    def test_success_parks_only_new_server_and_reports_lease_release(self):
        reservation = Mock()
        loaded = []
        def generation(endpoint, **_kwargs):
            return ["iris-model"] if endpoint == self.config.iris_endpoint else list(loaded)
        def api(*args, **kwargs):
            if args[1].endswith("/unload"):
                loaded.clear()
            return {"status": "ok", "active_requests": 0,
                    "waiting_requests": 0, "models_loading": 0}
        with (patch.object(observer, "_lease_command", side_effect=self.lease),
              patch.object(observer, "loaded_generation_models", side_effect=generation),
              patch.object(observer, "require_idle_iris_handoff"),
              patch.object(observer, "reserve_empty_mavis_port", return_value=reservation),
              patch.object(observer, "start_server", return_value={"pid": 1234}) as start,
              patch.object(observer, "read_json", return_value={"pid": 1234}),
              patch.object(observer, "inventory", return_value=[{"id": self.config.model}]),
              patch.object(observer, "endpoint_alive", side_effect=[True, False]),
              patch.object(observer, "load_model", side_effect=lambda _: loaded.append(self.config.model)),
              patch.object(observer, "request_json", side_effect=api),
              patch.object(observer, "park_mavis_server", return_value=Mock()) as park):
            with observer.shared_mavis_model(self.config, "test") as result:
                self.assertTrue(result["mavis_loaded"])
                observer.renew_gpu_lease()
            self.assertEqual(result["mavis_loaded"], False)
            self.assertTrue(result["iris_stayed_loaded"])
            self.assertTrue(result["gpu_lease_released"])
        start.assert_called_once_with(self.config, require_new=True, heartbeat=ANY)
        park.assert_called_once_with(self.config, expected_pid=1234,
                                     expected_state={"pid": 1234}, heartbeat=ANY)
        self.assertIn(("renew", "codex-mavis", "90"), self.lease.calls)
        self.assertIn(("release", "codex-mavis"), self.lease.calls)

    def test_observation_child_group_survival_retains_lease_after_server_stop(self):
        reservation = Mock()
        def generation(endpoint, **_kwargs):
            return ["iris-model"] if endpoint == self.config.iris_endpoint else []
        with (patch.object(observer, "_lease_command", side_effect=self.lease),
              patch.object(observer, "loaded_generation_models", side_effect=generation),
              patch.object(observer, "require_idle_iris_handoff"),
              patch.object(observer, "reserve_empty_mavis_port", return_value=reservation),
              patch.object(observer, "start_server", return_value={"pid": 1234}),
              patch.object(observer, "read_json", return_value={"pid": 1234}),
              patch.object(observer, "inventory", return_value=[{"id": self.config.model}]),
              patch.object(observer, "endpoint_alive", return_value=False),
              patch.object(observer, "load_model"),
              patch.object(observer, "request_json", return_value={
                  "status": "ok", "active_requests": 0,
                  "waiting_requests": 0, "models_loading": 0,
              }), patch.object(observer, "park_mavis_server", return_value=Mock()),
              patch.object(observer, "_failure_receipt",
                           return_value=Path("/tmp/e0-child-group.json")) as receipt):
            with self.assertRaisesRegex(RuntimeError, "lease retained"):
                with observer.shared_mavis_model(self.config, "test") as shared:
                    shared["observation_child_safe"] = False
                    shared["observation_child_group_remained"] = True
        receipt.assert_called_once()
        self.assertNotIn(("release", "codex-mavis"), self.lease.calls)

    def test_same_name_lease_takeover_blocks_cleanup_and_release(self):
        reservation = Mock()
        def generation(endpoint, **_kwargs):
            return ["iris-model"] if endpoint == self.config.iris_endpoint else []
        with (patch.object(observer, "_lease_command", side_effect=self.lease),
              patch.object(observer, "loaded_generation_models", side_effect=generation),
              patch.object(observer, "require_idle_iris_handoff"),
              patch.object(observer, "reserve_empty_mavis_port", return_value=reservation),
              patch.object(observer, "start_server", return_value={"pid": 1234}),
              patch.object(observer, "read_json", return_value={"pid": 1234}),
              patch.object(observer, "inventory", return_value=[{"id": self.config.model}]),
              patch.object(observer, "request_json", return_value={"status": "ok",
                  "active_requests": 0, "waiting_requests": 0, "models_loading": 0}),
              patch.object(observer, "load_model"),
              patch.object(observer, "park_mavis_server") as park,
              patch.object(observer, "stop_trial_server", side_effect=RuntimeError("ownership lost")),
              patch.object(observer, "_failure_receipt", return_value=Path("/tmp/failure.json"))):
            with self.assertRaisesRegex(RuntimeError, "lease purpose changed"):
                with observer.shared_mavis_model(self.config, "test"):
                    self.lease.other_purpose = True
        park.assert_not_called()
        self.assertNotIn(("release", "codex-mavis"), self.lease.calls)

    def test_game_start_during_blocking_load_aborts_own_server_before_release(self):
        reservation = Mock()
        blocked = threading.Event()
        def load(_config):
            self.lease.game_live = True
            blocked.wait(15)
        def abort(_config, state):
            self.assertEqual(state["pid"], 1234)
            blocked.set()
        def generation(endpoint, **_kwargs):
            return ["iris-model"] if endpoint == self.config.iris_endpoint else []
        with (patch.object(observer, "_lease_command", side_effect=self.lease),
              patch.object(observer, "loaded_generation_models", side_effect=generation),
              patch.object(observer, "require_idle_iris_handoff"),
              patch.object(observer, "reserve_empty_mavis_port", return_value=reservation),
              patch.object(observer, "start_server", return_value={"pid": 1234}),
              patch.object(observer, "read_json", return_value={"pid": 1234}),
              patch.object(observer, "inventory", return_value=[{"id": self.config.model}]),
              patch.object(observer, "load_model", side_effect=load),
              patch.object(observer, "request_json", return_value={"status": "ok",
                  "active_requests": 0, "waiting_requests": 0, "models_loading": 0}),
              patch.object(observer, "stop_trial_server", side_effect=abort) as stop,
              patch.object(observer, "park_mavis_server") as park):
            with self.assertRaisesRegex(RuntimeError, "game state is unsafe"):
                with observer.shared_mavis_model(self.config, "test"):
                    pass
        stop.assert_called_once()
        park.assert_not_called()
        self.assertIn(("release", "codex-mavis"), self.lease.calls)

    def test_failed_abort_retains_lease_and_writes_receipt(self):
        reservation = Mock()
        def generation(endpoint, **_kwargs):
            return ["iris-model"] if endpoint == self.config.iris_endpoint else []
        with (patch.object(observer, "_lease_command", side_effect=self.lease),
              patch.object(observer, "loaded_generation_models", side_effect=generation),
              patch.object(observer, "require_idle_iris_handoff"),
              patch.object(observer, "reserve_empty_mavis_port", return_value=reservation),
              patch.object(observer, "start_server", return_value={"pid": 1234}),
              patch.object(observer, "read_json", return_value={"pid": 1234}),
              patch.object(observer, "inventory", return_value=[{"id": self.config.model}]),
              patch.object(observer, "load_model"),
              patch.object(observer, "request_json", return_value={"status": "ok",
                  "active_requests": 0, "waiting_requests": 0, "models_loading": 0}),
              patch.object(observer, "park_mavis_server", side_effect=RuntimeError("park failed")),
              patch.object(observer, "stop_trial_server", side_effect=RuntimeError("abort failed")),
              patch.object(observer, "_failure_receipt", return_value=Path("/tmp/failure.json")) as receipt):
            with self.assertRaisesRegex(RuntimeError, "abort failed"):
                with observer.shared_mavis_model(self.config, "test"):
                    pass
        receipt.assert_called_once()
        self.assertNotIn(("release", "codex-mavis"), self.lease.calls)

    def test_load_error_with_loading_worker_aborts_before_releasing_lease(self):
        reservation = Mock()
        def generation(endpoint, **_kwargs):
            return ["iris-model"] if endpoint == self.config.iris_endpoint else []
        idle = {"status": "ok", "active_requests": 0,
                "waiting_requests": 0, "models_loading": 0}
        loading = {**idle, "models_loading": 1}
        with (patch.object(observer, "_lease_command", side_effect=self.lease),
              patch.object(observer, "loaded_generation_models", side_effect=generation),
              patch.object(observer, "require_idle_iris_handoff"),
              patch.object(observer, "reserve_empty_mavis_port", return_value=reservation),
              patch.object(observer, "start_server", return_value={"pid": 1234}),
              patch.object(observer, "read_json", return_value={"pid": 1234}),
              patch.object(observer, "inventory", return_value=[{"id": self.config.model}]),
              patch.object(observer, "endpoint_alive", return_value=True),
              patch.object(observer, "load_model", side_effect=RuntimeError("load response failed")),
              patch.object(observer, "request_json", side_effect=[idle, loading, loading]),
              patch.object(observer, "stop_trial_server") as stop,
              patch.object(observer, "park_mavis_server") as park):
            with self.assertRaisesRegex(RuntimeError, "load response failed"):
                with observer.shared_mavis_model(self.config, "test"):
                    pass
        stop.assert_called_once()
        park.assert_not_called()
        self.assertIn(("release", "codex-mavis"), self.lease.calls)

    def test_unstopped_startup_failure_retains_lease_and_receipt(self):
        failure = RuntimeError("launched server did not stop")
        failure.unsafe_gpu_work = True
        with (patch.object(observer, "_lease_command", side_effect=self.lease),
              patch.object(observer, "loaded_generation_models", return_value=["iris-model"]),
              patch.object(observer, "require_idle_iris_handoff"),
              patch.object(observer, "reserve_empty_mavis_port", return_value=Mock()),
              patch.object(observer, "start_server", side_effect=failure),
              patch.object(observer, "_failure_receipt", return_value=Path("/tmp/failure.json")) as receipt):
            with self.assertRaisesRegex(RuntimeError, "did not stop"):
                with observer.shared_mavis_model(self.config, "test"):
                    pass
        receipt.assert_called_once()
        self.assertNotIn(("release", "codex-mavis"), self.lease.calls)

    def test_renew_refuses_changed_purpose(self):
        with patch.object(observer, "_lease_command", side_effect=self.lease):
            with self.assertRaisesRegex(RuntimeError, "no active"):
                observer.renew_gpu_lease()
            self.lease.purpose = "ours"
            token = observer._PURPOSE.set("ours")
            try:
                self.lease.other_purpose = True
                with self.assertRaisesRegex(RuntimeError, "purpose changed"):
                    observer.renew_gpu_lease()
            finally:
                observer._PURPOSE.reset(token)
        self.assertFalse(any(call[0] == "renew" for call in self.lease.calls))


if __name__ == "__main__":
    unittest.main()
