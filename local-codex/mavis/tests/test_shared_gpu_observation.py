"""No-GPU fault tests for the installed observation wrapper."""

from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from mavis.runtime import RuntimeConfig


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
                raise subprocess.TimeoutExpired(["gpu-lease", "status"], 1)
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
    def setUp(self):
        self.config = RuntimeConfig(home=Path("/private/tmp/mavis-test"),
                                    allow_concurrent_local=True)
        self.lease = Lease()

    def test_status_call_has_bounded_timeout(self):
        with patch.object(observer.subprocess, "run",
                          side_effect=subprocess.TimeoutExpired(["gpu-lease", "status"], 1)) as run:
            with self.assertRaises(subprocess.TimeoutExpired):
                observer._lease_command("status")
        self.assertEqual(run.call_args.kwargs["timeout"], 1)

    def test_slow_status_during_observation_aborts_exact_child(self):
        child = Mock(pid=4321)
        def wait(*, timeout):
            self.lease.status_timeout = True
            raise subprocess.TimeoutExpired(["e0"], timeout)
        child.wait.side_effect = wait
        child.poll.return_value = None
        shared = {"iris_generation_models": ["iris-model"], "observation_child_safe": True}
        token = observer._PURPOSE.set("our-e0")
        self.lease.purpose = "our-e0"
        try:
            with (patch.object(observer, "_lease_command", side_effect=self.lease),
                  patch.object(observer, "loaded_generation_models", return_value=["iris-model"]),
                  patch.object(observer.subprocess, "Popen", return_value=child),
                  patch.object(observer, "_stop_spawned_process_group") as stop,
                  self.assertRaises(subprocess.TimeoutExpired)):
                observer.run_monitored_observation(
                    self.config, ["e0"], cwd=Path("/tmp"), env={}, stdout=None,
                    shared=shared, timeout=30,
                )
            stop.assert_called_once_with(child)
        finally:
            observer._PURPOSE.reset(token)

    def test_observation_child_checks_game_and_stops_only_its_own_group(self):
        child = Mock(pid=4321)
        def wait(*, timeout):
            self.lease.game_live = True
            raise __import__("subprocess").TimeoutExpired(["e0"], timeout)
        child.wait.side_effect = wait
        child.poll.return_value = None
        shared = {"iris_generation_models": ["iris-model"], "observation_child_safe": True}
        token = observer._PURPOSE.set("our-e0")
        self.lease.purpose = "our-e0"
        try:
            with (patch.object(observer, "_lease_command", side_effect=self.lease),
                  patch.object(observer, "loaded_generation_models", return_value=["iris-model"]),
                  patch.object(observer.subprocess, "Popen", return_value=child) as spawn,
                  patch.object(observer, "_stop_spawned_process_group") as stop,
                  self.assertRaisesRegex(RuntimeError, "game state is unsafe")):
                observer.run_monitored_observation(
                    self.config, ["e0"], cwd=Path("/tmp"), env={}, stdout=None,
                    shared=shared, timeout=30,
                )
            self.assertTrue(spawn.call_args.kwargs["start_new_session"])
            self.assertLessEqual(child.wait.call_args.kwargs["timeout"], 1)
            stop.assert_called_once_with(child)
            self.assertTrue(shared["observation_child_safe"])
        finally:
            observer._PURPOSE.reset(token)

    def test_unstopped_observation_child_retains_failure_state(self):
        child = Mock(pid=4321)
        def wait(*, timeout):
            self.lease.game_live = True
            raise __import__("subprocess").TimeoutExpired(["e0"], timeout)
        child.wait.side_effect = wait
        child.poll.return_value = None
        shared = {"iris_generation_models": ["iris-model"], "observation_child_safe": True}
        token = observer._PURPOSE.set("our-e0")
        self.lease.purpose = "our-e0"
        try:
            with (patch.object(observer, "_lease_command", side_effect=self.lease),
                  patch.object(observer, "loaded_generation_models", return_value=["iris-model"]),
                  patch.object(observer.subprocess, "Popen", return_value=child),
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
        child.wait.return_value = 0
        shared = {"iris_generation_models": ["iris-model"], "observation_child_safe": True}
        token = observer._PURPOSE.set("our-e0")
        self.lease.purpose = "our-e0"
        try:
            with (patch.object(observer, "_lease_command", side_effect=self.lease),
                  patch.object(observer, "loaded_generation_models", return_value=["iris-model"]),
                  patch.object(observer.subprocess, "Popen", return_value=child),
                  patch.object(observer.os, "killpg", side_effect=ProcessLookupError)):
                self.assertEqual(observer.run_monitored_observation(
                    self.config, ["e0"], cwd=Path("/tmp"), env={}, stdout=None,
                    shared=shared, timeout=30,
                ), 0)
        finally:
            observer._PURPOSE.reset(token)

    def test_surviving_observation_child_group_is_recorded_and_fails_closed(self):
        child = Mock(pid=4321)
        child.wait.return_value = 0
        child.poll.return_value = 0
        shared = {"iris_generation_models": ["iris-model"], "observation_child_safe": True}
        token = observer._PURPOSE.set("our-e0")
        self.lease.purpose = "our-e0"
        try:
            with (patch.object(observer, "_lease_command", side_effect=self.lease),
                  patch.object(observer, "loaded_generation_models", return_value=["iris-model"]),
                  patch.object(observer.subprocess, "Popen", return_value=child),
                  patch.object(observer.os, "killpg", return_value=None),
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
                    "cleanup_error": None,
                })
        finally:
            observer._PURPOSE.reset(token)

    def test_abort_refuses_another_sessions_active_request(self):
        with (patch.object(observer, "endpoint_alive", return_value=True),
              patch.object(observer, "request_json", return_value={
                  "status": "ok", "active_requests": 2,
                  "waiting_requests": 0, "models_loading": 1,
              }), patch.object(observer, "stop_trial_server") as stop,
              self.assertRaisesRegex(RuntimeError, "outside this observation")):
            observer._abort_observation_server(self.config, {"pid": 1234},
                                               own_loading=True)
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
        reservation.close.assert_called_once()
        start.assert_called_once_with(self.config, require_new=True)
        park.assert_not_called()
        self.assertIn(("release", "codex-mavis"), self.lease.calls)

    def test_partial_load_failure_unloads_and_parks_own_server_then_releases(self):
        reservation = Mock()
        loaded = []
        def generation(endpoint):
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
                                     expected_state={"pid": 1234})
        self.assertIn(("release", "codex-mavis"), self.lease.calls)

    def test_failed_park_still_releases_own_lease(self):
        reservation = Mock()
        def generation(endpoint):
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

    def test_success_parks_only_new_server_and_reports_lease_release(self):
        reservation = Mock()
        loaded = []
        def generation(endpoint):
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
        start.assert_called_once_with(self.config, require_new=True)
        park.assert_called_once_with(self.config, expected_pid=1234,
                                     expected_state={"pid": 1234})
        self.assertIn(("renew", "codex-mavis", "90"), self.lease.calls)
        self.assertIn(("release", "codex-mavis"), self.lease.calls)

    def test_observation_child_group_survival_retains_lease_after_server_stop(self):
        reservation = Mock()
        def generation(endpoint):
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
        def generation(endpoint):
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
        def generation(endpoint):
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
        def generation(endpoint):
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
        def generation(endpoint):
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
