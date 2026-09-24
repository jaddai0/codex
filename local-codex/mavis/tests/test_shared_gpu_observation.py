"""No-GPU fault tests for the installed observation wrapper."""

from pathlib import Path
import importlib.util
import sys
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

    def __call__(self, *args):
        self.calls.append(args)
        if args[0] == "status":
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
              patch.object(observer, "stop_trial_server") as abort):
            with self.assertRaisesRegex(RuntimeError, "park failed"):
                with observer.shared_mavis_model(self.config, "test"):
                    pass
        abort.assert_called_once()
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
              patch.object(observer, "request_json", side_effect=[idle, loading]),
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
