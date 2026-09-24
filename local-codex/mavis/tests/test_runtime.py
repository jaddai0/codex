from pathlib import Path
import os
import signal
import socket
import subprocess
import sys
import threading
import tempfile
import time
import unittest
from unittest.mock import patch

from mavis.runtime import (
    RuntimeConfig,
    admission,
    acquire_iris_model_drain,
    clear_trial_auth_settings,
    ensure_isolated_settings,
    handoff_lease_fd,
    loaded_generation_models,
    mavis_generation_lease,
    owns_running_server,
    require_idle_iris_handoff,
    require_installed_selected_model,
    release_iris_model_drain,
    park_mavis_server,
    reserve_empty_mavis_port,
    start_server,
    stop_server,
    stop_trial_server,
    wait_iris_model_drain,
)
import mavis.runtime as runtime
from mavis.storage import write_json


class FakeProcess:
    def __init__(self):
        self.pid = 1234
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.returncode = -signal.SIGTERM
        return self.returncode


class RuntimeTests(unittest.TestCase):
    def test_trial_auth_settings_are_private_and_residual_key_blocks_normal_start(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory), api_key="test-secret")
            self.assertNotIn("test-secret", repr(config))
            ensure_isolated_settings(config)
            settings_path = config.base_path / "settings.json"
            self.assertEqual(settings_path.stat().st_mode & 0o777, 0o600)
            settings = __import__("json").loads(settings_path.read_text())
            self.assertEqual(settings["auth"]["api_key"], "test-secret")
            self.assertIs(settings["auth"]["skip_api_key_verification"], False)
            with self.assertRaisesRegex(RuntimeError, "residual Mavis trial"):
                ensure_isolated_settings(RuntimeConfig(home=config.home))
            with self.assertRaisesRegex(RuntimeError, "another Mavis trial"):
                ensure_isolated_settings(RuntimeConfig(home=config.home, api_key="other"))
            clear_trial_auth_settings(config)
            self.assertNotIn("test-secret", settings_path.read_text())
            ensure_isolated_settings(RuntimeConfig(home=config.home))

    def test_trial_key_is_absent_from_server_argv_and_launch_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "omlx"
            binary.write_text("fixture")
            config = RuntimeConfig(home=root / "home", omlx_binary=binary,
                                   api_key="test-secret")
            process = FakeProcess()
            with patch.dict(os.environ, {"MAVIS_E0_TRIAL_API_KEY": "inherited-stale"}), \
                    patch.dict(runtime._TRIAL_PROCESSES, {}, clear=True), \
                    patch("mavis.runtime.endpoint_alive", side_effect=[False, True]) as alive, \
                    patch("mavis.runtime.port_in_use", return_value=False), \
                    patch("mavis.runtime.owns_running_server", return_value=True), \
                    patch("mavis.runtime.subprocess.Popen", return_value=process) as spawn:
                state = start_server(config, require_new=True)
            self.assertNotIn("test-secret", str(state))
            self.assertNotIn("test-secret", str(spawn.call_args.args[0]))
            self.assertNotIn("MAVIS_E0_TRIAL_API_KEY", spawn.call_args.kwargs["env"])
            self.assertEqual(alive.call_args_list[-1].kwargs, {"api_key": "test-secret"})

    def test_authenticated_inventory_sends_bearer_without_logging_key(self):
        import io
        from unittest.mock import Mock
        payload = io.BytesIO(b'[{"id":"model"}]')
        response = Mock()
        response.__enter__ = Mock(return_value=payload)
        response.__exit__ = Mock(return_value=False)
        with patch("mavis.runtime.urlopen", return_value=response) as open_url:
            self.assertEqual(runtime.inventory("http://127.0.0.1:8001/v1",
                                               api_key="test-secret"), [{"id": "model"}])
        request = open_url.call_args.args[0]
        self.assertTrue(request.full_url.endswith("/v1/models/status"))
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")

    def test_trial_inventory_avoids_cookie_only_admin_route(self):
        from urllib.error import HTTPError
        def route(_endpoint, path, **_kwargs):
            if path == "/admin/api/models":
                raise HTTPError(path, 401, "admin session required", {}, None)
            self.assertEqual(path, "/v1/models/status")
            return {"models": [{"id": "model", "loaded": True,
                                "estimated_size": 123, "last_access": 456}]}
        with patch("mavis.runtime.request_json", side_effect=route):
            self.assertEqual(runtime.inventory("http://127.0.0.1:8001/v1",
                                               api_key="test-secret")[0]["estimated_size"], 123)

    def test_memory_probe_drops_e0_trial_key_from_os_children(self):
        with patch.dict(os.environ, {"MAVIS_E0_TRIAL_API_KEY": "test-secret"}), \
                patch("mavis.runtime.subprocess.check_output",
                      side_effect=["4096\n", "Pages free: 1.\n"]) as check:
            self.assertEqual(runtime.available_memory_bytes(), 4096)
        self.assertEqual(check.call_count, 2)
        for call in check.call_args_list:
            self.assertNotIn("MAVIS_E0_TRIAL_API_KEY", call.kwargs["env"])

    def test_public_stop_fails_closed_without_exact_launch_handle(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory))
            write_json(config.state_path, {"pid": 1234})
            with patch.dict(runtime._TRIAL_PROCESSES, {}, clear=True), \
                    patch("mavis.runtime.os.killpg") as kill, \
                    self.assertRaisesRegex(RuntimeError, "exact launch ownership is unproven"):
                stop_server(config)
            kill.assert_not_called()

    def test_public_stop_reaps_registered_group_and_clears_state(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory))
            state = {"pid": 1234, "command": ["trial"]}
            write_json(config.state_path, state)
            process = FakeProcess()
            with patch.dict(runtime._TRIAL_PROCESSES, {1234: process}, clear=True), \
                    patch("mavis.runtime.owns_running_server", return_value=True), \
                    patch("mavis.runtime.request_json", return_value={
                        "status": "ok", "active_requests": 0,
                        "waiting_requests": 0, "models_loading": 0,
                    }), patch("mavis.runtime.os.getpgid", return_value=1234), \
                    patch("mavis.runtime.os.waitpid", return_value=(1234, 0)), \
                    patch("mavis.runtime.os.killpg",
                          side_effect=[None, None, ProcessLookupError]) as kill:
                stop_server(config)
            kill.assert_any_call(1234, signal.SIGTERM)
            self.assertFalse(config.state_path.exists())

    def test_public_stop_refuses_active_request_before_signaling(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory))
            write_json(config.state_path, {"pid": 1234})
            with patch.dict(runtime._TRIAL_PROCESSES, {1234: FakeProcess()}, clear=True), \
                    patch("mavis.runtime.owns_running_server", return_value=True), \
                    patch("mavis.runtime.request_json", return_value={
                        "status": "ok", "active_requests": 1,
                        "waiting_requests": 0, "models_loading": 0,
                    }), patch("mavis.runtime.os.killpg") as kill, \
                    self.assertRaisesRegex(RuntimeError, "active, waiting, or loading"):
                stop_server(config)
            kill.assert_not_called()

    def test_trial_abort_requires_process_handle_from_this_python_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory))
            state = {"pid": 1234, "command": ["trial"]}
            write_json(config.state_path, state)
            with patch.dict(runtime._TRIAL_PROCESSES, {}, clear=True), \
                    patch("mavis.runtime.os.killpg") as kill, \
                    self.assertRaisesRegex(RuntimeError, "no process handle"):
                stop_trial_server(config, state)
            kill.assert_not_called()

    def test_trial_abort_reaps_only_registered_spawned_group(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory))
            state = {"pid": 1234, "command": ["trial"]}
            write_json(config.state_path, state)
            process = FakeProcess()
            with patch.dict(runtime._TRIAL_PROCESSES, {1234: process}, clear=True), \
                    patch("mavis.runtime.owns_running_server", return_value=True), \
                    patch("mavis.runtime.os.getpgid", return_value=1234), \
                    patch("mavis.runtime.os.waitpid", return_value=(1234, 0)), \
                    patch("mavis.runtime.os.killpg", side_effect=[None, None, ProcessLookupError]) as kill:
                stop_trial_server(config, state)
                self.assertNotIn(1234, runtime._TRIAL_PROCESSES)
            kill.assert_any_call(1234, signal.SIGTERM)

    def test_trial_abort_rejects_changed_launch_record_without_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory))
            write_json(config.state_path, {"pid": 2222, "command": ["other"]})
            with patch("mavis.runtime.os.killpg") as kill, self.assertRaisesRegex(
                RuntimeError, "launch record changed"
            ):
                stop_trial_server(config, {"pid": 1111, "command": ["ours"]})
            kill.assert_not_called()

    def test_new_server_requirement_refuses_even_owned_preexisting_server(self):
        config = RuntimeConfig(home=Path("/tmp/mavis-owned-test"))
        with patch("mavis.runtime.endpoint_alive", return_value=True), \
                patch("mavis.runtime.owns_running_server", return_value=True), \
                patch("mavis.runtime.subprocess.Popen") as spawn, \
                self.assertRaisesRegex(RuntimeError, "already running"):
            start_server(config, require_new=True)
        spawn.assert_not_called()

    def test_empty_reservation_refuses_new_listener_without_signaling(self):
        with tempfile.TemporaryDirectory() as directory, socket.socket() as owner:
            owner.bind(("127.0.0.1", 0))
            port = owner.getsockname()[1]
            owner.listen(1)
            config = RuntimeConfig(home=Path(directory),
                                   endpoint=f"http://127.0.0.1:{port}/v1")
            with patch("mavis.runtime.os.killpg") as kill, self.assertRaisesRegex(
                RuntimeError, "occupied"
            ):
                reserve_empty_mavis_port(config)
            kill.assert_not_called()
            self.assertGreaterEqual(owner.fileno(), 0)

    def test_park_refuses_state_pid_change_before_signaling(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory))
            write_json(config.state_path, {"pid": 4321})
            with patch("mavis.runtime.os.killpg") as kill, self.assertRaisesRegex(
                RuntimeError, "PID changed"
            ):
                park_mavis_server(config, expected_pid=1234)
            kill.assert_not_called()

    def test_park_refuses_same_pid_different_launch_record(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory))
            write_json(config.state_path, {"pid": 1234, "command": ["other"]})
            with patch("mavis.runtime.os.killpg") as kill, self.assertRaisesRegex(
                RuntimeError, "launch record changed"
            ):
                park_mavis_server(config, expected_pid=1234,
                                  expected_state={"pid": 1234, "command": ["ours"]})
            kill.assert_not_called()

    def test_park_reserves_empty_port_without_starting_server(self):
        with tempfile.TemporaryDirectory() as directory:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            config = RuntimeConfig(home=Path(directory),
                                   endpoint=f"http://127.0.0.1:{port}/v1")
            reservation = park_mavis_server(config)
            try:
                with socket.socket() as probe:
                    self.assertEqual(probe.connect_ex(("127.0.0.1", port)), 0)
                self.assertFalse(config.state_path.exists())
            finally:
                reservation.close()

    def test_park_waits_for_process_group_and_exclusively_reserves_port(self):
        with tempfile.TemporaryDirectory() as directory:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            config = RuntimeConfig(home=Path(directory),
                                   endpoint=f"http://127.0.0.1:{port}/v1")
            child = subprocess.Popen(
                [sys.executable, "-c", "import socket,time; s=socket.socket(); "
                 "s.bind(('127.0.0.1', int(__import__('sys').argv[1]))); "
                 "s.listen(); time.sleep(60)", str(port)],
                start_new_session=True,
            )
            try:
                write_json(config.state_path, {"pid": child.pid})
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    with socket.socket() as probe:
                        if probe.connect_ex(("127.0.0.1", port)) == 0:
                            break
                    time.sleep(0.02)
                else:
                    self.fail("test listener did not start")
                idle = {"status": "ok", "active_requests": 0,
                        "waiting_requests": 0, "models_loading": 0}
                reaper = threading.Thread(target=child.wait, daemon=True)
                reaper.start()
                with patch("mavis.runtime.endpoint_alive", return_value=True), \
                        patch("mavis.runtime.request_json", return_value=idle), \
                        patch("mavis.runtime.owns_running_server", return_value=True):
                    reservation = park_mavis_server(config)
                try:
                    reaper.join(timeout=1)
                    self.assertIsNotNone(child.poll())
                    with socket.socket() as probe:
                        self.assertEqual(probe.connect_ex(("127.0.0.1", port)), 0)
                    with self.assertRaises(OSError):
                        with socket.socket() as competing:
                            competing.bind(("127.0.0.1", port))
                finally:
                    reservation.close()
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()

    def test_park_retries_denied_group_probe_until_process_disappears(self):
        with tempfile.TemporaryDirectory() as directory:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            config = RuntimeConfig(home=Path(directory),
                                   endpoint=f"http://127.0.0.1:{port}/v1")
            write_json(config.state_path, {"pid": 1234})
            with patch("mavis.runtime.endpoint_alive", return_value=False), \
                    patch("mavis.runtime._listener_pids", side_effect=[
                        set(), set(), {os.getpid()}]), \
                    patch("mavis.runtime.os.killpg", side_effect=[
                        PermissionError(1, "Operation not permitted"),
                        ProcessLookupError(3, "No such process")]), \
                    patch("mavis.runtime.time.sleep"):
                reservation = park_mavis_server(config, timeout=1)
            reservation.close()

    def test_loaded_generation_inventory_keeps_unknown_models_visible(self):
        rows = [{"id": "embed", "loaded": True, "engine_type": "embedding"},
                {"id": "main", "loaded": True, "engine_type": "vlm"},
                {"id": "unexpected", "loaded": True}]
        with patch("mavis.runtime.inventory", return_value=rows):
            self.assertEqual(loaded_generation_models("http://127.0.0.1:8001/v1"),
                             ["main", "unexpected"])

    def test_handoff_refuses_active_or_uncertain_iris_work(self):
        config = RuntimeConfig(home=Path("/tmp/mavis-handoff-test"))
        idle = {"status": "ok", "loaded_models": [config.model], "models_loading": 0,
                "active_requests": 0, "waiting_requests": 0}
        with patch("mavis.runtime.request_json", side_effect=[idle, idle]) as reader, \
                patch("mavis.runtime.loaded_generation_models", return_value=[config.model]), \
                patch("mavis.runtime.time.sleep"):
            require_idle_iris_handoff(config)
            self.assertEqual(reader.call_count, 2)
        for changed in ({"active_requests": 1}, {"waiting_requests": 1},
                        {"loaded_models": []}, {"active_requests": None},
                        {"models_loading": 1}):
            with self.subTest(changed=changed), patch(
                "mavis.runtime.request_json", return_value={**idle, **changed}
            ), self.assertRaisesRegex(RuntimeError, "handoff refused"):
                require_idle_iris_handoff(config, interval_seconds=0)

    def test_handoff_preserves_a_distinct_iris_generation_model(self):
        config = RuntimeConfig(home=Path("/tmp/mavis-handoff-test"), model="mavis-model")
        idle = {"status": "ok", "loaded_models": ["iris-qwen"], "models_loading": 0,
                "active_requests": 0, "waiting_requests": 0}
        with patch("mavis.runtime.request_json", return_value=idle), patch(
            "mavis.runtime.loaded_generation_models", return_value=["iris-qwen"]
        ), patch("mavis.runtime.time.sleep"):
            require_idle_iris_handoff(config, expected_models=["iris-qwen"])
        with patch("mavis.runtime.request_json", return_value=idle), patch(
            "mavis.runtime.loaded_generation_models", return_value=["changed"]
        ), patch("mavis.runtime.time.sleep"), self.assertRaisesRegex(
            RuntimeError, "inventory changed"
        ):
            require_idle_iris_handoff(config, expected_models=["iris-qwen"])
        with patch("mavis.runtime.request_json", side_effect=[
            idle, {**idle, "loaded_models": ["iris-qwen", "new-model"]}
        ]), patch("mavis.runtime.time.sleep"), self.assertRaisesRegex(
            RuntimeError, "status changed"
        ):
            require_idle_iris_handoff(config, expected_models=["iris-qwen"])

    def test_handoff_requires_installed_launcher_model_match(self):
        config = RuntimeConfig(home=Path("/tmp/mavis-handoff-test"))
        with patch("mavis.runtime.subprocess.run") as run:
            run.return_value.stdout = "another-model\n"
            with self.assertRaisesRegex(RuntimeError, "different Mavis model"):
                require_installed_selected_model(config)
            run.return_value.stdout = config.model + "\n"
            require_installed_selected_model(config)

    def test_drain_lease_blocks_until_accepted_status_and_confirms_release(self):
        config = RuntimeConfig(home=Path("/tmp/mavis-drain-test"))
        lease_id = "4f9c8ae0-62f6-4ad5-b3e0-4891db2fcd62"
        def response(state):
            return {"schema_version": "omlx.model-drain/v1", "model_id": config.model,
                    "lease_id": lease_id, "state": state}
        with patch("mavis.runtime.iris_drain_headers", return_value={"X-OMLX-Drain-Token": "secret"}), \
                patch("mavis.runtime.request_json", side_effect=[response("draining"),
                      response("draining"), response("drained"), response("released")]) as request, \
                patch("mavis.runtime.time.sleep"):
            self.assertEqual(acquire_iris_model_drain(config, owner="mavis-e0"), lease_id)
            wait_iris_model_drain(config, lease_id)
            release_iris_model_drain(config, lease_id)
            self.assertEqual(request.call_count, 4)
            self.assertEqual(request.call_args_list[0].kwargs["payload"],
                             {"owner": "mavis-e0"})
            self.assertEqual(request.call_args_list[-1].kwargs["payload"],
                             {"lease_id": lease_id})

    def test_drain_refuses_untrusted_or_changed_lease(self):
        config = RuntimeConfig(home=Path("/tmp/mavis-drain-test"))
        with patch("mavis.runtime.iris_drain_headers", return_value={}), \
                patch("mavis.runtime.request_json", return_value={"state": "drained"}), \
                self.assertRaisesRegex(RuntimeError, "invalid lease"):
            acquire_iris_model_drain(config, owner="mavis-e0")

    def test_handoff_lock_exposes_only_held_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory))
            with self.assertRaisesRegex(RuntimeError, "not held"):
                handoff_lease_fd()
            with mavis_generation_lease(config, purpose="test"):
                descriptor = handoff_lease_fd()
                self.assertGreaterEqual(descriptor, 0)
                with self.assertRaisesRegex(RuntimeError, "host lease"):
                    with mavis_generation_lease(config, purpose="competing"):
                        pass
            with self.assertRaisesRegex(RuntimeError, "not held"):
                handoff_lease_fd()

    def test_ownership_requires_exact_binary_base_path_and_port(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory), omlx_binary=Path("/opt/omlx"))
            write_json(config.state_path, {"pid": 12})
            with patch("mavis.runtime._listener_pids", return_value={12}), patch(
                "mavis.runtime._process_open_paths",
                return_value={config.base_path / "logs" / "server.log"},
            ):
                self.assertTrue(owns_running_server(config))
            with patch("mavis.runtime._listener_pids", return_value={12}), patch(
                "mavis.runtime._process_open_paths", return_value={Path("/tmp/other")}
            ):
                self.assertFalse(owns_running_server(config))

    def test_refuses_healthy_foreign_server(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory))
            with patch("mavis.runtime.endpoint_alive", return_value=True), patch(
                "mavis.runtime.owns_running_server", return_value=False
            ):
                with self.assertRaisesRegex(RuntimeError, "does not own"):
                    start_server(config)

    def test_start_timeout_terminates_only_spawned_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "omlx"
            binary.write_text("fixture")
            config = RuntimeConfig(home=root / "home", omlx_binary=binary)
            process = FakeProcess()
            with patch("mavis.runtime.endpoint_alive", return_value=False), patch(
                "mavis.runtime.port_in_use", return_value=False
            ), patch("mavis.runtime.subprocess.Popen", return_value=process) as popen, \
                    patch("mavis.runtime.os.getpgid", return_value=1234), \
                    patch("mavis.runtime.os.killpg", side_effect=[None, ProcessLookupError]) as kill:
                with self.assertRaises(TimeoutError):
                    start_server(config, wait_seconds=0)
            kill.assert_any_call(1234, signal.SIGTERM)
            child_env = popen.call_args.kwargs["env"]
            self.assertEqual(child_env["OMLX_BASE_PATH"], str(config.base_path))
            self.assertEqual(child_env["HOME"], str(config.home / "user-home"))
            self.assertFalse(config.state_path.exists())

    def test_start_state_write_failure_reaps_exact_spawned_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "omlx"
            binary.write_text("fixture")
            config = RuntimeConfig(home=root / "home", omlx_binary=binary)
            process = FakeProcess()
            def write_or_fail(path, value):
                if path == config.state_path:
                    raise OSError("disk full")
                write_json(path, value)
            with patch("mavis.runtime.endpoint_alive", return_value=False), \
                    patch("mavis.runtime.port_in_use", return_value=False), \
                    patch("mavis.runtime.subprocess.Popen", return_value=process), \
                    patch("mavis.runtime.write_json", side_effect=write_or_fail), \
                    patch("mavis.runtime.os.getpgid", return_value=1234), \
                    patch("mavis.runtime.os.killpg", side_effect=[None, ProcessLookupError]) as kill:
                with self.assertRaisesRegex(OSError, "disk full"):
                    start_server(config, require_new=True)
            kill.assert_any_call(1234, signal.SIGTERM)
            self.assertIsNotNone(process.returncode)

    def test_settings_pin_auth_server_model_and_cache_to_mavis(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory))
            write_json(config.base_path / "settings.json", {
                "version": "1.0", "server": {"distributed_inference_enabled": True}
            })
            ensure_isolated_settings(config)
            settings = __import__("json").loads(
                (config.base_path / "settings.json").read_text()
            )
            self.assertEqual(settings["server"], {"host": "127.0.0.1", "port": 8001,
                                                   "distributed_inference_enabled": False})
            self.assertTrue(settings["auth"]["skip_api_key_verification"])
            self.assertEqual(settings["model"]["model_dirs"], [str(config.model_dir)])
            self.assertEqual(settings["cache"]["ssd_cache_dir"], str(config.base_path / "cache"))

    def test_isolated_runtime_refuses_distributed_deployment_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory))
            write_json(config.base_path / "cluster" / "deployments.json", {
                "schema_version": 1, "deployments": []
            })
            with self.assertRaisesRegex(RuntimeError, "distributed deployment"):
                ensure_isolated_settings(config)
            self.assertFalse((config.base_path / "settings.json").exists())

    def test_admission_blocks_recent_iris_generation_but_ignores_embedding(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory), reserve_bytes=10)
            own = [{"id": config.model, "estimated_size": 100}]
            iris = [
                {"id": "embed", "model_type": "embedding", "loaded": True, "last_access": time.time()},
                {"id": "main", "model_type": "llm", "loaded": True, "last_access": 9990},
            ]
            with patch("mavis.runtime.inventory", side_effect=[own, iris]), patch(
                "mavis.runtime.available_memory_bytes", return_value=1000
            ), patch("mavis.runtime.endpoint_alive", return_value=True):
                result = admission(config, now=10000)
            self.assertFalse(result["allowed"])
            self.assertIn("IRIS already owns", result["reasons"][0])

    def test_concurrent_override_still_requires_proven_idle(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(
                home=Path(directory), reserve_bytes=10, allow_concurrent_local=True
            )
            own = [{"id": config.model, "estimated_size": 100}]
            iris = [{"id": "main", "model_type": "llm", "loaded": True, "last_access": 9990}]
            with patch("mavis.runtime.inventory", side_effect=[own, iris]), patch(
                "mavis.runtime.available_memory_bytes", return_value=1000
            ), patch("mavis.runtime.endpoint_alive", return_value=True):
                result = admission(config, now=10000)
            self.assertFalse(result["allowed"])
            self.assertIn("not proven idle", result["reasons"][0])

    def test_concurrent_override_accepts_distinct_iris_model_after_fifteen_minutes(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory), model="mavis-model",
                                   reserve_bytes=10, allow_concurrent_local=True)
            own = [{"id": "mavis-model", "estimated_size": 100}]
            iris = [{"id": "iris-qwen", "model_type": "llm", "loaded": True,
                     "last_access": 9099}]
            with patch("mavis.runtime.inventory", side_effect=[own, iris]), patch(
                "mavis.runtime.available_memory_bytes", return_value=1000
            ), patch("mavis.runtime.endpoint_alive", return_value=True):
                decision = admission(config, now=10000)
            self.assertTrue(decision["allowed"])

    def test_admission_requires_model_plus_reserve(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory), reserve_bytes=100)
            with patch(
                "mavis.runtime.inventory",
                return_value=[{"id": config.model, "estimated_size": 1000}],
            ), patch("mavis.runtime.available_memory_bytes", return_value=1099), patch(
                "mavis.runtime.endpoint_alive", return_value=False
            ):
                result = admission(config)
            self.assertEqual(
                result["reasons"],
                ["aggregate available memory does not satisfy model plus reserve"],
            )


if __name__ == "__main__":
    unittest.main()
