from pathlib import Path
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

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
    def test_trial_key_is_absent_from_listener_process_probes(self):
        completed = subprocess.CompletedProcess(["lsof"], 0, stdout="")
        with (patch.dict(os.environ, {"MAVIS_E0_TRIAL_API_KEY": "fixture-key"}),
              patch("mavis.runtime.subprocess.run", return_value=completed) as run):
            self.assertEqual(runtime._listener_pids(8001), set())
            self.assertEqual(runtime._process_open_paths(1234), set())
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            self.assertNotIn("MAVIS_E0_TRIAL_API_KEY", call.kwargs["env"])

    def test_trial_accepts_omlx_null_key_default_before_own_key_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory), api_key="test-secret")
            settings_path = config.base_path / "settings.json"
            write_json(settings_path, {"version": "1.0", "auth": {
                "api_key": None, "skip_api_key_verification": True,
            }})
            runtime.reject_residual_trial_auth(config)
            ensure_isolated_settings(config)
            self.assertEqual(runtime.read_json(settings_path)["auth"], {
                "api_key": "test-secret", "skip_api_key_verification": False,
            })

    def test_trial_rejects_unsafe_null_or_residual_auth_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory), api_key="test-secret")
            settings_path = config.base_path / "settings.json"
            for auth in (
                {"api_key": None, "skip_api_key_verification": False},
                {"api_key": None, "skip_api_key_verification": "true"},
                {"api_key": None},
                {"api_key": "", "skip_api_key_verification": True},
                {"api_key": "another-secret", "skip_api_key_verification": True},
                {"skip_api_key_verification": False},
            ):
                with self.subTest(auth=auth):
                    write_json(settings_path, {"version": "1.0", "auth": auth})
                    original = settings_path.read_bytes()
                    with self.assertRaisesRegex(RuntimeError, "residual Mavis trial"):
                        runtime.reject_residual_trial_auth(config)
                    self.assertEqual(settings_path.read_bytes(), original)

    def test_trial_rejects_persisted_pinned_preload_before_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "omlx"
            binary.write_text("fixture")
            config = RuntimeConfig(home=root / "home", omlx_binary=binary,
                                   api_key="test-secret")
            write_json(config.base_path / "model_settings.json", {
                "version": 1, "models": {"old-model": {"is_pinned": True}},
            })
            with (patch("mavis.runtime.endpoint_alive", return_value=False),
                  patch("mavis.runtime.port_in_use", return_value=False),
                  patch("mavis.runtime.subprocess.Popen") as spawn,
                  self.assertRaisesRegex(RuntimeError, "pinned-model preload")):
                start_server(config, require_new=True, heartbeat=lambda: None)
            spawn.assert_not_called()

    def test_trial_refuses_even_matching_residual_key_before_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "omlx"
            binary.write_text("fixture")
            config = RuntimeConfig(home=root / "home", omlx_binary=binary,
                                   api_key="test-secret")
            settings_path = config.base_path / "settings.json"
            write_json(settings_path, {"version": "1.0", "auth": {
                "api_key": "test-secret", "skip_api_key_verification": False,
            }})
            original = settings_path.read_bytes()
            with (patch("mavis.runtime.endpoint_alive", return_value=False),
                  patch("mavis.runtime.port_in_use", return_value=False),
                  patch("mavis.runtime.subprocess.Popen") as spawn,
                  self.assertRaisesRegex(RuntimeError, "residual Mavis trial") as caught):
                start_server(config, require_new=True, heartbeat=lambda: None)
            self.assertTrue(caught.exception.residual_trial_auth)
            self.assertEqual(settings_path.read_bytes(), original)
            spawn.assert_not_called()

    def test_trial_rejects_untrusted_pinned_record_and_accepts_false(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory), api_key="test-secret")
            path = config.base_path / "model_settings.json"
            for pinned in ("false", 1):
                write_json(path, {"version": 1, "models": {"model": {
                    "is_pinned": pinned,
                }}})
                with self.assertRaisesRegex(RuntimeError, "pinned-model state"):
                    runtime.reject_persisted_trial_preload(config)
            write_json(path, {"version": 1, "models": {"model": {
                "is_pinned": False,
            }}})
            runtime.reject_persisted_trial_preload(config)

    def test_post_launch_game_heartbeat_stops_exact_no_gpu_child_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "harmless-server"
            binary.write_text("#!/bin/sh\nsleep 30\n")
            binary.chmod(0o700)
            config = RuntimeConfig(home=root / "home", omlx_binary=binary,
                                   api_key="test-secret")
            write_json(config.base_path / "settings.json", {"version": "1.0", "auth": {
                "api_key": None, "skip_api_key_verification": True,
            }})
            observed: list[int] = []
            calls = 0
            def heartbeat() -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    observed.append(runtime.read_json(config.state_path)["pid"])
                    raise RuntimeError("game began during trial startup")
            with (patch("mavis.runtime.endpoint_alive", return_value=False),
                  patch("mavis.runtime.port_in_use", return_value=False),
                  self.assertRaisesRegex(RuntimeError, "game began")):
                start_server(config, require_new=True, heartbeat=heartbeat)
            self.assertEqual(calls, 2)
            self.assertFalse(config.state_path.exists())
            self.assertEqual(runtime._process_group_workers(observed[0]), set())
            with self.assertRaises(ChildProcessError):
                os.waitid(os.P_PID, observed[0], os.WEXITED | os.WNOHANG | os.WNOWAIT)

    def test_unsafe_heartbeat_kills_only_its_unreaped_no_gpu_group(self):
        child = subprocess.Popen(
            [sys.executable, "-c", "import signal,time; "
             "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
             "print('ready', flush=True); time.sleep(30)"],
            start_new_session=True, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True,
        )
        try:
            self.assertEqual(child.stdout.readline().strip(), "ready")
            runtime._stop_spawned_process_group(
                child, timeout=2, heartbeat=lambda: False,
            )
            self.assertEqual(child.returncode, -signal.SIGKILL)
        finally:
            child.stdout.close()
            if child.returncode is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()

    def test_game_during_park_status_aborts_registered_group_without_idle_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory), api_key="test-secret")
            state = {"pid": 1234}
            write_json(config.state_path, state)
            heartbeat = Mock(side_effect=[True, True, False])
            reservation = Mock()
            with (patch.dict(runtime._TRIAL_PROCESSES, {1234: FakeProcess()}, clear=True),
                  patch("mavis.runtime.endpoint_alive", return_value=True),
                  patch("mavis.runtime.request_json", return_value={
                      "status": "ok", "active_requests": 1,
                      "waiting_requests": 0, "models_loading": 0,
                  }), patch("mavis.runtime.stop_trial_server") as stop,
                  patch("mavis.runtime.reserve_empty_mavis_port",
                        return_value=reservation)):
                self.assertIs(runtime.park_mavis_server(
                    config, expected_pid=1234, expected_state=state,
                    heartbeat=heartbeat,
                ), reservation)
            stop.assert_called_once_with(config, state, timeout=1)
            self.assertEqual(heartbeat.call_count, 3)

    def test_park_reservation_failure_records_proven_group_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory), api_key="test-secret")
            state = {"pid": 1234}
            write_json(config.state_path, state)
            with (patch.dict(runtime._TRIAL_PROCESSES, {1234: FakeProcess()}, clear=True),
                  patch("mavis.runtime.endpoint_alive", return_value=True),
                  patch("mavis.runtime.request_json", return_value={
                      "status": "ok", "active_requests": 0,
                      "waiting_requests": 0, "models_loading": 0,
                  }), patch("mavis.runtime.stop_trial_server") as stop,
                  patch("mavis.runtime.reserve_empty_mavis_port",
                        side_effect=RuntimeError("foreign listener")),
                  self.assertRaisesRegex(RuntimeError, "group stopped") as caught):
                runtime.park_mavis_server(
                    config, expected_pid=1234, expected_state=state,
                )
            stop.assert_called_once_with(config, state, timeout=30, heartbeat=None)
            self.assertTrue(caught.exception.gpu_work_stopped)

    def test_exited_unreaped_leader_anchors_worker_group_until_stop(self):
        child = subprocess.Popen(
            [sys.executable, "-c", "import subprocess,sys; "
             "subprocess.Popen(['sleep', '30']); sys.exit(7)"],
            start_new_session=True, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 5
            while runtime._child_exit_unreaped(child) is None and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertIsNotNone(runtime._child_exit_unreaped(child))
            self.assertIsNone(child.returncode)
            self.assertTrue(runtime._process_group_workers(child.pid))
            runtime._stop_spawned_process_group(child, timeout=3)
            self.assertEqual(child.returncode, 7)
        finally:
            if child.returncode is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()

    def test_exited_unreaped_leader_with_empty_group_survives_macos_eperm(self):
        # Forced EPERM with an exited leader and no workers: macOS answers EPERM,
        # not ESRCH, for a group whose only member is an exited, unreaped leader.
        # That group has no GPU work left, so the stop must proceed.
        process = FakeProcess()
        exited = Mock(si_status=0)
        with patch("mavis.runtime._child_exit_unreaped", return_value=exited), \
                patch("mavis.runtime._process_group_workers", return_value=set()), \
                patch("mavis.runtime.os.killpg",
                      side_effect=PermissionError(1, "denied")) as kill:
            runtime._stop_spawned_process_group(process, timeout=1)
        kill.assert_called_once_with(1234, signal.SIGTERM)
        self.assertEqual(process.returncode, -signal.SIGTERM)

    def test_eperm_is_not_accepted_while_leader_runs_or_workers_remain(self):
        for exited, workers in ((None, set()), (Mock(si_status=0), {4321})):
            process = FakeProcess()
            with self.subTest(exited=exited, workers=workers), \
                    patch("mavis.runtime._child_exit_unreaped", return_value=exited), \
                    patch("mavis.runtime.os.getpgid", return_value=1234), \
                    patch("mavis.runtime._process_group_workers", return_value=workers), \
                    patch("mavis.runtime.os.killpg", side_effect=PermissionError(1, "denied")):
                with self.assertRaises(PermissionError):
                    runtime._stop_spawned_process_group(process, timeout=0.2)

    def test_reaped_leader_never_signals_numeric_group(self):
        process = FakeProcess()
        process.returncode = 7
        with patch("mavis.runtime.os.killpg") as kill, self.assertRaisesRegex(
            RuntimeError, "already reaped"
        ):
            runtime._stop_spawned_process_group(process)
        kill.assert_not_called()

    def test_leader_exits_between_waitid_and_getpgid_without_reaping(self):
        process = FakeProcess()
        exited = Mock(si_status=7)
        with patch("mavis.runtime._child_exit_unreaped",
                   side_effect=[None, exited, exited]), \
                patch("mavis.runtime.os.getpgid", side_effect=ProcessLookupError), \
                patch("mavis.runtime._process_group_workers", return_value=set()), \
                patch("mavis.runtime.os.killpg") as kill:
            runtime._stop_spawned_process_group(process, timeout=1)
        kill.assert_called_once_with(1234, signal.SIGTERM)
        self.assertIsNotNone(process.returncode)

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
                    patch("mavis.runtime._child_exit_unreaped", return_value=None), \
                    patch("mavis.runtime.subprocess.Popen", return_value=process) as spawn:
                state = start_server(config, require_new=True)
            self.assertNotIn("test-secret", str(state))
            self.assertNotIn("test-secret", str(spawn.call_args.args[0]))
            self.assertNotIn("MAVIS_E0_TRIAL_API_KEY", spawn.call_args.kwargs["env"])
            self.assertEqual(alive.call_args_list[-1].kwargs, {"api_key": "test-secret"})

    def test_iris_voice_session_probe_reads_bounded_loopback_status(self):
        import io
        from unittest.mock import Mock
        for payload, expected in ((b'{"sessionActive": true}', True),
                                  (b'{"sessionActive": false}', False),
                                  (b'{}', False)):
            response = Mock()
            response.__enter__ = Mock(return_value=io.BytesIO(payload))
            response.__exit__ = Mock(return_value=False)
            with self.subTest(payload=payload), \
                    patch("mavis.runtime.urlopen", return_value=response) as open_url:
                self.assertIs(runtime.iris_voice_session_active(), expected)
            self.assertEqual(open_url.call_args.args[0].full_url,
                             "http://127.0.0.1:8117/status")
        with patch("mavis.runtime.urlopen", side_effect=OSError("refused")):
            self.assertIsNone(runtime.iris_voice_session_active())

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
                    }), patch("mavis.runtime._child_exit_unreaped", return_value=None), \
                    patch("mavis.runtime._stop_spawned_process_group",
                          side_effect=lambda child, **_: child.wait()) as stop:
                stop_server(config)
            stop.assert_called_once_with(process, timeout=10.0)
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
                    patch("mavis.runtime._child_exit_unreaped", return_value=None), \
                    patch("mavis.runtime._stop_spawned_process_group",
                          side_effect=lambda child, **_: child.wait()) as stop:
                stop_trial_server(config, state)
                self.assertNotIn(1234, runtime._TRIAL_PROCESSES)
                self.assertFalse(config.state_path.exists())
            stop.assert_called_once_with(process, timeout=10.0)

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
                with patch.dict(runtime._TRIAL_PROCESSES, {child.pid: child}, clear=True), \
                        patch("mavis.runtime.endpoint_alive", return_value=True), \
                        patch("mavis.runtime.request_json", return_value=idle), \
                        patch("mavis.runtime.owns_running_server", return_value=True):
                    reservation = park_mavis_server(config)
                try:
                    self.assertIsNotNone(child.returncode)
                    self.assertFalse(config.state_path.exists())
                    with socket.socket() as probe:
                        self.assertEqual(probe.connect_ex(("127.0.0.1", port)), 0)
                    with self.assertRaises(OSError):
                        with socket.socket() as competing:
                            competing.bind(("127.0.0.1", port))
                finally:
                    reservation.close()
            finally:
                if child.returncode is None:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()

    def test_park_refuses_unregistered_process_without_group_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            config = RuntimeConfig(home=Path(directory),
                                   endpoint=f"http://127.0.0.1:{port}/v1")
            write_json(config.state_path, {"pid": 1234})
            with patch.dict(runtime._TRIAL_PROCESSES, {}, clear=True), \
                    patch("mavis.runtime.os.killpg") as kill, \
                    self.assertRaisesRegex(RuntimeError, "no process handle"):
                park_mavis_server(config, timeout=1)
            kill.assert_not_called()

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
                    patch("mavis.runtime._stop_spawned_process_group",
                          side_effect=lambda child: child.wait()) as stop:
                with self.assertRaises(TimeoutError):
                    start_server(config, wait_seconds=0)
            stop.assert_called_once_with(process)
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
                    patch("mavis.runtime._stop_spawned_process_group",
                          side_effect=lambda child: child.wait()) as stop:
                with self.assertRaisesRegex(OSError, "disk full"):
                    start_server(config, require_new=True)
            stop.assert_called_once_with(process)
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
