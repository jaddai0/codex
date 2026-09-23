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
    ensure_isolated_settings,
    handoff_lease_fd,
    loaded_generation_models,
    mavis_generation_lease,
    owns_running_server,
    require_idle_iris_handoff,
    require_installed_selected_model,
    release_iris_model_drain,
    park_mavis_server,
    start_server,
    wait_iris_model_drain,
)
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


class RuntimeTests(unittest.TestCase):
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
            ), patch("mavis.runtime.subprocess.Popen", return_value=process) as popen:
                with self.assertRaises(TimeoutError):
                    start_server(config, wait_seconds=0)
            self.assertTrue(process.terminated)
            child_env = popen.call_args.kwargs["env"]
            self.assertEqual(child_env["OMLX_BASE_PATH"], str(config.base_path))
            self.assertEqual(child_env["HOME"], str(config.home / "user-home"))

    def test_settings_pin_auth_server_model_and_cache_to_mavis(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory))
            ensure_isolated_settings(config)
            settings = __import__("json").loads(
                (config.base_path / "settings.json").read_text()
            )
            self.assertEqual(settings["server"], {"host": "127.0.0.1", "port": 8001})
            self.assertTrue(settings["auth"]["skip_api_key_verification"])
            self.assertEqual(settings["model"]["model_dirs"], [str(config.model_dir)])
            self.assertEqual(settings["cache"]["ssd_cache_dir"], str(config.base_path / "cache"))

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
