from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from mavis.runtime import (
    RuntimeConfig,
    admission,
    ensure_isolated_settings,
    owns_running_server,
    start_server,
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
