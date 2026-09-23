"""The shared GPU observer must not touch IRIS or leak its lease."""

from pathlib import Path
import importlib.util
import sys
import unittest
from unittest.mock import Mock, patch

from mavis.runtime import RuntimeConfig


SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "shared_gpu_observation.py"
)
spec = importlib.util.spec_from_file_location("shared_gpu_observation", SCRIPT)
assert spec and spec.loader
observer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = observer
spec.loader.exec_module(observer)


class SharedGpuObservationTests(unittest.TestCase):
    def setUp(self):
        self.config = RuntimeConfig(
            home=Path("/private/tmp/mavis-test"), allow_concurrent_local=True
        )
        self.status = "the GPU is free\n"
        self.holding = "codex-mavis has the GPU: test (90 min left)\n"

    def test_occupied_lease_does_not_touch_services(self):
        with (
            patch.object(
                observer, "_lease_command", return_value="claude-work has the GPU\n"
            ) as lease,
            patch.object(observer, "park_mavis_server") as park,
        ):
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                with observer.shared_mavis_model(self.config, "test"):
                    pass
        lease.assert_called_once_with("status")
        park.assert_not_called()

    def test_preflight_failure_releases_lease_without_parking_server(self):
        with (
            patch.object(
                observer,
                "_lease_command",
                side_effect=[self.status, "holding", self.holding, "released"],
            ) as lease,
            patch.object(
                observer, "require_idle_iris_handoff", side_effect=RuntimeError("busy")
            ),
            patch.object(observer, "park_mavis_server") as park,
            patch.object(observer, "_iris_loaded", return_value=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "busy"):
                with observer.shared_mavis_model(self.config, "test"):
                    pass
        park.assert_not_called()
        self.assertEqual(lease.call_args_list[-1].args, ("release", "codex-mavis"))

    def test_success_loads_and_unloads_only_mavis_then_releases(self):
        reservation = Mock()
        with (
            patch.object(
                observer,
                "_lease_command",
                side_effect=[self.status, "holding", self.holding, "released"],
            ) as lease,
            patch.object(observer, "require_idle_iris_handoff"),
            patch.object(observer, "_iris_loaded", return_value=True),
            patch.object(observer, "endpoint_alive", side_effect=[False, True, False]),
            patch.object(
                observer, "loaded_generation_models", return_value=[self.config.model]
            ),
            patch.object(
                observer, "park_mavis_server", return_value=reservation
            ) as park,
            patch.object(observer, "ensure_runtime") as start,
            patch.object(observer, "load_model") as load,
            patch.object(observer, "request_json") as request,
        ):
            with observer.shared_mavis_model(self.config, "test") as result:
                self.assertTrue(result["mavis_loaded"])
            self.assertTrue(result["iris_stayed_loaded"])
            self.assertTrue(result["gpu_lease_released"])
        start.assert_called_once_with(self.config, load=False)
        load.assert_called_once_with(self.config)
        self.assertEqual(park.call_count, 2)
        self.assertEqual(request.call_args.args[0], self.config.endpoint)
        self.assertEqual(lease.call_args_list[-1].args, ("release", "codex-mavis"))


if __name__ == "__main__":
    unittest.main()
