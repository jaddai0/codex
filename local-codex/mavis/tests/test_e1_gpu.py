"""Source-only fault checks for installed E1 shared GPU admission."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mavis import e1_gpu
from mavis.runtime import RuntimeConfig
from mavis.storage import read_json


class E1GPUAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = RuntimeConfig(home=Path(self.temp.name), model="model-a")
        self.commands = []
        self.held = False
        self.loaded = False
        self.iris = ["iris-qwen"]
        self.game = False
        self.unavailable = False
        self.owner = True
        self.purpose = ""
        self.fail_load = False
        self.fail_unload = False
        self.active_requests = 0

        def command(*args):
            self.commands.append(args)
            action = args[0]
            if action == "status":
                first = (f"codex-mavis has the GPU: {self.purpose} (89 min left)"
                         if self.held else "the GPU is free")
                game = ("IRIS: not answering" if self.unavailable else
                        "IRIS: a game is live" if self.game else "IRIS: idle")
                return first + "\n" + game + "\n"
            if action == "acquire":
                self.assertEqual(args[1], "codex-mavis")
                self.purpose = args[2]
                self.held = True
            elif action == "renew":
                self.assertTrue(self.held)
                self.assertEqual(args[1], "codex-mavis")
            elif action == "release":
                self.assertEqual(args[1], "codex-mavis")
                self.held = False
            return ""

        def generations(endpoint):
            return list(self.iris) if endpoint == self.config.iris_endpoint else (
                [self.config.model] if self.loaded else [])

        def load(config):
            self.assertTrue(config.allow_concurrent_local)
            self.assertEqual(config.idle_seconds, 900)
            self.loaded = True  # a failed HTTP response may follow a successful load
            if self.fail_load:
                raise RuntimeError("load response lost")

        def request(endpoint, path, **kwargs):
            if path == "/api/status":
                return {"status": "ok", "active_requests": self.active_requests,
                        "waiting_requests": 0, "models_loading": 0}
            self.assertEqual(path, "/v1/models/model-a/unload")
            self.assertEqual(kwargs["method"], "POST")
            if self.fail_unload:
                raise RuntimeError("unload failed")
            self.loaded = False
            return {}

        patchers = [
            patch.object(e1_gpu, "_lease_command", side_effect=command),
            patch.object(e1_gpu, "require_idle_iris_handoff"),
            patch.object(e1_gpu, "loaded_generation_models", side_effect=generations),
            patch.object(e1_gpu, "endpoint_alive", return_value=True),
            patch.object(e1_gpu, "owns_running_server", side_effect=lambda _config: self.owner),
            patch.object(e1_gpu, "ensure_runtime", return_value={}),
            patch.object(e1_gpu, "load_model", side_effect=load),
            patch.object(e1_gpu, "request_json", side_effect=request),
            patch.object(e1_gpu, "inventory", side_effect=lambda endpoint: [
                {"id": model, "loaded": True} for model in generations(endpoint)]),
        ]
        for patcher in patchers:
            started = patcher.start()
            if patcher is patchers[1]:
                self.handoff = started
            self.addCleanup(patcher.stop)

    def test_keeps_iris_loaded_and_unloads_only_its_model(self):
        with e1_gpu.admitted_e1_model(self.config, "e1:trial") as heartbeat:
            self.assertTrue(self.loaded)
            self.assertEqual(self.iris, ["iris-qwen"])
            heartbeat.next_renew = 0
            heartbeat(force=True)
        self.assertFalse(self.loaded)
        self.assertFalse(self.held)
        self.assertEqual(self.iris, ["iris-qwen"])
        self.assertEqual(self.handoff.call_args.kwargs["expected_models"], ["iris-qwen"])
        self.assertEqual([cmd[0] for cmd in self.commands].count("renew"), 1)
        self.assertEqual(self.commands[-1][0], "release")

    def test_game_or_unknown_game_state_refuses_before_acquire(self):
        for attribute in ("game", "unavailable"):
            setattr(self, attribute, True)
            with self.assertRaisesRegex(RuntimeError, "unsafe"):
                with e1_gpu.admitted_e1_model(self.config, "e1:trial"):
                    pass
            self.assertFalse(self.held)
            self.assertNotIn("acquire", [cmd[0] for cmd in self.commands])
            setattr(self, attribute, False)

    def test_preloaded_or_unowned_mavis_server_is_not_changed(self):
        for loaded, owner in ((True, True), (False, False)):
            self.loaded, self.owner = loaded, owner
            with self.assertRaises(RuntimeError):
                with e1_gpu.admitted_e1_model(self.config, "e1:trial"):
                    pass
            self.assertEqual(self.loaded, loaded)
            self.assertFalse(self.held)
            self.assertNotIn("unload", [cmd[0] for cmd in self.commands])

    def test_partial_load_failure_still_unloads_and_releases(self):
        self.fail_load = True
        with self.assertRaisesRegex(RuntimeError, "load response lost"):
            with e1_gpu.admitted_e1_model(self.config, "e1:trial"):
                pass
        self.assertFalse(self.loaded)
        self.assertFalse(self.held)
        self.assertEqual(self.iris, ["iris-qwen"])

    def test_same_name_lease_takeover_does_not_unload_or_release_new_owner(self):
        with self.assertRaisesRegex(RuntimeError, "purpose changed"):
            with e1_gpu.admitted_e1_model(self.config, "e1:trial") as heartbeat:
                self.purpose = "another-codex-session"
                heartbeat(force=True)
        self.assertTrue(self.loaded)
        self.assertTrue(self.held)
        self.assertNotIn("release", [cmd[0] for cmd in self.commands])
        receipts = list((self.config.home / "e1/admission-failures").glob("*.json"))
        self.assertEqual(len(receipts), 1)
        self.assertEqual(read_json(receipts[0])["mavis_inventory"][0]["id"], "model-a")

    def test_game_start_during_trial_unloads_and_releases(self):
        with self.assertRaisesRegex(RuntimeError, "game state"):
            with e1_gpu.admitted_e1_model(self.config, "e1:trial") as heartbeat:
                self.game = True
                heartbeat(force=True)
        self.assertFalse(self.loaded)
        self.assertFalse(self.held)

    def test_failed_unload_reports_error_and_still_releases(self):
        self.fail_unload = True
        with self.assertRaisesRegex(RuntimeError, "unload failed twice"):
            with e1_gpu.admitted_e1_model(self.config, "e1:trial"):
                pass
        self.assertTrue(self.loaded)
        self.assertFalse(self.held)
        receipts = list((self.config.home / "e1/admission-failures").glob("*.json"))
        self.assertEqual(len(receipts), 1)
        self.assertEqual(read_json(receipts[0])["iris_inventory"][0]["id"], "iris-qwen")

    def test_active_mavis_request_prevents_unload_and_records_failure(self):
        with self.assertRaisesRegex(RuntimeError, "active, waiting, or loading"):
            with e1_gpu.admitted_e1_model(self.config, "e1:trial"):
                self.active_requests = 1
        self.assertTrue(self.loaded)
        self.assertFalse(self.held)
        self.assertEqual(len(list((self.config.home / "e1/admission-failures").glob("*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
