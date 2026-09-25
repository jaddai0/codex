"""No-GPU fault checks for installed E1 shared GPU admission."""

from pathlib import Path
from dataclasses import replace
import os
import subprocess
import tempfile
import threading
import time
import unittest
import io
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from mavis import e1_gpu
from mavis import runtime
from mavis.runtime import RuntimeConfig
from mavis.storage import read_json

RAW_LEASE_COMMAND = e1_gpu._lease_command


class E1GPUAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = RuntimeConfig(home=Path(self.temp.name), model="model-a",
                                    api_key="fixture-private-key")
        self.state = {"pid": 1234, "command": ["trial-omlx"]}
        self.commands = []
        self.held = False
        self.loaded = False
        self.server = False
        self.iris = ["iris-qwen"]
        self.game = False
        self.unavailable = False
        self.purpose = ""
        self.fail_load = False
        self.fail_unload = False
        self.fail_abort = False
        self.active_requests = 0
        self.models_loading = 0
        self.load_gate = None

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

        def generations(endpoint, **_kwargs):
            return list(self.iris) if endpoint == self.config.iris_endpoint else (
                [self.config.model] if self.loaded else [])

        def load(config):
            self.assertTrue(config.allow_concurrent_local)
            self.assertEqual(config.idle_seconds, 900)
            if self.load_gate is not None:
                self.models_loading = 1
                self.load_gate.wait(20)
                self.models_loading = 0
            self.loaded = True  # an HTTP failure may follow a successful load
            if self.fail_load:
                raise RuntimeError("load response lost")

        def request(endpoint, path, **kwargs):
            if path == "/api/status":
                if endpoint == self.config.endpoint and not kwargs.get("headers"):
                    raise HTTPError(endpoint, 401, "authentication required", None,
                                    io.BytesIO())
                return {"status": "ok", "active_requests": self.active_requests,
                        "waiting_requests": 0, "models_loading": self.models_loading}
            self.assertEqual(path, "/v1/models/model-a/unload")
            self.assertEqual(kwargs["method"], "POST")
            if self.fail_unload:
                raise RuntimeError("unload failed")
            self.loaded = False
            return {}

        def start(config, *, require_new, heartbeat):
            self.assertTrue(require_new)
            self.assertEqual(config.api_key, "fixture-private-key")
            self.assertFalse(self.server)
            heartbeat()
            self.server = True
            heartbeat()
            return self.state

        def park(config, *, expected_pid, expected_state, heartbeat):
            self.assertEqual(expected_pid, 1234)
            self.assertEqual(expected_state, self.state)
            self.assertTrue(self.server)
            self.assertTrue(heartbeat())
            self.server = False
            self.loaded = False
            return Mock()

        def abort(config, state, **kwargs):
            self.assertEqual(state, self.state)
            if self.fail_abort:
                raise RuntimeError("trial abort failed")
            self.server = False
            self.loaded = False
            self.models_loading = 0
            if self.load_gate is not None:
                self.load_gate.set()

        patchers = [
            patch.object(e1_gpu, "_lease_command", side_effect=command),
            patch.object(e1_gpu, "require_idle_iris_handoff"),
            patch.object(e1_gpu, "loaded_generation_models", side_effect=generations),
            patch.object(e1_gpu, "endpoint_alive", side_effect=lambda _, **__: self.server),
            patch.object(e1_gpu, "reserve_empty_mavis_port", return_value=Mock()),
            patch.object(e1_gpu, "start_server", side_effect=start),
            patch.object(e1_gpu, "read_json", return_value=self.state),
            patch.object(e1_gpu, "park_mavis_server", side_effect=park),
            patch.object(e1_gpu, "stop_trial_server", side_effect=abort),
            patch.object(e1_gpu, "load_model", side_effect=load),
            patch.object(e1_gpu, "request_json", side_effect=request),
            patch.object(e1_gpu, "inventory", side_effect=lambda endpoint, **kwargs: (
                [{"id": "model-a", "loaded": self.loaded}] if endpoint == self.config.endpoint
                else [{"id": model, "loaded": True} for model in self.iris])),
            patch.object(e1_gpu, "_read_lease_record", side_effect=lambda: (
                {"holder": "codex-mavis", "purpose": self.purpose,
                 "expires": time.time() + 3600} if self.held else None)),
            patch.object(e1_gpu, "_iris_game_live", side_effect=lambda: (
                None if self.unavailable else bool(self.game))),
        ]
        self.patches = [p.start() for p in patchers]
        for patcher in patchers:
            self.addCleanup(patcher.stop)
        self.handoff = self.patches[1]
        self.reserve = self.patches[4]
        self.start = self.patches[5]
        self.park = self.patches[7]
        self.abort = self.patches[8]

    def test_new_server_preserves_iris_then_unloads_and_parks(self):
        with e1_gpu.admitted_e1_model(self.config, "e1:trial") as heartbeat:
            self.assertTrue(self.loaded)
            self.assertEqual(self.iris, ["iris-qwen"])
            heartbeat.next_renew = 0
            heartbeat(force=True)
        self.assertFalse(self.loaded)
        self.assertFalse(self.server)
        self.assertFalse(self.held)
        self.assertEqual(self.handoff.call_args.kwargs["expected_models"], ["iris-qwen"])
        self.assertEqual([cmd[0] for cmd in self.commands].count("renew"), 1)
        self.park.assert_called_once()
        self.abort.assert_not_called()

    def test_preexisting_server_fails_without_load_or_stop(self):
        self.server = True
        self.reserve.side_effect = RuntimeError("port occupied")
        with self.assertRaisesRegex(RuntimeError, "port occupied"):
            with e1_gpu.admitted_e1_model(self.config, "e1:trial"):
                pass
        self.start.assert_not_called()
        self.park.assert_not_called()
        self.abort.assert_not_called()
        self.assertFalse(self.held)

    def test_race_after_empty_reservation_fails_without_adopting_server(self):
        self.start.side_effect = RuntimeError("endpoint occupied by another session")
        with self.assertRaisesRegex(RuntimeError, "occupied by another session"):
            with e1_gpu.admitted_e1_model(self.config, "e1:trial"):
                pass
        self.park.assert_not_called()
        self.abort.assert_not_called()
        self.assertFalse(self.held)

    def test_game_or_unknown_game_state_refuses_before_acquire(self):
        for attribute in ("game", "unavailable"):
            setattr(self, attribute, True)
            with self.assertRaisesRegex(RuntimeError, "unsafe"):
                with e1_gpu.admitted_e1_model(self.config, "e1:trial"):
                    pass
            self.assertFalse(self.held)
            self.assertNotIn("acquire", [cmd[0] for cmd in self.commands])
            setattr(self, attribute, False)

    def test_partial_load_failure_stops_trial_server_then_releases(self):
        self.fail_load = True
        with self.assertRaisesRegex(RuntimeError, "load response lost"):
            with e1_gpu.admitted_e1_model(self.config, "e1:trial"):
                pass
        self.assertFalse(self.loaded)
        self.assertFalse(self.server)
        self.assertFalse(self.held)

    def test_game_starts_during_blocking_load_and_aborts_exact_trial(self):
        self.load_gate = threading.Event()
        original_load = self.patches[9]
        def loading(config):
            self.game = True
            self.models_loading = 1
            self.load_gate.wait(20)
            self.models_loading = 0
        original_load.side_effect = loading
        with self.assertRaisesRegex(RuntimeError, "game state is unsafe"):
            with e1_gpu.admitted_e1_model(self.config, "e1:trial"):
                pass
        self.abort.assert_called_once_with(self.config.__class__(
            **{**self.config.__dict__, "allow_concurrent_local": True, "idle_seconds": 900}
        ), self.state, timeout=1)
        self.assertFalse(self.server)
        self.assertFalse(self.held)

    def test_game_during_server_startup_stops_before_model_load(self):
        def start(config, *, require_new, heartbeat):
            self.server = True
            self.game = True
            try:
                heartbeat()
            finally:
                self.server = False  # start_server cleans up its exact child

        self.start.side_effect = start
        with self.assertRaisesRegex(RuntimeError, "game state is unsafe"):
            with e1_gpu.admitted_e1_model(self.config, "e1:trial"):
                pass
        self.patches[9].assert_not_called()
        self.assertFalse(self.server)
        self.assertFalse(self.held)

    def test_real_server_startup_uses_key_and_reaps_its_group_on_game(self):
        binary = Path(self.temp.name) / "harmless-server"
        binary.write_text("#!/bin/sh\nsleep 30\n")
        binary.chmod(0o700)
        self.config = replace(self.config, omlx_binary=binary)
        spawned: list[int] = []

        def start(config, *, require_new, heartbeat):
            def game_heartbeat():
                if config.state_path.is_file():
                    spawned.append(runtime.read_json(config.state_path)["pid"])
                    self.game = True
                heartbeat()
            return runtime.start_server(config, require_new=require_new,
                                        heartbeat=game_heartbeat)

        self.start.side_effect = start
        with (patch.object(runtime, "endpoint_alive", return_value=False),
              patch.object(runtime, "port_in_use", return_value=False),
              self.assertRaisesRegex(RuntimeError, "game state is unsafe")):
            with e1_gpu.admitted_e1_model(self.config, "e1:trial"):
                pass
        self.assertTrue(spawned)
        self.assertFalse(self.config.state_path.exists())
        with self.assertRaises(ProcessLookupError):
            os.kill(spawned[0], 0)
        self.assertFalse(self.held)

    def test_live_trial_checks_game_within_one_second_after_prior_heartbeat(self):
        detected = False
        with self.assertRaisesRegex(RuntimeError, "game state is unsafe"):
            with e1_gpu.admitted_e1_model(self.config, "e1:trial") as heartbeat:
                heartbeat(force=True)
                self.game = True
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    try:
                        heartbeat()
                    except RuntimeError as error:
                        self.assertIn("game state is unsafe", str(error))
                        detected = True
                        raise
                    time.sleep(0.05)
        self.assertTrue(detected, "E1 heartbeat did not detect the game within one second")
        self.assertFalse(self.server)
        self.assertFalse(self.held)

    def test_slow_gpu_status_is_bounded_and_fails_closed(self):
        with patch.object(e1_gpu.subprocess, "run", side_effect=subprocess.TimeoutExpired(
            [str(e1_gpu.LEASE), "status"], 8,
        )) as run:
            with self.assertRaises(subprocess.TimeoutExpired):
                RAW_LEASE_COMMAND("status")
        # gpu-lease status can spend 1 s asking IRIS and 5 s scanning processes;
        # it measured 3.5 s during a large model load (2026-09-25). Monitoring
        # does not call it, so this looser bound never sits on the safety path.
        self.assertEqual(run.call_args.kwargs["timeout"], 8)

    def test_slow_gpu_status_cannot_delay_monitor_game_detection(self):
        # A blocked monitor must still notice a game within its bound even while
        # `gpu-lease status` is stuck in its process scan.
        slow = threading.Event()
        entered = threading.Event()
        original_abort = self.abort.side_effect
        original_command = self.patches[0].side_effect

        def abort(config, state, **kwargs):
            original_abort(config, state, **kwargs)
            slow.set()

        def command(*args):
            if args[0] == "status" and entered.is_set():
                slow.wait(5)
            return original_command(*args)

        def hashing():
            entered.set()
            self.game = True
            slow.wait(5)

        self.abort.side_effect = abort
        self.patches[0].side_effect = command
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "game state is unsafe"):
            with e1_gpu.admitted_e1_model(self.config, "e1:trial") as heartbeat:
                heartbeat.monitor_read_only(hashing)
        self.assertLess(time.monotonic() - started, 1.5)
        self.abort.assert_called_once()
        self.assertFalse(self.server)

    def test_monitor_sees_lost_lease_before_the_slow_command_returns(self):
        heartbeat = e1_gpu._LeaseHeartbeat("e1:trial:abc")
        with patch.object(e1_gpu, "_lease_command") as slow, \
                patch.object(e1_gpu, "_read_lease_record", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "purpose changed"):
                heartbeat.monitor()
            slow.assert_not_called()

    def test_monitor_treats_unanswered_iris_as_unsafe(self):
        heartbeat = e1_gpu._LeaseHeartbeat("e1:trial:abc")
        with patch.object(e1_gpu, "_read_lease_record", return_value={
                "holder": "codex-mavis", "purpose": "e1:trial:abc",
                "expires": time.time() + 3600}), \
                patch.object(e1_gpu, "_iris_game_live", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "game state is unsafe"):
                heartbeat.monitor()

    def test_game_during_post_load_hash_stops_exact_idle_server(self):
        gate = threading.Event()
        entered = threading.Event()
        original_abort = self.abort.side_effect

        def abort(config, state, **kwargs):
            original_abort(config, state, **kwargs)
            gate.set()

        def hashing():
            entered.set()
            self.game = True
            gate.wait(5)
            return {"bound": True}

        self.abort.side_effect = abort
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "game state is unsafe"):
            with e1_gpu.admitted_e1_model(self.config, "e1:trial") as heartbeat:
                heartbeat.monitor_read_only(hashing)
        self.assertTrue(entered.is_set())
        self.assertLess(time.monotonic() - started, 1.5)
        self.abort.assert_called_once()
        self.assertFalse(self.server)
        self.assertFalse(self.held)

    def test_slow_status_during_hash_stops_server_and_retains_unverified_lease(self):
        gate = threading.Event()
        entered = threading.Event()
        original_abort = self.abort.side_effect
        original_command = self.patches[0].side_effect

        def abort(config, state, **kwargs):
            original_abort(config, state, **kwargs)
            gate.set()

        def command(*args):
            if args[0] == "status" and entered.is_set():
                raise subprocess.TimeoutExpired([str(e1_gpu.LEASE), "status"], 0.5)
            return original_command(*args)

        def hashing():
            entered.set()
            gate.wait(5)

        self.abort.side_effect = abort
        self.patches[0].side_effect = command
        with self.assertRaisesRegex(RuntimeError, "inventory receipt"):
            with e1_gpu.admitted_e1_model(self.config, "e1:trial") as heartbeat:
                heartbeat.monitor_read_only(hashing)
        self.abort.assert_called_once()
        self.assertFalse(self.server)
        self.assertTrue(self.held)

    def test_same_name_lease_takeover_stops_own_server_but_not_new_lease(self):
        with self.assertRaisesRegex(RuntimeError, "purpose changed"):
            with e1_gpu.admitted_e1_model(self.config, "e1:trial") as heartbeat:
                self.purpose = "another-codex-session"
                heartbeat(force=True)
        self.abort.assert_called_once()
        self.assertFalse(self.server)
        self.assertTrue(self.held)
        self.assertNotIn("release", [cmd[0] for cmd in self.commands])
        receipts = list((self.config.home / "e1/admission-failures").glob("*.json"))
        self.assertEqual(len(receipts), 1)
        self.assertEqual(read_json(receipts[0])["mavis_inventory"], [{"id": "model-a", "loaded": False}])

    def test_active_request_prevents_stop_and_retains_lease_with_receipt(self):
        with self.assertRaisesRegex(RuntimeError, "lease retained"):
            with e1_gpu.admitted_e1_model(self.config, "e1:trial"):
                self.active_requests = 1
        self.assertTrue(self.server)
        self.assertTrue(self.loaded)
        self.assertTrue(self.held)
        self.abort.assert_not_called()
        receipts = list((self.config.home / "e1/admission-failures").glob("*.json"))
        self.assertEqual(len(receipts), 1)
        self.assertFalse(read_json(receipts[0])["gpu_work_stopped"])

    def test_game_stops_exact_trial_without_waiting_for_active_request(self):
        original_request = self.patches[10].side_effect
        observed = {"active": 0}

        def request(endpoint, path, **kwargs):
            if path == "/api/status" and self.game:
                observed["active"] += 1
                self.active_requests = 1 if observed["active"] == 1 else 0
            return original_request(endpoint, path, **kwargs)

        self.patches[10].side_effect = request
        with self.assertRaisesRegex(RuntimeError, "game state is unsafe"):
            with e1_gpu.admitted_e1_model(self.config, "e1:trial"):
                self.active_requests = 1
                self.game = True
        self.assertEqual(observed["active"], 0)
        self.abort.assert_called_once()
        self.assertFalse(self.server)
        self.assertFalse(self.held)

    def test_game_during_blocking_unload_stops_private_trial(self):
        unload_gate = threading.Event()
        entered = threading.Event()
        original_request = self.patches[10].side_effect
        original_abort = self.abort.side_effect

        def request(endpoint, path, **kwargs):
            if path.endswith("/unload"):
                self.assertEqual(kwargs["headers"]["Authorization"],
                                 "Bearer fixture-private-key")
                entered.set()
                self.game = True
                unload_gate.wait(5)
                return {}
            return original_request(endpoint, path, **kwargs)

        def abort(config, state, **kwargs):
            original_abort(config, state, **kwargs)
            unload_gate.set()

        self.patches[10].side_effect = request
        self.abort.side_effect = abort
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "game state is unsafe"):
            with e1_gpu.admitted_e1_model(self.config, "e1:trial"):
                pass
        self.assertTrue(entered.is_set())
        self.assertLess(time.monotonic() - started, 1.5)
        self.abort.assert_called_once()
        self.assertFalse(self.server)
        self.assertFalse(self.held)

    def test_game_during_parking_triggers_exact_group_stop(self):
        original_abort = self.abort.side_effect

        def park(config, *, expected_pid, expected_state, heartbeat):
            self.assertEqual(expected_pid, 1234)
            self.game = True
            self.assertFalse(heartbeat())
            original_abort(config, expected_state, timeout=1)
            return Mock()

        self.park.side_effect = park
        with self.assertRaisesRegex(RuntimeError, "game state is unsafe"):
            with e1_gpu.admitted_e1_model(self.config, "e1:trial"):
                pass
        self.assertFalse(self.server)
        self.assertFalse(self.held)

    def test_failed_abort_retains_lease_and_writes_receipt(self):
        self.fail_unload = True
        self.fail_abort = True
        with self.assertRaisesRegex(RuntimeError, "lease retained"):
            with e1_gpu.admitted_e1_model(self.config, "e1:trial"):
                pass
        self.assertTrue(self.held)
        self.assertTrue(self.server)
        self.assertEqual(len(list((self.config.home / "e1/admission-failures").glob("*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
