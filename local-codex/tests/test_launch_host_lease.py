"""Model-free checks of the installed launcher host-lease boundary."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mavis"))

from generation_lease import generation_lease
from mavis.maintenance_runtime import host_admission, tick


class LaunchHostLeaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.home = self.root / "mavis-home"
        self.share = self.root / "share"
        self.share.mkdir()
        for name in ("launch_core.py", "generation_lease.py"):
            shutil.copy(ROOT / name, self.share / name)
        (self.share / "prepare_runtime.py").write_text(
            "from pathlib import Path\n"
            "import os, sys, time\n"
            "def accepted_main_profile(_home): return None\n"
            "def atomic_write(path, text): Path(path).write_text(text)\n"
            "if __name__ == '__main__':\n"
            "  phase = 'resolve' if '--resolve-model' in sys.argv else 'receipt'\n"
            "  root = Path(os.environ['STUB_PHASE_ROOT'])\n"
            "  (root / (phase + '.started')).write_text('ready')\n"
            "  while not (root / (phase + '.continue')).exists(): time.sleep(.01)\n"
            "  print('model-a' if phase == 'resolve' else '')\n"
        )
        mavis = self.share / "mavis"
        mavis.mkdir()
        (mavis / "__init__.py").write_text("")
        (mavis / "__main__.py").write_text(
            "from pathlib import Path\n"
            "import os, time\n"
            "root = Path(os.environ['STUB_PHASE_ROOT'])\n"
            "(root / 'ensure.started').write_text('ready')\n"
            "while not (root / 'ensure.continue').exists(): time.sleep(.01)\n"
        )
        self.core = self.root / "core"
        self.core.write_text(
            "#!/usr/bin/env python3\n"
            "from pathlib import Path\n"
            "import os, time\n"
            "root = Path(os.environ['STUB_PHASE_ROOT'])\n"
            "(root / 'core.started').write_text(os.environ['CODEX_HOME'])\n"
            "while not (root / 'core.continue').exists(): time.sleep(.01)\n"
        )
        self.core.chmod(0o755)
        self.env = {
            **os.environ,
            "HOME": str(self.root),
            "LOCAL_CODEX_SHARE_DIR": str(self.share),
            "LOCAL_CODEX_HOME": str(self.root / "codex-home"),
            "MAVIS_HOME": str(self.home),
            "LOCAL_CODEX_BIN": str(self.core),
            "STUB_PHASE_ROOT": str(self.root),
        }
        self.env.pop("MAVIS_GENERATION_LEASE_FD", None)

    def _start(self, *, env=None, pass_fds=()):
        process = subprocess.Popen(
            ["zsh", str(ROOT / "bin" / "local-codex")],
            env=env or self.env, pass_fds=pass_fds,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self.addCleanup(lambda: self._stop(process))
        return process

    @staticmethod
    def _stop(process):
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=2)

    def _phase(self, process, name):
        marker = self.root / f"{name}.started"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not marker.exists():
            if process.poll() is not None:
                self.fail(f"launcher exited before {name}: {process.stderr.read()}")
            time.sleep(.01)
        self.assertTrue(marker.exists(), f"launcher never reached {name}")

    def _assert_maintenance_defers(self):
        with patch("mavis.maintenance_runtime.inventory", side_effect=AssertionError("IRIS must not be probed")):
            allowed, reason = host_admission(self.home)
            maintenance = tick(self.home)
        self.assertFalse(allowed)
        self.assertIn("Mavis foreground generation", reason)
        self.assertEqual(maintenance["status"], "deferred")
        self.assertEqual(maintenance["reason"], reason)

    def test_no_profile_launcher_holds_lease_from_resolution_through_core_exit(self):
        process = self._start()
        for phase in ("resolve", "ensure", "receipt", "core"):
            self._phase(process, phase)
            self._assert_maintenance_defers()
            (self.root / f"{phase}.continue").write_text("go")
        stdout, stderr = process.communicate(timeout=5)
        self.assertEqual(process.returncode, 0, stderr or stdout)
        self.assertEqual((self.root / "core.started").read_text(),
                         str((self.root / "codex-home").resolve()))
        with patch("mavis.maintenance_runtime.inventory", return_value=[]), patch(
            "mavis.maintenance_runtime.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, stdout=""),
        ):
            self.assertTrue(host_admission(self.home)[0])

    def test_valid_parent_handoff_keeps_observer_lease_through_desktop_launch(self):
        with generation_lease(self.home, purpose="observer") as descriptor:
            env = {**self.env, "MAVIS_GENERATION_LEASE_FD": str(descriptor)}
            process = self._start(env=env, pass_fds=(descriptor,))
            for phase in ("resolve", "ensure", "receipt", "core"):
                self._phase(process, phase)
                self._assert_maintenance_defers()
                (self.root / f"{phase}.continue").write_text("go")
            _stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0, stderr)
            self._assert_maintenance_defers()

    def test_interrupted_preparation_stops_child_before_lease_release(self):
        process = self._start()
        self._phase(process, "resolve")
        (self.root / "resolve.continue").write_text("go")
        self._phase(process, "ensure")
        self._assert_maintenance_defers()
        process.terminate()
        _stdout, stderr = process.communicate(timeout=5)
        self.assertEqual(process.returncode, 2, stderr)
        self.assertIn("interrupted during model preparation", stderr)
        self.assertFalse((self.root / "receipt.started").exists())
        with patch("mavis.maintenance_runtime.inventory", return_value=[]), patch(
            "mavis.maintenance_runtime.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, stdout=""),
        ):
            self.assertTrue(host_admission(self.home)[0])

    def test_fresh_descriptor_cannot_claim_another_owners_handoff(self):
        with generation_lease(self.home, purpose="observer"):
            descriptor = os.open(self.home / "generation.lock", os.O_RDWR)
            try:
                env = {**self.env, "MAVIS_GENERATION_LEASE_FD": str(descriptor)}
                process = self._start(env=env, pass_fds=(descriptor,))
                _stdout, stderr = process.communicate(timeout=5)
            finally:
                os.close(descriptor)
            self.assertEqual(process.returncode, 2)
            self.assertIn("inherited Mavis host lease is not owned", stderr)
            self.assertFalse((self.root / "resolve.started").exists())

    def test_unrelated_descriptor_cannot_bypass_host_lease(self):
        with generation_lease(self.home, purpose="observer"):
            unrelated = self.root / "unrelated.lock"
            descriptor = os.open(unrelated, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                env = {**self.env, "MAVIS_GENERATION_LEASE_FD": str(descriptor)}
                process = self._start(env=env, pass_fds=(descriptor,))
                _stdout, stderr = process.communicate(timeout=5)
            finally:
                os.close(descriptor)
            self.assertEqual(process.returncode, 2)
            self.assertIn("not generation.lock", stderr)
            self.assertFalse((self.root / "resolve.started").exists())


if __name__ == "__main__":
    unittest.main()
