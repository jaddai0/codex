from datetime import datetime, timedelta, timezone
import fcntl
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from mavis.archive_hygiene import ArchiveRetention
from mavis.maintenance import MaintenanceQueue
from mavis.maintenance_runtime import RUNTIME_KINDS, host_admission, tick
from mavis.storage import read_json, sha256_file
from mavis.transcripts import TranscriptArchive


class MaintenanceRuntimeTests(unittest.TestCase):
    def _closed_two_segments(self, home: Path):
        archive = TranscriptArchive(home, "conversation-1")
        first = archive.append_segment([{"role": "user", "content": "first shard"}])
        second = archive.append_segment([{"role": "user", "content": "second shard"}])
        retention = ArchiveRetention(home)
        retention.register("project-1", ["conversation-1"], [])
        closed = datetime.now(timezone.utc) + timedelta(seconds=1)
        retention.close("project-1", now=closed)
        return first, second, retention, closed + timedelta(days=31)

    def test_one_shard_per_tick_restart_and_period_due(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            first, second, retention, now = self._closed_two_segments(home)
            idle = lambda: (True, "fixture Mavis and IRIS idle")
            first_tick = tick(home, now=now, admit=idle)
            self.assertEqual(first_tick["status"], "shard")
            self.assertFalse(first.exists())
            self.assertTrue(second.exists())
            self.assertEqual(sha256_file(Path(first_tick["receipt_path"])),
                             first_tick["receipt_sha256"])
            queue = MaintenanceQueue(home)  # A new process reads the saved checkpoint.
            job = queue.get(first_tick["job_id"])
            self.assertEqual(job["state"], "paused")
            self.assertEqual(job["checkpoint"]["project_id"], "project-1")
            second_tick = tick(home, now=now + timedelta(seconds=1), admit=idle)
            self.assertEqual(second_tick["status"], "shard")
            self.assertFalse(second.exists())
            third_tick = tick(home, now=now + timedelta(seconds=2), admit=idle)
            self.assertTrue(third_tick["shard"]["complete"])
            self.assertEqual(read_json(retention._path("project-1"))["last_sweep_at"],
                             (now + timedelta(seconds=2)).isoformat())
            fourth_tick = tick(home, now=now + timedelta(seconds=3), admit=idle)
            self.assertEqual(fourth_tick["status"], "complete-period")
            self.assertEqual(tick(home, now=now + timedelta(seconds=4), admit=idle)["status"],
                             "not-due")
            self.assertEqual(queue.get(first_tick["job_id"])["state"], "queued")

    def test_due_intervals_allowlist_and_expired_lease(self):
        origin = datetime(2026, 1, 31, 12, tzinfo=timezone.utc)
        due = {"immediate": origin, "daily": origin + timedelta(days=1),
               "weekly": origin + timedelta(days=7),
               "monthly": datetime(2026, 2, 28, 12, tzinfo=timezone.utc)}
        for interval, expected in due.items():
            with self.subTest(interval=interval), tempfile.TemporaryDirectory() as directory:
                queue = MaintenanceQueue(Path(directory))
                model = queue.enqueue("immediate", "model-eval", {}, now=origin)
                job = queue.enqueue(interval, "archive-sweep", {}, now=origin)
                with self.assertRaisesRegex(ValueError, "approved host kind"):
                    queue.claim_due(kinds={"model-eval"}, now=origin)
                self.assertEqual(job["next_due_at"], expected.isoformat())
                self.assertIsNone(queue.claim_due(kinds=RUNTIME_KINDS,
                                                 now=expected - timedelta(seconds=1)))
                claimed = queue.claim_due(kinds=RUNTIME_KINDS, now=expected)
                self.assertEqual(claimed["job_id"], job["job_id"])
                self.assertEqual(queue.get(model["job_id"])["state"], "queued")
                self.assertIsNone(queue.claim_due(kinds=RUNTIME_KINDS,
                                                 now=expected + timedelta(seconds=60)))
                recovered = queue.claim_due(kinds=RUNTIME_KINDS,
                                            now=expected + timedelta(seconds=301))
                self.assertEqual(recovered["job_id"], job["job_id"])
                finished = queue.finish_runtime(job["job_id"], success=True,
                                                detail="period complete",
                                                now=expected + timedelta(seconds=301))
                if interval == "immediate":
                    self.assertEqual(finished["state"], "complete")
                else:
                    self.assertEqual(finished["state"], "queued")
                    self.assertGreater(datetime.fromisoformat(finished["next_due_at"]),
                                       expected + timedelta(seconds=301))
                    if interval == "monthly":
                        self.assertEqual(finished["next_due_at"],
                                         datetime(2026, 3, 31, 12, tzinfo=timezone.utc).isoformat())

    def test_foreground_defer_and_mid_shard_yield(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            first, second, _, now = self._closed_two_segments(home)
            deferred = tick(home, now=now, admit=lambda: (False, "Mavis foreground"))
            self.assertEqual(deferred["status"], "deferred")
            self.assertTrue(first.exists())
            self.assertFalse((home / "maintenance/queue.sqlite3").exists())
            checks = iter([(True, "idle"), (True, "idle"), (False, "foreground")])
            yielded = tick(home, now=now + timedelta(seconds=1), admit=lambda: next(checks))
            self.assertEqual(yielded["status"], "deferred")
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())
            self.assertEqual(MaintenanceQueue(home).get(yielded["job_id"])["state"], "paused")

    def test_runner_lock_defers_a_second_tick_with_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            lock_path = home / "maintenance/tick.lock"
            lock_path.parent.mkdir(parents=True)
            with lock_path.open("a+") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = tick(home, admit=lambda: (True, "fixture idle"))
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            self.assertEqual(result["status"], "deferred")
            self.assertIn("runner lock", result["reason"])
            self.assertEqual(sha256_file(Path(result["receipt_path"])),
                             result["receipt_sha256"])

    def test_receipt_binds_installed_manifest_when_present(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / "home"
            share = Path(directory) / "share"
            share.mkdir()
            manifest = share / "install-manifest.json"
            manifest.write_text('{"schema_version":"mavis.installed-core/v1"}\n')
            with patch.dict(os.environ, {"LOCAL_CODEX_SHARE_DIR": str(share)}):
                result = tick(home, admit=lambda: (False, "foreground"))
            self.assertEqual(result["installed_manifest_sha256"], sha256_file(manifest))
            self.assertEqual(read_json(Path(result["receipt_path"]))["installed_manifest_sha256"],
                             sha256_file(manifest))

    def test_failed_shard_is_retained_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            first, _, _, now = self._closed_two_segments(home)
            first.write_bytes(b"corrupted raw evidence")
            idle = lambda: (True, "fixture idle")
            failed = tick(home, now=now, admit=idle)
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(MaintenanceQueue(home).get(failed["job_id"])["state"], "failed")
            repeat = tick(home, now=now + timedelta(hours=1), admit=idle)
            self.assertEqual(repeat["status"], "failed-retained")
            self.assertEqual(first.read_bytes(), b"corrupted raw evidence")

    def test_host_admission_uses_lease_and_defers_loaded_iris_or_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            with (home / "generation.lock").open("a+") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertFalse(host_admission(home)[0])
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            with patch("mavis.maintenance_runtime.inventory", return_value=[
                {"id": "iris-main", "loaded": True, "model_type": "text-generation"}
            ]):
                self.assertIn("loaded generation model", host_admission(home)[1])
            with patch("mavis.maintenance_runtime.inventory", return_value=[]), patch(
                "mavis.maintenance_runtime.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, "node diagnostics/lyria/bench.mjs\n", ""),
            ):
                self.assertIn("benchmark", host_admission(home)[1])
            with patch("mavis.maintenance_runtime.inventory", return_value=[]), patch(
                "mavis.maintenance_runtime.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0,
                    "node diagnostics/lyria/replay.mjs --variant=candidate\n", ""),
            ):
                self.assertIn("benchmark", host_admission(home)[1])
            with patch("mavis.maintenance_runtime.inventory", side_effect=OSError("down")):
                self.assertIn("unavailable", host_admission(home)[1])


if __name__ == "__main__":
    unittest.main()
