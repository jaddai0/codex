from pathlib import Path
import time
import tempfile
import unittest

from mavis.maintenance import MaintenanceQueue


class MaintenanceQueueTests(unittest.TestCase):
    def test_foreground_yields_at_checkpoint_and_job_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = MaintenanceQueue(Path(directory))
            job = queue.enqueue("daily", "focused-regression", {"case": "failure-1"})
            self.assertIsNone(queue.claim_next(foreground_active=True))
            claimed = queue.claim_next(foreground_active=False)
            self.assertEqual(claimed["job_id"], job["job_id"])
            paused = queue.checkpoint(job["job_id"], {"shard": 2}, foreground_active=True)
            self.assertEqual(paused["state"], "paused")
            resumed = queue.claim_next(foreground_active=False)
            self.assertEqual(resumed["checkpoint"], {"shard": 2})
            complete = queue.finish(job["job_id"], success=True, detail="held-out check passed")
            self.assertEqual(complete["state"], "complete")

    def test_intervals_and_failures_are_durable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = MaintenanceQueue(root)
            ids = [queue.enqueue(interval, "review", {})["job_id"] for interval in ("immediate", "daily", "weekly", "monthly")]
            claimed = queue.claim_next(foreground_active=False)
            queue.finish(claimed["job_id"], success=False, detail="fixture failure")
            reopened = MaintenanceQueue(root)
            self.assertEqual(reopened.get(ids[0])["state"], "failed")
            self.assertEqual(len(reopened.list()), 4)

    def test_cancel_queued_job_preserves_payload_and_records_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = MaintenanceQueue(Path(directory))
            job = queue.enqueue("daily", "review", {"key": "value"})
            time.sleep(0.01)
            cancelled = queue.cancel(job["job_id"], reason="obsolete")
            self.assertEqual(cancelled["state"], "cancelled")
            self.assertEqual(cancelled["detail"], "obsolete")
            self.assertEqual(cancelled["payload"], {"key": "value"})
            self.assertEqual(cancelled["checkpoint"], {})
            self.assertNotEqual(cancelled["updated_at"], job["created_at"])
            self.assertEqual(queue.get(job["job_id"])["state"], "cancelled")

    def test_cancel_paused_job_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = MaintenanceQueue(Path(directory))
            job = queue.enqueue("weekly", "compaction", {"progress": 3})
            self.assertIsNotNone(queue.claim_next(foreground_active=False))
            paused = queue.checkpoint(
                job["job_id"], {"progress": 5}, foreground_active=True
            )
            self.assertEqual(paused["state"], "paused")
            time.sleep(0.01)
            cancelled = queue.cancel(job["job_id"], reason="maintenance window closed")
            self.assertEqual(cancelled["state"], "cancelled")
            self.assertEqual(cancelled["checkpoint"], {"progress": 5})
            self.assertEqual(cancelled["payload"], {"progress": 3})
            self.assertEqual(cancelled["detail"], "maintenance window closed")

    def test_cancel_rejects_running_job(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = MaintenanceQueue(Path(directory))
            job = queue.enqueue("monthly", "long-run", {})
            self.assertIsNotNone(queue.claim_next(foreground_active=False))
            with self.assertRaisesRegex(ValueError, "cannot cancel"):
                queue.cancel(job["job_id"], reason="too early")

    def test_cancel_rejects_complete_job(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = MaintenanceQueue(Path(directory))
            job = queue.enqueue("daily", "check", {})
            self.assertIsNotNone(queue.claim_next(foreground_active=False))
            queue.finish(job["job_id"], success=True, detail="done")
            with self.assertRaisesRegex(ValueError, "cannot cancel"):
                queue.cancel(job["job_id"], reason="too late")

    def test_cancel_rejects_failed_job(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = MaintenanceQueue(Path(directory))
            job = queue.enqueue("daily", "check", {})
            self.assertIsNotNone(queue.claim_next(foreground_active=False))
            queue.finish(job["job_id"], success=False, detail="boom")
            with self.assertRaisesRegex(ValueError, "cannot cancel"):
                queue.cancel(job["job_id"], reason="duplicate")

    def test_cancel_rejects_already_cancelled_job(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = MaintenanceQueue(Path(directory))
            job = queue.enqueue("weekly", "review", {})
            queue.cancel(job["job_id"], reason="obsolete")
            with self.assertRaisesRegex(ValueError, "cannot cancel"):
                queue.cancel(job["job_id"], reason="again")

    def test_cancel_unknown_job_raises_key_error(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = MaintenanceQueue(Path(directory))
            queue.enqueue("daily", "review", {})
            with self.assertRaises(KeyError):
                queue.cancel("missing", reason="nobody")

    def test_claim_does_not_run_cancelled_job(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = MaintenanceQueue(Path(directory))
            job = queue.enqueue("daily", "review", {})
            cancelled = queue.cancel(job["job_id"], reason="obsolete")
            self.assertEqual(cancelled["state"], "cancelled")
            self.assertIsNone(queue.claim_next(foreground_active=False))

    def test_cancel_and_claim_serialize_in_both_orders(self):
        from concurrent.futures import ThreadPoolExecutor
        import threading
        from unittest.mock import patch

        for first in ("cancel", "claim"):
            with self.subTest(first=first), tempfile.TemporaryDirectory() as directory:
                queue = MaintenanceQueue(Path(directory))
                job = queue.enqueue("daily", "review", {})
                acquired = threading.Event()
                contender_started = threading.Event()
                release = threading.Event()
                role = threading.local()
                connect = queue._connect

                class ConnectionProxy:
                    def __init__(self, connection):
                        self.connection = connection

                    def __getattr__(self, name):
                        return getattr(self.connection, name)

                    def execute(self, sql, *args):
                        if sql == "BEGIN IMMEDIATE" and role.name != first:
                            contender_started.set()
                        result = self.connection.execute(sql, *args)
                        if sql == "BEGIN IMMEDIATE" and role.name == first:
                            acquired.set()
                            if not release.wait(5):
                                raise TimeoutError("contending operation did not start")
                        return result

                def run_cancel():
                    role.name = "cancel"
                    try:
                        return queue.cancel(job["job_id"], reason="obsolete")
                    except ValueError:
                        return None

                def run_claim():
                    role.name = "claim"
                    return queue.claim_next(foreground_active=False)

                operations = {"cancel": run_cancel, "claim": run_claim}
                with patch.object(queue, "_connect", side_effect=lambda: ConnectionProxy(connect())):
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        winner = pool.submit(operations[first])
                        self.assertTrue(acquired.wait(5))
                        loser = pool.submit(operations["claim" if first == "cancel" else "cancel"])
                        self.assertTrue(contender_started.wait(5))
                        release.set()
                        first_result = winner.result(timeout=5)
                        second_result = loser.result(timeout=5)

                self.assertEqual(queue.get(job["job_id"])["state"],
                                 "cancelled" if first == "cancel" else "running")
                self.assertEqual(first_result["job_id"], job["job_id"])
                self.assertIsNone(second_result)

    def test_claim_skips_cancelled_job_and_returns_next_queued(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = MaintenanceQueue(Path(directory))
            first = queue.enqueue("daily", "review", {})
            second = queue.enqueue("weekly", "review", {})
            queue.cancel(first["job_id"], reason="obsolete")
            claimed = queue.claim_next(foreground_active=False)
            self.assertIsNotNone(claimed)
            self.assertEqual(claimed["job_id"], second["job_id"])
            self.assertEqual(queue.get(first["job_id"])["state"], "cancelled")


if __name__ == "__main__":
    unittest.main()
