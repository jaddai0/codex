from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
