from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from mavis.helpers import HelperSession


class HelperSessionTests(unittest.TestCase):
    def test_librarian_context_expires_after_sixty_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            session = HelperSession(Path(directory), "librarian", ttl_seconds=60)
            session.store_query_context("where", [{"path": "a", "line": 1}], now=100)
            self.assertIsNotNone(session.followup_context(now=159.9))
            self.assertIsNone(session.followup_context(now=160))
            self.assertFalse(session.cache_path.exists())

    def test_helper_roots_are_role_isolated_and_pressure_clears_early(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            librarian = HelperSession(root, "librarian")
            reader = HelperSession(root, "output-reader")
            self.assertNotEqual(librarian.root, reader.root)
            librarian.store_query_context("q", [], now=10)
            librarian.clear_for_compute_pressure()
            self.assertIsNone(librarian.followup_context(now=11))

    def test_context_file_disappears_after_deadline_without_followup_read(self):
        with tempfile.TemporaryDirectory() as directory:
            session = HelperSession(Path(directory), "librarian", ttl_seconds=0.25)
            session.store_query_context("question", [{"path": "source", "line": 1}])
            self.assertTrue(session.cache_path.is_file())
            deadline = time.monotonic() + 3
            while session.cache_path.exists() and time.monotonic() < deadline:
                time.sleep(0.03)
            self.assertFalse(session.cache_path.exists(), "expiry worker did not remove context")

    def test_expiry_worker_survives_the_one_shot_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            script = ("from pathlib import Path; from mavis.helpers import HelperSession; "
                      "HelperSession(Path(__import__('sys').argv[1]), 'librarian', 0.8)"
                      ".store_query_context('question', [])")
            subprocess.run([sys.executable, "-c", script, str(home)], check=True)
            cache = home / "helpers" / "librarian" / "query-context.json"
            self.assertTrue(cache.exists())
            deadline = time.monotonic() + 3
            while cache.exists() and time.monotonic() < deadline:
                time.sleep(0.03)
            self.assertFalse(cache.exists(), "one-shot process left context at rest")

    def test_old_expiry_worker_does_not_delete_replaced_context(self):
        with tempfile.TemporaryDirectory() as directory:
            session = HelperSession(Path(directory), "librarian", ttl_seconds=0.5)
            session.store_query_context("old", [])
            time.sleep(0.25)
            session.store_query_context("new", [])
            time.sleep(0.33)
            self.assertTrue(session.cache_path.exists())
            self.assertEqual(session.followup_context()["query"], "new")
            deadline = time.monotonic() + 2
            while session.cache_path.exists() and time.monotonic() < deadline:
                time.sleep(0.03)
            self.assertFalse(session.cache_path.exists())


if __name__ == "__main__":
    unittest.main()
