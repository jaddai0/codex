from pathlib import Path
import tempfile
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


if __name__ == "__main__":
    unittest.main()
