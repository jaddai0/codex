from pathlib import Path
import tempfile
import unittest

from mavis.transcripts import TranscriptArchive


class TranscriptArchiveTests(unittest.TestCase):
    def test_archives_full_segments_links_manifest_and_searches_exact_text(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = TranscriptArchive(Path(directory), "conversation-1")
            segment = archive.append_segment(
                [{"role": "user", "content": "retain canary fact blue heron"}]
            )
            handoff = archive.write_handoff(
                {
                    "goals": ["finish"],
                    "accepted_decisions": ["preserve IRIS"],
                    "completed_requirements": [],
                    "current_changes": [],
                    "recent_work": [],
                    "unresolved_failures": [],
                    "evidence_links": [str(segment)],
                }
            )
            self.assertTrue(handoff.is_file())
            matches = archive.search("blue heron")
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0]["line"], 1)

    def test_handoff_requires_every_recovery_field(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = TranscriptArchive(Path(directory), "conversation-1")
            archive.append_segment([{"role": "user", "content": "x"}])
            with self.assertRaisesRegex(ValueError, "accepted_decisions"):
                archive.write_handoff({"goals": []})


if __name__ == "__main__":
    unittest.main()
