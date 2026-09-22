from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

from mavis.transcripts import TranscriptArchive


class TranscriptArchiveTests(unittest.TestCase):
    def test_parallel_rollout_imports_publish_one_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rollout = root / "rollout.jsonl"
            rollout.write_text(
                json.dumps(
                    {"type": "session_meta", "payload": {"id": "conversation-1"}}
                )
                + "\n"
            )
            script = "from pathlib import Path; from mavis.transcripts import TranscriptArchive; import sys; TranscriptArchive(Path(sys.argv[1]), 'conversation-1').import_rollout(Path(sys.argv[2]))"
            processes = [
                subprocess.Popen(
                    [sys.executable, "-c", script, str(root), str(rollout)]
                )
                for _ in range(2)
            ]
            for process in processes:
                self.assertEqual(process.wait(timeout=10), 0)
            archive = TranscriptArchive(root, "conversation-1")
            self.assertEqual(
                len(json.loads(archive.manifest_path.read_text())["segments"]), 1
            )

    def test_rollout_import_is_incremental_and_detects_changed_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rollout = root / "rollout.jsonl"
            first = {"type": "session_meta", "payload": {"id": "conversation-1"}}
            rollout.write_text(json.dumps(first) + "\n", encoding="utf-8")
            archive = TranscriptArchive(root, "conversation-1")
            self.assertIsNotNone(archive.import_rollout(rollout))
            self.assertIsNone(archive.import_rollout(rollout))
            rollout.write_text(
                rollout.read_text()
                + json.dumps({"type": "event_msg", "payload": {"text": "second"}})
                + "\n"
            )
            self.assertIsNotNone(archive.import_rollout(rollout))
            self.assertEqual(len(archive.search("second")), 1)
            self.assertEqual(
                len(json.loads(archive.manifest_path.read_text())["segments"]),
                2,
            )
            rollout.write_text(
                rollout.read_text().replace("session_meta", "other_event")
            )
            with self.assertRaisesRegex(ValueError, "changed before"):
                archive.import_rollout(rollout)

    def test_rollout_import_rejects_partial_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rollout = root / "rollout.jsonl"
            rollout.write_text('{"type":', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                TranscriptArchive(root, "conversation-1").import_rollout(rollout)

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
            for path, expected in (
                (archive.root, 0o700),
                (segment.parent, 0o700),
                (handoff.parent, 0o700),
                (segment, 0o600),
                (handoff, 0o600),
                (archive.manifest_path, 0o600),
            ):
                self.assertEqual(path.stat().st_mode & 0o777, expected)
            matches = archive.search("blue heron")
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0]["line"], 1)

    def test_handoff_requires_every_recovery_field(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = TranscriptArchive(Path(directory), "conversation-1")
            archive.append_segment([{"role": "user", "content": "x"}])
            with self.assertRaisesRegex(ValueError, "accepted_decisions"):
                archive.write_handoff({"goals": []})

    def test_search_rejects_changed_archived_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = TranscriptArchive(Path(directory), "conversation-1")
            segment = archive.append_segment(
                [{"role": "user", "content": "original decision"}]
            )
            segment.write_text(
                '{"role":"user","content":"altered decision"}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                archive.search("decision")


if __name__ == "__main__":
    unittest.main()
