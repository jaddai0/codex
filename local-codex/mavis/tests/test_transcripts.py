import hashlib
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from mavis.transcripts import TranscriptArchive


class TranscriptArchiveTests(unittest.TestCase):
    def test_search_rejects_symlinked_transcripts_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = TranscriptArchive(root, "conversation-1")
            archive.append_segment([{"role": "user", "content": "private decision"}])
            outside = root / "outside-transcripts"
            archive.root.parent.rename(outside)
            archive.root.parent.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "archive directory escapes"):
                archive.search("private decision")

    def test_search_rejects_symlinked_archive_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = TranscriptArchive(root, "conversation-1")
            archive.append_segment([{"role": "user", "content": "private decision"}])
            outside = root / "outside-archive"
            archive.root.rename(outside)
            archive.root.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "archive directory escapes"):
                archive.search("private decision")

    def test_search_rejects_symlinked_segments_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = TranscriptArchive(root, "conversation-1")
            archive.append_segment([{"role": "user", "content": "safe decision"}])
            external_dir = root / "outside"
            external_dir.mkdir()
            external = external_dir / "leak.jsonl"
            external.write_text("private decision\n", encoding="utf-8")
            for segment in (archive.root / "segments").iterdir():
                segment.unlink()
            (archive.root / "segments").rmdir()
            (archive.root / "segments").symlink_to(external_dir, target_is_directory=True)
            manifest = json.loads(archive.manifest_path.read_text())
            manifest["segments"][0]["path"] = str(archive.root / "segments" / "leak.jsonl")
            manifest["segments"][0]["sha256"] = hashlib.sha256(external.read_bytes()).hexdigest()
            archive.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "directory escapes the archive"):
                archive.search("private decision")

    def test_search_rejects_forged_external_segment_even_when_hash_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = TranscriptArchive(root, "conversation-1")
            archive.append_segment([{"role": "user", "content": "visible decision"}])
            external = root / "leaked.jsonl"
            external.write_text("a decision outside\n", encoding="utf-8")
            digest = hashlib.sha256(external.read_bytes()).hexdigest()
            manifest = json.loads(archive.manifest_path.read_text())
            manifest["segments"][0]["path"] = str(external)
            manifest["segments"][0]["sha256"] = digest
            archive.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes the archive"):
                archive.search("decision")

    def test_search_rejects_symlinked_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = TranscriptArchive(root, "conversation-1")
            segment = archive.append_segment(
                [{"role": "user", "content": "decision visible"}]
            )
            target = root / "real.jsonl"
            target.write_text("hidden decision\n", encoding="utf-8")
            segment.unlink()
            segment.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "escapes the archive"):
                archive.search("decision")

    def test_search_stops_scanning_after_page_but_verifies_later_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = TranscriptArchive(Path(directory), "conversation-1")
            first = archive.append_segment(
                [{"role": "user", "content": "decision first"}]
            )
            later = archive.append_segment(
                [{"role": "user", "content": "decision later"}]
            )
            original = Path.read_text

            def no_later_scan(path, *args, **kwargs):
                if path.resolve() == later.resolve():
                    raise AssertionError("later segment content was scanned after page filled")
                return original(path, *args, **kwargs)

            with patch.object(Path, "read_text", no_later_scan):
                page = archive.search("decision", limit=1)
            self.assertEqual(Path(page[0]["path"]).resolve(), first.resolve())
            self.assertIn("decision later", archive.search("decision", offset=1)[0]["text"])
            later.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                archive.search("decision", limit=1)

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

    def test_search_pagination_page1_and_page2(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = TranscriptArchive(Path(directory), "conversation-1")
            archive.append_segment(
                [
                    {"role": "user", "content": "line one decision alpha"},
                    {"role": "user", "content": "line two decision beta"},
                    {"role": "user", "content": "line three decision gamma"},
                ]
            )
            all_results = archive.search("decision")
            self.assertEqual(len(all_results), 3)
            page1 = archive.search("decision", limit=2, offset=0)
            self.assertEqual(len(page1), 2)
            self.assertIn("decision", page1[0]["text"])
            self.assertIn("decision", page1[1]["text"])
            page2 = archive.search("decision", limit=2, offset=2)
            self.assertEqual(len(page2), 1)
            self.assertIn("decision", page2[0]["text"])

    def test_search_offset_beyond_results_returns_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = TranscriptArchive(Path(directory), "conversation-1")
            archive.append_segment(
                [{"role": "user", "content": "single decision point"}]
            )
            results = archive.search("decision", limit=10, offset=100)
            self.assertEqual(results, [])

    def test_search_invalid_negative_offset_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = TranscriptArchive(Path(directory), "conversation-1")
            archive.append_segment(
                [{"role": "user", "content": "decision one"}]
            )
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                archive.search("decision", offset=-1)

    def test_search_invalid_limit_out_of_range_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = TranscriptArchive(Path(directory), "conversation-1")
            archive.append_segment(
                [{"role": "user", "content": "decision one"}]
            )
            with self.assertRaisesRegex(ValueError, "limit must be 1..200"):
                archive.search("decision", limit=0)
            with self.assertRaisesRegex(ValueError, "limit must be 1..200"):
                archive.search("decision", limit=201)


if __name__ == "__main__":
    unittest.main()
