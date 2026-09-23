from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from mavis.archive_hygiene import ArchiveRetention, register_reproducible_cache, storage_pressure
from mavis.evidence import run_command
from mavis.helper_interfaces import LibrarianEvidence
from mavis.helpers import HelperSession
from mavis.maintenance import MaintenanceQueue
from mavis.transcripts import TranscriptArchive


class ArchiveRetentionTests(unittest.TestCase):
    def _closed(self, home):
        archive = TranscriptArchive(home, "conversation-1")
        original = archive.append_segment([{"role": "user", "content": "decision blue heron" * 100}])
        retention = ArchiveRetention(home)
        retention.register("project-1", ["conversation-1"], [])
        closed = datetime.now(timezone.utc) + timedelta(seconds=1)
        retention.close("project-1", now=closed)
        return archive, original, retention, closed

    def test_closed_archive_reconstructs_and_remains_searchable_and_citable(self):
        with tempfile.TemporaryDirectory() as directory:
            archive, original, retention, closed = self._closed(Path(directory))
            before = original.read_bytes()
            with self.assertRaisesRegex(ValueError, "30 days"):
                retention.compact("project-1", now=closed + timedelta(days=29))
            outcome = retention.compact("project-1", now=closed + timedelta(days=31))
            self.assertEqual(outcome["segments"][0]["status"], "compressed")
            self.assertFalse(original.exists())
            manifest = json.loads(archive.manifest_path.read_text())
            segment = manifest["segments"][0]
            self.assertEqual(segment["sha256"], hashlib.sha256(before).hexdigest())
            self.assertEqual(b"".join(line.encode() for line in archive.segment_lines(segment)), before)
            evidence = LibrarianEvidence(archive).search("blue heron")
            self.assertEqual(len(evidence), 1)
            LibrarianEvidence(archive).validate_answer({
                "answer": "The decision mentions blue heron.", "uncertainty": "Only one segment was checked.",
                "citations": [{key: evidence[0][key] for key in ("path", "line", "sha256")}],
            }, evidence)
            Path(segment["path"]).write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "compressed transcript segment"):
                archive.search("blue heron")

    def test_restore_rebuilds_exact_raw_bytes_and_reopens_project(self):
        with tempfile.TemporaryDirectory() as directory:
            archive, original, retention, closed = self._closed(Path(directory))
            before = original.read_bytes()
            retention.compact("project-1", now=closed + timedelta(days=31))
            self.assertFalse(original.exists())
            restored = retention.restore("project-1")
            self.assertEqual(len(restored["restored_segments"]), 1)
            self.assertEqual(original.read_bytes(), before)
            self.assertEqual(len(archive.search("blue heron")), 1)
            self.assertEqual(retention.reopen("project-1")["state"], "open")

    def test_valid_interrupted_compression_is_recovered(self):
        with tempfile.TemporaryDirectory() as directory:
            archive, original, retention, closed = self._closed(Path(directory))
            compressed = original.with_name(original.name + ".gz")
            with gzip.open(compressed, "wb") as destination:
                destination.write(original.read_bytes())
            result = retention.compact("project-1", now=closed + timedelta(days=31))
            self.assertEqual(result["segments"][0]["status"], "compressed")
            self.assertFalse(original.exists())
            self.assertEqual(len(archive.search("blue heron")), 1)

    def test_handoff_reference_keeps_raw_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            archive = TranscriptArchive(home, "conversation-1")
            first = archive.append_segment([{"role": "user", "content": "keep cited"}])
            other = archive.append_segment([{"role": "user", "content": "compress uncited"}])
            archive.write_handoff({
                "goals": [], "accepted_decisions": [], "completed_requirements": [],
                "current_changes": [], "recent_work": [], "unresolved_failures": [],
                "evidence_links": [str(first)],
            })
            retention = ArchiveRetention(home)
            retention.register("project-1", ["conversation-1"], [])
            closed = datetime.now(timezone.utc) + timedelta(seconds=1)
            retention.close("project-1", now=closed)
            result = retention.compact("project-1", now=closed + timedelta(days=31))
            self.assertEqual([item["status"] for item in result["segments"]], ["referenced", "compressed"])
            self.assertTrue(first.is_file())
            self.assertFalse(other.exists())
            self.assertEqual(len(archive.search("cited")), 2)

    def test_open_or_active_objective_cannot_compact(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            TranscriptArchive(home, "conversation-1").append_segment([{"content": "evidence"}])
            retention = ArchiveRetention(home)
            retention.register("project-1", ["conversation-1"], ["objective-1"])
            objectives = home / "objectives"
            objectives.mkdir()
            (objectives / "objective-1.json").write_text(json.dumps({
                "objective_id": "objective-1", "state": "running",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }))
            with self.assertRaisesRegex(ValueError, "active"):
                retention.close("project-1")
            with self.assertRaisesRegex(ValueError, "closed"):
                retention.compact("project-1", now=datetime.now(timezone.utc) + timedelta(days=31))

    def test_new_segment_after_close_invalidates_frozen_project(self):
        with tempfile.TemporaryDirectory() as directory:
            archive, original, retention, closed = self._closed(Path(directory))
            archive.append_segment([{"content": "later evidence"}])
            with self.assertRaisesRegex(ValueError, "segment set changed"):
                retention.compact("project-1", now=closed + timedelta(days=31))
            self.assertTrue(original.exists())

    def test_external_objective_reference_protects_raw_source(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            archive, original, retention, closed = self._closed(home)
            objectives = home / "objectives"
            objectives.mkdir()
            (objectives / "other.json").write_text(json.dumps({
                "objective_id": "other", "state": "running", "evidence_links": [str(original)],
            }))
            outcome = retention.compact("project-1", now=closed + timedelta(days=31))
            self.assertEqual(outcome["segments"][0]["status"], "referenced")
            self.assertTrue(original.is_file())

    def test_live_librarian_followup_citation_protects_raw_source(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            archive, original, retention, closed = self._closed(home)
            HelperSession(home, "librarian").store_query_context(
                "where was the decision", [{"path": str(original), "line": 1}], now=time.time())
            result = retention.compact("project-1", now=closed + timedelta(days=31))
            self.assertEqual(result["segments"][0]["status"], "referenced")
            self.assertTrue(original.is_file())

    def test_overlapping_project_registration_and_legacy_overlap_block_compression(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            archive, original, retention, closed = self._closed(home)
            with self.assertRaisesRegex(ValueError, "another archive project"):
                retention.register("project-2", ["conversation-1"], [])
            (retention.root / "project-2.json").write_text(json.dumps({
                "schema_version": "mavis.archive-project/v1", "project_id": "project-2",
                "conversation_ids": ["conversation-1"], "state": "open",
            }))
            with self.assertRaisesRegex(ValueError, "another archive project"):
                retention.compact("project-1", now=closed + timedelta(days=31))
            self.assertTrue(original.is_file())

    def test_concurrent_registration_cannot_claim_same_conversation_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            retention = ArchiveRetention(Path(directory))
            def claim(project_id):
                try:
                    retention.register(project_id, ["conversation-1"], [])
                    return "registered"
                except ValueError:
                    return "overlap-refused"
            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(pool.map(claim, ("project-a", "project-b")))
            self.assertEqual(sorted(outcomes), ["overlap-refused", "registered"])

    def test_idle_sweep_compresses_due_project_and_foreground_skips_trigger(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            archive, original, retention, closed = self._closed(home)
            queue = MaintenanceQueue(home)
            with patch.object(ArchiveRetention, "sweep_due", wraps=retention.sweep_due) as sweep:
                self.assertIsNone(queue.claim_next(foreground_active=True))
                sweep.assert_not_called()
            report = retention.sweep_due(now=closed + timedelta(days=31))
            self.assertEqual(report["projects"][0]["status"], "checked")
            self.assertFalse(original.exists())
            with patch.object(ArchiveRetention, "sweep_due", return_value=report) as sweep:
                self.assertIsNone(queue.claim_next(foreground_active=False))
                sweep.assert_called_once()
            self.assertEqual(retention.sweep_due(now=closed + timedelta(days=31))["projects"], [])

    def test_reproducible_cache_is_only_pruned_on_pressure(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            cache = home / "cache" / "reproducible"
            cache.mkdir(parents=True)
            item = cache / "fixture"
            item.write_bytes(b"cache")
            evidence = home / "transcripts" / "evidence.jsonl"
            evidence.parent.mkdir()
            evidence.write_bytes(b"unique evidence")
            register_reproducible_cache(home, item, evidence)
            unregistered = cache / "unique"
            unregistered.write_bytes(b"do not prune")
            with patch("mavis.archive_hygiene.shutil.disk_usage", return_value=shutil_usage(100 * 1024**3, 50 * 1024**3)):
                report = storage_pressure(home, prune=True)
            self.assertFalse(report["pressure"])
            self.assertTrue(item.exists())
            with patch("mavis.archive_hygiene.shutil.disk_usage", return_value=shutil_usage(100 * 1024**3, 1 * 1024**3)):
                report = storage_pressure(home, prune=True)
            self.assertTrue(report["pressure"])
            self.assertEqual(report["freed_cache_bytes"], 5)
            self.assertFalse(item.exists())
            self.assertTrue(unregistered.exists())
            self.assertTrue(evidence.is_file())

    def test_evidence_write_surfaces_pressure_and_prunes_only_reproducible_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            cache = home / "cache" / "reproducible"
            cache.mkdir(parents=True)
            (cache / "throwaway").write_text("cache")
            source = home / "retained-source.txt"
            source.write_text("source")
            register_reproducible_cache(home, cache / "throwaway", source)
            with patch("mavis.archive_hygiene.shutil.disk_usage", return_value=shutil_usage(100 * 1024**3, 1 * 1024**3)):
                with self.assertWarnsRegex(RuntimeWarning, "storage pressure"):
                    receipt = run_command(home, "objective-1", ["python3", "-c", "print('OK')"], home)
            payload = json.loads(receipt.read_text())
            self.assertTrue(payload["storage_pressure_before"]["pressure"])
            self.assertFalse((cache / "throwaway").exists())
            self.assertTrue((home / "storage-pressure.json").is_file())
            self.assertTrue((Path(payload["raw_output"]["path"]) / "stdout.log").is_file())

    def test_malformed_optional_cache_registry_never_blocks_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            cache = home / "cache" / "reproducible"
            cache.mkdir(parents=True)
            (cache / "manifest.json").write_text('{"entries": [{"path": "missing"}]}')
            with self.assertWarnsRegex(RuntimeWarning, "registry has invalid entry"):
                segment = TranscriptArchive(home, "conversation-1").append_segment([{"content": "evidence"}])
            self.assertTrue(segment.is_file())
            (cache / "manifest.json").write_text("{broken json")
            with self.assertWarnsRegex(RuntimeWarning, "registered cache skipped"):
                receipt = run_command(home, "objective-1", ["python3", "-c", "print('OK')"], home)
            self.assertTrue(receipt.is_file())


def shutil_usage(total, free):
    return type("Usage", (), {"total": total, "free": free, "used": total - free})()


if __name__ == "__main__":
    unittest.main()
