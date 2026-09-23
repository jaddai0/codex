import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mavis.evidence import run_command
from mavis.helper_interfaces import LibrarianEvidence, OutputReader
from mavis.storage import sha256_file
from mavis.transcripts import TranscriptArchive


class LibrarianEvidenceTests(unittest.TestCase):
    def test_validate_answer_rejects_symlinked_transcripts_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = TranscriptArchive(root, "conversation-1")
            archive.append_segment([{"role": "user", "content": "private decision"}])
            librarian = LibrarianEvidence(archive)
            evidence = librarian.search("private decision")
            citation = {key: evidence[0][key] for key in ("path", "line", "sha256")}
            outside = root / "outside-transcripts"
            archive.root.parent.rename(outside)
            archive.root.parent.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "outside"):
                librarian.validate_answer(
                    {
                        "answer": "private decision",
                        "uncertainty": "single line only",
                        "citations": [citation],
                    },
                    evidence,
                )

    def test_validate_answer_rejects_symlinked_archive_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = TranscriptArchive(root, "conversation-1")
            archive.append_segment([{"role": "user", "content": "private decision"}])
            librarian = LibrarianEvidence(archive)
            evidence = librarian.search("private decision")
            citation = {key: evidence[0][key] for key in ("path", "line", "sha256")}
            outside = root / "outside-archive"
            archive.root.rename(outside)
            archive.root.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "outside"):
                librarian.validate_answer(
                    {
                        "answer": "private decision",
                        "uncertainty": "single line only",
                        "citations": [citation],
                    },
                    evidence,
                )

    def test_validate_answer_rejects_symlinked_segments_directory(self):
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
            path = archive.root / "segments" / "leak.jsonl"
            digest = sha256_file(external)
            manifest = json.loads(archive.manifest_path.read_text())
            manifest["segments"][0]["path"] = str(path)
            manifest["segments"][0]["sha256"] = digest
            archive.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            evidence = [{"path": str(path), "line": 1, "sha256": digest, "text": "private decision"}]
            answer = {
                "answer": "private decision",
                "uncertainty": "single line only",
                "citations": [{"path": str(path), "line": 1, "sha256": digest}],
            }
            with self.assertRaisesRegex(ValueError, "outside"):
                LibrarianEvidence(archive).validate_answer(answer, evidence)

    def test_cited_answer_requires_archive_source_and_uncertainty(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = TranscriptArchive(Path(directory), "conversation-1")
            segment = archive.append_segment(
                [{"role": "user", "content": "Decision: keep the old release."}]
            )
            librarian = LibrarianEvidence(archive)
            evidence = librarian.search("old release")
            citation = {key: evidence[0][key] for key in ("path", "line", "sha256")}
            answer = {
                "answer": "The old release was kept.",
                "uncertainty": "This line does not explain why.",
                "citations": [citation],
            }
            self.assertEqual(librarian.validate_answer(answer, evidence), answer)
            with self.assertRaisesRegex(ValueError, "integer"):
                librarian.validate_answer(
                    {**answer, "citations": [{**citation, "line": True}]}, evidence
                )
            with self.assertRaisesRegex(ValueError, "uncertainty"):
                librarian.validate_answer({**answer, "uncertainty": ""}, evidence)
            with self.assertRaisesRegex(ValueError, "outside"):
                librarian.validate_answer(
                    {**answer, "citations": [{**citation, "line": 2}]}, evidence
                )
            unrelated = Path(directory) / "unrelated.txt"
            unrelated.write_text("Decision: replace the release.\n", encoding="utf-8")
            forged = {
                "path": str(unrelated),
                "line": 1,
                "sha256": sha256_file(unrelated),
            }
            with self.assertRaisesRegex(ValueError, "outside"):
                librarian.validate_answer(
                    {**answer, "citations": [forged]}, evidence + [forged]
                )
            segment.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside"):
                librarian.validate_answer(answer, evidence)

    def test_validate_answer_rejects_forged_external_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = TranscriptArchive(root, "conversation-1")
            archive.append_segment([{"role": "user", "content": "keep the old release"}])
            # The crafted manifest names an external file that was never part of
            # the archive, while carrying that file's own correct hash. The path
            # guard rejects it before any hash comparison happens.
            external = root / "leaked.jsonl"
            external.write_text("keep the old release\n", encoding="utf-8")
            digest = hashlib.sha256(external.read_bytes()).hexdigest()
            manifest = json.loads(archive.manifest_path.read_text())
            manifest["segments"][0]["path"] = str(external)
            manifest["segments"][0]["sha256"] = digest
            archive.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            evidence = [
                {
                    "path": str(external),
                    "line": 1,
                    "sha256": digest,
                    "text": "keep the old release",
                }
            ]
            citation = {"path": str(external), "line": 1, "sha256": digest}
            librarian = LibrarianEvidence(archive)
            with self.assertRaisesRegex(ValueError, "outside"):
                librarian.validate_answer(
                    {
                        "answer": "The old release was kept.",
                        "uncertainty": "This line does not explain why.",
                        "citations": [citation],
                    },
                    evidence,
                )

    def test_validate_answer_rejects_symlinked_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = TranscriptArchive(root, "conversation-1")
            segment = archive.append_segment(
                [{"role": "user", "content": "Decision: keep the old release."}]
            )
            line_text = segment.read_text(encoding="utf-8")
            # Search first to capture a genuine, in-segments evidence entry.
            librarian = LibrarianEvidence(archive)
            evidence = librarian.search("old release")
            citation = {key: evidence[0][key] for key in ("path", "line", "sha256")}
            # Swap the real file for a symlink to an identical-content sibling
            # outside the segments directory; the hash stays correct.
            external = root / "mirror.jsonl"
            external.write_text(line_text, encoding="utf-8")
            segment.unlink()
            segment.symlink_to(external)
            answer = {
                "answer": "The old release was kept.",
                "uncertainty": "This line does not explain why.",
                "citations": [citation],
            }
            with self.assertRaisesRegex(ValueError, "outside"):
                librarian.validate_answer(answer, evidence)

    def test_missing_archive_returns_no_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                LibrarianEvidence(TranscriptArchive(Path(directory), "missing")).search(
                    "decision"
                ),
                [],
            )

    def test_librarian_search_validates_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = TranscriptArchive(Path(directory), "conversation-1")
            archive.append_segment(
                [
                    {"role": "user", "content": "decision alpha"},
                    {"role": "user", "content": "decision beta"},
                    {"role": "user", "content": "decision gamma"},
                ]
            )
            librarian = LibrarianEvidence(archive)
            page1 = librarian.search("decision", limit=2, offset=0)
            self.assertEqual(len(page1), 2)
            page2 = librarian.search("decision", limit=2, offset=2)
            self.assertEqual(len(page2), 1)
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                librarian.search("decision", offset=-1)
            empty = librarian.search("decision", limit=10, offset=100)
            self.assertEqual(empty, [])


class OutputReaderTests(unittest.TestCase):
    def test_zero_failure_summary_remains_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = run_command(
                root,
                "reader-test",
                ["python3", "-c", "print('10 passed, 0 failed, 0 errors')"],
                root,
            )
            result = OutputReader.read(receipt)
            self.assertEqual(result["verdict"], "pass")
            self.assertEqual(result["failure_lines"], [])
            self.assertEqual(
                result["count_lines"][0]["text"], "10 passed, 0 failed, 0 errors"
            )

    def test_mixed_zero_count_and_explicit_failure_remains_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = run_command(
                root,
                "reader-test",
                [
                    "python3",
                    "-c",
                    "print('10 passed, 0 failed\\nFAILED: later stage\\n9 passed, 1 failed')",
                ],
                root,
            )
            result = OutputReader.read(receipt)
            self.assertEqual(result["verdict"], "fail")
            self.assertEqual(
                [item["text"] for item in result["failure_lines"]],
                ["FAILED: later stage", "9 passed, 1 failed"],
            )

    def test_buried_failure_cannot_be_rewritten_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = run_command(
                root,
                "reader-test",
                [
                    "python3",
                    "-c",
                    "print('ok\\n' * 1000 + 'FAILED: one hidden test\\n' + 'ok\\n' * 1000)",
                ],
                root,
            )
            result = OutputReader.read(receipt)
            self.assertEqual(result["verdict"], "fail")
            self.assertTrue(
                any("FAILED" in item["text"] for item in result["failure_lines"])
            )
            with self.assertRaisesRegex(ValueError, "cannot change"):
                OutputReader.read(receipt, {"verdict": "pass", "summary": "all clear"})
            with self.assertRaisesRegex(ValueError, "cannot replace"):
                OutputReader.read(receipt, {"verdict": "fail", "summary": "all clear"})

    def test_receipt_preserves_counts_exit_and_raw_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = run_command(
                root,
                "reader-test",
                ["python3", "-c", "print('2 passed, 1 skipped')"],
                root,
            )
            result = OutputReader.read(
                receipt, {"verdict": "pass", "summary": "Two tests passed."}
            )
            self.assertEqual(result["exit_status"], 0)
            self.assertEqual(result["count_lines"][0]["text"], "2 passed, 1 skipped")
            self.assertEqual(result["summary"], "Two tests passed.")
            self.assertTrue(Path(result["raw_output"]["path"]).is_dir())

    def test_nonzero_exit_and_changed_raw_output_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = run_command(
                root, "reader-test", ["python3", "-c", "raise SystemExit(7)"], root
            )
            self.assertEqual(OutputReader.read(receipt)["verdict"], "fail")
            (receipt.parent / "stdout.log").write_text("later", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                OutputReader.read(receipt)


if __name__ == "__main__":
    unittest.main()
