from pathlib import Path
import tempfile
import unittest

from mavis.evidence import run_command
from mavis.helper_interfaces import LibrarianEvidence, OutputReader
from mavis.storage import sha256_file
from mavis.transcripts import TranscriptArchive


class LibrarianEvidenceTests(unittest.TestCase):
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

    def test_missing_archive_returns_no_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                LibrarianEvidence(TranscriptArchive(Path(directory), "missing")).search(
                    "decision"
                ),
                [],
            )


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
