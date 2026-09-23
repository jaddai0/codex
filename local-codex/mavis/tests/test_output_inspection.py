import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from mavis.cli import main
from mavis.evidence import run_command
from mavis.output_inspection import inspect_output


def host_receipt(root: Path, output: str, *, exit_status: int = 0) -> tuple[Path, Path]:
    repo = root / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Mavis Test"], cwd=repo, check=True)
    (repo / "file.txt").write_text("fixture", encoding="utf-8")
    subprocess.run(["git", "add", "file.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    home = root / "home"
    receipt = run_command(
        home, "objective-1",
        ["python3", "-c", f"print({output!r}); raise SystemExit({exit_status})"],
        repo, acceptance_check_ids=["check-1"],
    )
    return home, receipt


class OutputInspectionTests(unittest.TestCase):
    def test_cli_inspects_known_format_without_model_or_receipt_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            home, receipt = host_receipt(Path(directory), "2 passed, 0 failed")
            before = receipt.read_bytes()
            output = io.StringIO()
            with (patch.dict(os.environ, {"MAVIS_HOME": str(home)}),
                  contextlib.redirect_stdout(output)):
                self.assertEqual(main(["output", "inspect", str(receipt),
                                       "--model-id", "small", "--model-path", str(home)]), 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["host"]["verdict"], "pass")
            self.assertEqual(result["host"]["count_lines"][0]["text"], "2 passed, 0 failed")
            self.assertEqual(result["model"], {"status": "skipped_known_format", "observations": []})
            self.assertEqual(result["host"]["raw_output"]["path"], str(receipt.parent.resolve()))
            self.assertEqual(receipt.read_bytes(), before)

    def test_irregular_model_can_only_cite_exact_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            home, receipt = host_receipt(Path(directory), "custom stage complete")
            called = []

            def ask(endpoint, model_id, messages, timeout):
                called.append((endpoint, model_id, timeout))
                lines = json.loads(messages[1]["content"])["lines"]
                return {"verdict": "uncertain", "summary": "", "observations": [lines[0]]}

            result = inspect_output(
                home, receipt, model_id="small", model_path=home,
                ask=ask, binding_reader=lambda *args: {"model_path": str(home)},
            )
            self.assertEqual(called, [("http://127.0.0.1:8001/v1", "small", 90.0)])
            self.assertEqual(result["route"], "irregular")
            self.assertEqual(result["host"]["verdict"], "uncertain")
            self.assertEqual(result["model"]["observations"],
                             [{"line": 1, "text": "custom stage complete"}])

            def fake_success(*args):
                return {"verdict": "pass", "summary": "Looks good", "observations": []}

            with self.assertRaisesRegex(ValueError, "cannot change"):
                inspect_output(home, receipt, model_id="small", model_path=home,
                               ask=fake_success, binding_reader=lambda *args: {})

            def invented_line(*args):
                return {"verdict": "uncertain", "summary": "",
                        "observations": [{"line": 1, "text": "invented"}]}

            with self.assertRaisesRegex(ValueError, "exact raw log line"):
                inspect_output(home, receipt, model_id="small", model_path=home,
                               ask=invented_line, binding_reader=lambda *args: {})

    def test_tampered_and_external_receipts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            home, receipt = host_receipt(Path(directory), "custom stage complete")
            external = Path(directory) / "external.json"
            external.write_bytes(receipt.read_bytes())
            with self.assertRaisesRegex(ValueError, "under Mavis evidence"):
                inspect_output(home, external)
            (receipt.parent / "stdout.log").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash or byte count"):
                inspect_output(home, receipt)

    def test_model_requires_explicit_pair_and_full_log_context(self):
        with tempfile.TemporaryDirectory() as directory:
            home, receipt = host_receipt(Path(directory), "custom stage complete")
            with self.assertRaisesRegex(ValueError, "both model ID and model path"):
                inspect_output(home, receipt, model_id="small")
            (receipt.parent / "stdout.log").write_text("x" * 40000 + "\n", encoding="utf-8")
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            from mavis.storage import sha256_file
            payload["raw_output"]["sha256"] = (
                sha256_file(receipt.parent / "stdout.log") + ":" +
                sha256_file(receipt.parent / "stderr.log")
            )
            payload["raw_output"]["bytes"] = (
                (receipt.parent / "stdout.log").stat().st_size +
                (receipt.parent / "stderr.log").stat().st_size
            )
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            result = inspect_output(
                home, receipt, model_id="small", model_path=home,
                ask=lambda *args: self.fail("model must not receive truncated output"),
                binding_reader=lambda *args: self.fail("no model binding for oversized output"),
            )
            self.assertEqual(result["model"]["status"], "inconclusive")
            self.assertEqual(result["host"]["raw_output"]["bytes"],
                             payload["raw_output"]["bytes"])

    def test_unavailable_local_model_keeps_verified_host_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            home, receipt = host_receipt(Path(directory), "custom stage complete")

            def unavailable(*args):
                raise ValueError("local model is not loaded")

            result = inspect_output(
                home, receipt, model_id="small", model_path=home,
                ask=lambda *args: self.fail("unavailable model must not be called"),
                binding_reader=unavailable,
            )
            self.assertEqual(result["host"]["verdict"], "uncertain")
            self.assertEqual(result["model"], {
                "status": "unavailable", "reason": "local model is not loaded",
                "observations": [],
            })


if __name__ == "__main__":
    unittest.main()
