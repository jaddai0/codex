from pathlib import Path
import json
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from mavis.evaluations import E0_CASES, E0Evaluator
from mavis.e0_tasks import prepare_small_repository
from mavis.runtime import RuntimeConfig
from mavis.storage import sha256_file


class E0EvaluationTests(unittest.TestCase):
    def test_small_repository_fixture_has_failing_baseline_and_protected_dirty_note(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            manifest = json.loads(prepare_small_repository(Path(directory)).read_text())
            repo = Path(manifest["repo"])
            self.assertNotEqual(manifest["baseline_exit_status"], 0)
            self.assertEqual(
                sha256_file(Path(manifest["protected_dirty_file"])),
                manifest["protected_dirty_sha256"],
            )
            self.assertIn(
                "user-notes.txt",
                subprocess.check_output(
                    ["git", "-C", str(repo), "status", "--porcelain"], text=True
                ),
            )
            result = subprocess.run(
                manifest["test_command"], cwd=repo, capture_output=True, text=True
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("test_discount_reduces_price", result.stderr)

    def test_full_e0_never_passes_when_real_worker_cases_are_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory) / "runtime")
            evaluator = E0Evaluator(Path(directory), config)
            with (
                patch("mavis.evaluations.endpoint_alive", return_value=False),
                patch("mavis.evaluations._listener_pids", return_value=set()),
            ):
                result = evaluator.run()
            self.assertEqual(result["status"], "reject")
            self.assertEqual(result["mandatory_cases"], list(E0_CASES))
            statuses = {item["case"]: item["status"] for item in result["results"]}
            self.assertEqual(statuses["small-repository"], "blocked")
            self.assertEqual(statuses["external-harness"], "blocked")

    def test_buried_failure_and_compaction_fixtures_do_not_claim_real_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = E0Evaluator(
                Path(directory), RuntimeConfig(home=Path(directory))
            )
            self.assertEqual(evaluator.run_case("buried-failure")["status"], "blocked")
            self.assertEqual(
                evaluator.run_case("compaction-restart")["status"], "blocked"
            )
            self.assertTrue(
                (
                    Path(directory) / "evaluations" / "e0" / "buried-failure.raw.log"
                ).is_file()
            )

    def test_fabricated_success_case_checks_an_existing_objective(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = E0Evaluator(
                Path(directory), RuntimeConfig(home=Path(directory))
            )
            result = evaluator.run_case("fabricated-success-rejection")
            self.assertEqual(result["status"], "pass")
            self.assertTrue(
                (
                    Path(directory)
                    / "e0-fixtures"
                    / "fabricated"
                    / "objectives"
                    / "e0-fabricated.json"
                ).is_file()
            )

    def test_tool_roundtrip_fails_closed_without_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = E0Evaluator(
                Path(directory), RuntimeConfig(home=Path(directory))
            )
            with patch("mavis.evaluations.endpoint_alive", return_value=False):
                result = evaluator.run_case("tool-roundtrip")
            self.assertEqual(result["status"], "blocked")

    def test_tool_roundtrip_does_not_implicitly_load_model(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = E0Evaluator(
                Path(directory), RuntimeConfig(home=Path(directory))
            )
            with (
                patch("mavis.evaluations.endpoint_alive", return_value=True),
                patch(
                    "mavis.evaluations.inventory",
                    return_value=[{"id": evaluator.config.model, "loaded": False}],
                ),
                patch("mavis.evaluations._post_json") as request,
            ):
                result = evaluator.run_case("tool-roundtrip")
            self.assertEqual(result["status"], "blocked")
            request.assert_not_called()

    def test_one_case_does_not_replace_full_suite_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = E0Evaluator(
                Path(directory), RuntimeConfig(home=Path(directory))
            )
            with (
                patch("mavis.evaluations.endpoint_alive", return_value=False),
                patch("mavis.evaluations._listener_pids", return_value=set()),
            ):
                evaluator.run()
            summary = evaluator.root / "summary.json"
            before = summary.read_bytes()
            evaluator.run("fabricated-success-rejection")
            self.assertEqual(summary.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
