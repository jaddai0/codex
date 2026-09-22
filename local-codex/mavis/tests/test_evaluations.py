from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mavis.evaluations import E0_CASES, E0Evaluator
from mavis.runtime import RuntimeConfig


class E0EvaluationTests(unittest.TestCase):
    def test_full_e0_never_passes_when_real_worker_cases_are_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory) / "runtime")
            evaluator = E0Evaluator(Path(directory), config)
            with patch("mavis.evaluations.endpoint_alive", return_value=False), patch(
                "mavis.evaluations._listener_pids", return_value=set()
            ):
                result = evaluator.run()
            self.assertEqual(result["status"], "reject")
            self.assertEqual(result["mandatory_cases"], list(E0_CASES))
            statuses = {item["case"]: item["status"] for item in result["results"]}
            self.assertEqual(statuses["small-repository"], "blocked")
            self.assertEqual(statuses["external-harness"], "blocked")

    def test_buried_failure_and_compaction_restart_produce_real_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = E0Evaluator(Path(directory), RuntimeConfig(home=Path(directory)))
            self.assertEqual(evaluator.run_case("buried-failure")["status"], "pass")
            self.assertEqual(evaluator.run_case("compaction-restart")["status"], "pass")
            self.assertTrue((Path(directory) / "evaluations" / "e0" / "buried-failure.raw.log").is_file())

    def test_tool_roundtrip_fails_closed_without_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = E0Evaluator(Path(directory), RuntimeConfig(home=Path(directory)))
            with patch("mavis.evaluations.endpoint_alive", return_value=False):
                result = evaluator.run_case("tool-roundtrip")
            self.assertEqual(result["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
