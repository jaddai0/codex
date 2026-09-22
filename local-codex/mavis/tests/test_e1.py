from pathlib import Path
import subprocess
import tempfile
import unittest

from mavis.e1 import E1Runner
from mavis.evidence import run_command
from mavis.storage import read_json, sha256_file, write_json


class E1RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "source"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(self.repo),
                "config",
                "user.email",
                "test@example.invalid",
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "config", "user.name", "Test"], check=True
        )
        (self.repo / "result.txt").write_text("fail\n")
        subprocess.run(["git", "-C", str(self.repo), "add", "result.txt"], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-qm", "fixture"], check=True
        )
        revision = subprocess.check_output(
            ["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True
        ).strip()
        check = {
            "id": "accept",
            "argv": [
                "python3",
                "-c",
                "from pathlib import Path; import sys; good=Path('result.txt').read_text().strip()=='pass'; print('OK' if good else 'FAILED'); sys.exit(0 if good else 1)",
            ],
            "timeout_seconds": 10,
        }
        failed_receipt = run_command(
            self.root / "failure-home",
            "original-failure",
            check["argv"],
            self.repo,
            acceptance_check_ids=["accept"],
            timeout=10,
        )
        cases = [
            {
                "id": id_,
                "source": str(self.repo),
                "revision": revision,
                "checks": [check],
            }
            for id_ in ("regression", "held-a", "held-b")
        ]
        self.manifest = self.root / "cases.json"
        write_json(
            self.manifest,
            {
                "schema_version": "mavis.e1-cases/v1",
                "scoring": "held_out_pass_fraction/v1",
                "regression": "regression",
                "failure_receipt": {
                    "path": str(failed_receipt),
                    "sha256": sha256_file(failed_receipt),
                },
                "held_out": ["held-a", "held-b"],
                "minimum_gain": 0.5,
                "cases": cases,
            },
        )
        self.home = self.root / "home"
        self.runner = E1Runner(self.home)
        base = {"prompts": {"system": "A"}, "tool_settings": {}, "retrieval": {}}
        self.runner.store.seed_active("main", base)
        base["prompts"]["system"] = "B"
        self.candidate = self.root / "candidate.json"
        write_json(self.candidate, base)

    def _freeze(self):
        return self.runner.freeze(
            "repair",
            "main",
            "prompts",
            self.candidate,
            self.manifest,
            "Fix a real omission",
        )

    def test_complete_comparison_retains_raw_receipts_and_coverage(self):
        frozen = self._freeze()
        self.assertEqual(len(frozen["candidate_requirements"]), 1)
        self.runner.prepare("repair", "baseline")
        self.runner.prepare("repair", "candidate")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.runner.compare("repair", "native-job", self.manifest)
        for case in ("regression", "held-a", "held-b"):
            self.runner.check("repair", "baseline", case)
            candidate_checkout = (
                self.home / "e1" / "repair" / "checkouts" / "candidate" / case
            )
            (candidate_checkout / "result.txt").write_text("pass\n")
            result = self.runner.check("repair", "candidate", case)
            self.assertTrue(result["passed"])
            receipt = read_json(Path(result["checks"][0]["receipt"]))
            self.assertTrue(
                (Path(receipt["raw_output"]["path"]) / "stdout.log").is_file()
            )
        self.assertEqual(self.runner.coverage("repair")["state"], "complete")
        report = self.root / "native-report.json"
        report.write_text('{"native":"placeholder requiring gateway review"}')
        record = self.runner.compare("repair", "native-job", report)
        self.assertEqual(record["state"], "compared")
        self.assertEqual(record["comparison"]["baseline"]["target_score"], 0)
        self.assertEqual(record["comparison"]["candidate"]["target_score"], 1)
        self.assertEqual(
            self.runner.store.active("main")["configuration"], record["baseline"]
        )

    def test_changed_manifest_fails_closed(self):
        self._freeze()
        frozen = self.home / "e1" / "repair" / "cases.json"
        frozen.write_text("{}")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            self.runner.coverage("repair")

    def test_changed_original_failure_output_fails_closed(self):
        failure = read_json(self.manifest)["failure_receipt"]
        receipt = read_json(Path(failure["path"]))
        (Path(receipt["raw_output"]["path"]) / "stdout.log").write_text("changed")
        with self.assertRaisesRegex(ValueError, "original failure"):
            self._freeze()

    def test_manifest_rejects_overlapping_split(self):
        manifest = read_json(self.manifest)
        manifest["held_out"] = ["regression", "held-a"]
        write_json(self.manifest, manifest)
        with self.assertRaisesRegex(ValueError, "separate"):
            self._freeze()


if __name__ == "__main__":
    unittest.main()
