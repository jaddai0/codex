from pathlib import Path
import subprocess
import tempfile
import unittest

from mavis.e1 import E1Runner
from mavis.experiments import (
    candidate_assignment_requirements,
    review_assignment_requirements,
)
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
        with self.assertRaisesRegex(ValueError, "host receipt"):
            self._freeze()

    def test_original_failure_must_be_from_case_source(self):
        manifest = read_json(self.manifest)
        path = Path(manifest["failure_receipt"]["path"])
        receipt = read_json(path)
        receipt["cwd"] = str(self.root)
        write_json(path, receipt)
        manifest["failure_receipt"]["sha256"] = sha256_file(path)
        write_json(self.manifest, manifest)
        with self.assertRaisesRegex(ValueError, "host receipt"):
            self._freeze()

    def test_review_stage_and_promotion_recheck_underlying_raw_output(self):
        self._freeze()
        self.runner.prepare("repair", "baseline")
        self.runner.prepare("repair", "candidate")
        for case in ("regression", "held-a", "held-b"):
            self.runner.check("repair", "baseline", case)
            (
                self.home
                / "e1"
                / "repair"
                / "checkouts"
                / "candidate"
                / case
                / "result.txt"
            ).write_text("pass\n")
            self.runner.check("repair", "candidate", case)
        report = self.root / "native-report.json"
        report.write_text('{"report":"native fixture"}')
        record = self.runner.compare("repair", "candidate-job", report)
        receipt_path = self.home / "verifications" / "experiments" / "repair.json"
        write_json(
            receipt_path,
            {
                "schema_version": "mavis.experiment-review/v1",
                "experiment_id": "repair",
                "comparison_digest": record["comparison"]["comparison_digest"],
                "baseline_sha256": record["baseline"]["sha256"],
                "candidate_sha256": record["candidate"]["sha256"],
                "candidate_job_id": "candidate-job",
                "gateway_worker_job_id": "review-job",
                "verifier_job_id": "terra-review",
                "verdict": "accepted",
            },
        )

        def gateway_status(job_id):
            candidate = job_id == "candidate-job"
            return {
                "job_id": job_id,
                "state": "completed",
                "exit_code": 0,
                "accepted": True,
                "receipt": {"job_id": job_id, "exit_code": 0},
                "acceptance": {
                    "accepted": True,
                    "job_id": job_id,
                    "verifier": "terra",
                    "verifier_job_id": "terra-candidate"
                    if candidate
                    else "terra-review",
                    "target_sha256": "a" * 64,
                    "evidence_sha256": "b" * 64,
                    "report_sha256_on_disk": "c" * 64,
                    "verifier_verdict_sha256": "d" * 64,
                },
                "mavis_binding": {
                    "objective_id": "repair",
                    "requirements": candidate_assignment_requirements(record)
                    if candidate
                    else review_assignment_requirements(record),
                    "report_sha256": record["comparison"]["candidate"]["evidence"][
                        "sha256"
                    ]
                    if candidate
                    else sha256_file(receipt_path),
                },
            }

        self.runner.store.gateway_status_reader = gateway_status
        self.runner.store.review("repair", receipt_path)
        case_result = read_json(
            self.home / "e1" / "repair" / "results" / "baseline" / "held-a.json"
        )
        host_receipt = Path(case_result["checks"][0]["receipt"])
        stdout = host_receipt.parent / "stdout.log"
        original = stdout.read_bytes()
        stdout.write_bytes(b"tampered after review")
        with self.assertRaisesRegex(
            ValueError, "host receipt|underlying E1|raw output"
        ):
            self.runner.store.stage("repair")
        stdout.write_bytes(original)
        self.runner.store.stage("repair")
        stdout.write_bytes(b"tampered after staging")
        with self.assertRaisesRegex(
            ValueError, "host receipt|underlying E1|raw output"
        ):
            self.runner.store.promote("repair", between_objectives=True)

    def test_manifest_rejects_overlapping_split(self):
        manifest = read_json(self.manifest)
        manifest["held_out"] = ["regression", "held-a"]
        write_json(self.manifest, manifest)
        with self.assertRaisesRegex(ValueError, "separate"):
            self._freeze()

    def test_native_candidate_dispatch_and_compare_use_gateway_report(self):
        self._freeze()
        self.runner.prepare("repair", "baseline")
        self.runner.prepare("repair", "candidate")
        job_dir = self.root / "gateway-job"
        job_dir.mkdir()
        report = job_dir / "report.md"
        report.write_text("Native worker report\n")
        assignments = []

        def start(arguments):
            assignments.append(arguments)
            write_json(job_dir / "assignment.json", {key: value for key, value in arguments.items() if key != "job_id"})
            return {"success": True, "started": True, "accepted": False,
                    "job_id": arguments["job_id"], "job_dir": str(job_dir),
                    "report_path": str(report), "assignment_path": str(job_dir / "assignment.json")}

        self.runner.gateway_assignment_starter = start
        dispatch = self.runner.dispatch_candidate(
            "repair", "regression", job_id="candidate-native", lane="minimax",
            model="minimax/MiniMax-M3", task="Repair the frozen case")
        self.assertEqual(assignments[0]["mavis_requirements"],
                         candidate_assignment_requirements(self.runner.store.load("repair")))
        self.assertEqual(assignments[0]["cwd"], dispatch["checkout"])
        self.assertEqual(assignments[0]["mavis_owner"],
                         {"provider": "minimax", "model": "minimax/MiniMax-M3", "harness": "opencode"})
        for case in ("regression", "held-a", "held-b"):
            self.runner.check("repair", "baseline", case)
            checkout = self.home / "e1" / "repair" / "checkouts" / "candidate" / case
            (checkout / "result.txt").write_text("pass\n")
            self.runner.check("repair", "candidate", case)
        binding = {"objective_id": "repair", "starting_revision": dispatch["starting_revision"],
                   "cwd": dispatch["checkout"], "requirements": assignments[0]["mavis_requirements"],
                   "report_sha256": sha256_file(report),
                   "assignment_sha256": dispatch["assignment_sha256"],
                   "changed_revision": dispatch["starting_revision"],
                   "owned_paths": [dispatch["checkout"]],
                   "owner": assignments[0]["mavis_owner"],
                   "required_checks": ["accept"]}
        self.runner.store.gateway_status_reader = lambda job_id: {
            "job_id": job_id, "state": "completed", "exit_code": 0,
            "accepted": True, "mavis_binding": binding,
            "receipt": {"job_id": job_id, "exit_code": 0},
            "acceptance": {"accepted": True, "job_id": job_id, "verifier": "terra",
                           "verifier_job_id": "terra-job", "target_sha256": "a" * 64,
                           "evidence_sha256": "b" * 64, "report_sha256_on_disk": "c" * 64,
                           "verifier_verdict_sha256": "d" * 64}}
        binding["report_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "receipt or report"):
            self.runner.compare_native("repair", "regression")
        binding["report_sha256"] = sha256_file(report)
        compared = self.runner.compare_native("repair", "regression")
        self.assertEqual(compared["comparison"]["candidate"]["candidate_job_id"], "candidate-native")
        self.assertEqual((self.home / "e1" / "repair" / "candidate-worker-report.json").read_text(),
                         report.read_text())

    def test_native_dispatch_rejects_unbound_start_receipt(self):
        self._freeze()
        self.runner.prepare("repair", "candidate")
        self.runner.gateway_assignment_starter = lambda arguments: {
            "success": True, "started": True, "accepted": False,
            "job_id": "different-job", "job_dir": str(self.root / "job"),
            "report_path": str(self.root / "job" / "report.md"),
            "assignment_path": str(self.root / "job" / "assignment.json")}
        with self.assertRaisesRegex(ValueError, "exact Mavis candidate job"):
            self.runner.dispatch_candidate("repair", "regression", job_id="expected-job",
                                           lane="minimax", model="minimax/MiniMax-M3", task="Fix")


if __name__ == "__main__":
    unittest.main()
