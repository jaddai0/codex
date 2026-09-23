# ruff: noqa: E402
"""First-profile E1 trials require current installed E0 and independent review."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mavis"))
sys.path.insert(0, str(ROOT / "mavis/tests"))

from mavis import e1_bootstrap
from mavis.evaluations import E0_CASES
from mavis.package_provenance import package_tree_sha256
from mavis.storage import read_json, sha256_file, write_json
from test_e1 import E1RunnerTests
import trial_runtime


class E1BootstrapTests(unittest.TestCase):
    def setUp(self):
        fixture = E1RunnerTests("test_changed_manifest_fails_closed")
        fixture.setUp()
        self.addCleanup(fixture.temp.cleanup)
        self.fixture = fixture
        manifest = read_json(fixture.manifest)
        for case in manifest["cases"]:
            case["task"] = "task"
        write_json(fixture.manifest, manifest)
        fixture._freeze()
        fixture.runner.prepare("repair", "baseline")
        fixture.runner.prepare("repair", "candidate")
        self.home = fixture.home
        self.identity = {
            "model_id": "model-a", "architecture": "qwen",
            "weights_fingerprint": "weights-a", "tokenizer_fingerprint": "tok-a",
            "chat_template_fingerprint": "template-a", "quantization": "4bit",
        }
        self.fingerprint = {"core_sha256": "a" * 64, "service_sha256": "b" * 64}
        self.share = fixture.root / "share"
        self.share.mkdir()
        (self.share / "base-instructions.md").write_text("Base instructions\n")
        (self.share / "persona.toml").write_text('name = "Mavis"\n')
        for name in ("launch_core.py", "prepare_runtime.py", "generation_lease.py"):
            shutil.copyfile(ROOT / name, self.share / name)
        shutil.copytree(ROOT / "mavis/mavis", self.share / "mavis",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        self.core = self.share / "local-codex-core"
        self.core.write_text("#!/bin/sh\nexit 0\n")
        self.core.chmod(0o755)
        self.fingerprint["core_sha256"] = sha256_file(self.core)
        self.launcher = self.share / "Mavis.command"
        self.launcher.write_text("#!/bin/sh\n")
        self.package = self.share / "install-manifest.json"
        write_json(self.package, {
            "schema_version": "mavis.installed-core/v1",
            "core_binary": str(self.core.resolve()), "core_sha256": sha256_file(self.core),
            "launcher": str(self.launcher), "launcher_sha256": sha256_file(self.launcher),
            "trial_runtime_sha256": sha256_file(ROOT / "trial_runtime.py"),
            "launch_core_sha256": sha256_file(self.share / "launch_core.py"),
            "prepare_runtime_sha256": sha256_file(self.share / "prepare_runtime.py"),
            "generation_lease_sha256": sha256_file(self.share / "generation_lease.py"),
            "base_instructions_sha256": sha256_file(self.share / "base-instructions.md"),
            "persona_sha256": sha256_file(self.share / "persona.toml"),
            "mavis_package_sha256": package_tree_sha256(self.share / "mavis"),
        })
        e0 = self.home / "evaluations/e0"
        cases = {}
        for case in E0_CASES:
            path = e0 / f"{case}.json"
            write_json(path, {"schema_version": "mavis.evaluation-case/v1",
                              "suite": "e0", "case": case, "status": "pass", "evidence": ["host"]})
            cases[case] = sha256_file(path)
        self.summary = e0 / "summary.json"
        write_json(self.summary, {"schema_version": "mavis.evaluation-suite/v1",
                                  "suite": "e0", "status": "pass", "mandatory_cases": list(E0_CASES),
                                  "results": [{"case": x, "status": "pass"} for x in E0_CASES],
                                  "installed_candidate": self.fingerprint, "model_id": "model-a",
                                  "case_receipts": cases})
        baseline = fixture.runner.store.active("main")["configuration"]
        self.review = self.home / "verifications/e1-bootstrap/main.json"
        write_json(self.review, {
            "schema_version": "mavis.e1-bootstrap-review/v1", "verdict": "accepted",
            "e0_summary_sha256": sha256_file(self.summary),
            "baseline_sha256": baseline["sha256"],
            "package_manifest_sha256": sha256_file(self.package),
            "installed_candidate_digest": e1_bootstrap._digest(self.fingerprint),
            "model_identity_digest": e1_bootstrap._digest(self.identity),
            "gateway_worker_job_id": "review-job", "verifier_job_id": "terra-job",
        })
        self.records = [{"id": "model-a", "model_type": "llm", "loaded": True}]
        self.patches = [
            patch.object(e1_bootstrap, "installed_candidate_fingerprint", return_value=self.fingerprint),
            patch.object(e1_bootstrap, "_package_manifest_path", return_value=self.package),
            patch.object(e1_bootstrap, "harness_job_status", side_effect=self._status),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def _status(self, job_id):
        return {
            "job_id": job_id, "state": "completed", "exit_code": 0, "accepted": True,
            "receipt": {"job_id": job_id, "exit_code": 0},
            "acceptance": {"accepted": True, "job_id": job_id, "verifier": "terra",
                           "verifier_job_id": "terra-job", "target_sha256": "a" * 64,
                           "evidence_sha256": "b" * 64, "report_sha256_on_disk": sha256_file(self.review),
                           "verifier_verdict_sha256": "d" * 64},
            "mavis_binding": {"objective_id": "e1-bootstrap-main",
                              "report_sha256": sha256_file(self.review)},
        }

    def _create(self):
        return e1_bootstrap.create_bootstrap(self.home, self.identity, self.review)

    def test_first_profile_uses_same_reviewed_model_for_both_trial_arms(self):
        bootstrap = self._create()
        self.assertEqual(bootstrap["model_identity"], self.identity)
        receipts = []
        for arm in ("baseline", "candidate"):
            binding = trial_runtime.trial_binding(self.home, "repair", arm, "regression")
            self.assertEqual(binding["profile_source"], "e0-bootstrap")
            receipt_path = trial_runtime.prepare_trial(
                binding, mavis_home=self.home, share=self.share, core_binary=self.core,
                base_url="http://127.0.0.1:8001/v1", records=self.records,
                package_manifest=self.package,
            )
            receipt = read_json(receipt_path)
            trial_runtime.validate_trial_receipt(receipt)
            receipts.append(receipt)
        self.assertEqual(receipts[0]["bootstrap_receipt_sha256"], receipts[1]["bootstrap_receipt_sha256"])
        self.assertEqual(receipts[0]["selected_model"], receipts[1]["selected_model"])
        self.assertNotEqual(receipts[0]["instructions_sha256"], receipts[1]["instructions_sha256"])
        self.assertIsNone(receipts[0]["accepted_profile_id"])
        self.assertFalse((self.home / "profiles/main/active.json").exists())

    def test_missing_or_stale_e0_rejects_before_first_trial(self):
        with self.assertRaises(Exception):
            trial_runtime.trial_binding(self.home, "repair", "baseline", "regression")
        self._create()
        self.fingerprint["service_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "current installed E0"):
            trial_runtime.trial_binding(self.home, "repair", "baseline", "regression")

    def test_old_all_pass_summary_without_installed_binding_requires_fresh_e0(self):
        summary = read_json(self.summary)
        for key in ("installed_candidate", "model_id", "case_receipts"):
            summary.pop(key)
        write_json(self.summary, summary)
        with self.assertRaisesRegex(ValueError, "current installed E0"):
            self._create()
        self.assertFalse((self.home / "e1/bootstrap/main.json").exists())

    def test_tampered_summary_review_and_package_fail_closed(self):
        self._create()
        for path in (self.summary, self.review, self.package):
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.assertRaises(ValueError):
                trial_runtime.trial_binding(self.home, "repair", "baseline", "regression")
            path.write_bytes(original)

    def test_review_change_after_binding_rejects_preparation(self):
        self._create()
        binding = trial_runtime.trial_binding(self.home, "repair", "baseline", "regression")
        self.review.write_bytes(self.review.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            trial_runtime.prepare_trial(
                binding, mavis_home=self.home, share=self.share, core_binary=self.core,
                base_url="http://127.0.0.1:8001/v1", records=self.records,
                package_manifest=self.package,
            )
        self.assertFalse((self.home / "e1/repair/runtime/baseline/regression").exists())

    def test_e0_change_after_preparation_rejects_launch_validation(self):
        self._create()
        binding = trial_runtime.trial_binding(self.home, "repair", "baseline", "regression")
        receipt_path = trial_runtime.prepare_trial(
            binding, mavis_home=self.home, share=self.share, core_binary=self.core,
            base_url="http://127.0.0.1:8001/v1", records=self.records,
            package_manifest=self.package,
        )
        trial_runtime.validate_trial_receipt(read_json(receipt_path))
        self.summary.write_bytes(self.summary.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            trial_runtime.validate_trial_receipt(read_json(receipt_path))

    def test_wrong_model_or_review_rejected(self):
        wrong = {**self.identity, "model_id": "model-b"}
        with self.assertRaisesRegex(ValueError, "model differs"):
            e1_bootstrap.create_bootstrap(self.home, wrong, self.review)
        review = read_json(self.review)
        review["verifier_job_id"] = "wrong"
        write_json(self.review, review)
        with self.assertRaisesRegex(ValueError, "independent gateway acceptance"):
            self._create()

    def test_review_acceptance_must_attest_exact_review_bytes(self):
        status = self._status("review-job")
        status["acceptance"]["report_sha256_on_disk"] = "f" * 64
        with patch.object(e1_bootstrap, "harness_job_status", return_value=status):
            with self.assertRaisesRegex(ValueError, "independent gateway acceptance"):
                self._create()

    def test_accepted_profile_path_remains_authoritative(self):
        self._create()
        pointer = self.home / "profiles/main/active.json"
        write_json(pointer, {"version": 1, "path": "/invalid"})
        with self.assertRaises(ValueError):
            trial_runtime.trial_binding(self.home, "repair", "baseline", "regression")
        with self.assertRaises(ValueError):
            e1_bootstrap.validate_bootstrap(self.home)


if __name__ == "__main__":
    unittest.main()
