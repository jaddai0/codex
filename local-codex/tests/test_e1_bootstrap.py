# ruff: noqa: E402
"""First-profile E1 trials require current installed E0 and independent review."""

import json
from pathlib import Path
import shutil
import subprocess
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
        self.fingerprint = {"core_sha256": "a" * 64, "service_sha256": "b" * 64}
        self.share = fixture.root / "share"
        self.share.mkdir()
        (self.share / "base-instructions.md").write_text("Base instructions\n")
        (self.share / "persona.toml").write_text('name = "Mavis"\n')
        for name in ("launch_core.py", "prepare_runtime.py", "generation_lease.py"):
            shutil.copyfile(ROOT / name, self.share / name)
        shutil.copytree(ROOT / "mavis/mavis", self.share / "mavis",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        source = fixture.root / "installed-source"
        shutil.copytree(self.share / "mavis", source / "local-codex/mavis/mavis")
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "config", "user.name", "Mavis Test"], check=True)
        subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(source), "add", "local-codex/mavis/mavis"], check=True)
        subprocess.run(["git", "-C", str(source), "commit", "-qm", "installed source"], check=True)
        source_revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
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
            "source_repository": str(source.resolve()),
            "source_revision": source_revision,
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
        self.model_root = fixture.root / "models"
        self.model_path = self.model_root / "model-a"
        self.model_path.mkdir(parents=True)
        write_json(self.model_path / "config.json", {
            "architectures": ["QwenFixture"], "quantization": {"bits": 4},
        })
        write_json(self.model_path / "tokenizer.json", {"version": "fixture"})
        write_json(self.model_path / "tokenizer_config.json", {"bos_token": "<s>"})
        (self.model_path / "chat_template.jinja").write_text("{{ prompt }}")
        (self.model_path / "model-00001.safetensors").write_bytes(b"weights-one")
        (self.model_path / "model-00002.safetensors").write_bytes(b"weights-two")
        self.records = [{"id": "model-a", "model_type": "llm", "loaded": True,
                         "model_path": str(self.model_path.resolve())}]
        self.owner = {"provider": "minimax", "model": "minimax/MiniMax-M3", "harness": "opencode"}
        self.patches = [
            patch.object(e1_bootstrap, "installed_candidate_fingerprint", return_value=self.fingerprint),
            patch.object(e1_bootstrap, "_package_manifest_path", return_value=self.package),
            patch.object(e1_bootstrap, "_model_root_path", return_value=self.model_root),
            patch.object(e1_bootstrap, "harness_job_status", side_effect=self._status),
            patch.object(trial_runtime, "inventory", side_effect=lambda _: self.records),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.assignment = e1_bootstrap.prepare_bootstrap_review(
            self.home, self.owner, inventory_reader=lambda _: self.records
        )
        self.assignment_path = self.home / "e1/bootstrap/review-assignment.json"
        self.identity = self.assignment["model_identity"]
        baseline = fixture.runner.store.active("main")["configuration"]
        self.review = self.home / "verifications/e1-bootstrap/main.json"
        self.gateway_job_dir = fixture.root / "gateway-job"
        self.gateway_report = self.gateway_job_dir / "review.json"
        self.gateway_assignment = self.gateway_job_dir / "assignment.json"

        def start_review(arguments):
            write_json(self.gateway_assignment, arguments)
            return {
                "success": True, "started": True, "accepted": False,
                "job_id": "review-job", "job_dir": str(self.gateway_job_dir),
                "report_path": str(self.gateway_report),
                "assignment_path": str(self.gateway_assignment),
            }

        e1_bootstrap.dispatch_bootstrap_review(
            self.home, "review-job", starter=start_review
        )
        write_json(self.review, {
            "schema_version": "mavis.e1-bootstrap-review/v1", "verdict": "accepted",
            "e0_summary_sha256": sha256_file(self.summary),
            "baseline_sha256": baseline["sha256"],
            "package_manifest_sha256": sha256_file(self.package),
            "installed_candidate_digest": e1_bootstrap._digest(self.fingerprint),
            "model_identity_digest": e1_bootstrap._digest(self.identity),
            "assignment_sha256": sha256_file(self.assignment_path),
            "gateway_worker_job_id": "review-job", "verifier_job_id": "review-job-terra",
        })
        shutil.copyfile(self.review, self.gateway_report)

    def _status(self, job_id):
        return {
            "job_id": job_id, "state": "completed", "exit_code": 0, "accepted": True,
            "receipt": {"job_id": job_id, "exit_code": 0},
            "acceptance": {"accepted": True, "job_id": job_id, "verifier": "terra",
                           "verifier_job_id": "review-job-terra", "target_sha256": sha256_file(self.assignment_path),
                           "evidence_sha256": "b" * 64, "report_sha256_on_disk": sha256_file(self.review),
                           "verifier_verdict_sha256": "d" * 64},
            "mavis_binding": {
                "schema_version": "model-gateway-mavis-objective-binding/v1",
                "objective_id": "e1-bootstrap-main",
                "cwd": self.assignment["cwd"], "owner": self.owner,
                "starting_revision": self.assignment["starting_revision"],
                "changed_revision": self.assignment["starting_revision"],
                "owned_paths": self.assignment["owned_paths"],
                "requirements": self.assignment["requirements"],
                "required_checks": self.assignment["required_checks"],
                "assignment_sha256": sha256_file(self.gateway_assignment),
                "report_sha256": sha256_file(self.review),
                "target_sha256": sha256_file(self.assignment_path),
            },
        }

    def _create(self):
        return e1_bootstrap.create_bootstrap(self.home, self.review)

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
        self.records[0]["id"] = "model-b"
        with self.assertRaisesRegex(ValueError, "unique model_path"):
            e1_bootstrap._inventory_model(self.records, "model-a")
        self.records[0]["id"] = "model-a"
        review = read_json(self.review)
        review["verifier_job_id"] = "wrong"
        write_json(self.review, review)
        with self.assertRaisesRegex(ValueError, "differs from gateway report bytes"):
            self._create()

    def test_same_model_id_with_changed_weight_bytes_rejected(self):
        self._create()
        (self.model_path / "model-00002.safetensors").write_bytes(b"changed-bytes")
        with self.assertRaisesRegex(ValueError, "artifact bytes changed"):
            trial_runtime.trial_binding(self.home, "repair", "baseline", "regression")

    def test_symlinked_model_file_rejected(self):
        self._create()
        target = self.model_path / "model-00002.safetensors"
        target.unlink()
        target.symlink_to(self.model_path / "model-00001.safetensors")
        with self.assertRaisesRegex(ValueError, "symlink"):
            trial_runtime.trial_binding(self.home, "repair", "baseline", "regression")

    def test_gateway_assignment_fields_must_match_frozen_file(self):
        for field, changed in (("owner", {"provider": "minimax", "model": "other", "harness": "opencode"}),
                               ("scope", "other"), ("requirements", []),
                               ("assignment_sha256", "f" * 64)):
            with self.subTest(field=field):
                status = self._status("review-job")
                status["mavis_binding"][field] = changed
                with patch.object(e1_bootstrap, "harness_job_status", return_value=status):
                    with self.assertRaisesRegex(ValueError, "independent gateway acceptance"):
                        self._create()

    def test_changed_frozen_assignment_rejected_after_bootstrap(self):
        self._create()
        self.assignment_path.write_bytes(self.assignment_path.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            trial_runtime.trial_binding(self.home, "repair", "baseline", "regression")

    def test_preparation_rechecks_inventory_model_path(self):
        self._create()
        binding = trial_runtime.trial_binding(self.home, "repair", "baseline", "regression")
        elsewhere = self.model_root / "other-model"
        shutil.copytree(self.model_path, elsewhere)
        self.records[0]["model_path"] = str(elsewhere.resolve())
        with self.assertRaisesRegex(ValueError, "model_path changed"):
            trial_runtime.prepare_trial(
                binding, mavis_home=self.home, share=self.share, core_binary=self.core,
                base_url="http://127.0.0.1:8001/v1", records=self.records,
                package_manifest=self.package,
            )

    def test_launch_rechecks_live_inventory_model_path(self):
        self._create()
        binding = trial_runtime.trial_binding(self.home, "repair", "baseline", "regression")
        receipt_path = trial_runtime.prepare_trial(
            binding, mavis_home=self.home, share=self.share, core_binary=self.core,
            base_url="http://127.0.0.1:8001/v1", records=self.records,
            package_manifest=self.package,
        )
        receipt = read_json(receipt_path)
        trial_runtime.validate_trial_receipt(receipt)
        elsewhere = self.model_root / "other-model"
        shutil.copytree(self.model_path, elsewhere)
        self.records[0]["model_path"] = str(elsewhere.resolve())
        with self.assertRaisesRegex(ValueError, "model_path changed before launch"):
            trial_runtime.validate_trial_receipt(receipt)

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
