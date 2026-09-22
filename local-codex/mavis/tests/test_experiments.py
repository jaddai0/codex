from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest

from mavis.experiments import ExperimentStore, _digest
from mavis.storage import sha256_file, write_json


BASE = {"prompts": {"system": "A"}, "tool_settings": {"max_calls": 4},
        "retrieval": {"top_k": 3}}


class ExperimentLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.home = Path(self.temporary.name)
        self.gateway_verifier = "terra-2"
        self.store = ExperimentStore(self.home, gateway_status_reader=self._gateway_status)
        self.store.seed_active("main", BASE)
        self.candidate = deepcopy(BASE)
        self.candidate["prompts"]["system"] = "B"
        self.record = self.store.create("fix-1", "main", "prompts", self.candidate,
                                        hypothesis="Repair a recurring omission",
                                        workload={"revision": "abc123", "suite": "E1"},
                                        split={"held_out": ["case-1", "case-2"], "minimum_gain": 0.1})

    def _gateway_status(self, worker_job_id):
        record = self.store._load("fix-1")
        return {"job_id": worker_job_id, "state": "completed", "exit_code": 0,
                "accepted": True,
                "receipt": {"job_id": worker_job_id, "exit_code": 0},
                "acceptance": {"accepted": True, "job_id": worker_job_id,
                               "verifier": "terra", "verifier_job_id": self.gateway_verifier,
                               "target_sha256": "a" * 64, "evidence_sha256": "b" * 64,
                               "report_sha256_on_disk": "c" * 64,
                               "verifier_verdict_sha256": "d" * 64},
                "mavis_binding": {"objective_id": "fix-1", "requirements": [
                    {"id": "experiment-comparison", "text": record["comparison"]["comparison_digest"]}]}}

    def _result(self, arm, score, *, mandatory=True):
        path = self.home / f"{arm}.log"
        path.write_text(f"{arm} exact independent check output\n")
        return {"configuration_sha256": self.record[arm]["sha256"],
                "workload_digest": _digest(self.record["workload"]),
                "case_ids": ["case-1", "case-2"], "mandatory_passed": mandatory,
                "target_score": score,
                "evidence": {"path": str(path), "sha256": sha256_file(path)}}

    def _compare(self):
        return self.store.compare("fix-1", baseline_result=self._result("baseline", 0.3, mandatory=False),
                                  candidate_result=self._result("candidate", 0.8))

    def _review(self, *, verifier="terra-2", candidate_job="worker-1", verdict="accepted"):
        record = self.store.load("fix-1")
        path = self.home / "verifications" / "experiments" / "fix-1.json"
        write_json(path, {"schema_version": "mavis.experiment-review/v1", "experiment_id": "fix-1",
                          "comparison_digest": record["comparison"]["comparison_digest"],
                          "baseline_sha256": record["baseline"]["sha256"],
                          "candidate_sha256": record["candidate"]["sha256"],
                          "candidate_job_id": candidate_job,
                          "gateway_worker_job_id": "review-worker-1", "verifier_job_id": verifier,
                          "verdict": verdict})
        return path

    def test_frozen_comparison_review_stage_promote_and_rollback(self):
        original = self.store.active("main")
        self._compare()
        self.store.review("fix-1", self._review())
        self.store.stage("fix-1")
        self.assertEqual(self.store.active("main"), original)
        with self.assertRaisesRegex(ValueError, "between objectives"):
            self.store.promote("fix-1", between_objectives=False)
        promoted = self.store.promote("fix-1", between_objectives=True)
        self.assertEqual(promoted["state"], "promoted")
        self.assertEqual(self.store.active("main")["configuration"], self.record["candidate"])
        restored = self.store.rollback("fix-1", reason="critical regression")
        self.assertEqual(restored["state"], "rolled-back")
        self.assertEqual(self.store.active("main"), original)
        reopened = ExperimentStore(self.home, gateway_status_reader=self._gateway_status)
        self.assertEqual(reopened.load("fix-1")["state"], "rolled-back")

    def test_scope_and_baseline_are_frozen(self):
        self.assertEqual(os.stat(self.store.root).st_mode & 0o777, 0o700)
        self.assertEqual(os.stat(self.record["baseline"]["path"]).st_mode & 0o777, 0o600)
        with self.assertRaisesRegex(ValueError, "only its declared"):
            changed = deepcopy(self.candidate)
            changed["retrieval"]["top_k"] = 5
            self.store.create("bad", "main", "prompts", changed, hypothesis="bad",
                              workload={"revision": "abc123"},
                              split={"held_out": ["case"], "minimum_gain": 0.1})
        with self.assertRaises(FileExistsError):
            self.store.seed_active("main", BASE)
        Path(self.record["baseline"]["path"]).write_text("{}")
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            self.store.load("fix-1")

    def test_comparison_rejects_mismatched_or_regressed_evidence(self):
        baseline = self._result("baseline", 0.7)
        candidate = self._result("candidate", 0.71)
        with self.assertRaisesRegex(ValueError, "did not improve"):
            self.store.compare("fix-1", baseline_result=baseline, candidate_result=candidate)
        candidate["target_score"] = 0.9
        candidate["case_ids"] = ["case-2", "case-1"]
        with self.assertRaisesRegex(ValueError, "same mandatory"):
            self.store.compare("fix-1", baseline_result=baseline, candidate_result=candidate)
        candidate["case_ids"] = ["case-1", "case-2"]
        candidate["mandatory_passed"] = False
        with self.assertRaisesRegex(ValueError, "candidate failed"):
            self.store.compare("fix-1", baseline_result=baseline, candidate_result=candidate)
        candidate["mandatory_passed"] = True
        Path(candidate["evidence"]["path"]).write_text("changed after evaluation")
        with self.assertRaisesRegex(ValueError, "evidence hash changed"):
            self.store.compare("fix-1", baseline_result=baseline, candidate_result=candidate)

    def test_review_requires_separate_bound_receipt_and_rechecks_it(self):
        self._compare()
        with self.assertRaisesRegex(ValueError, "separate verification store"):
            self.store.review("fix-1", self.home / "forged.json")
        bad = self._review(verifier="worker-1")
        with self.assertRaisesRegex(ValueError, "mismatched"):
            self.store.review("fix-1", bad)
        good = self._review()
        self.store.review("fix-1", good)
        good.write_text("{}")
        with self.assertRaisesRegex(ValueError, "review receipt changed"):
            self.store.stage("fix-1")

    def test_gateway_review_revocation_blocks_stage_and_promotion(self):
        self._compare()
        self.store.review("fix-1", self._review())
        self.gateway_verifier = "different-terra-job"
        with self.assertRaisesRegex(ValueError, "status changed"):
            self.store.stage("fix-1")
        self.gateway_verifier = "terra-2"
        self.store.stage("fix-1")
        self.gateway_verifier = "different-terra-job"
        with self.assertRaisesRegex(ValueError, "status changed"):
            self.store.promote("fix-1", between_objectives=True)
        self.assertEqual(self.store.active("main")["configuration"], self.record["baseline"])

    def test_stale_baseline_blocks_stage(self):
        self._compare()
        self.store.review("fix-1", self._review())
        active = self.store.active("main")
        active["configuration"] = self.store._snapshot(self.candidate)
        write_json(self.store._active_path("main"), active)
        with self.assertRaisesRegex(ValueError, "active baseline changed"):
            self.store.stage("fix-1")

    def test_interrupted_promotion_is_detected_and_can_finish(self):
        self._compare()
        self.store.review("fix-1", self._review())
        self.store.stage("fix-1")
        previous = self.store.active("main")
        write_json(self.store._active_path("main"), {
            "schema_version": "mavis.experiment-active/v1", "scope": "main",
            "configuration": self.record["candidate"], "experiment_id": "fix-1",
            "previous": previous, "updated_at": "interrupted",
        })
        with self.assertRaisesRegex(ValueError, "incomplete promotion"):
            self.store.active("main")
        self.store.promote("fix-1", between_objectives=True)
        self.assertEqual(self.store.active("main")["configuration"], self.record["candidate"])

    def test_other_reversible_configuration_kinds(self):
        for kind, value in (("tool_settings", {"max_calls": 5}),
                            ("retrieval", {"top_k": 4})):
            candidate = deepcopy(BASE)
            candidate[kind] = value
            created = self.store.create(f"change-{kind}", "main", kind, candidate,
                                        hypothesis=f"Tune {kind}", workload={"revision": "abc123"},
                                        split={"held_out": ["case-1"], "minimum_gain": 0.1})
            self.assertEqual(created["kind"], kind)

    def test_maintenance_checkpoint_yields_and_resumes_without_repeating(self):
        job = self.store.enqueue("fix-1")
        self.assertIsNone(self.store.queue.claim_next(foreground_active=True))
        claimed = self.store.queue.claim_next(foreground_active=False)
        self.assertEqual(claimed["job_id"], job["job_id"])
        paused = self.store.checkpoint(job["job_id"], foreground_active=True)
        self.assertEqual(paused["state"], "paused")
        self.assertIsNone(self.store.queue.claim_next(foreground_active=True))
        self.store.queue.claim_next(foreground_active=False)
        self.assertEqual(self.store.resume_checkpoint(job["job_id"])["state"], "candidate")
        self._compare()
        with self.assertRaisesRegex(ValueError, "checkpoint no longer matches"):
            self.store.resume_checkpoint(job["job_id"])
        self.store.checkpoint(job["job_id"], foreground_active=False)
        self.assertEqual(self.store.resume_checkpoint(job["job_id"])["state"], "compared")


if __name__ == "__main__":
    unittest.main()
