from copy import deepcopy
import fcntl
import os
from pathlib import Path
import tempfile
import unittest

from mavis.experiments import (ExperimentStore, _digest,
                               candidate_assignment_requirements, review_assignment_requirements)
from mavis.storage import sha256_file, write_json


BASE = {"prompts": {"system": "A"}, "tool_settings": {"max_calls": 4},
        "retrieval": {"top_k": 3}}


class ExperimentLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.home = Path(self.temporary.name)
        self.gateway_verifier = "glm-2"
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
        candidate = worker_job_id == "candidate-worker-1"
        requirements = (candidate_assignment_requirements(record) if candidate
                        else review_assignment_requirements(record))
        verifier = "glm-candidate-1" if candidate else self.gateway_verifier
        review_report = self.home / "verifications" / "experiments" / "fix-1.json"
        return {"job_id": worker_job_id, "state": "completed", "exit_code": 0,
                "accepted": True,
                "receipt": {"job_id": worker_job_id, "exit_code": 0},
                "acceptance": {"accepted": True, "job_id": worker_job_id,
                               "verifier": "glm-codex", "verifier_job_id": verifier,
                               "target_sha256": "a" * 64, "evidence_sha256": "b" * 64,
                               "report_sha256_on_disk": "c" * 64,
                               "verifier_verdict_sha256": "d" * 64},
                "mavis_binding": {"objective_id": "fix-1", "requirements": requirements,
                                  "report_sha256": (record["comparison"]["candidate"]["evidence"]["sha256"]
                                                    if candidate else sha256_file(review_report))}}

    def _result(self, arm, score, *, mandatory=True):
        path = self.home / f"{arm}.log"
        path.write_text(f"{arm} exact independent check output\n")
        result = {"configuration_sha256": self.record[arm]["sha256"],
                "workload_digest": _digest(self.record["workload"]),
                "case_ids": ["case-1", "case-2"], "mandatory_passed": mandatory,
                "target_score": score,
                "evidence": {"path": str(path), "sha256": sha256_file(path)}}
        if arm == "candidate":
            result["candidate_job_id"] = "candidate-worker-1"
        return result

    def _compare(self):
        return self.store.compare("fix-1", baseline_result=self._result("baseline", 0.3, mandatory=False),
                                  candidate_result=self._result("candidate", 0.8))

    def _review(self, *, verifier="glm-2", candidate_job="candidate-worker-1", verdict="accepted"):
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

    def test_promotion_and_rollback_require_a_live_objective_boundary(self):
        self._compare()
        self.store.review("fix-1", self._review())
        self.store.stage("fix-1")
        objective = self.home / "objectives" / "work-1.json"
        write_json(objective, {"objective_id": "work-1", "state": "running"})
        with self.assertRaisesRegex(ValueError, "between objectives"):
            self.store.promote("fix-1", between_objectives=True)
        objective.unlink()
        with (self.home / "generation.lock").open("a+") as lease:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "generation owns"):
                self.store.promote("fix-1", between_objectives=True)
        self.store.promote("fix-1", between_objectives=True)
        write_json(objective, {"objective_id": "work-1", "state": "awaiting verification"})
        with self.assertRaisesRegex(ValueError, "between objectives"):
            self.store.rollback("fix-1", reason="regression")
        objective.unlink()
        self.assertEqual(self.store.rollback("fix-1", reason="regression")["state"], "rolled-back")

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
        bad = self._review(verifier="candidate-worker-1")
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
        self.gateway_verifier = "different-glm-job"
        with self.assertRaisesRegex(ValueError, "status changed"):
            self.store.stage("fix-1")
        self.gateway_verifier = "glm-2"
        self.store.stage("fix-1")
        self.gateway_verifier = "different-glm-job"
        with self.assertRaisesRegex(ValueError, "status changed"):
            self.store.promote("fix-1", between_objectives=True)
        self.assertEqual(self.store.active("main")["configuration"], self.record["baseline"])

    def test_review_rejects_gateway_incompatible_requirement_shape(self):
        self._compare()
        receipt = self._review()
        def incompatible(job_id):
            status = self._gateway_status(job_id)
            if job_id == "review-worker-1":
                status["mavis_binding"]["requirements"] = [
                    {"id": "experiment-comparison", "text": self.store._load("fix-1")["comparison"]["comparison_digest"]}]
            return status
        self.store.gateway_status_reader = incompatible
        with self.assertRaisesRegex(ValueError, "exact comparison"):
            self.store.review("fix-1", receipt)

    def test_assignment_requirements_are_canonical_gateway_strings(self):
        record = self._compare()
        candidate = candidate_assignment_requirements(record)
        review = review_assignment_requirements(record)
        self.assertEqual(candidate, [f"experiment-candidate-snapshot:{record['candidate']['sha256']}"])
        self.assertEqual(review, [
            f"experiment-comparison:{record['comparison']['comparison_digest']}",
            "experiment-candidate-job:candidate-worker-1",
            f"experiment-candidate-report:{record['comparison']['candidate']['evidence']['sha256']}",
        ])
        self.assertTrue(all(isinstance(item, str) and item for item in candidate + review))
        self.assertEqual(len(set(review)), len(review))

    def test_review_rejects_substituted_candidate_job_and_report(self):
        self._compare()
        with self.assertRaisesRegex(ValueError, "mismatched"):
            self.store.review("fix-1", self._review(candidate_job="unrelated-worker"))
        receipt = self._review()
        def wrong_candidate_report(job_id):
            status = self._gateway_status(job_id)
            if job_id == "candidate-worker-1":
                status["mavis_binding"]["report_sha256"] = "f" * 64
            return status
        self.store.gateway_status_reader = wrong_candidate_report
        with self.assertRaisesRegex(ValueError, "candidate evidence"):
            self.store.review("fix-1", receipt)

    def test_review_receipt_must_be_exact_gateway_report(self):
        self._compare()
        receipt = self._review()
        def unrelated_report(job_id):
            status = self._gateway_status(job_id)
            if job_id == "review-worker-1":
                status["mavis_binding"]["report_sha256"] = "f" * 64
            return status
        self.store.gateway_status_reader = unrelated_report
        with self.assertRaisesRegex(ValueError, "exact report"):
            self.store.review("fix-1", receipt)

    def test_candidate_gateway_revocation_blocks_staging(self):
        self._compare()
        self.store.review("fix-1", self._review())
        def revoked_candidate(job_id):
            status = self._gateway_status(job_id)
            if job_id == "candidate-worker-1":
                status["accepted"] = False
            return status
        self.store.gateway_status_reader = revoked_candidate
        with self.assertRaisesRegex(ValueError, "not accepted"):
            self.store.stage("fix-1")

    def test_stale_baseline_blocks_stage(self):
        self._compare()
        self.store.review("fix-1", self._review())
        active = self.store.active("main")
        active["configuration"] = self.store._snapshot(self.candidate)
        write_json(self.store._active_path("main"), active)
        with self.assertRaisesRegex(ValueError, "active baseline changed"):
            self.store.stage("fix-1")

    def test_comparison_rechecks_active_seed_under_profile_boundary(self):
        active = self.store.active("main")
        active["configuration"] = self.store._snapshot(self.candidate)
        write_json(self.store._active_path("main"), active)
        with self.assertRaisesRegex(ValueError, "active baseline changed"):
            self._compare()
        self.assertEqual(self.store.load("fix-1")["state"], "candidate")

    def test_comparison_is_serialized_with_generation_host_lease(self):
        with (self.home / "generation.lock").open("a+") as lease:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "generation owns"):
                self._compare()
        self.assertEqual(self.store.load("fix-1")["state"], "candidate")

    def test_comparison_uses_its_own_scope_not_main(self):
        self.store.seed_active("scratch", BASE)
        candidate = deepcopy(BASE)
        candidate["retrieval"]["top_k"] = 8
        record = self.store.create("scratch-fix", "scratch", "retrieval", candidate,
                                   hypothesis="Improve retrieval", workload={"suite": "E1"},
                                   split={"held_out": ["case-1"], "minimum_gain": 0.1})
        active = self.store.active("main")
        active["configuration"] = self.store._snapshot(self.candidate)
        write_json(self.store._active_path("main"), active)
        def result(arm, score):
            path = self.home / f"scratch-{arm}.log"
            path.write_text("host result\n")
            value = {"configuration_sha256": record[arm]["sha256"],
                     "workload_digest": _digest(record["workload"]),
                     "case_ids": ["case-1"], "mandatory_passed": True,
                     "target_score": score,
                     "evidence": {"path": str(path), "sha256": sha256_file(path)}}
            if arm == "candidate":
                value["candidate_job_id"] = "scratch-worker"
            return value
        compared = self.store.compare("scratch-fix", baseline_result=result("baseline", 0.2),
                                      candidate_result=result("candidate", 0.8))
        self.assertEqual(compared["state"], "compared")

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
