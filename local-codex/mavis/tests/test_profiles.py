import json
import fcntl
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mavis.experiments import (ExperimentStore, _digest,
                               candidate_assignment_requirements, review_assignment_requirements)
from mavis.profiles import ProfileStore
from mavis.storage import read_json, sha256_file, write_json


def profile(profile_id, experiments=None, prompt=""):
    return {
        "profile_id": profile_id,
        "model_identity": {
            "model_id": "qwen-local",
            "architecture": "qwen",
            "weights_fingerprint": "weights",
            "tokenizer_fingerprint": "tokenizer",
            "chat_template_fingerprint": "template",
            "quantization": "4bit",
        },
        "runtime": {"name": "omlx", "version": "1"},
        "prompts": {"system": prompt},
        "tool_settings": {},
        "context_policy": {"retrieval": {}},
        "experiments": experiments or [],
    }


class ProfileStoreTests(unittest.TestCase):
    def _first_profile(self, root):
        self.experiments = ExperimentStore(root, gateway_status_reader=self._gateway_status)
        baseline = {"prompts": {"system": "original accepted instructions"},
                    "tool_settings": {}, "retrieval": {}}
        seed = self.experiments.seed_active("main", baseline)
        bootstrap_path = root / "e1/bootstrap/main.json"
        review_path = root / "verifications/e1-bootstrap/main.json"
        write_json(bootstrap_path, {"schema_version": "mavis.e1-bootstrap/v1"})
        write_json(review_path, {"verdict": "accepted"})
        bootstrap = {"prompts": baseline["prompts"],
                     "model_identity": profile("seed")["model_identity"],
                     "_source_path": str(bootstrap_path),
                     "_source_sha256": sha256_file(bootstrap_path),
                     "independent_review": {"path": str(review_path),
                                            "sha256": sha256_file(review_path)}}
        store = ProfileStore(root, gateway_status_reader=self._gateway_status)
        with patch("mavis.e1_bootstrap.validate_bootstrap", return_value=bootstrap):
            first = store.activate_bootstrap_baseline()
        return seed, baseline, store, first

    def test_first_profile_rollback_restores_nonempty_bootstrap_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed, baseline, store, first = self._first_profile(root)
            self.assertEqual(read_json(first)["prompts"], baseline["prompts"])
            self.assertEqual(store.active_version("main"), 1)
            exp, _ = self._promotion_evidence(root, "first-candidate", "improved instructions")
            self.assertEqual(store.active_version("main"), 2)
            self.assertEqual(read_json(exp)["state"], "promoted")
            self.assertEqual(self.experiments.assert_promoted("first-candidate")["state"], "promoted")
            baseline_hash = sha256_file(first)
            candidate_hash = sha256_file(root / "profiles/main/v2.json")
            self.assertEqual(self.experiments.rollback("first-candidate", reason="critical regression")["state"],
                             "rolled-back")
            self.assertEqual(self.experiments.active("main"), seed)
            self.assertEqual(store.active_version("main"), 1)
            self.assertEqual(read_json(first)["prompts"]["system"], "original accepted instructions")
            self.assertEqual(sha256_file(first), baseline_hash)
            self.assertEqual(sha256_file(root / "profiles/main/v2.json"), candidate_hash)

    def test_interrupted_first_profile_rollback_fails_closed_and_resumes(self):
        from mavis import profile_transition

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed, _, store, _ = self._first_profile(root)
            self._promotion_evidence(root, "first-candidate", "improved instructions")
            original_write = profile_transition.write_json
            calls = 0

            def interrupted(path, payload):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise OSError("injected interruption")
                return original_write(path, payload)

            with patch.object(profile_transition, "write_json", side_effect=interrupted):
                with self.assertRaisesRegex(OSError, "injected interruption"):
                    self.experiments.rollback("first-candidate", reason="critical regression")
            self.assertTrue(profile_transition.pending(root))
            with self.assertRaisesRegex(ValueError, "interrupted profile transition"):
                self.experiments.active("main")
            with self.assertRaisesRegex(ValueError, "interrupted profile transition"):
                self.experiments.load("first-candidate")
            with self.assertRaisesRegex(ValueError, "interrupted profile transition"):
                store.restore("main", 1)
            self.assertEqual(self.experiments.rollback("first-candidate", reason="critical regression")["state"],
                             "rolled-back")
            self.assertFalse(profile_transition.pending(root))
            self.assertEqual(self.experiments.active("main"), seed)
            self.assertEqual(store.active_version("main"), 1)

    def _gateway_status(self, worker_job_id):
        candidate = worker_job_id.startswith("candidate-")
        experiment_id = worker_job_id.removeprefix("candidate-").removeprefix("review-")
        record = self.experiments._load(experiment_id)
        requirements = (candidate_assignment_requirements(record) if candidate
                        else review_assignment_requirements(record))
        review_report = self.experiments.home / "verifications" / "experiments" / f"{experiment_id}.json"
        return {"job_id": worker_job_id, "state": "completed", "exit_code": 0,
                "accepted": True,
                "receipt": {"job_id": worker_job_id, "exit_code": 0},
                "acceptance": {"accepted": True, "job_id": worker_job_id,
                               "verifier": "terra", "verifier_job_id": f"terra-{experiment_id}",
                               "target_sha256": "a" * 64, "evidence_sha256": "b" * 64,
                               "report_sha256_on_disk": "c" * 64,
                               "verifier_verdict_sha256": "d" * 64},
                "mavis_binding": {"objective_id": experiment_id, "requirements": requirements,
                                  "report_sha256": (record["comparison"]["candidate"]["evidence"]["sha256"]
                                                    if candidate else sha256_file(review_report))}}

    def _promotion_evidence(self, root: Path, experiment_id: str, prompt: str, *, promote=True):
        active = self.experiments.active("main")
        candidate = self.experiments._read_snapshot(active["configuration"])
        candidate["prompts"] = {"system": prompt}
        record = self.experiments.create(experiment_id, "main", "prompts", candidate,
                                         hypothesis="Improve prompt", workload={"revision": "abc123"},
                                         split={"held_out": ["case-1"], "minimum_gain": 0.1})
        results = {}
        for arm, score in (("baseline", 0.2), ("candidate", 0.8)):
            evidence = root / f"{experiment_id}-{arm}.log"
            evidence.write_text(f"{arm} check output")
            results[arm] = {"configuration_sha256": record[arm]["sha256"],
                            "workload_digest": _digest(record["workload"]),
                            "case_ids": ["case-1"], "mandatory_passed": arm == "candidate",
                            "target_score": score,
                            "evidence": {"path": str(evidence), "sha256": sha256_file(evidence)}}
            if arm == "candidate":
                results[arm]["candidate_job_id"] = f"candidate-{experiment_id}"
        record = self.experiments.compare(experiment_id, baseline_result=results["baseline"],
                                          candidate_result=results["candidate"])
        verifier = root / "verifications" / "experiments" / f"{experiment_id}.json"
        write_json(verifier, {"schema_version": "mavis.experiment-review/v1",
                              "experiment_id": experiment_id,
                              "comparison_digest": record["comparison"]["comparison_digest"],
                              "baseline_sha256": record["baseline"]["sha256"],
                              "candidate_sha256": record["candidate"]["sha256"],
                              "candidate_job_id": f"candidate-{experiment_id}",
                              "gateway_worker_job_id": f"review-{experiment_id}",
                              "verifier_job_id": f"terra-{experiment_id}",
                              "verdict": "accepted"})
        self.experiments.review(experiment_id, verifier)
        self.experiments.stage(experiment_id)
        if ProfileStore(root).active_version("main") is not None:
            ProfileStore(root).create_candidate(
                "main", profile(experiment_id, [experiment_id], prompt),
            )
        if promote:
            self.experiments.promote(experiment_id, between_objectives=True)
        return self.experiments._record_path(experiment_id), verifier

    def test_interrupted_first_profile_promotion_fails_closed_and_resumes(self):
        from mavis import profile_transition

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, store, first = self._first_profile(root)
            self._promotion_evidence(root, "first-candidate", "improved instructions", promote=False)
            baseline_hash = sha256_file(first)
            original_write = profile_transition.write_json
            calls = 0

            def interrupted(path, payload):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise OSError("injected interruption")
                return original_write(path, payload)

            with patch.object(profile_transition, "write_json", side_effect=interrupted):
                with self.assertRaisesRegex(OSError, "injected interruption"):
                    self.experiments.promote("first-candidate", between_objectives=True)
            self.assertTrue(profile_transition.pending(root))
            with self.assertRaisesRegex(ValueError, "interrupted profile transition"):
                self.experiments.active("main")
            with self.assertRaisesRegex(ValueError, "interrupted profile transition"):
                self.experiments.load("first-candidate")
            with self.assertRaisesRegex(ValueError, "interrupted profile transition"):
                store.create_candidate("main", profile("later"))
            self.assertEqual(self.experiments.promote("first-candidate", between_objectives=True)["state"],
                             "promoted")
            self.assertFalse(profile_transition.pending(root))
            self.assertEqual(store.active_version("main"), 2)
            self.assertEqual(sha256_file(first), baseline_hash)

    def test_switching_back_restores_exact_accepted_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.experiments = ExperimentStore(root, gateway_status_reader=self._gateway_status)
            self.experiments.seed_active("main", {"prompts": {"system": "initial"},
                                                  "tool_settings": {}, "retrieval": {}})
            store = ProfileStore(root, gateway_status_reader=self._gateway_status)
            exp_a, verify_a = self._promotion_evidence(root, "exp-a", "A")
            first = store.create_candidate("main", profile("a", ["exp-a"], "A"))
            store.activate("main", 1, exp_a, verify_a)
            exp_b, verify_b = self._promotion_evidence(root, "exp-b", "B")
            store.activate("main", 2, exp_b, verify_b)
            prior = read_json(first)
            prior["status"] = "previous"  # Existing on-disk v2 behavior before immutable profiles.
            write_json(first, prior)
            self.experiments.rollback("exp-b", reason="critical regression")
            restored = store.restore("main", 1)
            self.assertEqual(restored["profile_id"], "a")
            self.assertEqual(restored["status"], "active")
            self.assertIsNone(restored["previous_version"])
            self.assertEqual(first.read_text(), (root / "profiles/main/v1.json").read_text())
            self.assertEqual(store.active_version("main"), 1)

    def test_old_file_only_promotion_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ProfileStore(root, gateway_status_reader=lambda _: {})
            store.create_candidate("main", profile("a", ["exp-a"], "A"))
            experiment = root / "experiments" / "exp-a.json"
            verifier = root / "verifications" / "exp-a.json"
            write_json(experiment, {"schema_version": "mavis.experiment/v1",
                                    "experiment_id": "exp-a", "promotion_decision": "promote"})
            write_json(verifier, {"schema_version": "mavis.verifier/v1", "verdict": "accepted"})
            with self.assertRaisesRegex(ValueError, "lifecycle record"):
                store.activate("main", 1, experiment, verifier)
            self.assertIsNone(store.active_version("main"))

    def test_profile_activation_requires_exact_candidate_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.experiments = ExperimentStore(root, gateway_status_reader=self._gateway_status)
            self.experiments.seed_active("main", {"prompts": {"system": "initial"},
                                                  "tool_settings": {}, "retrieval": {}})
            experiment, verifier = self._promotion_evidence(root, "exp-a", "A")
            store = ProfileStore(root, gateway_status_reader=self._gateway_status)
            store.create_candidate("main", profile("mismatch", ["exp-a"], "different"))
            with self.assertRaisesRegex(ValueError, "differs from promoted candidate"):
                store.activate("main", 1, experiment, verifier)
            self.assertIsNone(store.active_version("main"))

    def test_activation_rejects_unretained_arbitrary_evidence_names(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory))
            store.create_candidate("main", profile("a", ["exp-a"]))
            with self.assertRaisesRegex(ValueError, "experiment store"):
                store.activate("main", 1, Path("exp-a"), Path("verify-a"))

    def test_family_inheritance_never_inherits_pass_or_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory))
            path = store.create_candidate("main", profile("child"), inherited_from="parent")
            payload = json.loads(path.read_text())
            self.assertEqual(
                payload["candidate_inheritance"],
                {"profile_id": "parent", "passed_status_inherited": False, "adapters_inherited": False},
            )

    def test_main_profile_rejects_unapplied_candidate_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.experiments = ExperimentStore(root, gateway_status_reader=self._gateway_status)
            self.experiments.seed_active("main", {"prompts": {"system": "initial"},
                                                  "tool_settings": {}, "retrieval": {}})
            experiment, verifier = self._promotion_evidence(root, "exp-a", "A")
            store = ProfileStore(root, gateway_status_reader=self._gateway_status)
            payload = profile("a", ["exp-a"], "A")
            payload["context_policy"]["compaction"] = "unsupported"
            store.create_candidate("main", payload)
            with self.assertRaisesRegex(ValueError, "cannot apply"):
                store.activate("main", 1, experiment, verifier)
            self.assertIsNone(store.active_version("main"))

    def test_profile_pointer_cannot_change_during_active_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.experiments = ExperimentStore(root, gateway_status_reader=self._gateway_status)
            self.experiments.seed_active("main", {"prompts": {"system": "initial"},
                                                  "tool_settings": {}, "retrieval": {}})
            experiment, verifier = self._promotion_evidence(root, "exp-a", "A")
            store = ProfileStore(root, gateway_status_reader=self._gateway_status)
            store.create_candidate("main", profile("a", ["exp-a"], "A"))
            objective = root / "objectives" / "work-1.json"
            write_json(objective, {"objective_id": "work-1", "state": "running"})
            with self.assertRaisesRegex(ValueError, "between objectives"):
                store.activate("main", 1, experiment, verifier)
            objective.unlink()
            with (root / "generation.lock").open("a+") as lease:
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "generation owns"):
                    store.activate("main", 1, experiment, verifier)
            self.assertIsNone(store.active_version("main"))
            store.activate("main", 1, experiment, verifier)
            self.assertEqual(store.active_version("main"), 1)


if __name__ == "__main__":
    unittest.main()
