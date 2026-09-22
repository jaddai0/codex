from pathlib import Path
import json
import subprocess
import tempfile
import unittest

from mavis.evidence import run_command
from mavis.gateway import GatewayUnavailable
from mavis.objectives import ObjectiveStore


def objective(objective_id="obj-1"):
    return {
        "schema_version": "mavis.objective/v1",
        "objective_id": objective_id,
        "blueprint": "approved plan",
        "requirements": [{"id": "r1", "text": "prove it"}],
        "dependencies": [],
        "scope": {"paths": ["src"]},
        "acceptance_checks": [{"id": "c1", "command": ["true"]}],
        "unresolved_decisions": [],
        "state": "queued",
    }


class ObjectiveStoreTests(unittest.TestCase):
    def gateway_status(self, worker_job_id="worker-job-1", mavis_binding=None):
        digest = "a" * 64
        status = {
            "job_id": worker_job_id,
            "state": "completed",
            "exit_code": 0,
            "accepted": True,
            "receipt": {"job_id": worker_job_id, "exit_code": 0},
            "acceptance": {
                "job_id": worker_job_id,
                "accepted": True,
                "verifier": "terra",
                "verifier_job_id": "terra-job-1",
                "target_sha256": digest,
                "evidence_sha256": "b" * 64,
                "report_sha256_on_disk": "c" * 64,
                "verifier_verdict_sha256": "d" * 64,
            },
        }
        if mavis_binding is not None:
            status["mavis_binding"] = mavis_binding
        return status

    def gateway_binding(self, fixture, revision, objective_id="obj-1"):
        return {
            "schema_version": "model-gateway-mavis-objective-binding/v1",
            "objective_id": objective_id,
            "cwd": str(fixture),
            "starting_revision": revision,
            "changed_revision": revision,
            "owned_paths": ["src"],
            "requirements": ["r1"],
            "required_checks": ["c1"],
            "owner": {"provider": "minimax", "model": "M3", "harness": "opencode"},
            "assignment_sha256": "e" * 64,
            "report_sha256": "c" * 64,
            "target_sha256": "a" * 64,
        }

    def assignment(self):
        return {
            "schema_version": "mavis.worker-assignment/v1",
            "assignment_id": "assignment-1",
            "objective_id": "obj-1",
            "requirements": ["r1"],
            "starting_revision": "abcdef0",
            "owner": {"provider": "minimax", "model": "M3", "harness": "opencode"},
            "checkout": {"path": "/tmp/fixture", "owned_paths": ["src"]},
            "allowed_effects": ["edit fixture"],
            "expected_artifacts": ["src/result.txt"],
            "escalate_when": ["same failure repeats"],
            "state": "queued",
        }

    def test_rejects_acceptance_without_receipts_and_independent_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ObjectiveStore(Path(directory))
            store.create(objective())
            store.transition("obj-1", "running", "started")
            store.transition("obj-1", "awaiting verification", "checks complete")
            with self.assertRaisesRegex(ValueError, "evidence receipts"):
                store.transition("obj-1", "accepted", "worker says done")

    def test_same_failure_without_new_evidence_escalates(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ObjectiveStore(Path(directory))
            store.create(objective())
            store.transition("obj-1", "running", "started")
            store.record_attempt("obj-1", "same", ["first.log"], "first repair")
            record = store.record_attempt("obj-1", "same", [], "repeated repair")
            self.assertEqual(record["state"], "escalated")

    def test_worker_cannot_verify_own_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ObjectiveStore(Path(directory))
            store.create(objective())
            store.add_assignment(
                "obj-1",
                self.assignment(),
            )
            with self.assertRaisesRegex(ValueError, "independent"):
                store.add_verification(
                    "obj-1",
                    {
                        "schema_version": "mavis.verifier/v1",
                        "verification_id": "verify-1",
                        "objective_id": "obj-1",
                        "revision": "abcdef0",
                        "requirements": ["r1"],
                        "protected_fixtures": ["/tmp/fixture"],
                        "checks": ["independent inspection"],
                        "required_receipts": ["/tmp/receipt"],
                        "verifier": {"provider": "minimax", "model": "M3", "harness": "opencode"},
                        "verdict": "accepted",
                    },
                )

    def test_model_supplied_verifier_cannot_authorize_accepted_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ObjectiveStore(root)
            store.create(objective())
            store.add_assignment("obj-1", self.assignment())
            fixture = root / "fixture"
            fixture.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=fixture, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=fixture, check=True)
            subprocess.run(["git", "config", "user.name", "Mavis Test"], cwd=fixture, check=True)
            (fixture / "test.txt").write_text("fixture")
            subprocess.run(["git", "add", "test.txt"], cwd=fixture, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture"], cwd=fixture, check=True)
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=fixture, text=True).strip()
            receipt = run_command(
                root,
                "obj-1",
                ["python3", "-c", "print('1 passed')"],
                fixture,
                acceptance_check_ids=["c1"],
            )
            store.add_receipt("obj-1", receipt)
            store.add_verification(
                "obj-1",
                {
                    "schema_version": "mavis.verifier/v1",
                    "verification_id": "verify-1",
                    "objective_id": "obj-1",
                    "revision": revision,
                    "requirements": ["r1"],
                    "protected_fixtures": [str(fixture)],
                    "checks": ["independent inspection"],
                    "required_receipts": [str(receipt.resolve())],
                    "verifier": {"provider": "codex", "model": "gpt-5.6-terra", "harness": "codex"},
                    "verdict": "accepted",
                },
            )
            store.transition("obj-1", "running", "started")
            store.transition("obj-1", "awaiting verification", "checked")
            with self.assertRaisesRegex(ValueError, "Mavis host gateway verification"):
                store.transition("obj-1", "accepted", "forged verifier JSON")

    def test_accepts_host_gateway_receipt_bound_to_terra_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gateway_available = True

            def gateway_reader(job_id):
                if not gateway_available:
                    raise GatewayUnavailable("fixture gateway unavailable")
                return self.gateway_status(job_id, binding)

            store = ObjectiveStore(root, gateway_status_reader=gateway_reader)
            store.create(objective())
            fixture = root / "fixture"
            fixture.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=fixture, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=fixture, check=True)
            subprocess.run(["git", "config", "user.name", "Mavis Test"], cwd=fixture, check=True)
            (fixture / "test.txt").write_text("fixture")
            subprocess.run(["git", "add", "test.txt"], cwd=fixture, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture"], cwd=fixture, check=True)
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=fixture, text=True).strip()
            assignment = self.assignment()
            assignment["starting_revision"] = revision
            assignment["checkout"] = {"path": str(fixture), "owned_paths": ["src"]}
            store.add_assignment("obj-1", assignment)
            binding = self.gateway_binding(fixture, revision)
            receipt = run_command(
                root,
                "obj-1",
                ["python3", "-c", "print('1 passed')"],
                fixture,
                acceptance_check_ids=["c1"],
            )
            store.add_receipt("obj-1", receipt)
            store.record_gateway_verification("obj-1", "worker-job-1")
            store.transition("obj-1", "running", "started")
            store.transition("obj-1", "awaiting verification", "checked")
            gateway_available = False
            with self.assertRaisesRegex(GatewayUnavailable, "unavailable"):
                store.transition("obj-1", "accepted", "gateway unavailable")
            gateway_available = True
            record = store.transition("obj-1", "accepted", "verified")
            self.assertEqual(record["state"], "accepted")

    def test_gateway_receipt_rejects_same_job_as_terra_verifier(self):
        status = self.gateway_status()
        status["acceptance"]["verifier_job_id"] = "worker-job-1"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ObjectiveStore(root, gateway_status_reader=lambda _: status)
            store.create(objective())
            fixture = root / "fixture"
            fixture.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=fixture, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=fixture, check=True)
            subprocess.run(["git", "config", "user.name", "Mavis Test"], cwd=fixture, check=True)
            (fixture / "test.txt").write_text("fixture")
            subprocess.run(["git", "add", "test.txt"], cwd=fixture, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture"], cwd=fixture, check=True)
            receipt = run_command(root, "obj-1", ["python3", "-c", "print('1 passed')"], fixture, acceptance_check_ids=["c1"])
            store.add_receipt("obj-1", receipt)
            with self.assertRaisesRegex(ValueError, "distinct Terra"):
                store.record_gateway_verification("obj-1", "worker-job-1")

    def test_gateway_binding_rejects_an_accepted_job_for_another_objective(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            fixture.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=fixture, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=fixture, check=True)
            subprocess.run(["git", "config", "user.name", "Mavis Test"], cwd=fixture, check=True)
            (fixture / "test.txt").write_text("fixture")
            subprocess.run(["git", "add", "test.txt"], cwd=fixture, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture"], cwd=fixture, check=True)
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=fixture, text=True).strip()
            binding = self.gateway_binding(fixture, revision, objective_id="other-objective")
            store = ObjectiveStore(root, gateway_status_reader=lambda job_id: self.gateway_status(job_id, binding))
            store.create(objective())
            assignment = self.assignment()
            assignment["starting_revision"] = revision
            assignment["checkout"] = {"path": str(fixture), "owned_paths": ["src"]}
            store.add_assignment("obj-1", assignment)
            receipt = run_command(root, "obj-1", ["python3", "-c", "print('1 passed')"], fixture, acceptance_check_ids=["c1"])
            store.add_receipt("obj-1", receipt)
            with self.assertRaisesRegex(ValueError, "different objective"):
                store.record_gateway_verification("obj-1", "worker-job-1")
            binding["objective_id"] = "obj-1"
            other_checkout = root / "other-checkout"
            other_checkout.mkdir()
            binding["cwd"] = str(other_checkout)
            with self.assertRaisesRegex(ValueError, "checkout does not match"):
                store.record_gateway_verification("obj-1", "worker-job-1")
            binding["cwd"] = str(fixture)
            binding["changed_revision"] = "0" * 40
            with self.assertRaisesRegex(ValueError, "revision does not match host evidence"):
                store.record_gateway_verification("obj-1", "worker-job-1")
            binding["changed_revision"] = revision
            binding["owner"] = {"provider": "terra", "model": "gpt-5.6-terra", "harness": "codex"}
            with self.assertRaisesRegex(ValueError, "owner does not match"):
                store.record_gateway_verification("obj-1", "worker-job-1")

    def test_rejects_fabricated_minimal_receipt_outside_host_evidence_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ObjectiveStore(root)
            store.create(objective())
            forged = root / "receipt.json"
            forged.write_text('{"schema_version":"mavis.evidence-receipt/v1","verdict":"pass"}')
            with self.assertRaisesRegex(ValueError, "host-recorded"):
                store.add_receipt("obj-1", forged)

    def test_detects_raw_output_tampering_before_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ObjectiveStore(root)
            store.create(objective())
            fixture = root / "fixture"
            fixture.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=fixture, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=fixture, check=True)
            subprocess.run(["git", "config", "user.name", "Mavis Test"], cwd=fixture, check=True)
            (fixture / "x").write_text("x")
            subprocess.run(["git", "add", "x"], cwd=fixture, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture"], cwd=fixture, check=True)
            receipt = run_command(root, "obj-1", ["python3", "-c", "print('1 passed')"], fixture, acceptance_check_ids=["c1"])
            store.add_receipt("obj-1", receipt)
            (receipt.parent / "stdout.log").write_text("fabricated 99 passed")
            with self.assertRaisesRegex(ValueError, "raw output"):
                store._assert_acceptance(store.load("obj-1"))

    def test_rejects_receipt_that_calls_failed_command_a_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ObjectiveStore(root)
            store.create(objective())
            fixture = root / "fixture"
            fixture.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=fixture, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=fixture, check=True)
            subprocess.run(["git", "config", "user.name", "Mavis Test"], cwd=fixture, check=True)
            (fixture / "x").write_text("x")
            subprocess.run(["git", "add", "x"], cwd=fixture, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture"], cwd=fixture, check=True)
            receipt_path = run_command(
                root, "obj-1", ["python3", "-c", "print('FAILED'); raise SystemExit(1)"],
                fixture, acceptance_check_ids=["c1"],
            )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["verdict"] = "pass"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "verdict does not match"):
                store.add_receipt("obj-1", receipt_path)


if __name__ == "__main__":
    unittest.main()
