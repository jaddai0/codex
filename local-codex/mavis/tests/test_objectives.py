from pathlib import Path
import json
import subprocess
import tempfile
import unittest

from mavis.evidence import run_command
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

    def test_accepts_matching_receipt_and_independent_verification(self):
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
            record = store.transition("obj-1", "accepted", "verified")
            self.assertEqual(record["state"], "accepted")

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
