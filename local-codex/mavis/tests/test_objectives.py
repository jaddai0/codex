import json
from pathlib import Path
import tempfile
import unittest

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
                {
                    "schema_version": "mavis.worker-assignment/v1",
                    "owner": {"provider": "minimax"},
                    "requirements": ["r1"],
                },
            )
            with self.assertRaisesRegex(ValueError, "independent"):
                store.add_verification(
                    "obj-1",
                    {"verifier": {"provider": "minimax"}, "verdict": "accepted"},
                )

    def test_accepts_matching_receipt_and_independent_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ObjectiveStore(root)
            store.create(objective())
            store.add_assignment(
                "obj-1",
                {"schema_version": "mavis.worker-assignment/v1", "owner": {"provider": "minimax"}, "requirements": ["r1"]},
            )
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps({"schema_version": "mavis.evidence-receipt/v1", "verdict": "pass", "changed_revision": "abcdef0"}))
            store.add_receipt("obj-1", receipt)
            store.add_verification("obj-1", {"verifier": {"provider": "terra"}, "verdict": "accepted", "revision": "abcdef0"})
            store.transition("obj-1", "running", "started")
            store.transition("obj-1", "awaiting verification", "checked")
            record = store.transition("obj-1", "accepted", "verified")
            self.assertEqual(record["state"], "accepted")


if __name__ == "__main__":
    unittest.main()
