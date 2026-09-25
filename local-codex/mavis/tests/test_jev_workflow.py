"""A real objective event uses Jev advice without granting it authority."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from mavis.cli import main
from mavis.evidence import run_command
from mavis.gateway import GatewayUnavailable
from mavis.jev import advise
from mavis.objectives import ObjectiveStore


def objective():
    return {
        "schema_version": "mavis.objective/v1",
        "objective_id": "jev-workflow",
        "blueprint": "keep the host acceptance rule",
        "requirements": [{"id": "r1", "text": "repair the failure"}],
        "dependencies": [],
        "scope": {"paths": ["src"]},
        "acceptance_checks": [{"id": "c1", "command": ["true"]}],
        "unresolved_decisions": [],
    }


class JevWorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="mavis-jev-workflow-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.home = self.root / "service"
        self.home.mkdir()
        payload = self.root / "objective.json"
        payload.write_text(json.dumps(objective()))
        self.environment = patch.dict(os.environ, {"MAVIS_HOME": str(self.home)}, clear=False)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.call("objective", "create", str(payload))
        self.call("objective", "transition", "jev-workflow", "running", "--reason", "start")
        self.fixture = self.root / "fixture"
        self.fixture.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=self.fixture, check=True)
        subprocess.run(["git", "-c", "user.name=Mavis Test",
                        "-c", "user.email=mavis@example.invalid", "commit", "--allow-empty",
                        "-qm", "seed"], cwd=self.fixture, check=True)
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                           cwd=self.fixture, text=True).strip()
        self.store = ObjectiveStore(self.home)
        self.store.add_assignment("jev-workflow", {
            "schema_version": "mavis.worker-assignment/v1",
            "assignment_id": "repair-1", "objective_id": "jev-workflow",
            "requirements": ["r1"], "starting_revision": revision,
            "owner": {"provider": "minimax", "model": "M3", "harness": "opencode"},
            "checkout": {"path": str(self.fixture), "owned_paths": ["src"]},
            "allowed_effects": ["edit fixture"], "expected_artifacts": ["src/result.txt"],
            "escalate_when": ["same failure repeats"], "state": "running",
        })
        receipt = run_command(self.home, "jev-workflow",
                              ["python3", "-c", "print('FAILED'); raise SystemExit(1)"],
                              self.fixture, acceptance_check_ids=["c1"])
        self.store.add_receipt("jev-workflow", receipt)

    def call(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(list(args)), 0)
        return json.loads(output.getvalue())

    def repeat_failure(self):
        first = self.call("objective", "attempt", "jev-workflow", "secret fingerprint",
                          "first repair", "--evidence", "first.log")
        self.assertEqual(first["state"], "running")
        return self.call("objective", "attempt", "jev-workflow", "secret fingerprint",
                         "same repair")

    def linked_event(self, record):
        self.assertEqual(len(record["jev_advice_receipts"]), 1)
        retained = record["jev_advice_receipts"][0]
        path = Path(retained["path"])
        self.assertTrue(path.is_file())
        event = json.loads(path.read_text())
        self.assertEqual(event["objective_id"], "jev-workflow")
        self.assertEqual(event["attempt_number"], 2)
        self.assertEqual(event["deterministic_disposition"], "escalated")
        self.assertNotIn("secret fingerprint", path.read_text())
        return event

    def test_contrary_answer_cannot_cancel_host_escalation(self):
        answer = {
            "success": True, "decision": "answered", "advisory": True,
            "binding": False, "grants_permission": False,
            "answers": {"decision": {"type": "noul", "noul": 0.0}},
        }
        with patch("mavis.jev.mavis_jev_decisions", return_value=answer) as gateway:
            record = self.repeat_failure()
        self.assertEqual(record["state"], "escalated")
        self.assertEqual(record["jev_latest_advice"], {
            "event_id": record["jev_advice_events"][0]["event_id"],
            "attempt_number": 2, "status": "answered",
            "consider_escalation_probability": 0.0,
            "deterministic_disposition": "escalated",
        })
        gateway.assert_called_once_with("escalation", {
            "retry_count": 2, "unresolved_count": 0, "verifier_accepted": False,
        }, 0.01)
        event = self.linked_event(record)
        self.assertEqual(event["outcome"]["status"], "answered")
        self.assertTrue(Path(event["outcome"]["advice_receipt"]).is_file())
        with self.assertRaisesRegex(ValueError, "invalid objective transition"):
            self.call("objective", "transition", "jev-workflow", "accepted",
                      "--reason", "Jev says yes")

    def test_refusal_keeps_escalation_and_auditable_receipt(self):
        refusal = {"success": False, "decision": "cap_blocked", "advisory": True,
                   "binding": False, "grants_permission": False}
        with patch("mavis.jev.mavis_jev_decisions", return_value=refusal):
            record = self.repeat_failure()
        self.assertEqual(record["state"], "escalated")
        self.assertEqual(self.linked_event(record)["outcome"]["status"], "refused")

    def test_transport_failure_does_not_change_host_rule(self):
        with patch("mavis.jev.mavis_jev_decisions",
                   side_effect=GatewayUnavailable("offline")):
            record = self.repeat_failure()
        self.assertEqual(record["state"], "escalated")
        self.assertEqual(self.linked_event(record)["outcome"],
                         {"status": "blocked", "reason": "GatewayUnavailable"})

    def test_malformed_answer_does_not_change_host_rule(self):
        invalid = {"success": True, "decision": "answered", "advisory": True,
                   "binding": True, "grants_permission": True, "answers": {}}
        with patch("mavis.jev.mavis_jev_decisions", return_value=invalid):
            record = self.repeat_failure()
        self.assertEqual(record["state"], "escalated")
        self.assertEqual(self.linked_event(record)["outcome"],
                         {"status": "blocked", "reason": "ValueError"})

    def test_no_assignment_or_host_failure_does_not_dispatch(self):
        record = self.store.load("jev-workflow")
        record["assignments"] = []
        self.store.save(record)
        with patch("mavis.jev.mavis_jev_decisions") as gateway:
            result = self.repeat_failure()
        gateway.assert_not_called()
        self.assertEqual(result["state"], "escalated")
        self.assertEqual(result["jev_latest_advice"]["reason"],
                         "no_worker_assignment")
        self.assertEqual(result["jev_advice_events"], [])

    def test_prior_assignment_failure_cannot_authorize_new_paid_advice(self):
        previous = self.store.load("jev-workflow")["assignments"][-1].copy()
        previous["assignment_id"] = "repair-2"
        self.store.add_assignment("jev-workflow", previous)
        with patch("mavis.jev.mavis_jev_decisions") as gateway:
            result = self.repeat_failure()
        gateway.assert_not_called()
        self.assertEqual(result["state"], "escalated")
        self.assertEqual(result["jev_latest_advice"]["reason"],
                         "no_failed_host_check")

    def test_accepted_objective_cannot_be_reopened_by_attempt(self):
        record = self.store.load("jev-workflow")
        record["state"] = "accepted"
        self.store.save(record)
        with patch("mavis.jev.mavis_jev_decisions") as gateway:
            with self.assertRaisesRegex(ValueError, "closed objectives"):
                self.call("objective", "attempt", "jev-workflow", "failure", "repair")
        gateway.assert_not_called()
        self.assertEqual(self.store.load("jev-workflow")["state"], "accepted")

    def test_concurrent_attempt_cannot_duplicate_or_make_late_advice_current(self):
        started = threading.Event()
        release = threading.Event()
        results = []

        def delayed_advice(*args, **kwargs):
            started.set()
            if not release.wait(5):
                raise RuntimeError("test advice wait expired")
            return advise(*args, **kwargs)

        store = ObjectiveStore(self.home, advice_hook=delayed_advice)
        store.record_attempt("jev-workflow", "same", ["first.log"], "first repair")
        answer = {"success": True, "decision": "answered", "advisory": True,
                  "binding": False, "grants_permission": False,
                  "answers": {"decision": {"type": "noul", "noul": 0.9}}}
        with patch("mavis.jev.mavis_jev_decisions", return_value=answer) as gateway:
            worker = threading.Thread(target=lambda: results.append(
                store.record_attempt("jev-workflow", "same", [], "second repair")))
            worker.start()
            self.assertTrue(started.wait(5))
            pending = store.load("jev-workflow")
            self.assertEqual(pending["jev_advice_events"][0]["status"], "pending")
            store.transition("jev-workflow", "running", "resume after escalation")
            store.record_attempt("jev-workflow", "different", ["new evidence"],
                                 "third repair")
            release.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(results), 1)
        gateway.assert_called_once()
        latest = store.load("jev-workflow")
        self.assertEqual(len(latest["attempts"]), 3)
        self.assertEqual(latest["state"], "running")
        self.assertEqual(latest["jev_latest_advice"]["status"], "superseded")
        self.assertEqual(self.linked_event(latest)["applicability"], "superseded")

    def test_unrelated_receipt_inside_service_is_rejected(self):
        answer = {"success": True, "decision": "answered", "advisory": True,
                  "binding": False, "grants_permission": False,
                  "answers": {"decision": {"type": "noul", "noul": 0.9}}}

        def wrong_event(*args, **kwargs):
            return advise(*args, **{**kwargs, "event_id": "other-event"})

        store = ObjectiveStore(self.home, advice_hook=wrong_event)
        with patch("mavis.jev.mavis_jev_decisions", return_value=answer):
            store.record_attempt("jev-workflow", "same", ["first.log"], "first repair")
            record = store.record_attempt("jev-workflow", "same", [], "second repair")
        self.assertEqual(record["state"], "escalated")
        self.assertEqual(self.linked_event(record)["outcome"],
                         {"status": "blocked", "reason": "ValueError"})

    def test_interrupted_dispatch_leaves_one_visible_pending_intent(self):
        def interrupted(*_args, **_kwargs):
            raise KeyboardInterrupt()

        store = ObjectiveStore(self.home, advice_hook=interrupted)
        store.record_attempt("jev-workflow", "same", ["first.log"], "first repair")
        with self.assertRaises(KeyboardInterrupt):
            store.record_attempt("jev-workflow", "same", [], "second repair")
        pending = store.load("jev-workflow")
        self.assertEqual(pending["state"], "escalated")
        self.assertEqual(len(pending["jev_advice_events"]), 1)
        self.assertEqual(pending["jev_advice_events"][0]["status"], "pending")
        self.assertEqual(pending["jev_latest_advice"]["status"], "pending")
        with self.assertRaisesRegex(ValueError, "invalid objective transition"):
            store.transition("jev-workflow", "accepted", "unverified")
        pending["jev_advice_events"][0]["created_at"] = "2000-01-01T00:00:00+00:00"
        store.save(pending)
        settled = self.call("objective", "jev-reconcile", "jev-workflow")
        self.assertEqual(settled["jev_advice_events"][0]["status"], "unknown")
        self.assertEqual(settled["jev_latest_advice"]["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
