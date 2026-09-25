"""Jev advice is retained without becoming an acceptance authority."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mavis.jev import advise


QUESTIONS = {
    "escalate": {
        "type": "noul",
        "instructions": "Does this failure need escalation?",
        "criteria": {"true": "yes", "false": "no"},
    }
}


class JevAdviceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mavis-jev-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.service = self.root / "service"
        self.project = self.root / "project"
        self.service.mkdir()
        self.project.mkdir()

    def test_answered_advice_has_local_receipt_and_no_authority(self):
        state = {"failure": "tests still fail after a changed approach"}
        gateway = {
            "success": True, "decision": "answered", "advisory": True,
            "binding": False, "grants_permission": False,
            "answers": {"escalate": {"type": "noul", "noul": 0.93}},
            "actual_cost_usd": 0.00002,
        }
        with patch("mavis.jev.jev_decisions", return_value=gateway) as call:
            result = advise(self.service, self.project, "escalation", state,
                            QUESTIONS, 0.01)
        call.assert_called_once_with(state, QUESTIONS, 0.01)
        self.assertEqual(result["decision"], "answered")
        self.assertFalse(result["binding"])
        receipt = Path(result["receipt"])
        record = json.loads(receipt.read_text())
        self.assertEqual(record["gateway_response"], gateway)
        self.assertNotIn(state["failure"], receipt.read_text())
        self.assertEqual(receipt.stat().st_mode & 0o777, 0o600)

    def test_gateway_cap_refusal_is_recorded_without_answers(self):
        gateway = {
            "success": False, "decision": "cap_blocked", "advisory": True,
            "binding": False, "grants_permission": False,
        }
        with patch("mavis.jev.jev_decisions", return_value=gateway):
            result = advise(self.service, self.project, "tool_selection", {},
                            QUESTIONS, 0.01)
        self.assertEqual(result["decision"], "cap_blocked")
        self.assertEqual(result["answers"], {})
        self.assertTrue(Path(result["receipt"]).is_file())

    def test_claim_of_permission_is_rejected_without_receipt(self):
        gateway = {
            "success": True, "decision": "answered", "advisory": True,
            "binding": True, "grants_permission": True,
            "answers": {"escalate": {"type": "noul", "noul": 0.5}},
        }
        with patch("mavis.jev.jev_decisions", return_value=gateway):
            with self.assertRaisesRegex(ValueError, "claimed authority"):
                advise(self.service, self.project, "escalation", {}, QUESTIONS, 0.01)
        self.assertFalse((self.service / "jev").exists())

    def test_unbounded_cost_is_rejected_before_gateway(self):
        with patch("mavis.jev.jev_decisions") as call:
            with self.assertRaisesRegex(ValueError, "estimated cost"):
                advise(self.service, self.project, "escalation", {}, QUESTIONS, 2)
        call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
