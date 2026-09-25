"""Jev advice is retained without becoming an acceptance authority."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mavis.jev import advise


SIGNALS = {"failed_checks": 2, "retry_count": 1}


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
        gateway = {
            "success": True, "decision": "answered", "advisory": True,
            "binding": False, "grants_permission": False,
            "answers": {"decision": {"type": "noul", "noul": 0.93}},
            "actual_cost_usd": 0.00002,
        }
        with patch("mavis.jev.mavis_jev_decisions", return_value=gateway) as call:
            result = advise(self.service, self.project, "escalation", SIGNALS, 0.01)
        call.assert_called_once_with("escalation", SIGNALS, 0.01)
        self.assertEqual(result["decision"], "answered")
        self.assertFalse(result["binding"])
        receipt = Path(result["receipt"])
        record = json.loads(receipt.read_text())
        self.assertEqual(record["gateway_response"], gateway)
        self.assertNotIn("failed_checks", receipt.read_text())
        self.assertEqual(receipt.stat().st_mode & 0o777, 0o600)

    def test_gateway_cap_refusal_is_recorded_without_answers(self):
        gateway = {
            "success": False, "decision": "cap_blocked", "advisory": True,
            "binding": False, "grants_permission": False,
        }
        with patch("mavis.jev.mavis_jev_decisions", return_value=gateway):
            result = advise(self.service, self.project, "tool_selection", {"tool_available": False}, 0.01)
        self.assertEqual(result["decision"], "cap_blocked")
        self.assertEqual(result["answers"], {})
        self.assertTrue(Path(result["receipt"]).is_file())

    def test_claim_of_permission_is_rejected_without_receipt(self):
        gateway = {
            "success": True, "decision": "answered", "advisory": True,
            "binding": True, "grants_permission": True,
            "answers": {"decision": {"type": "noul", "noul": 0.5}},
        }
        with patch("mavis.jev.mavis_jev_decisions", return_value=gateway):
            with self.assertRaisesRegex(ValueError, "claimed authority"):
                advise(self.service, self.project, "escalation", SIGNALS, 0.01)
        self.assertFalse((self.service / "jev").exists())

    def test_unbounded_cost_is_rejected_before_gateway(self):
        with patch("mavis.jev.mavis_jev_decisions") as call:
            with self.assertRaisesRegex(ValueError, "estimated cost"):
                advise(self.service, self.project, "escalation", SIGNALS, 2)
        call.assert_not_called()

    def test_state_rejects_arbitrary_project_data_before_gateway(self):
        with patch("mavis.jev.mavis_jev_decisions") as call:
            with self.assertRaisesRegex(ValueError, "unknown field"):
                advise(self.service, self.project, "escalation",
                       {"client_secret": "short"}, 0.01)
            with self.assertRaisesRegex(ValueError, "bounded counts"):
                advise(self.service, self.project, "escalation",
                       {"failed_checks": "Jane Doe, 123 Main Street"}, 0.01)
        call.assert_not_called()

    def test_receipt_discards_provider_detail(self):
        gateway = {
            "success": False, "decision": "unavailable", "failure_class": "transport",
            "detail": "remote body that must not be repeated locally",
        }
        with patch("mavis.jev.mavis_jev_decisions", return_value=gateway):
            result = advise(self.service, self.project, "escalation", SIGNALS, 0.01)
        self.assertNotIn("remote body", Path(result["receipt"]).read_text())


if __name__ == "__main__":
    unittest.main()
