"""Jev advice is retained without becoming an acceptance authority."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mavis.jev import advise, normalize_context, _digest
from mavis.gateway import mavis_jev_decisions


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

    def test_context_is_canonical_and_only_its_digest_is_retained(self):
        gateway = {"success": False, "decision": "cap_blocked", "advisory": True,
                   "binding": False, "grants_permission": False}
        context = {"version": "mavis-jev-context/v1",
                   "facts": ["source_changed", "check_failed"],
                   "required_tool": "native_ui", "offered_tool": "http"}
        expected = {**context, "facts": ["check_failed", "source_changed"]}
        with patch("mavis.jev.mavis_jev_decisions", return_value=gateway) as call:
            result = advise(self.service, self.project, "escalation", SIGNALS, 0.01,
                            context=context)
        call.assert_called_once_with("escalation", SIGNALS, 0.01, context=expected)
        retained = json.loads(Path(result["receipt"]).read_text())
        self.assertEqual(retained["context_version"], "mavis-jev-context/v1")
        self.assertEqual(retained["context_sha256"], _digest(expected))
        self.assertNotIn("context", retained)
        self.assertNotIn("source_changed", Path(result["receipt"]).read_text())

    def test_invalid_context_rejected_before_gateway(self):
        valid = {"version": "mavis-jev-context/v1", "facts": ["check_failed"]}
        invalid = [
            {**valid, "summary": "Jane Doe /private/file"},
            {**valid, "facts": ["/private/file"]},
            {**valid, "facts": ["check_failed", "check_failed"]},
            {**valid, "required_tool": "http"},
            {**valid, "required_tool": "http", "offered_tool": {"name": "curl"}},
            {**valid, "facts": ["x" * 600]},
            {**valid, "version": "future"},
        ]
        with patch("mavis.jev.mavis_jev_decisions") as call:
            for context in invalid:
                with self.subTest(context=context), self.assertRaises(ValueError):
                    advise(self.service, self.project, "escalation", SIGNALS, 0.01,
                           context=context)
        call.assert_not_called()
        self.assertFalse((self.service / "jev").exists())

    def test_gateway_wrapper_keeps_old_wire_shape_and_positional_timeout(self):
        with patch("mavis.gateway._gateway_tool", return_value={}) as call:
            mavis_jev_decisions("escalation", SIGNALS, 0.01, 7.0)
            call.assert_called_once_with("mavis_jev_decisions", {
                "purpose": "escalation", "signals": SIGNALS,
                "estimated_cost_usd": 0.01,
            }, 7.0, require_success=False)
            call.reset_mock()
            context = normalize_context({"version": "mavis-jev-context/v1",
                                         "facts": ["check_failed"]})
            mavis_jev_decisions("escalation", SIGNALS, 0.01, 7.0, context=context)
            call.assert_called_once_with("mavis_jev_decisions", {
                "purpose": "escalation", "signals": SIGNALS,
                "estimated_cost_usd": 0.01, "context": context,
            }, 7.0, require_success=False)


if __name__ == "__main__":
    unittest.main()
