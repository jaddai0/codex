"""Durable objective state with bounded repair and independent acceptance."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .storage import read_json, require_safe_id, write_json


STATES = {
    "queued",
    "running",
    "awaiting verification",
    "accepted",
    "needs repair",
    "escalated",
    "blocked",
    "cancelled",
}
TRANSITIONS = {
    "queued": {"running", "cancelled", "blocked"},
    "running": {"awaiting verification", "needs repair", "escalated", "blocked", "cancelled"},
    "awaiting verification": {"accepted", "needs repair", "blocked", "cancelled"},
    "needs repair": {"running", "escalated", "blocked", "cancelled"},
    "escalated": {"running", "blocked", "cancelled"},
    "blocked": {"running", "cancelled"},
    "accepted": set(),
    "cancelled": set(),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ObjectiveStore:
    def __init__(self, home: Path):
        self.root = Path(home) / "objectives"

    def _path(self, objective_id: str) -> Path:
        return self.root / f"{require_safe_id(objective_id, 'objective id')}.json"

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("schema_version") != "mavis.objective/v1":
            raise ValueError("unsupported objective schema")
        objective_id = require_safe_id(str(payload.get("objective_id") or ""), "objective id")
        path = self._path(objective_id)
        if path.exists():
            raise FileExistsError(path)
        if not payload.get("requirements") or not payload.get("acceptance_checks"):
            raise ValueError("objective requires requirements and acceptance checks")
        record = deepcopy(payload)
        record["state"] = "queued"
        record["created_at"] = _now()
        record["updated_at"] = record["created_at"]
        record["attempts"] = []
        record["assignments"] = []
        record["evidence_receipts"] = []
        record["verifications"] = []
        write_json(path, record)
        return record

    def load(self, objective_id: str) -> dict[str, Any]:
        return read_json(self._path(objective_id))

    def save(self, record: dict[str, Any]) -> None:
        record["updated_at"] = _now()
        write_json(self._path(str(record["objective_id"])), record)

    def transition(self, objective_id: str, state: str, reason: str) -> dict[str, Any]:
        if state not in STATES:
            raise ValueError(f"unknown objective state: {state}")
        record = self.load(objective_id)
        current = str(record["state"])
        if state not in TRANSITIONS[current]:
            raise ValueError(f"invalid objective transition: {current} -> {state}")
        if state == "accepted":
            self._assert_acceptance(record)
        record["state"] = state
        record.setdefault("state_history", []).append(
            {"from": current, "to": state, "reason": reason, "at": _now()}
        )
        self.save(record)
        return record

    def add_assignment(self, objective_id: str, assignment: dict[str, Any]) -> dict[str, Any]:
        if assignment.get("schema_version") != "mavis.worker-assignment/v1":
            raise ValueError("unsupported worker assignment schema")
        if not assignment.get("owner") or not assignment.get("requirements"):
            raise ValueError("assignment must record owner and requirements")
        record = self.load(objective_id)
        record["assignments"].append(deepcopy(assignment))
        self.save(record)
        return record

    def add_receipt(self, objective_id: str, receipt_path: Path) -> dict[str, Any]:
        record = self.load(objective_id)
        receipt = read_json(receipt_path)
        if receipt.get("schema_version") != "mavis.evidence-receipt/v1":
            raise ValueError("unsupported evidence receipt schema")
        record["evidence_receipts"].append(str(Path(receipt_path).resolve()))
        self.save(record)
        return record

    def record_attempt(
        self,
        objective_id: str,
        failure_fingerprint: str,
        evidence: list[str],
        approach: str,
    ) -> dict[str, Any]:
        record = self.load(objective_id)
        attempts = record["attempts"]
        attempts.append(
            {
                "failure_fingerprint": failure_fingerprint,
                "evidence": evidence,
                "approach": approach,
                "at": _now(),
            }
        )
        if len(attempts) >= 2:
            previous = attempts[-2]
            if previous["failure_fingerprint"] == failure_fingerprint and not evidence:
                record["state"] = "escalated"
                record["escalation_reason"] = "same failure repeated without new evidence"
        self.save(record)
        return record

    def add_verification(self, objective_id: str, verification: dict[str, Any]) -> dict[str, Any]:
        record = self.load(objective_id)
        worker_owners = {
            str(item.get("owner", {}).get("provider")) for item in record["assignments"]
        }
        verifier = str(verification.get("verifier", {}).get("provider") or "")
        if not verifier or verifier in worker_owners:
            raise ValueError("verifier must be independent from implementation owners")
        record["verifications"].append(deepcopy(verification))
        self.save(record)
        return record

    @staticmethod
    def _assert_acceptance(record: dict[str, Any]) -> None:
        receipts = [read_json(Path(path)) for path in record["evidence_receipts"]]
        if not receipts or any(item.get("verdict") != "pass" for item in receipts):
            raise ValueError("all retained evidence receipts must pass before acceptance")
        verifications = record.get("verifications") or []
        if not verifications or verifications[-1].get("verdict") != "accepted":
            raise ValueError("independent verification must accept the current milestone")
        expected_revision = verifications[-1].get("revision")
        if expected_revision and any(
            receipt.get("changed_revision") not in {None, expected_revision}
            for receipt in receipts
        ):
            raise ValueError("verification revision does not match evidence receipts")
