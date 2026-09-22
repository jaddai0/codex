"""Durable objective state with bounded repair and independent acceptance."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .storage import read_json, require_safe_id, sha256_file, write_json


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
        _validate_assignment(assignment, objective_id)
        record = self.load(objective_id)
        record["assignments"].append(deepcopy(assignment))
        self.save(record)
        return record

    def add_receipt(self, objective_id: str, receipt_path: Path) -> dict[str, Any]:
        record = self.load(objective_id)
        resolved = Path(receipt_path).resolve()
        evidence_root = (self.root.parent / "evidence" / objective_id).resolve()
        if evidence_root not in resolved.parents:
            raise ValueError("evidence receipt must be host-recorded under the objective evidence root")
        receipt = read_json(resolved)
        _validate_receipt(receipt, objective_id, resolved)
        record["evidence_receipts"].append(
            {"path": str(resolved), "sha256": sha256_file(resolved)}
        )
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
        _validate_verification(verification, objective_id)
        worker_owners = {_owner_identity(item.get("owner", {})) for item in record["assignments"]}
        verifier = _owner_identity(verification.get("verifier", {}))
        if verifier in worker_owners:
            raise ValueError("verifier must be independent from implementation owners")
        record["verifications"].append(deepcopy(verification))
        self.save(record)
        return record

    @staticmethod
    def _assert_acceptance(record: dict[str, Any]) -> None:
        receipts = []
        for retained in record["evidence_receipts"]:
            if not isinstance(retained, dict) or set(retained) != {"path", "sha256"}:
                raise ValueError("evidence receipt retention record is invalid")
            path = Path(retained["path"])
            if sha256_file(path) != retained["sha256"]:
                raise ValueError("retained evidence receipt hash changed after recording")
            receipt = read_json(path)
            _validate_receipt(receipt, str(record["objective_id"]), path)
            receipts.append(receipt)
        if not receipts or any(item.get("verdict") != "pass" for item in receipts):
            raise ValueError("all retained evidence receipts must pass before acceptance")
        verifications = record.get("verifications") or []
        if not verifications or verifications[-1].get("verdict") != "accepted":
            raise ValueError("independent verification must accept the current milestone")
        verification = verifications[-1]
        _validate_verification(verification, str(record["objective_id"]))
        expected_revision = verification["revision"]
        if any(receipt.get("changed_revision") != expected_revision for receipt in receipts):
            raise ValueError("verification revision does not match evidence receipts")
        required_checks = {
            str(check.get("id")) for check in record["acceptance_checks"] if check.get("id")
        }
        covered_checks = {
            check_id for receipt in receipts for check_id in receipt["acceptance_check_ids"]
        }
        if not required_checks or not required_checks.issubset(covered_checks):
            raise ValueError("retained evidence does not cover every acceptance check")
        retained_paths = {item["path"] for item in record["evidence_receipts"]}
        if set(verification["required_receipts"]) != retained_paths:
            raise ValueError("verification is not bound to the exact retained receipts")
        required_requirements = {str(item["id"]) for item in record["requirements"]}
        if set(verification["requirements"]) != required_requirements:
            raise ValueError("verification does not cover every objective requirement")


def _require_keys(payload: dict[str, Any], required: set[str], label: str) -> None:
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{label} missing required fields: {', '.join(missing)}")


def _owner_identity(owner: object) -> tuple[str, str, str]:
    if not isinstance(owner, dict):
        raise ValueError("owner identity must be an object")
    identity = tuple(str(owner.get(key) or "") for key in ("provider", "model", "harness"))
    if any(not value for value in identity):
        raise ValueError("owner identity requires provider, model, and harness")
    return identity


def _validate_assignment(assignment: dict[str, Any], objective_id: str) -> None:
    required = {
        "schema_version", "assignment_id", "objective_id", "requirements",
        "starting_revision", "owner", "checkout", "allowed_effects",
        "expected_artifacts", "escalate_when", "state",
    }
    _require_keys(assignment, required, "worker assignment")
    if assignment["schema_version"] != "mavis.worker-assignment/v1":
        raise ValueError("unsupported worker assignment schema")
    if assignment["objective_id"] != objective_id:
        raise ValueError("worker assignment objective does not match")
    require_safe_id(str(assignment["assignment_id"]), "assignment id")
    _owner_identity(assignment["owner"])
    if not assignment["requirements"] or not assignment["expected_artifacts"] or not assignment["escalate_when"]:
        raise ValueError("assignment lists must not be empty")
    checkout = assignment["checkout"]
    if not isinstance(checkout, dict) or not checkout.get("path") or not checkout.get("owned_paths"):
        raise ValueError("assignment checkout must bind a path and owned paths")


def _validate_receipt(receipt: dict[str, Any], objective_id: str, receipt_path: Path) -> None:
    required = {
        "schema_version", "receipt_id", "objective_id", "command", "cwd",
        "exit_status", "started_at", "finished_at", "raw_output",
        "changed_revision", "artifact_hashes", "acceptance_check_ids",
        "producer", "verdict",
    }
    _require_keys(receipt, required, "evidence receipt")
    if receipt["schema_version"] != "mavis.evidence-receipt/v1" or receipt["producer"] != "mavis-host-command/v1":
        raise ValueError("unsupported evidence receipt producer or schema")
    if receipt["objective_id"] != objective_id:
        raise ValueError("evidence receipt objective does not match")
    if not isinstance(receipt["command"], list) or not receipt["command"]:
        raise ValueError("evidence receipt command is missing")
    if not isinstance(receipt["exit_status"], int):
        raise ValueError("evidence receipt exit status is invalid")
    if not isinstance(receipt["changed_revision"], str) or len(receipt["changed_revision"]) < 7:
        raise ValueError("evidence receipt must bind a git revision")
    if not receipt["acceptance_check_ids"]:
        raise ValueError("evidence receipt must cover an acceptance check")
    raw = receipt["raw_output"]
    evidence_dir = receipt_path.parent.resolve()
    if not isinstance(raw, dict) or Path(str(raw.get("path"))).resolve() != evidence_dir:
        raise ValueError("raw output path is not bound to the receipt directory")
    stdout_path, stderr_path = evidence_dir / "stdout.log", evidence_dir / "stderr.log"
    if not stdout_path.is_file() or not stderr_path.is_file():
        raise ValueError("raw output files are missing")
    actual_hash = sha256_file(stdout_path) + ":" + sha256_file(stderr_path)
    actual_bytes = stdout_path.stat().st_size + stderr_path.stat().st_size
    if raw.get("sha256") != actual_hash or raw.get("bytes") != actual_bytes:
        raise ValueError("raw output hash or byte count does not match")
    for artifact, expected_hash in receipt["artifact_hashes"].items():
        path = Path(artifact)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"artifact hash does not match: {artifact}")


def _validate_verification(verification: dict[str, Any], objective_id: str) -> None:
    required = {
        "schema_version", "verification_id", "objective_id", "revision",
        "requirements", "protected_fixtures", "checks", "required_receipts",
        "verifier", "verdict",
    }
    _require_keys(verification, required, "verification")
    if verification["schema_version"] != "mavis.verifier/v1":
        raise ValueError("unsupported verification schema")
    if verification["objective_id"] != objective_id:
        raise ValueError("verification objective does not match")
    require_safe_id(str(verification["verification_id"]), "verification id")
    _owner_identity(verification["verifier"])
    if not isinstance(verification["revision"], str) or len(verification["revision"]) < 7:
        raise ValueError("verification must bind a git revision")
    if verification["verdict"] not in {"accepted", "rejected"}:
        raise ValueError("verification verdict must be accepted or rejected")
    for key in ("requirements", "checks", "required_receipts"):
        if not isinstance(verification[key], list) or not verification[key]:
            raise ValueError(f"verification {key} must not be empty")
