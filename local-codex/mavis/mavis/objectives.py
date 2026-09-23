"""Durable objective state with bounded repair and independent acceptance."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .evidence import parse_test_output
from .gateway import GatewayUnavailable, harness_job_status
from .storage import profile_boundary_lock, read_json, require_safe_id, sha256_file, write_json


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
    "running": {
        "awaiting verification",
        "needs repair",
        "escalated",
        "blocked",
        "cancelled",
    },
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
    def __init__(
        self,
        home: Path,
        gateway_status_reader: Callable[[str], dict[str, Any]] | None = None,
    ):
        self.home = Path(home)
        self.root = self.home / "objectives"
        self.gateway_status_reader = gateway_status_reader or harness_job_status

    def _path(self, objective_id: str) -> Path:
        return self.root / f"{require_safe_id(objective_id, 'objective id')}.json"

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("schema_version") != "mavis.objective/v1":
            raise ValueError("unsupported objective schema")
        objective_id = require_safe_id(
            str(payload.get("objective_id") or ""), "objective id"
        )
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
        record["gateway_verifications"] = []
        write_json(path, record)
        return record

    def load(self, objective_id: str) -> dict[str, Any]:
        return read_json(self._path(objective_id))

    def bind_session(self, objective_id: str, session_id: str) -> Path:
        """Bind one Codex session to an existing objective for compaction handoffs."""
        self.load(objective_id)
        path = (
            self.root.parent
            / "objective_sessions"
            / f"{require_safe_id(session_id, 'session id')}.json"
        )
        write_json(
            path,
            {
                "schema_version": "mavis.objective-session/v1",
                "objective_id": objective_id,
                "session_id": session_id,
            },
        )
        return path

    def handoff_snapshot(self, session_id: str) -> dict[str, Any] | None:
        path = (
            self.root.parent
            / "objective_sessions"
            / f"{require_safe_id(session_id, 'session id')}.json"
        )
        if not path.is_file():
            return None
        binding = read_json(path)
        if (
            binding.get("schema_version") != "mavis.objective-session/v1"
            or binding.get("session_id") != session_id
        ):
            raise ValueError("objective session binding is invalid")
        objective_id = str(binding.get("objective_id") or "")
        record = self.load(objective_id)
        attempts = record.get("attempts") or []
        unresolved = list(record.get("unresolved_decisions") or [])
        if attempts and record.get("state") != "accepted":
            unresolved.append(
                f"Latest failure: {attempts[-1].get('failure_fingerprint', 'unknown')}"
            )
        if record.get("escalation_reason"):
            unresolved.append(str(record["escalation_reason"]))
        return {
            "objective_id": objective_id,
            "objective_state": record.get("state"),
            "goals": [record["blueprint"]],
            "accepted_decisions": [],
            "completed_requirements": [item["id"] for item in record["requirements"]]
            if record.get("state") == "accepted"
            else [],
            "current_changes": [],
            "recent_work": [
                {
                    "state": record.get("state"),
                    "updated_at": record.get("updated_at"),
                    "attempts": len(attempts),
                }
            ],
            "unresolved_failures": unresolved,
            "evidence_links": [str(self._path(objective_id).resolve())]
            + [str(item["path"]) for item in record.get("evidence_receipts") or []],
            "unknown_fields": ["accepted_decisions", "current_changes"],
        }

    def save(self, record: dict[str, Any]) -> None:
        record["updated_at"] = _now()
        write_json(self._path(str(record["objective_id"])), record)

    def transition(self, objective_id: str, state: str, reason: str) -> dict[str, Any]:
        if state not in STATES:
            raise ValueError(f"unknown objective state: {state}")
        with profile_boundary_lock(self.home):
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

    def add_assignment(
        self, objective_id: str, assignment: dict[str, Any]
    ) -> dict[str, Any]:
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
            raise ValueError(
                "evidence receipt must be host-recorded under the objective evidence root"
            )
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
                record["escalation_reason"] = (
                    "same failure repeated without new evidence"
                )
        self.save(record)
        return record

    def add_verification(
        self, objective_id: str, verification: dict[str, Any]
    ) -> dict[str, Any]:
        record = self.load(objective_id)
        _validate_verification(verification, objective_id)
        worker_owners = {
            _owner_identity(item.get("owner", {})) for item in record["assignments"]
        }
        verifier = _owner_identity(verification.get("verifier", {}))
        if verifier in worker_owners:
            raise ValueError("verifier must be independent from implementation owners")
        record["verifications"].append(deepcopy(verification))
        self.save(record)
        return record

    def record_gateway_verification(
        self, objective_id: str, worker_job_id: str
    ) -> dict[str, Any]:
        """Record a host-owned, hash-bound status response from the configured gateway."""
        require_safe_id(worker_job_id, "gateway worker job id")
        record = self.load(objective_id)
        receipts = self._validated_receipts(record)
        if not receipts:
            raise ValueError("gateway verification requires retained evidence receipts")
        revisions = {str(receipt["changed_revision"]) for receipt in receipts}
        if len(revisions) != 1:
            raise ValueError("gateway verification requires one evidence revision")
        try:
            status = self.gateway_status_reader(worker_job_id)
        except GatewayUnavailable:
            raise
        except Exception as error:
            raise GatewayUnavailable("configured model gateway is unavailable") from error
        _validate_gateway_status(status, worker_job_id)
        self._validate_objective_gateway_binding(record, status, revisions)
        receipt_id = f"gateway-{uuid4().hex}"
        path = self.home / "evidence" / objective_id / receipt_id / "gateway-status.json"
        payload = {
            "schema_version": "mavis.gateway-verification-receipt/v1",
            "receipt_id": receipt_id,
            "objective_id": objective_id,
            "worker_job_id": worker_job_id,
            "revision": revisions.pop(),
            "host_receipts": deepcopy(record["evidence_receipts"]),
            "gateway_status": deepcopy(status),
            "gateway_status_sha256": _sha256_json(status),
            "recorded_at": _now(),
            "producer": "mavis-host-gateway/v1",
        }
        write_json(path, payload)
        record.setdefault("gateway_verifications", []).append(
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
        )
        self.save(record)
        return record

    def _validated_receipts(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        receipts = []
        for retained in record.get("evidence_receipts") or []:
            if not isinstance(retained, dict) or set(retained) != {"path", "sha256"}:
                raise ValueError("evidence receipt retention record is invalid")
            path = Path(retained["path"])
            if sha256_file(path) != retained["sha256"]:
                raise ValueError(
                    "retained evidence receipt hash changed after recording"
                )
            receipt = read_json(path)
            _validate_receipt(receipt, str(record["objective_id"]), path)
            receipts.append(receipt)
        return receipts

    def _validate_objective_gateway_binding(
        self,
        record: dict[str, Any],
        status: dict[str, Any],
        evidence_revisions: set[str],
    ) -> None:
        binding = status.get("mavis_binding")
        required = {
            "schema_version",
            "objective_id",
            "cwd",
            "starting_revision",
            "changed_revision",
            "owned_paths",
            "requirements",
            "required_checks",
            "owner",
            "assignment_sha256",
            "report_sha256",
            "target_sha256",
        }
        if not isinstance(binding, dict) or set(binding) != required:
            raise ValueError("gateway status lacks an objective binding")
        if binding["schema_version"] != "model-gateway-mavis-objective-binding/v1":
            raise ValueError("gateway objective binding schema is invalid")
        if binding["objective_id"] != record["objective_id"]:
            raise ValueError("gateway objective binding names a different objective")
        assignments = record.get("assignments") or []
        if not assignments:
            raise ValueError("gateway objective binding requires a worker assignment")
        assignment = assignments[-1]
        checkout = assignment.get("checkout")
        if not isinstance(checkout, dict):
            raise ValueError("worker assignment checkout is invalid")
        try:
            gateway_cwd = Path(str(binding["cwd"])).resolve(strict=True)
            assignment_cwd = Path(str(checkout["path"])).resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ValueError("gateway objective binding checkout is invalid") from error
        if gateway_cwd != assignment_cwd:
            raise ValueError("gateway objective binding checkout does not match")
        if binding["starting_revision"] != assignment.get("starting_revision"):
            raise ValueError("gateway objective binding starting revision does not match")
        if _owner_identity(binding["owner"]) != _owner_identity(assignment.get("owner")):
            raise ValueError("gateway objective binding owner does not match")
        if binding["changed_revision"] not in evidence_revisions:
            raise ValueError("gateway objective binding revision does not match host evidence")
        if set(binding["owned_paths"]) != set(checkout.get("owned_paths") or []):
            raise ValueError("gateway objective binding owned paths do not match")
        requirements = {str(item["id"]) for item in record["requirements"]}
        if set(binding["requirements"]) != requirements or set(assignment.get("requirements") or []) != requirements:
            raise ValueError("gateway objective binding requirements do not match")
        checks = {str(item["id"]) for item in record["acceptance_checks"]}
        if set(binding["required_checks"]) != checks:
            raise ValueError("gateway objective binding acceptance checks do not match")
        acceptance = status["acceptance"]
        for field, gateway_field in (
            ("target_sha256", "target_sha256"),
            ("report_sha256", "report_sha256_on_disk"),
        ):
            if binding[field] != acceptance[gateway_field]:
                raise ValueError(f"gateway objective binding {field} does not match")
        _require_sha256(binding["assignment_sha256"], "gateway objective assignment")

    def _assert_acceptance(self, record: dict[str, Any]) -> None:
        receipts = self._validated_receipts(record)
        if not receipts or any(item.get("verdict") != "pass" for item in receipts):
            raise ValueError(
                "all retained evidence receipts must pass before acceptance"
            )
        gateway_verifications = record.get("gateway_verifications") or []
        if not gateway_verifications:
            raise ValueError("Mavis host gateway verification receipt is required")
        retained_gateway = gateway_verifications[-1]
        if not isinstance(retained_gateway, dict) or set(retained_gateway) != {"path", "sha256"}:
            raise ValueError("gateway verification retention record is invalid")
        gateway_path = Path(retained_gateway["path"])
        if sha256_file(gateway_path) != retained_gateway["sha256"]:
            raise ValueError("retained gateway verification hash changed after recording")
        gateway_receipt = read_json(gateway_path)
        _validate_gateway_receipt(gateway_receipt, str(record["objective_id"]))
        expected_revision = gateway_receipt["revision"]
        if any(
            receipt.get("changed_revision") != expected_revision for receipt in receipts
        ):
            raise ValueError("gateway verification revision does not match evidence receipts")
        required_checks = {
            str(check.get("id"))
            for check in record["acceptance_checks"]
            if check.get("id")
        }
        covered_checks = {
            check_id
            for receipt in receipts
            for check_id in receipt["acceptance_check_ids"]
        }
        if not required_checks or not required_checks.issubset(covered_checks):
            raise ValueError("retained evidence does not cover every acceptance check")
        if gateway_receipt["host_receipts"] != record["evidence_receipts"]:
            raise ValueError("gateway verification is not bound to the exact retained receipts")
        worker_job_id = gateway_receipt["worker_job_id"]
        try:
            fresh_status = self.gateway_status_reader(worker_job_id)
        except GatewayUnavailable:
            raise
        except Exception as error:
            raise GatewayUnavailable("configured model gateway is unavailable") from error
        _validate_gateway_status(fresh_status, worker_job_id)
        self._validate_objective_gateway_binding(
            record, fresh_status, {str(receipt["changed_revision"]) for receipt in receipts}
        )
        if _sha256_json(fresh_status) != gateway_receipt["gateway_status_sha256"]:
            raise ValueError("gateway verification status changed after recording")


def _require_keys(payload: dict[str, Any], required: set[str], label: str) -> None:
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{label} missing required fields: {', '.join(missing)}")


def _owner_identity(owner: object) -> tuple[str, str, str]:
    if not isinstance(owner, dict):
        raise ValueError("owner identity must be an object")
    identity = tuple(
        str(owner.get(key) or "") for key in ("provider", "model", "harness")
    )
    if any(not value for value in identity):
        raise ValueError("owner identity requires provider, model, and harness")
    return identity


def _sha256_json(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 hash")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be a SHA-256 hash") from error
    return value


def _validate_gateway_status(status: dict[str, Any], worker_job_id: str) -> None:
    if not isinstance(status, dict):
        raise ValueError("gateway status is invalid")
    if status.get("job_id") != worker_job_id:
        raise ValueError("gateway status is not bound to the requested worker job")
    worker_receipt = status.get("receipt")
    if (
        not isinstance(worker_receipt, dict)
        or worker_receipt.get("job_id") != worker_job_id
        or worker_receipt.get("exit_code") != 0
    ):
        raise ValueError("gateway status lacks a successful receipt for the worker job")
    if status.get("state") != "completed" or status.get("exit_code") != 0:
        raise ValueError("gateway worker job did not complete successfully")
    if status.get("accepted") is not True:
        raise ValueError("gateway worker job is not accepted")
    acceptance = status.get("acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("accepted") is not True:
        raise ValueError("gateway acceptance receipt is invalid")
    if acceptance.get("job_id") != worker_job_id:
        raise ValueError("gateway acceptance is not bound to the worker job")
    verifier_job_id = acceptance.get("verifier_job_id")
    if (
        acceptance.get("verifier") != "terra"
        or not isinstance(verifier_job_id, str)
        or not verifier_job_id
        or verifier_job_id == worker_job_id
    ):
        raise ValueError("gateway acceptance requires a distinct Terra verifier job")
    for field in (
        "target_sha256",
        "evidence_sha256",
        "report_sha256_on_disk",
        "verifier_verdict_sha256",
    ):
        _require_sha256(acceptance.get(field), f"gateway acceptance {field}")


def _validate_gateway_receipt(receipt: dict[str, Any], objective_id: str) -> None:
    required = {
        "schema_version",
        "receipt_id",
        "objective_id",
        "worker_job_id",
        "revision",
        "host_receipts",
        "gateway_status",
        "gateway_status_sha256",
        "recorded_at",
        "producer",
    }
    _require_keys(receipt, required, "gateway verification receipt")
    if (
        receipt["schema_version"] != "mavis.gateway-verification-receipt/v1"
        or receipt["producer"] != "mavis-host-gateway/v1"
        or receipt["objective_id"] != objective_id
    ):
        raise ValueError("gateway verification receipt is not Mavis-host owned")
    require_safe_id(str(receipt["receipt_id"]), "gateway receipt id")
    require_safe_id(str(receipt["worker_job_id"]), "gateway worker job id")
    if not isinstance(receipt["revision"], str) or len(receipt["revision"]) < 7:
        raise ValueError("gateway verification receipt must bind a git revision")
    if not isinstance(receipt["host_receipts"], list) or not receipt["host_receipts"]:
        raise ValueError("gateway verification receipt must bind host receipts")
    for retained in receipt["host_receipts"]:
        if not isinstance(retained, dict) or set(retained) != {"path", "sha256"}:
            raise ValueError("gateway verification receipt host receipt is invalid")
        _require_sha256(retained["sha256"], "gateway verification receipt host hash")
    _validate_gateway_status(receipt["gateway_status"], receipt["worker_job_id"])
    if _sha256_json(receipt["gateway_status"]) != _require_sha256(
        receipt["gateway_status_sha256"], "gateway verification status"
    ):
        raise ValueError("gateway verification status hash does not match")


def _validate_assignment(assignment: dict[str, Any], objective_id: str) -> None:
    required = {
        "schema_version",
        "assignment_id",
        "objective_id",
        "requirements",
        "starting_revision",
        "owner",
        "checkout",
        "allowed_effects",
        "expected_artifacts",
        "escalate_when",
        "state",
    }
    _require_keys(assignment, required, "worker assignment")
    if assignment["schema_version"] != "mavis.worker-assignment/v1":
        raise ValueError("unsupported worker assignment schema")
    if assignment["objective_id"] != objective_id:
        raise ValueError("worker assignment objective does not match")
    require_safe_id(str(assignment["assignment_id"]), "assignment id")
    _owner_identity(assignment["owner"])
    if (
        not assignment["requirements"]
        or not assignment["expected_artifacts"]
        or not assignment["escalate_when"]
    ):
        raise ValueError("assignment lists must not be empty")
    checkout = assignment["checkout"]
    if (
        not isinstance(checkout, dict)
        or not checkout.get("path")
        or not checkout.get("owned_paths")
    ):
        raise ValueError("assignment checkout must bind a path and owned paths")


def _validate_receipt(
    receipt: dict[str, Any], objective_id: str, receipt_path: Path
) -> None:
    required = {
        "schema_version",
        "receipt_id",
        "objective_id",
        "command",
        "cwd",
        "exit_status",
        "timed_out",
        "started_at",
        "finished_at",
        "raw_output",
        "changed_revision",
        "artifact_hashes",
        "acceptance_check_ids",
        "producer",
        "verdict",
    }
    _require_keys(receipt, required, "evidence receipt")
    if (
        receipt["schema_version"] != "mavis.evidence-receipt/v1"
        or receipt["producer"] != "mavis-host-command/v1"
    ):
        raise ValueError("unsupported evidence receipt producer or schema")
    if receipt["objective_id"] != objective_id:
        raise ValueError("evidence receipt objective does not match")
    if not isinstance(receipt["command"], list) or not receipt["command"]:
        raise ValueError("evidence receipt command is missing")
    if not isinstance(receipt["exit_status"], int):
        raise ValueError("evidence receipt exit status is invalid")
    if not isinstance(receipt["timed_out"], bool):
        raise ValueError("evidence receipt timeout flag is invalid")
    if (
        not isinstance(receipt["changed_revision"], str)
        or len(receipt["changed_revision"]) < 7
    ):
        raise ValueError("evidence receipt must bind a git revision")
    if not receipt["acceptance_check_ids"]:
        raise ValueError("evidence receipt must cover an acceptance check")
    raw = receipt["raw_output"]
    evidence_dir = receipt_path.parent.resolve()
    if (
        not isinstance(raw, dict)
        or Path(str(raw.get("path"))).resolve() != evidence_dir
    ):
        raise ValueError("raw output path is not bound to the receipt directory")
    stdout_path, stderr_path = evidence_dir / "stdout.log", evidence_dir / "stderr.log"
    if not stdout_path.is_file() or not stderr_path.is_file():
        raise ValueError("raw output files are missing")
    actual_hash = sha256_file(stdout_path) + ":" + sha256_file(stderr_path)
    actual_bytes = stdout_path.stat().st_size + stderr_path.stat().st_size
    if raw.get("sha256") != actual_hash or raw.get("bytes") != actual_bytes:
        raise ValueError("raw output hash or byte count does not match")
    combined = (
        stdout_path.read_text(encoding="utf-8", errors="replace")
        + "\n"
        + stderr_path.read_text(encoding="utf-8", errors="replace")
    )
    if receipt["verdict"] != parse_test_output(
        combined, receipt["exit_status"], receipt["timed_out"]
    ):
        raise ValueError(
            "evidence receipt verdict does not match its raw output and exit status"
        )
    for artifact, expected_hash in receipt["artifact_hashes"].items():
        path = Path(artifact)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"artifact hash does not match: {artifact}")


def _validate_verification(verification: dict[str, Any], objective_id: str) -> None:
    required = {
        "schema_version",
        "verification_id",
        "objective_id",
        "revision",
        "requirements",
        "protected_fixtures",
        "checks",
        "required_receipts",
        "verifier",
        "verdict",
    }
    _require_keys(verification, required, "verification")
    if verification["schema_version"] != "mavis.verifier/v1":
        raise ValueError("unsupported verification schema")
    if verification["objective_id"] != objective_id:
        raise ValueError("verification objective does not match")
    require_safe_id(str(verification["verification_id"]), "verification id")
    _owner_identity(verification["verifier"])
    if (
        not isinstance(verification["revision"], str)
        or len(verification["revision"]) < 7
    ):
        raise ValueError("verification must bind a git revision")
    if verification["verdict"] not in {"accepted", "rejected"}:
        raise ValueError("verification verdict must be accepted or rejected")
    for key in ("requirements", "checks", "required_receipts"):
        if not isinstance(verification[key], list) or not verification[key]:
            raise ValueError(f"verification {key} must not be empty")
