"""Bounded advisory decisions through Mavis's configured model gateway."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

from .gateway import mavis_jev_decisions
from .storage import require_safe_id, write_json


PURPOSES = frozenset({
    "evidence_relevance", "tool_selection", "failure_classification",
    "escalation", "memory_category", "completion_suspicion",
})
COUNT_SIGNAL_KEYS = frozenset({
    "failed_checks", "passed_checks", "retry_count", "candidate_count",
    "evidence_count", "unresolved_count",
})
FLAG_SIGNAL_KEYS = frozenset({
    "tool_available",
    "required_checks_complete", "verifier_accepted", "change_detected",
})
SIGNAL_KEYS = COUNT_SIGNAL_KEYS | FLAG_SIGNAL_KEYS


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def advise(service_home: Path, project_root: Path, purpose: str,
           signals: dict[str, Any],
           estimated_cost_usd: float, *, event_id: str | None = None) -> dict[str, Any]:
    """Keep a local receipt; Jev's answer cannot change permissions or completion."""
    if purpose not in PURPOSES:
        raise ValueError("unknown Jev advisory purpose")
    if event_id is not None:
        require_safe_id(event_id, "Jev event id")
    if not isinstance(signals, dict) or not signals or len(signals) > 10:
        raise ValueError("Jev signals must be a nonempty small object")
    if set(signals) - SIGNAL_KEYS:
        raise ValueError("Jev signals include an unknown field")
    for key, value in signals.items():
        if ((key in COUNT_SIGNAL_KEYS and
             (type(value) is not int or not 0 <= value <= 1_000_000)) or
            (key in FLAG_SIGNAL_KEYS and type(value) is not bool)):
            raise ValueError("Jev signal values must be bounded counts or flags")
    if (isinstance(estimated_cost_usd, bool) or not isinstance(estimated_cost_usd, (int, float))
            or not math.isfinite(estimated_cost_usd)
            or not 0 < estimated_cost_usd <= 1):
        raise ValueError("Jev estimated cost must be between $0 and $1")
    response = mavis_jev_decisions(purpose, signals, float(estimated_cost_usd))
    if not isinstance(response, dict) or not isinstance(response.get("success"), bool):
        raise ValueError("Jev gateway response is malformed")
    if response["success"]:
        if (response.get("decision") != "answered" or response.get("advisory") is not True
                or response.get("binding") is not False
                or response.get("grants_permission") is not False
                or not isinstance(response.get("answers"), dict)):
            raise ValueError("Jev gateway response claimed authority or lacked typed answers")
    elif (response.get("advisory") is False or response.get("binding") is True
          or response.get("grants_permission") is True):
        raise ValueError("Jev gateway refusal claimed authority")

    decision = response.get("decision") or "policy_rejected"
    kept_fields = (
        "success", "decision", "failure_class", "advisory", "binding",
        "grants_permission", "answers", "response_model", "actual_cost_usd",
    )
    retained_response = {key: response[key] for key in kept_fields if key in response}
    raw_response = response.get("raw_response")
    if isinstance(raw_response, dict) and isinstance(raw_response.get("sha256"), str):
        retained_response["raw_response_sha256"] = raw_response["sha256"]
    usage_receipt = response.get("usage_receipt")
    if isinstance(usage_receipt, dict):
        retained_response["usage_receipt"] = {
            key: usage_receipt[key]
            for key in ("cost_usd", "response_sha256", "at_epoch")
            if key in usage_receipt
        }
    record = {
        "schema_version": "mavis.jev-advice/v1",
        "advisory": True,
        "binding": False,
        "grants_permission": False,
        "purpose": purpose,
        "signals_sha256": _digest(signals),
        "project_root_sha256": _digest(str(project_root.resolve())),
        "estimated_cost_usd": float(estimated_cost_usd),
        "received_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "gateway_response": retained_response,
    }
    if event_id is not None:
        record["event_id"] = event_id
    receipt = Path(service_home) / "jev" / f"{uuid4().hex}.json"
    write_json(receipt, record)
    return {"success": response["success"], "decision": decision,
            "advisory": True, "binding": False, "grants_permission": False,
            "answers": response.get("answers", {}) if response["success"] else {},
            "receipt": str(receipt)}
