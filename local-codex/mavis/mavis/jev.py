"""Bounded advisory decisions through Mavis's configured model gateway."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from .gateway import jev_decisions
from .storage import write_json


PURPOSES = frozenset({
    "evidence_relevance", "tool_selection", "failure_classification",
    "escalation", "memory_category", "completion_suspicion",
})
STATE_KEYS = frozenset({"summary", "signals"})


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def advise(service_home: Path, project_root: Path, purpose: str,
           state: dict[str, Any], questions: dict[str, Any],
           estimated_cost_usd: float) -> dict[str, Any]:
    """Keep a local receipt; Jev's answer cannot change permissions or completion."""
    if purpose not in PURPOSES:
        raise ValueError("unknown Jev advisory purpose")
    if not isinstance(state, dict) or not isinstance(questions, dict) or not questions:
        raise ValueError("Jev requires a state object and typed questions")
    if set(state) - STATE_KEYS or not isinstance(state.get("summary"), str):
        raise ValueError("Jev state allows only a redacted summary and signals")
    if not 0 < len(state["summary"]) <= 500 or "\n" in state["summary"]:
        raise ValueError("Jev summary must be one short line")
    signals = state.get("signals", {})
    if not isinstance(signals, dict) or len(signals) > 20:
        raise ValueError("Jev signals must be a small object")
    for key, value in signals.items():
        if not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,40}", key):
            raise ValueError("Jev signal names must be simple labels")
        if (not isinstance(value, (str, bool, int, float)) or
                isinstance(value, float) and not math.isfinite(value) or
                isinstance(value, str) and (len(value) > 80 or "\n" in value)):
            raise ValueError("Jev signal values must be short labels or numbers")
    if (isinstance(estimated_cost_usd, bool) or not isinstance(estimated_cost_usd, (int, float))
            or not math.isfinite(estimated_cost_usd)
            or not 0 < estimated_cost_usd <= 1):
        raise ValueError("Jev estimated cost must be between $0 and $1")
    if len(json.dumps({"state": state, "questions": questions}, ensure_ascii=False)) > 16000:
        raise ValueError("Jev request exceeds the bounded evidence packet")

    response = jev_decisions(state, questions, float(estimated_cost_usd))
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
        "state_sha256": _digest(state),
        "questions_sha256": _digest(questions),
        "project_root_sha256": _digest(str(project_root.resolve())),
        "estimated_cost_usd": float(estimated_cost_usd),
        "received_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "gateway_response": retained_response,
    }
    receipt = Path(service_home) / "jev" / f"{uuid4().hex}.json"
    write_json(receipt, record)
    return {"success": response["success"], "decision": decision,
            "advisory": True, "binding": False, "grants_permission": False,
            "answers": response.get("answers", {}) if response["success"] else {},
            "receipt": str(receipt)}
