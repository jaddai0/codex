"""Evidence-producing evaluation gates for Mavis.

E0 is deliberately fail-closed. Cases with no real worker/runtime evidence are
reported as blocked; a deterministic unit test cannot stand in for a local
model completing a coding task or an external harness operating in its native
environment.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Callable
from urllib.request import Request, urlopen

from .evidence import parse_test_output
from .objectives import ObjectiveStore
from .runtime import RuntimeConfig, _listener_pids, endpoint_alive, owns_running_server
from .storage import sha256_file, write_json
from .transcripts import TranscriptArchive


E0_CASES = (
    "tool-roundtrip",
    "small-repository",
    "dirty-work-preservation",
    "seeded-failure-repair",
    "fabricated-success-rejection",
    "buried-failure",
    "compaction-restart",
    "repeated-no-progress",
    "external-harness",
    "isolation-recovery",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _post_json(url: str, payload: dict[str, Any], timeout: float = 180.0) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        parsed = json.load(response)
    if not isinstance(parsed, dict):
        raise RuntimeError("Responses API returned a non-object")
    return parsed


class E0Evaluator:
    def __init__(self, home: Path, config: RuntimeConfig):
        self.home = Path(home)
        self.config = config
        self.root = self.home / "evaluations" / "e0"

    def _receipt(self, case: str, status: str, evidence: list[str], **extra: Any) -> dict[str, Any]:
        if case not in E0_CASES:
            raise ValueError(f"unknown E0 case: {case}")
        payload = {
            "schema_version": "mavis.evaluation-case/v1",
            "suite": "e0",
            "case": case,
            "status": status,
            "evidence": evidence,
            "observed_at": _now(),
            **extra,
        }
        write_json(self.root / f"{case}.json", payload)
        return payload

    def run_case(self, case: str) -> dict[str, Any]:
        methods: dict[str, Callable[[], dict[str, Any]]] = {
            "tool-roundtrip": self._tool_roundtrip,
            "fabricated-success-rejection": self._fabricated_success,
            "buried-failure": self._buried_failure,
            "compaction-restart": self._compaction_restart,
            "repeated-no-progress": self._repeated_no_progress,
            "isolation-recovery": self._isolation_recovery,
        }
        if case in methods:
            try:
                return methods[case]()
            except Exception as exc:
                return self._receipt(case, "reject", [f"{type(exc).__name__}: {exc}"])
        return self._receipt(
            case,
            "blocked",
            [
                "This case requires a real Mavis or native-harness assignment receipt; no synthetic fixture is accepted as a substitute."
            ],
        )

    def run(self, case: str | None = None) -> dict[str, Any]:
        if case is not None:
            return self.run_case(case)
        results = [self.run_case(item) for item in E0_CASES]
        passed = len(results) == len(E0_CASES) and all(item["status"] == "pass" for item in results)
        summary = {
            "schema_version": "mavis.evaluation-suite/v1",
            "suite": "e0",
            "status": "pass" if passed else "reject",
            "mandatory_cases": list(E0_CASES),
            "results": [{"case": item["case"], "status": item["status"]} for item in results],
            "observed_at": _now(),
        }
        write_json(self.root / "summary.json", summary)
        return summary

    def _tool_roundtrip(self) -> dict[str, Any]:
        if not endpoint_alive(self.config.endpoint):
            return self._receipt("tool-roundtrip", "blocked", ["Mavis endpoint is unavailable"])
        payload = {
            "model": self.config.model,
            "input": "Call the mavis_probe tool with value exactly E0_CANARY. Do not answer in text.",
            "store": True,
            "max_output_tokens": 128,
            "tools": [
                {
                    "type": "function",
                    "name": "mavis_probe",
                    "description": "Return the supplied canary value.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["value"],
                        "properties": {"value": {"type": "string"}},
                    },
                }
            ],
            "tool_choice": "required",
        }
        first = _post_json(self.config.endpoint.rstrip("/") + "/responses", payload)
        calls = [item for item in first.get("output", []) if item.get("type") == "function_call"]
        if len(calls) != 1 or calls[0].get("name") != "mavis_probe":
            raise RuntimeError("model did not emit the required function call")
        arguments = json.loads(calls[0].get("arguments") or "{}")
        if arguments.get("value") != "E0_CANARY":
            raise RuntimeError("tool arguments did not preserve the exact canary")
        second = _post_json(
            self.config.endpoint.rstrip("/") + "/responses",
            {
                "model": self.config.model,
                "previous_response_id": first["id"],
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": calls[0]["call_id"],
                        "output": "E0_CANARY_OK",
                    }
                ],
                "store": True,
                "max_output_tokens": 64,
            },
        )
        if second.get("status") != "completed":
            raise RuntimeError("tool-result continuation did not complete")
        return self._receipt(
            "tool-roundtrip",
            "pass",
            [f"Responses {first['id']} -> {second['id']} completed one exact function-call round trip"],
            response_ids=[first["id"], second["id"]],
        )

    def _fabricated_success(self) -> dict[str, Any]:
        store = ObjectiveStore(self.home / "e0-fixtures" / "fabricated")
        objective_id = "e0-fabricated"
        if store._path(objective_id).exists():
            store._path(objective_id).unlink()
        store.create({
            "schema_version": "mavis.objective/v1",
            "objective_id": objective_id,
            "blueprint": "Reject forged completion evidence",
            "requirements": [{"id": "r1", "text": "reject fabricated success"}],
            "dependencies": [], "scope": {},
            "acceptance_checks": [{"id": "c1"}],
            "unresolved_decisions": [],
        })
        forged = store.root.parent / "evidence" / objective_id / "forged"
        forged.mkdir(parents=True, exist_ok=True)
        forged_receipt = forged / "receipt.json"
        write_json(forged_receipt, {
            "schema_version": "mavis.evidence-receipt/v1",
            "verdict": "pass", "changed_revision": "abcdef0",
        })
        try:
            store.add_receipt(objective_id, forged_receipt)
        except ValueError:
            return self._receipt("fabricated-success-rejection", "pass", ["Objective evidence gate rejected a forged in-root receipt with missing host fields"])
        raise RuntimeError("fabricated receipt was accepted")

    def _buried_failure(self) -> dict[str, Any]:
        text = "\n".join(["ok detail"] * 5000 + ["FAILED buried_case"] + ["ok detail"] * 5000)
        if parse_test_output(text, 0) != "fail":
            raise RuntimeError("buried failure was converted to success")
        artifact = self.root / "buried-failure.raw.log"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(text, encoding="utf-8")
        return self._receipt("buried-failure", "blocked", [
            f"Parser fixture retained the failure in {artifact} (sha256 {sha256_file(artifact)})",
            "A real harness tool-output capture and model inspection are still required.",
        ])

    def _compaction_restart(self) -> dict[str, Any]:
        archive = TranscriptArchive(self.home, "e0-compaction")
        segment = archive.append_segment(
            [{"role": "user", "content": "EARLY_DECISION_7B9A"}]
        )
        archive.write_handoff(
            {
                "goals": ["recover the early decision"],
                "accepted_decisions": ["EARLY_DECISION_7B9A"],
                "completed_requirements": [],
                "current_changes": [],
                "recent_work": [],
                "unresolved_failures": [],
                "evidence_links": ["segment"],
            }
        )
        restarted = TranscriptArchive(self.home, "e0-compaction")
        hits = restarted.search("EARLY_DECISION_7B9A", 5)
        if not hits:
            raise RuntimeError("early decision was not recovered after archive restart")
        return self._receipt("compaction-restart", "blocked", [
            f"Archive fixture recovered the early decision from {segment}",
            "A real Codex compaction and resumed Mavis session are still required.",
        ])

    def _repeated_no_progress(self) -> dict[str, Any]:
        fixture_home = self.home / "e0-fixtures" / "escalation"
        store = ObjectiveStore(fixture_home)
        objective_id = "e0-escalation"
        path = store._path(objective_id)
        if path.exists():
            path.unlink()
        store.create(
            {
                "schema_version": "mavis.objective/v1",
                "objective_id": objective_id,
                "blueprint": "E0 repeated failure fixture",
                "requirements": [{"id": "r1", "text": "repair"}],
                "dependencies": [],
                "scope": {},
                "acceptance_checks": [{"id": "c1"}],
                "unresolved_decisions": [],
                "state": "queued",
            }
        )
        store.record_attempt(objective_id, "same", ["first evidence"], "approach one")
        result = store.record_attempt(objective_id, "same", [], "approach one repeated")
        if result["state"] != "escalated":
            raise RuntimeError("repeated no-progress failure did not escalate")
        return self._receipt("repeated-no-progress", "pass", ["Second identical failure without new evidence escalated"])

    def _isolation_recovery(self) -> dict[str, Any]:
        iris_alive = endpoint_alive(self.config.iris_endpoint)
        mavis_alive = endpoint_alive(self.config.endpoint)
        distinct = not (_listener_pids(8000) & _listener_pids(8001))
        mavis_owned = owns_running_server(self.config) if mavis_alive else False
        status = "pass" if iris_alive and mavis_alive and distinct and mavis_owned else "reject"
        return self._receipt(
            "isolation-recovery",
            status,
            [
                f"iris_alive={iris_alive}",
                f"mavis_alive={mavis_alive}",
                f"distinct_listener_pids={distinct}",
                f"mavis_owned={mavis_owned}",
            ],
        )
