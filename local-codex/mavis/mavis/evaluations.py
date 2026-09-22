"""Evidence-producing evaluation gates for Mavis.

E0 is deliberately fail-closed. Cases with no real worker/runtime evidence are
reported as blocked; a deterministic unit test cannot stand in for a local
model completing a coding task or an external harness operating in its native
environment.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable
from urllib.request import Request, urlopen

from .evidence import run_command
from .objectives import ObjectiveStore
from .runtime import RuntimeConfig, _listener_pids, admission, endpoint_alive, inventory, owns_running_server
from .storage import sha256_file, write_json


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


def installed_candidate_fingerprint() -> dict[str, str]:
    """Bind E0 live work to the exact installed core, launcher, and service code."""
    share = Path.home() / ".local" / "share" / "local-codex"
    package = share / "mavis"
    digest = hashlib.sha256()
    files = sorted(package.rglob("*.py"))
    if not files:
        raise FileNotFoundError("installed Mavis service package is missing")
    for path in files:
        digest.update(str(path.relative_to(package)).encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return {
        "core_sha256": sha256_file(share / "local-codex-core"),
        "launcher_sha256": sha256_file(Path.home() / "Desktop" / "Mavis.command"),
        "service_sha256": digest.hexdigest(),
    }


def native_review_completed(log: str, verdict: str) -> bool:
    """Recognize a completed native ZCode stream or its structured CLI result."""
    events = []
    for line in log.splitlines():
        if line.startswith("{"):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                events = []
                break
    if any(event.get("type") == "turn.completed" for event in events):
        return verdict.lstrip().startswith("ACCEPT")
    start = log.find("{")
    if start < 0:
        return False
    try:
        result = json.loads(log[start:])
    except json.JSONDecodeError:
        return False
    projection = result.get("projection") or {}
    response = result.get("response")
    return bool(
        isinstance(result.get("sessionId"), str)
        and result["sessionId"].startswith("sess_")
        and isinstance(result.get("turnId"), str)
        and result["turnId"].startswith("turn_")
        and isinstance(result.get("eventCount"), int)
        and result["eventCount"] > 0
        and projection.get("status") == "idle"
        and isinstance(response, str)
        and response == verdict
        and response.lstrip().startswith("ACCEPT")
    )


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
            "small-repository": lambda: self._small_repository("small-repository"),
            "dirty-work-preservation": lambda: self._small_repository("dirty-work-preservation"),
            "seeded-failure-repair": lambda: self._small_repository("seeded-failure-repair"),
            "external-harness": self._external_harness,
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

    def _small_repository(self, case: str) -> dict[str, Any]:
        """Recheck an installed Mavis repair and its separate native Terra review."""
        candidates = sorted(
            (self.root / "tasks").glob("*/manifest.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for manifest_path in candidates:
            task_root = manifest_path.parent
            mavis_log = task_root / "installed-mavis-repair.jsonl"
            terra_log = task_root / "terra-review.jsonl"
            terra_result = task_root / "terra-review.txt"
            installed_run = task_root / "installed-run.json"
            if not all(path.is_file() for path in (mavis_log, terra_log, terra_result, installed_run)):
                continue
            observed_run = json.loads(installed_run.read_text())
            if observed_run.get("candidate") != installed_candidate_fingerprint():
                continue
            if observed_run.get("mavis_log_sha256") != sha256_file(mavis_log) or observed_run.get("terra_log_sha256") != sha256_file(terra_log):
                raise ValueError("E0 installed task logs changed after observation")
            manifest = json.loads(manifest_path.read_text())
            repo = Path(manifest["repo"]).resolve()
            if repo != (task_root / "repo").resolve() or not repo.is_dir():
                raise ValueError("E0 task repository path is invalid")
            if manifest["baseline_exit_status"] == 0 or sha256_file(Path(manifest["baseline_log"])) != manifest["baseline_log_sha256"]:
                raise ValueError("E0 failing baseline is invalid")
            if sha256_file(Path(manifest["protected_dirty_file"])) != manifest["protected_dirty_sha256"]:
                raise ValueError("E0 protected dirty note changed")
            revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            if revision != manifest["starting_revision"]:
                raise ValueError("E0 task revision changed unexpectedly")
            changed = set(subprocess.check_output(["git", "-C", str(repo), "diff", "--name-only"], text=True).splitlines())
            expected = set(manifest["owned_paths"]) | {"user-notes.txt"}
            if changed != expected:
                raise ValueError("E0 changed paths do not match the allowed repair and protected note")
            if subprocess.check_output(["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard"], text=True).strip():
                raise ValueError("E0 task contains unexpected untracked files")
            mavis_events = [json.loads(line) for line in mavis_log.read_text().splitlines() if line.startswith("{")]
            terra_output = terra_log.read_text()
            mavis_finished = any(event.get("type") == "turn.completed" for event in mavis_events)
            terra_finished = native_review_completed(terra_output, terra_result.read_text())
            patch_seen = any(
                event.get("type") == "item.completed"
                and event.get("item", {}).get("type") == "file_change"
                and any(Path(change.get("path", "")).resolve() == repo / "package" / "pricing.py" for change in event["item"].get("changes", []))
                for event in mavis_events
            )
            if not (mavis_finished and terra_finished and patch_seen):
                raise ValueError("E0 installed repair or independent Terra review is incomplete")
            tests = subprocess.run(
                manifest["test_command"], cwd=repo, capture_output=True, text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=False,
            )
            if tests.returncode != 0 or "Ran 2 tests" not in tests.stderr or "OK" not in tests.stderr:
                raise ValueError("E0 seeded repair tests failed")
            return self._receipt(case, "pass", [
                f"Installed Mavis repaired {repo / 'package' / 'pricing.py'}; exact tests passed and protected note hash held",
                f"Separate native Terra review accepted; logs sha256 {sha256_file(mavis_log)} and {sha256_file(terra_log)}",
            ], manifest=str(manifest_path), review=str(terra_result))
        return self._receipt(case, "blocked", ["No complete installed Mavis repair and independent Terra review found"])

    def _external_harness(self) -> dict[str, Any]:
        """Revalidate a native gateway worker and distinct Terra closure."""
        store = ObjectiveStore(self.home)
        paths = sorted(
            store.root.glob("e0-gateway-bound-*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in paths:
            record = json.loads(path.read_text())
            if record.get("state") != "accepted":
                continue
            store._assert_acceptance(record)
            retained = record["gateway_verifications"][-1]
            gateway_receipt = json.loads(Path(retained["path"]).read_text())
            status = gateway_receipt["gateway_status"]
            return self._receipt("external-harness", "pass", [
                f"Native worker {status['job_id']} completed with host receipt and an accepted separate Terra job {status['acceptance']['verifier_job_id']}",
                f"Mavis objective {record['objective_id']} revalidated against the live configured gateway and exact host evidence",
            ], objective_id=record["objective_id"], gateway_receipt=retained["path"])
        return self._receipt("external-harness", "blocked", ["No accepted native gateway worker bound to a Mavis objective was found"])

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
        selected = next(
            (item for item in inventory(self.config.endpoint) if item.get("id") == self.config.model),
            None,
        )
        if not selected or not selected.get("loaded"):
            return self._receipt("tool-roundtrip", "blocked", ["Mavis model is not loaded; E0 will not trigger an implicit model load"])
        decision = admission(self.config)
        if not decision["allowed"]:
            return self._receipt("tool-roundtrip", "blocked", decision["reasons"])
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
                "max_output_tokens": 256,
            },
        )
        if second.get("status") != "completed":
            write_json(self.root / "tool-roundtrip-incomplete.json", {
                "schema_version": "mavis.e0-tool-roundtrip-incomplete/v1",
                "first_response_id": first["id"],
                "second_response_id": second.get("id"),
                "status": second.get("status"),
                "incomplete_details": second.get("incomplete_details"),
                "usage": second.get("usage"),
            })
            raise RuntimeError(
                f"tool-result continuation did not complete: {second.get('status')}"
            )
        return self._receipt(
            "tool-roundtrip",
            "pass",
            [f"Responses {first['id']} -> {second['id']} completed one exact function-call round trip"],
            response_ids=[first["id"], second["id"]],
        )

    def _fabricated_success(self) -> dict[str, Any]:
        fixture_home = self.home / "e0-fixtures" / "fabricated"
        store = ObjectiveStore(fixture_home)
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
            pass
        else:
            raise RuntimeError("fabricated receipt was accepted")
        repo = fixture_home / "repo"
        repo.mkdir(parents=True, exist_ok=True)
        if not (repo / ".git").is_dir():
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "-c", "user.name=Mavis E0", "-c", "user.email=mavis@local.invalid", "commit", "--allow-empty", "-qm", "fixture"], check=True)
        revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        real_receipt = run_command(fixture_home, objective_id, ["python3", "-c", "print('1 passed')"], repo, acceptance_check_ids=["c1"])
        store.add_receipt(objective_id, real_receipt)
        store.add_verification(objective_id, {
            "schema_version": "mavis.verifier/v1",
            "verification_id": "forged-review",
            "objective_id": objective_id,
            "revision": revision,
            "requirements": ["r1"],
            "protected_fixtures": [],
            "checks": ["c1"],
            "required_receipts": [str(real_receipt)],
            "verifier": {"provider": "fake", "model": "fake", "harness": "fake"},
            "verdict": "accepted",
        })
        store.transition(objective_id, "running", "fabrication test")
        store.transition(objective_id, "awaiting verification", "fabrication test")
        try:
            store.transition(objective_id, "accepted", "forged verifier document")
        except ValueError:
            return self._receipt("fabricated-success-rejection", "pass", [
                "Incomplete in-root receipt was rejected",
                "A host test receipt plus model-supplied accepted verifier JSON could not authorize acceptance without a bound native gateway/Terra receipt",
            ])
        raise RuntimeError("forged verifier JSON moved objective to accepted")

    def _buried_failure(self) -> dict[str, Any]:
        for result_path in sorted(
            self.root.glob("buried-live-*/result.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ):
            result = json.loads(result_path.read_text())
            if result.get("candidate") != installed_candidate_fingerprint():
                continue
            if result.get("candidate_after") != result["candidate"]:
                continue
            rollout = Path(result.get("rollout", ""))
            repo = Path(result.get("repo", ""))
            if not rollout.is_file() or not (repo / "produce_log.py").is_file():
                continue
            if not (result.get("mavis_exit") == 0 and result.get("iris_loaded") is True
                    and result.get("mavis_loaded") is False):
                continue
            events = [json.loads(line) for line in rollout.read_text().splitlines() if line.strip()]
            if not events or events[0].get("payload", {}).get("cwd") != str(repo):
                continue
            commands = [event for event in events if event.get("type") == "response_item"
                        and event.get("payload", {}).get("type") == "function_call"
                        and event.get("payload", {}).get("name") == "exec_command"
                        and "produce_log.py" in event.get("payload", {}).get("arguments", "")]
            if len(commands) != 1:
                continue
            outputs = [event.get("payload", {}).get("output", "") for event in events
                       if event.get("type") == "response_item"
                       and event.get("payload", {}).get("type") == "function_call_output"]
            primary = next((output for output in outputs if "Complete raw output:" in output
                            and "Process exited with code 1" in output), None)
            if primary is None:
                continue
            match = re.search(r"Complete raw output: (\S+\.raw) \((\d+) bytes; closed\)", primary)
            if match is None:
                continue
            raw = Path(match.group(1))
            if raw.parent != self.home / "tool-output" or not raw.is_file():
                continue
            content = raw.read_bytes()
            if len(content) != int(match.group(2)):
                continue
            marker_match = re.fullmatch(
                rb"A{800000}\r?\n(MAVIS_E0_FAILURE_[0-9a-f]{24})\r?\nZ{800000}\r?\n",
                content,
            )
            if marker_match is None:
                continue
            marker = marker_match.group(1).decode()
            final = [event.get("payload", {}).get("last_agent_message") for event in events
                     if event.get("type") == "event_msg"
                     and event.get("payload", {}).get("type") == "task_complete"]
            if not final or marker not in (final[-1] or "") or "code `1`" not in final[-1]:
                continue
            if marker in primary:
                continue
            return self._receipt("buried-failure", "pass", [
                f"Installed Mavis command ran once and reported exit 1 in {rollout}",
                f"Complete {len(content)}-byte raw output at {raw} contained the buried marker; final answer matched",
                f"IRIS restored and Mavis unloaded in {result_path}",
            ], installed_candidate=result["candidate"], raw_sha256=sha256_file(raw))
        return self._receipt("buried-failure", "blocked", [
            "No candidate-matched installed harness run retained the full raw output and identified its buried failure."
        ])

    def _compaction_restart(self) -> dict[str, Any]:
        for result_path in sorted(
            self.root.glob("compaction-live-*/result.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ):
            result = json.loads(result_path.read_text())
            rollout = Path(result.get("rollout", ""))
            session = result.get("session_id")
            fact = result.get("fact")
            if not (rollout.is_file() and isinstance(session, str) and isinstance(fact, str)):
                continue
            if result.get("candidate") != installed_candidate_fingerprint():
                continue
            if result.get("candidate_after") != result["candidate"]:
                continue
            if not (result.get("first_exit") == result.get("resume_exit") == 0
                    and result.get("iris_loaded") is True
                    and result.get("mavis_loaded") is False
                    and result.get("exact_recovery") is True
                    and result.get("resumed_answer") == fact):
                continue
            events = [json.loads(line) for line in rollout.read_text().splitlines() if line.strip()]
            if not events or events[0].get("payload", {}).get("id") != session:
                continue
            if events[0].get("payload", {}).get("cwd") != result.get("workspace"):
                continue
            compact_at = next((i for i, event in enumerate(events) if event.get("type") == "compacted"), None)
            if compact_at is None:
                continue
            before, after = events[:compact_at], events[compact_at + 1:]
            def has_user(items: list[dict[str, Any]], expected: str) -> bool:
                return any(event.get("type") == "response_item"
                           and event.get("payload", {}).get("role") == "user"
                           and expected in json.dumps(event.get("payload", {}).get("content", []))
                           for event in items)
            def has_answer(items: list[dict[str, Any]], expected: str) -> bool:
                return any(event.get("type") == "event_msg"
                           and event.get("payload", {}).get("type") == "task_complete"
                           and event["payload"].get("last_agent_message") == expected
                           for event in items)
            if not (has_user(before, fact) and has_answer(before, "ACK")
                    and has_user(after, "What exact fact did I give before compaction?")
                    and has_answer(after, fact)):
                continue
            handoffs = [Path(path) for path in result.get("handoffs", [])]
            handoff_root = self.home / "transcripts" / session / "handoffs"
            if not handoffs or not all(path.is_file() and path.parent == handoff_root for path in handoffs):
                continue
            segments = []
            for path in handoffs:
                handoff = json.loads(path.read_text())
                segments.extend(Path(link) for link in handoff.get("evidence_links", []))
            segment_root = self.home / "transcripts" / session / "segments"
            if not any(path.is_file() and path.parent == segment_root
                       and fact in path.read_text() for path in segments):
                continue
            return self._receipt("compaction-restart", "pass", [
                f"Installed TUI rollout {rollout} compacted and resumed the same session",
                f"Handoff {handoffs[0]} linked an archive containing the fact; resumed answer matched",
                f"IRIS restored and Mavis unloaded in {result_path}",
            ], installed_candidate=result["candidate"])
        return self._receipt("compaction-restart", "blocked", [
            "No candidate-matched installed TUI compaction, handoff, and exact-session resume passed inspection."
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
        proof_path = self.root / "isolation-recovery-live.json"
        if not proof_path.is_file():
            return self._receipt("isolation-recovery", "blocked", ["No host-observed Mavis service outage and recovery receipt exists"])
        proof = json.loads(proof_path.read_text())
        if proof.get("schema_version") != "mavis.e0-isolation-recovery/v1" or proof.get("candidate") != installed_candidate_fingerprint():
            return self._receipt("isolation-recovery", "blocked", ["The installed candidate changed since service recovery was observed"])
        before = proof.get("before") or {}
        after = proof.get("after") or {}
        iris_alive = endpoint_alive(self.config.iris_endpoint)
        mavis_alive = endpoint_alive(self.config.endpoint)
        iris_pids = _listener_pids(8000)
        mavis_pids = _listener_pids(8001)
        distinct = not (iris_pids & mavis_pids)
        mavis_owned = owns_running_server(self.config) if mavis_alive else False
        observed = (
            before.get("iris_pids") == after.get("iris_pids") == sorted(iris_pids)
            and before.get("iris_model_loaded") is True
            and after.get("iris_model_loaded") is True
            and before.get("mavis_pids") != after.get("mavis_pids")
            and after.get("mavis_pids") == sorted(mavis_pids)
            and proof.get("outage_observed") is True
        )
        status = "pass" if iris_alive and mavis_alive and distinct and mavis_owned and observed else "reject"
        return self._receipt(
            "isolation-recovery",
            status,
            [
                f"iris_alive={iris_alive}",
                f"mavis_alive={mavis_alive}",
                f"distinct_listener_pids={distinct}",
                f"mavis_owned={mavis_owned}",
                f"outage_and_new_service_observed={observed}",
            ],
            recovery_receipt=str(proof_path),
        )
