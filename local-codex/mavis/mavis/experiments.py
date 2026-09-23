"""Durable, reversible experiments for prompt, tool, and retrieval configuration.

This module records evidence and gates local configuration changes. It never runs
models, changes protected evaluations, or treats a caller's reviewer name as proof.
A separate host must issue the independent review receipt.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterator

from .gateway import harness_job_status
from .maintenance import MaintenanceQueue
from .objectives import _validate_gateway_status
from .storage import profile_boundary_lock, read_json, require_safe_id, sha256_file, write_json


KINDS = {"prompts", "tool_settings", "retrieval"}


def candidate_assignment_requirements(record: dict[str, Any]) -> list[str]:
    """String IDs for the gateway candidate worker's Mavis assignment."""
    return [f"experiment-candidate-snapshot:{record['candidate']['sha256']}"]


def review_assignment_requirements(record: dict[str, Any]) -> list[str]:
    """String IDs for the gateway review worker's Mavis assignment."""
    candidate = record["comparison"]["candidate"]
    return [
        f"experiment-comparison:{record['comparison']['comparison_digest']}",
        f"experiment-candidate-job:{candidate['candidate_job_id']}",
        f"experiment-candidate-report:{candidate['evidence']['sha256']}",
    ]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@contextmanager
def _profile_transition_lease(home: Path) -> Iterator[None]:
    """Hold the same host lock as accepted main sessions and E1 trials."""
    home = Path(home).resolve()
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (home / "generation.lock").open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another Mavis local generation owns the host lease") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class ExperimentStore:
    def __init__(self, home: Path, gateway_status_reader: Callable[[str], dict[str, Any]] | None = None):
        self.home = Path(home)
        self.root = self.home / "experiments"
        self.queue = MaintenanceQueue(self.home)
        self.gateway_status_reader = gateway_status_reader or harness_job_status

    @contextmanager
    def _locked(self) -> Iterator[None]:
        for path in (self.root, self.root / "records", self.root / "snapshots",
                     self.root / "active", self.root / "staged"):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(path, 0o700)
        with (self.root / ".lock").open("a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _record_path(self, experiment_id: str) -> Path:
        return self.root / "records" / f"{require_safe_id(experiment_id, 'experiment id')}.json"

    def _active_path(self, scope: str) -> Path:
        return self.root / "active" / f"{require_safe_id(scope, 'experiment scope')}.json"

    def _assert_objective_boundary(self) -> None:
        """A persisted unfinished objective cannot change its active profile."""
        root = self.home / "objectives"
        for path in root.glob("*.json"):
            if path.is_symlink():
                raise ValueError("objective state has a symlink")
            record = read_json(path)
            if (record.get("objective_id") != path.stem
                    or record.get("state") not in {"queued", "running", "awaiting verification",
                                                    "accepted", "needs repair", "escalated",
                                                    "blocked", "cancelled"}):
                raise ValueError("objective state is invalid")
            if record["state"] not in {"queued", "accepted", "cancelled"}:
                raise ValueError("promotion or rollback is allowed only between objectives")

    def _snapshot(self, config: dict[str, Any]) -> dict[str, str]:
        if not isinstance(config, dict) or not all(key in config for key in KINDS):
            raise ValueError("configuration needs prompts, tool_settings, and retrieval sections")
        if not all(isinstance(config[key], dict) for key in KINDS):
            raise ValueError("configuration sections must be objects")
        digest = _digest(config)
        path = self.root / "snapshots" / f"{digest}.json"
        if path.exists():
            if _digest(read_json(path)) != digest:
                raise ValueError("frozen configuration snapshot changed")
        else:
            write_json(path, config)
        return {"path": str(path.resolve()), "sha256": sha256_file(path), "digest": digest}

    def _read_snapshot(self, ref: dict[str, str]) -> dict[str, Any]:
        path = Path(ref["path"])
        if path.resolve().parent != (self.root / "snapshots").resolve() or sha256_file(path) != ref["sha256"]:
            raise ValueError("frozen configuration snapshot changed")
        config = read_json(path)
        if _digest(config) != ref["digest"]:
            raise ValueError("frozen configuration content changed")
        return config

    def _load(self, experiment_id: str) -> dict[str, Any]:
        return read_json(self._record_path(experiment_id))

    def load(self, experiment_id: str) -> dict[str, Any]:
        with self._locked():
            record = self._load(experiment_id)
            self._read_snapshot(record["baseline"])
            self._read_snapshot(record["candidate"])
            return record

    def seed_active(self, scope: str, config: dict[str, Any]) -> dict[str, Any]:
        """Bootstrap an accepted configuration exactly once; later changes use promotion."""
        with self._locked():
            path = self._active_path(scope)
            if path.exists():
                raise FileExistsError(path)
            ref = self._snapshot(config)
            active = {"schema_version": "mavis.experiment-active/v1", "scope": scope,
                      "configuration": ref, "experiment_id": None, "previous": None,
                      "updated_at": _now()}
            write_json(path, active)
            return active

    def active(self, scope: str) -> dict[str, Any]:
        with self._locked():
            active = read_json(self._active_path(scope))
            self._read_snapshot(active["configuration"])
            experiment_id = active.get("experiment_id")
            if experiment_id is not None:
                record = self._load(experiment_id)
                if record["state"] != "promoted" or record["candidate"] != active["configuration"]:
                    raise ValueError("active configuration has an incomplete promotion")
            return active

    def create(self, experiment_id: str, scope: str, kind: str, candidate_config: dict[str, Any],
               *, hypothesis: str, workload: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
        if kind not in KINDS or not hypothesis.strip():
            raise ValueError("experiment needs a supported change kind and hypothesis")
        if not isinstance(workload, dict) or not workload or not isinstance(split, dict) or not split.get("held_out"):
            raise ValueError("experiment needs a frozen workload and held-out split")
        gain = split.get("minimum_gain")
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or gain <= 0:
            raise ValueError("held-out split needs a positive minimum gain")
        with self._locked():
            path = self._record_path(experiment_id)
            if path.exists():
                raise FileExistsError(path)
            active = read_json(self._active_path(scope))
            baseline = self._read_snapshot(active["configuration"])
            if set(candidate_config) != set(baseline) or any(candidate_config[key] != baseline[key] for key in KINDS - {kind}):
                raise ValueError("candidate may change only its declared configuration section")
            if candidate_config[kind] == baseline[kind]:
                raise ValueError("candidate has no change")
            candidate = self._snapshot(candidate_config)
            record = {"schema_version": "mavis.experiment-lifecycle/v1", "experiment_id": experiment_id,
                      "scope": scope, "kind": kind, "hypothesis": hypothesis,
                      "baseline": active["configuration"], "candidate": candidate,
                      "workload": deepcopy(workload), "evaluation_split": deepcopy(split),
                      "workload_digest": _digest(workload), "evaluation_split_digest": _digest(split),
                      "state": "candidate", "comparison": None, "review": None,
                      "created_at": _now(), "updated_at": _now(), "history": []}
            write_json(path, record)
            return record

    def compare(self, experiment_id: str, *, baseline_result: dict[str, Any],
                candidate_result: dict[str, Any]) -> dict[str, Any]:
        """Require matched held-out cases, mandatory passes, and real improvement."""
        with self._locked():
            record = self._load(experiment_id)
            if record["state"] != "candidate":
                raise ValueError("only an unevaluated candidate can be compared")
            cases = record["evaluation_split"]["held_out"]
            if not isinstance(cases, list) or not cases or len(cases) != len(set(cases)):
                raise ValueError("held-out case IDs must be unique")
            for arm, result, ref in (("baseline", baseline_result, record["baseline"]),
                                     ("candidate", candidate_result, record["candidate"])):
                if result.get("configuration_sha256") != ref["sha256"] or result.get("workload_digest") != _digest(record["workload"]):
                    raise ValueError(f"{arm} result is not bound to frozen configuration and workload")
                if result.get("case_ids") != cases or not isinstance(result.get("mandatory_passed"), bool):
                    raise ValueError(f"{arm} did not cover the same mandatory held-out cases")
                if arm == "candidate" and result["mandatory_passed"] is not True:
                    raise ValueError("candidate failed a mandatory held-out case")
                score = result.get("target_score")
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                    raise ValueError(f"{arm} target score is invalid")
                evidence = result.get("evidence")
                if not isinstance(evidence, dict) or not evidence.get("path") or not evidence.get("sha256"):
                    raise ValueError(f"{arm} evidence is missing")
                path = Path(evidence["path"]).resolve()
                if not path.is_file() or sha256_file(path) != evidence["sha256"]:
                    raise ValueError(f"{arm} evidence hash changed")
            minimum_gain = record["evaluation_split"].get("minimum_gain")
            if (isinstance(minimum_gain, bool) or not isinstance(minimum_gain, (int, float))
                    or not math.isfinite(minimum_gain) or minimum_gain <= 0):
                raise ValueError("held-out split needs a positive minimum gain")
            if candidate_result["target_score"] - baseline_result["target_score"] < minimum_gain:
                raise ValueError("candidate did not improve the targeted held-out outcome")
            candidate_job_id = candidate_result.get("candidate_job_id")
            if not isinstance(candidate_job_id, str):
                raise ValueError("candidate result requires a native candidate job ID")
            require_safe_id(candidate_job_id, "candidate job id")
            comparison = {"baseline": deepcopy(baseline_result), "candidate": deepcopy(candidate_result),
                          "comparison_digest": _digest({"baseline": baseline_result, "candidate": candidate_result}),
                          "recorded_at": _now()}
            record.update(state="compared", comparison=comparison, updated_at=_now())
            write_json(self._record_path(experiment_id), record)
            return record

    def _check_comparison(self, record: dict[str, Any]) -> None:
        self._read_snapshot(record["baseline"])
        self._read_snapshot(record["candidate"])
        if (record.get("workload_digest") != _digest(record["workload"])
                or record.get("evaluation_split_digest") != _digest(record["evaluation_split"])):
            raise ValueError("frozen workload or held-out split changed")
        comparison = record.get("comparison")
        if not isinstance(comparison, dict) or comparison.get("comparison_digest") != _digest({
            "baseline": comparison.get("baseline"), "candidate": comparison.get("candidate")}):
            raise ValueError("comparison changed")
        for arm in ("baseline", "candidate"):
            result = comparison[arm]
            ref = record[arm]
            if (result.get("configuration_sha256") != ref["sha256"]
                    or result.get("workload_digest") != record["workload_digest"]
                    or result.get("case_ids") != record["evaluation_split"]["held_out"]):
                raise ValueError("comparison lost its frozen workload binding")
            evidence = comparison[arm]["evidence"]
            if sha256_file(Path(evidence["path"])) != evidence["sha256"]:
                raise ValueError("comparison evidence changed")
        if record["workload"].get("manifest_sha256") is not None:
            from .e1 import validate_e1_bundle

            digest = validate_e1_bundle(self.home, record)
            if any(comparison[arm].get("e1_bundle_digest") != digest for arm in ("baseline", "candidate")):
                raise ValueError("underlying E1 evidence changed")

    def review(self, experiment_id: str, receipt_path: Path) -> dict[str, Any]:
        """Import a separate host receipt; a reviewer string alone cannot pass."""
        with self._locked():
            record = self._load(experiment_id)
            if record["state"] != "compared":
                raise ValueError("only a compared candidate can be reviewed")
            self._check_comparison(record)
            if record["comparison"]["candidate"].get("e1_native_trial"):
                from .e1_review import validate_review

                checked = validate_review(self.home, record, receipt_path, self.gateway_status_reader)
                path = Path(receipt_path).resolve()
                receipt = checked["receipt"]
                status = checked["status"]
                record["review"] = {
                    "path": str(path), "sha256": sha256_file(path),
                    "verdict": receipt["verdict"],
                    "verifier_job_id": status["acceptance"]["verifier_job_id"],
                    "gateway_worker_job_id": receipt["gateway_worker_job_id"],
                    "gateway_status_digest": _digest(status),
                    "assignment_sha256": checked["assignment_sha256"],
                    "e1_native_trial": True,
                }
                record["state"] = "reviewed" if receipt["verdict"] == "accepted" else "rejected"
                record["updated_at"] = _now()
                write_json(self._record_path(experiment_id), record)
                return record
            path = Path(receipt_path).resolve()
            if (self.home / "verifications" / "experiments").resolve() != path.parent:
                raise ValueError("review receipt must be in the separate verification store")
            receipt = read_json(path)
            if (receipt.get("schema_version") != "mavis.experiment-review/v1"
                    or receipt.get("experiment_id") != experiment_id
                    or receipt.get("comparison_digest") != record["comparison"]["comparison_digest"]
                    or receipt.get("candidate_sha256") != record["candidate"]["sha256"]
                    or receipt.get("baseline_sha256") != record["baseline"]["sha256"]
                    or receipt.get("verdict") not in {"accepted", "rejected"}
                    or receipt.get("candidate_job_id") != record["comparison"]["candidate"]["candidate_job_id"]
                    or not receipt.get("gateway_worker_job_id")
                    or not receipt.get("verifier_job_id")
                    or receipt.get("verifier_job_id") == receipt.get("candidate_job_id")
                    or receipt.get("gateway_worker_job_id") == receipt.get("candidate_job_id")):
                raise ValueError("independent review receipt is missing or mismatched")
            candidate_status = self._candidate_status(receipt["candidate_job_id"], record)
            status = self._review_status(receipt["gateway_worker_job_id"], record)
            if status["mavis_binding"].get("report_sha256") != sha256_file(path):
                raise ValueError("review receipt is not the gateway worker's exact report")
            if receipt["verifier_job_id"] != status["acceptance"]["verifier_job_id"]:
                raise ValueError("review verifier does not match the native gateway receipt")
            record["review"] = {"path": str(path), "sha256": sha256_file(path),
                                "verdict": receipt["verdict"], "verifier_job_id": receipt["verifier_job_id"],
                                "gateway_worker_job_id": receipt["gateway_worker_job_id"],
                                "gateway_status_digest": _digest(status),
                                "candidate_job_id": receipt["candidate_job_id"],
                                "candidate_status_digest": _digest(candidate_status)}
            record["state"] = "reviewed" if receipt["verdict"] == "accepted" else "rejected"
            record["updated_at"] = _now()
            write_json(self._record_path(experiment_id), record)
            return record

    def _review_status(self, worker_job_id: str, record: dict[str, Any]) -> dict[str, Any]:
        status = self.gateway_status_reader(worker_job_id)
        if not isinstance(status, dict):
            raise ValueError("native gateway returned no review status")
        _validate_gateway_status(status, worker_job_id)
        acceptance = status.get("acceptance")
        binding = status.get("mavis_binding")
        requirements = binding.get("requirements") if isinstance(binding, dict) else None
        expected_requirements = review_assignment_requirements(record)
        if (status.get("job_id") != worker_job_id or status.get("state") != "completed"
                or status.get("exit_code") != 0 or status.get("accepted") is not True
                or not isinstance(acceptance, dict) or acceptance.get("accepted") is not True
                or acceptance.get("job_id") != worker_job_id or acceptance.get("verifier") != "terra"
                or not acceptance.get("verifier_job_id") or acceptance["verifier_job_id"] == worker_job_id
                or not isinstance(binding, dict) or binding.get("objective_id") != record["experiment_id"]
                or not isinstance(requirements, list)
                or not all(item in requirements for item in expected_requirements)):
            raise ValueError("native gateway did not independently accept this exact comparison")
        return status

    def _candidate_status(self, candidate_job_id: str, record: dict[str, Any]) -> dict[str, Any]:
        status = self.gateway_status_reader(candidate_job_id)
        if not isinstance(status, dict):
            raise ValueError("native gateway returned no candidate status")
        _validate_gateway_status(status, candidate_job_id)
        binding = status.get("mavis_binding")
        candidate = record["comparison"]["candidate"]
        if (not isinstance(binding, dict) or binding.get("objective_id") != record["experiment_id"]
                or binding.get("report_sha256") != candidate["evidence"]["sha256"]
                or not isinstance(binding.get("requirements"), list)
                or not all(item in binding["requirements"] for item in candidate_assignment_requirements(record))):
            raise ValueError("native candidate job is not bound to candidate evidence")
        return status

    def _check_review(self, record: dict[str, Any]) -> None:
        review = record.get("review") or {}
        if sha256_file(Path(review["path"])) != review["sha256"]:
            raise ValueError("review receipt changed")
        if review.get("e1_native_trial"):
            from .e1_review import validate_review

            checked = validate_review(self.home, record, Path(review["path"]), self.gateway_status_reader)
            if (checked["receipt"].get("verdict") != "accepted"
                    or checked["assignment_sha256"] != review["assignment_sha256"]
                    or _digest(checked["status"]) != review["gateway_status_digest"]
                    or checked["receipt"].get("gateway_worker_job_id") != review["gateway_worker_job_id"]
                    or checked["status"]["acceptance"].get("verifier_job_id") != review["verifier_job_id"]):
                raise ValueError("independent E1 review status changed")
            return
        receipt = read_json(Path(review["path"]))
        candidate_status = self._candidate_status(review["candidate_job_id"], record)
        status = self._review_status(review["gateway_worker_job_id"], record)
        if (receipt.get("verdict") != "accepted" or _digest(status) != review["gateway_status_digest"]
                or _digest(candidate_status) != review["candidate_status_digest"]
                or status["mavis_binding"].get("report_sha256") != review["sha256"]
                or receipt.get("candidate_job_id") != review["candidate_job_id"]
                or receipt.get("verifier_job_id") != status["acceptance"]["verifier_job_id"]):
            raise ValueError("independent review status changed")

    def stage(self, experiment_id: str) -> dict[str, Any]:
        with self._locked():
            record = self._load(experiment_id)
            if record["state"] != "reviewed":
                raise ValueError("candidate requires accepted independent review")
            self._check_comparison(record)
            self._check_review(record)
            active = read_json(self._active_path(record["scope"]))
            if active["configuration"] != record["baseline"]:
                raise ValueError("active baseline changed since experiment began")
            staged_path = self.root / "staged" / f"{record['scope']}.json"
            if staged_path.exists():
                staged = read_json(staged_path)
                if staged.get("experiment_id") != experiment_id or staged.get("baseline") != record["baseline"]:
                    raise ValueError("scope already has a staged candidate")
            else:
                staged = {"schema_version": "mavis.experiment-stage/v1", "experiment_id": experiment_id,
                          "baseline": record["baseline"], "candidate": record["candidate"], "at": _now()}
                write_json(staged_path, staged)
            record["state"] = "staged"
            record["updated_at"] = _now()
            write_json(self._record_path(experiment_id), record)
            return record

    def promote(self, experiment_id: str, *, between_objectives: bool) -> dict[str, Any]:
        if not between_objectives:
            raise ValueError("promotion is allowed only between objectives")
        with _profile_transition_lease(self.home), profile_boundary_lock(self.home), self._locked():
            self._assert_objective_boundary()
            record = self._load(experiment_id)
            if record["state"] != "staged":
                raise ValueError("only a staged candidate can be promoted")
            self._check_comparison(record)
            self._check_review(record)
            staged_path = self.root / "staged" / f"{record['scope']}.json"
            staged = read_json(staged_path)
            active_path = self._active_path(record["scope"])
            active = read_json(active_path)
            if staged.get("experiment_id") != experiment_id:
                raise ValueError("staged candidate or active baseline changed")
            self._read_snapshot(record["candidate"])
            if active["configuration"] == record["baseline"]:
                previous = deepcopy(active)
                write_json(active_path, {"schema_version": "mavis.experiment-active/v1", "scope": record["scope"],
                                         "configuration": record["candidate"], "experiment_id": experiment_id,
                                         "previous": previous, "updated_at": _now()})
            elif active.get("experiment_id") != experiment_id or active["configuration"] != record["candidate"]:
                raise ValueError("staged candidate or active baseline changed")
            record["state"] = "promoted"
            record["updated_at"] = _now()
            record["history"].append({"event": "promoted", "at": record["updated_at"]})
            write_json(self._record_path(experiment_id), record)
            staged_path.unlink()
            return record

    def rollback(self, experiment_id: str, *, reason: str) -> dict[str, Any]:
        if not reason.strip():
            raise ValueError("rollback requires a reason")
        with _profile_transition_lease(self.home), profile_boundary_lock(self.home), self._locked():
            self._assert_objective_boundary()
            record = self._load(experiment_id)
            active_path = self._active_path(record["scope"])
            active = read_json(active_path)
            if record["state"] != "promoted":
                raise ValueError("only the current promoted candidate can be rolled back")
            if active.get("experiment_id") == experiment_id:
                previous = active.get("previous")
                if not isinstance(previous, dict) or previous.get("configuration") != record["baseline"]:
                    raise ValueError("previous accepted configuration is missing")
                self._read_snapshot(previous["configuration"])
                write_json(active_path, previous)
            elif active["configuration"] != record["baseline"]:
                raise ValueError("only the current promoted candidate can be rolled back")
            record["state"] = "rolled-back"
            record["updated_at"] = _now()
            record["history"].append({"event": "rolled-back", "reason": reason, "at": record["updated_at"]})
            write_json(self._record_path(experiment_id), record)
            return record

    def assert_promoted(self, experiment_id: str) -> dict[str, Any]:
        """Revalidate a promoted candidate before another store activates it."""
        with self._locked():
            record = self._load(experiment_id)
            if record["state"] != "promoted":
                raise ValueError("experiment is not promoted")
            self._check_comparison(record)
            self._check_review(record)
            active = read_json(self._active_path(record["scope"]))
            if active.get("experiment_id") != experiment_id or active.get("configuration") != record["candidate"]:
                raise ValueError("experiment is not the active promoted candidate")
            return record

    def enqueue(self, experiment_id: str, interval: str = "daily") -> dict[str, Any]:
        record = self.load(experiment_id)
        if record["state"] in {"promoted", "rejected", "rolled-back"}:
            raise ValueError("finished experiment cannot be queued")
        return self.queue.enqueue(interval, "experiment", {"experiment_id": experiment_id})

    def checkpoint(self, job_id: str, *, foreground_active: bool) -> dict[str, Any]:
        job = self.queue.get(job_id)
        if job["kind"] != "experiment" or job["state"] != "running":
            raise ValueError("only a running experiment job can checkpoint")
        record = self.load(job["payload"]["experiment_id"])
        return self.queue.checkpoint(job_id, {"experiment_id": record["experiment_id"],
                                             "state": record["state"],
                                             "record_sha256": sha256_file(self._record_path(record["experiment_id"]))},
                                     foreground_active=foreground_active)

    def resume_checkpoint(self, job_id: str) -> dict[str, Any]:
        job = self.queue.get(job_id)
        if job["kind"] != "experiment" or job["state"] != "running":
            raise ValueError("experiment job must be claimed before resuming")
        checkpoint = job["checkpoint"]
        record = self.load(job["payload"]["experiment_id"])
        if checkpoint and (checkpoint.get("experiment_id") != record["experiment_id"]
                           or checkpoint.get("state") != record["state"]
                           or checkpoint.get("record_sha256") != sha256_file(self._record_path(record["experiment_id"]))):
            raise ValueError("experiment checkpoint no longer matches durable state")
        return record
