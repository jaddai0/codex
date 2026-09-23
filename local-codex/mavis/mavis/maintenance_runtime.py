"""A model-free, one-shard maintenance tick with host-observed admission."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import fcntl
import os
from pathlib import Path
import subprocess
from typing import Callable
import uuid

from .archive_hygiene import ArchiveRetention, ForegroundYield, RETENTION_DAYS, _parse, _stamp
from .maintenance import HOST_KINDS, MaintenanceQueue
from .runtime import handoff_lease_fd, inventory
from .storage import read_json, sha256_file, write_json


RUNTIME_KINDS = HOST_KINDS
BENCHMARK_MARKERS = ("bench.mjs", "replay.mjs", "benchmark.py",
                     "mlx_lm.benchmark", "mlx_lm.generate")


def _lock_available(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return True
    finally:
        os.close(descriptor)


def host_admission(home: Path, *, iris_endpoint: str = "http://127.0.0.1:8000/v1",
                   lease_held: bool = False) -> tuple[bool, str]:
    """Defer unless the Mavis lease and read-only IRIS signals are clear."""
    if lease_held:
        handoff_lease_fd()
    if not lease_held and not _lock_available(Path(home) / "generation.lock"):
        return False, "Mavis foreground generation owns its host lease"
    try:
        records = inventory(iris_endpoint)
    except (OSError, RuntimeError, ValueError) as error:
        return False, f"IRIS model state is unavailable: {type(error).__name__}"
    if any(record.get("loaded") and record.get("model_type") != "embedding" for record in records):
        return False, "IRIS has a loaded generation model"
    try:
        processes = subprocess.run(
            ["ps", "-axo", "command="], capture_output=True, text=True,
            timeout=3, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, f"benchmark process state is unavailable: {type(error).__name__}"
    if processes.returncode != 0:
        return False, "benchmark process state is unavailable"
    if any(marker in command for command in processes.stdout.splitlines()
           for marker in BENCHMARK_MARKERS):
        return False, "a local model benchmark process is active"
    lease_state = "Mavis lease held by caller" if lease_held else "Mavis lease free"
    return True, f"{lease_state}; no loaded IRIS generation model or known benchmark"


def _due_project(retention: ArchiveRetention, now: datetime) -> str | None:
    if not retention.root.exists():
        return None
    if retention.root.is_symlink():
        raise ValueError("archive project registry is a symlink")
    for path in sorted(retention.root.glob("*.json")):
        if path.is_symlink():
            raise ValueError("archive project record is a symlink")
        record = read_json(path)
        project_id = record.get("project_id")
        if (record.get("schema_version") != "mavis.archive-project/v1"
                or not isinstance(project_id, str)
                or retention._path(project_id) != path):
            raise ValueError("archive project registry contains a mismatched record")
        if record.get("state") != "closed":
            continue
        if now - _parse(record["closed_at"]) < timedelta(days=RETENTION_DAYS):
            continue
        last = record.get("last_sweep_at")
        if last and _parse(last).date() >= now.date():
            continue
        return project_id
    return None


def _mark_project_checked(retention: ArchiveRetention, project_id: str, now: datetime) -> None:
    with retention._registry_lock():
        path = retention._path(project_id)
        current = read_json(path)
        if (current.get("state") != "closed"
                or now - _parse(current["closed_at"]) < timedelta(days=RETENTION_DAYS)):
            raise ValueError("archive project changed before sweep completion")
        current["last_sweep_at"] = _stamp(now)
        write_json(path, current)


def tick(
    home: Path, *, now: datetime | None = None,
    admit: Callable[[], tuple[bool, str]] | None = None,
) -> dict[str, object]:
    """Run one due archive segment, or durably record why no work ran."""
    home = Path(home).resolve()
    instant = now or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        raise ValueError("maintenance clock must include a timezone")
    instant = instant.astimezone(timezone.utc)
    root = home / "maintenance"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    reader = admit or (lambda: host_admission(home))
    lock_path = root / "tick.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    receipt: dict[str, object] = {
        "schema_version": "mavis.maintenance-tick/v1", "checked_at": _stamp(instant),
        "kind": "archive-sweep", "model_free": True,
        "iris_request_level_idle_proven": False,
    }
    share = os.environ.get("LOCAL_CODEX_SHARE_DIR")
    if share:
        manifest = Path(share) / "install-manifest.json"
        receipt["installed_manifest_sha256"] = sha256_file(manifest) if manifest.is_file() else None
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            receipt.update(status="deferred", reason="another maintenance tick owns the runner lock")
        else:
            queue = MaintenanceQueue(home)
            allowed, reason = reader()
            if not allowed:
                receipt.update(status="deferred", reason=reason)
            else:
                jobs = [job for job in queue.list() if job["kind"] == "archive-sweep"]
                if not jobs:
                    queue.enqueue("daily", "archive-sweep", {}, now=instant, due_now=True)
                elif any(job["state"] == "failed" for job in jobs):
                    receipt.update(status="failed-retained", reason="archive sweep job requires operator review")
                if "status" not in receipt:
                    job = queue.claim_due(kinds=RUNTIME_KINDS, now=instant)
                    if job is None:
                        receipt.update(status="not-due")
                    else:
                        receipt["job_id"] = job["job_id"]
                        allowed, reason = reader()
                        if not allowed:
                            queue.yield_runtime(job["job_id"], job["checkpoint"], now=instant,
                                                delay=timedelta(minutes=1))
                            receipt.update(status="deferred", reason=reason)
                        else:
                            retention = ArchiveRetention(home)
                            try:
                                checkpoint = job["checkpoint"]
                                project_id = checkpoint.get("project_id") or _due_project(retention, instant)
                                if project_id is None:
                                    completed = queue.finish_runtime(job["job_id"], success=True,
                                                                     detail="no due closed archive remains", now=instant)
                                    receipt.update(status="complete-period", next_due_at=completed["next_due_at"])
                                else:
                                    cursor = checkpoint.get("cursor")
                                    if cursor is not None and (not isinstance(cursor, list) or len(cursor) != 2):
                                        raise ValueError("archive shard checkpoint is invalid")
                                    shard = retention.compact_one(
                                        project_id, now=instant,
                                        cursor=tuple(cursor) if cursor else None,
                                        admit=lambda: reader()[0],
                                    )
                                    if shard["complete"]:
                                        if not reader()[0]:
                                            raise ForegroundYield("foreground work began before sweep completion")
                                        _mark_project_checked(retention, project_id, instant)
                                        next_checkpoint = {}
                                    else:
                                        next_checkpoint = {"project_id": project_id,
                                                           "cursor": shard["cursor"]}
                                    queue.yield_runtime(job["job_id"], next_checkpoint, now=instant)
                                    receipt.update(status="shard", project_id=project_id,
                                                   shard=shard, checkpoint=next_checkpoint)
                            except ForegroundYield as error:
                                queue.yield_runtime(job["job_id"], job["checkpoint"], now=instant,
                                                    delay=timedelta(minutes=1))
                                receipt.update(status="deferred", reason=str(error))
                            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                                queue.finish_runtime(job["job_id"], success=False,
                                                     detail=f"{type(error).__name__}: {error}", now=instant)
                                receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        os.close(descriptor)
    receipt_path = root / "runs" / f"{instant.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex}.json"
    write_json(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path)}
