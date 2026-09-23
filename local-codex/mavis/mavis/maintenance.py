"""Resumable maintenance queue that yields to foreground work."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import calendar
import json
from pathlib import Path
import sqlite3
from typing import Any
import uuid


INTERVALS = {"immediate", "daily", "weekly", "monthly"}
HOST_KINDS = {"archive-sweep"}
STATES = {"queued", "running", "paused", "complete", "failed", "cancelled"}
CANCELLABLE_STATES = {"queued", "paused"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _next_due(instant: datetime, interval: str, *, anchor_day: int | None = None) -> datetime:
    if interval == "immediate":
        return instant
    if interval == "daily":
        return instant + timedelta(days=1)
    if interval == "weekly":
        return instant + timedelta(days=7)
    if interval == "monthly":
        month = instant.month + 1
        year = instant.year + (month > 12)
        month = (month - 1) % 12 + 1
        day = min(anchor_day or instant.day, calendar.monthrange(year, month)[1])
        return instant.replace(year=year, month=month, day=day)
    raise ValueError(f"unknown maintenance interval: {interval}")


class MaintenanceQueue:
    def __init__(self, home: Path):
        self.root = Path(home) / "maintenance"
        self.database = self.root / "queue.sqlite3"

    def _connect(self) -> sqlite3.Connection:
        self.root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS jobs (
              job_id TEXT PRIMARY KEY,
              interval TEXT NOT NULL,
              kind TEXT NOT NULL,
              payload TEXT NOT NULL,
              state TEXT NOT NULL,
              checkpoint TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              detail TEXT NOT NULL
            );
            """
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
        if "next_due_at" not in columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN next_due_at TEXT")
            connection.execute("UPDATE jobs SET next_due_at=created_at WHERE next_due_at IS NULL")
        if "lease_until" not in columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN lease_until TEXT")
            connection.execute("UPDATE jobs SET lease_until=updated_at WHERE state='running' AND lease_until IS NULL")
        if "next_due_at" not in columns or "lease_until" not in columns:
            connection.commit()
        return connection

    def enqueue(self, interval: str, kind: str, payload: dict[str, Any], *,
                now: datetime | None = None, due_now: bool = False) -> dict[str, Any]:
        if interval not in INTERVALS:
            raise ValueError(f"unknown maintenance interval: {interval}")
        if not kind.strip():
            raise ValueError("maintenance kind must not be empty")
        job_id = uuid.uuid4().hex
        instant = now or datetime.now(timezone.utc)
        if instant.tzinfo is None:
            raise ValueError("maintenance clock must include a timezone")
        stamp = instant.astimezone(timezone.utc).isoformat()
        due = (instant if due_now else _next_due(instant, interval)).astimezone(timezone.utc).isoformat()
        row = {
            "job_id": job_id,
            "interval": interval,
            "kind": kind,
            "payload": payload,
            "state": "queued",
            "checkpoint": {},
            "created_at": stamp,
            "updated_at": stamp,
            "next_due_at": due,
            "lease_until": None,
            "detail": "",
        }
        connection = self._connect()
        try:
            connection.execute(
                "INSERT INTO jobs (job_id,interval,kind,payload,state,checkpoint,created_at,updated_at,detail,next_due_at,lease_until) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (job_id, interval, kind, json.dumps(payload, sort_keys=True), "queued", "{}", stamp, stamp, "", due, None),
            )
            connection.commit()
        finally:
            connection.close()
        return row

    def _row(self, row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        result["checkpoint"] = json.loads(result["checkpoint"])
        return result

    def get(self, job_id: str) -> dict[str, Any]:
        connection = self._connect()
        try:
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            return self._row(row)
        finally:
            connection.close()

    def claim_next(self, *, foreground_active: bool) -> dict[str, Any] | None:
        if foreground_active:
            return None
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM jobs WHERE state IN ('queued','paused') ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            connection.execute(
                "UPDATE jobs SET state='running', updated_at=? WHERE job_id=?",
                (_now(), row["job_id"]),
            )
            connection.commit()
            return self.get(row["job_id"])
        finally:
            connection.close()

    def claim_due(self, *, kinds: set[str], now: datetime,
                  lease_seconds: int = 300) -> dict[str, Any] | None:
        """Claim only allowlisted, due jobs; recover expired owner leases."""
        if not kinds or not kinds <= HOST_KINDS:
            raise ValueError("maintenance due claim requires an approved host kind allowlist")
        if now.tzinfo is None:
            raise ValueError("maintenance clock must include a timezone")
        stamp = now.astimezone(timezone.utc).isoformat()
        lease_until = (now.astimezone(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            placeholders = ",".join("?" for _ in kinds)
            row = connection.execute(
                f"SELECT * FROM jobs WHERE kind IN ({placeholders}) AND "
                "((state IN ('queued','paused') AND next_due_at<=?) "
                "OR (state='running' AND lease_until<=?)) "
                "ORDER BY next_due_at,created_at LIMIT 1",
                (*sorted(kinds), stamp, stamp),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            connection.execute(
                "UPDATE jobs SET state='running', updated_at=?, lease_until=? WHERE job_id=?",
                (stamp, lease_until, row["job_id"]),
            )
            connection.commit()
            return self.get(row["job_id"])
        finally:
            connection.close()

    def yield_runtime(self, job_id: str, checkpoint: dict[str, Any], *,
                      now: datetime, delay: timedelta = timedelta(0)) -> dict[str, Any]:
        """Persist one shard cursor before another bounded tick may claim it."""
        stamp = now.astimezone(timezone.utc).isoformat()
        due = (now.astimezone(timezone.utc) + delay).isoformat()
        connection = self._connect()
        try:
            cursor = connection.execute(
                "UPDATE jobs SET state='paused', checkpoint=?, updated_at=?, "
                "next_due_at=?, lease_until=NULL WHERE job_id=? AND state='running'",
                (json.dumps(checkpoint, sort_keys=True), stamp, due, job_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("only a running maintenance job can yield")
            connection.commit()
        finally:
            connection.close()
        return self.get(job_id)

    def finish_runtime(self, job_id: str, *, success: bool, detail: str,
                       now: datetime) -> dict[str, Any]:
        """Advance a successful periodic job from its prior due time."""
        current = self.get(job_id)
        if current["state"] != "running":
            raise ValueError("only a running maintenance job can finish")
        stamp = now.astimezone(timezone.utc).isoformat()
        if success and current["interval"] != "immediate":
            due = datetime.fromisoformat(current["next_due_at"])
            anchor_day = datetime.fromisoformat(current["created_at"]).day
            while due <= now.astimezone(timezone.utc):
                due = _next_due(due, current["interval"], anchor_day=anchor_day)
            state, next_due_at = "queued", due.isoformat()
        else:
            state, next_due_at = ("complete" if success else "failed"), current["next_due_at"]
        connection = self._connect()
        try:
            cursor = connection.execute(
                "UPDATE jobs SET state=?, detail=?, updated_at=?, next_due_at=?, "
                "checkpoint='{}', lease_until=NULL WHERE job_id=? AND state='running'",
                (state, detail, stamp, next_due_at, job_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("maintenance job changed before finish")
            connection.commit()
        finally:
            connection.close()
        return self.get(job_id)

    def checkpoint(self, job_id: str, checkpoint: dict[str, Any], *, foreground_active: bool) -> dict[str, Any]:
        target = "paused" if foreground_active else "running"
        connection = self._connect()
        try:
            current = connection.execute("SELECT state FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if current is None:
                raise KeyError(job_id)
            if current["state"] != "running":
                raise ValueError("only a running maintenance job can checkpoint")
            connection.execute(
                "UPDATE jobs SET state=?, checkpoint=?, updated_at=? WHERE job_id=?",
                (target, json.dumps(checkpoint, sort_keys=True), _now(), job_id),
            )
            connection.commit()
        finally:
            connection.close()
        return self.get(job_id)

    def finish(self, job_id: str, *, success: bool, detail: str) -> dict[str, Any]:
        connection = self._connect()
        try:
            current = connection.execute("SELECT state FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if current is None:
                raise KeyError(job_id)
            if current["state"] != "running":
                raise ValueError("only a running maintenance job can finish")
            connection.execute(
                "UPDATE jobs SET state=?, detail=?, updated_at=? WHERE job_id=?",
                ("complete" if success else "failed", detail, _now(), job_id),
            )
            connection.commit()
        finally:
            connection.close()
        return self.get(job_id)

    def cancel(self, job_id: str, *, reason: str) -> dict[str, Any]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT state FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if current is None:
                raise KeyError(job_id)
            if current["state"] not in CANCELLABLE_STATES:
                raise ValueError(
                    f"cannot cancel maintenance job in state: {current['state']}"
                )
            connection.execute(
                "UPDATE jobs SET state='cancelled', detail=?, updated_at=? "
                "WHERE job_id=? AND state IN ('queued','paused')",
                (reason, _now(), job_id),
            )
            connection.commit()
        finally:
            connection.close()
        return self.get(job_id)

    def list(self, state: str | None = None) -> list[dict[str, Any]]:
        if state is not None and state not in STATES:
            raise ValueError(f"unknown maintenance state: {state}")
        connection = self._connect()
        try:
            if state:
                rows = connection.execute("SELECT * FROM jobs WHERE state=? ORDER BY created_at", (state,))
            else:
                rows = connection.execute("SELECT * FROM jobs ORDER BY created_at")
            return [self._row(row) for row in rows]
        finally:
            connection.close()
