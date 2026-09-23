"""Resumable maintenance queue that yields to foreground work."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any
import uuid


INTERVALS = {"immediate", "daily", "weekly", "monthly"}
STATES = {"queued", "running", "paused", "complete", "failed", "cancelled"}
CANCELLABLE_STATES = {"queued", "paused"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        return connection

    def enqueue(self, interval: str, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        if interval not in INTERVALS:
            raise ValueError(f"unknown maintenance interval: {interval}")
        if not kind.strip():
            raise ValueError("maintenance kind must not be empty")
        job_id = uuid.uuid4().hex
        now = _now()
        row = {
            "job_id": job_id,
            "interval": interval,
            "kind": kind,
            "payload": payload,
            "state": "queued",
            "checkpoint": {},
            "created_at": now,
            "updated_at": now,
            "detail": "",
        }
        connection = self._connect()
        try:
            connection.execute(
                "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)",
                (job_id, interval, kind, json.dumps(payload, sort_keys=True), "queued", "{}", now, now, ""),
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
