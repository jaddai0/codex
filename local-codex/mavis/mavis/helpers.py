"""Isolated helper identities and short-lived follow-up context."""

from __future__ import annotations

import json
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import uuid

from .storage import require_safe_id, write_json


HELPER_ROLES = {"librarian", "output-reader"}
_EXPIRY_WORKERS: list[subprocess.Popen[bytes]] = []


class HelperSession:
    def __init__(self, home: Path, role: str, ttl_seconds: int = 60,
                 *, context_id: str | None = None):
        if role not in HELPER_ROLES:
            raise ValueError(f"unknown helper role: {role}")
        if ttl_seconds <= 0:
            raise ValueError("helper context TTL must be positive")
        self.home = Path(home)
        self.role = role
        self.root = self.home / "helpers" / role
        self.ttl_seconds = ttl_seconds
        self.context_id = (require_safe_id(context_id, "helper context id")
                           if context_id is not None else None)
        self.cache_path = (self.root / "contexts" / self.context_id / "query-context.json"
                           if self.context_id is not None else self.root / "query-context.json")
        self.lock_path = self.root / "query-context.lock"

    @contextmanager
    def _locked(self, *, create: bool = True):
        if create:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            descriptor = os.open(self.lock_path,
                                 (os.O_CREAT | os.O_RDWR) if create else os.O_RDWR,
                                 0o600)
        except FileNotFoundError:
            yield False
            return
        with os.fdopen(descriptor, "rb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield True
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def store_query_context(self, query: str, citations: list[dict[str, Any]], *, now: float | None = None) -> None:
        observed = time.time() if now is None else now
        lease_id = uuid.uuid4().hex
        expires_at = observed + self.ttl_seconds
        with self._locked():
            if ((self.root / "contexts").is_symlink()
                    or self.cache_path.parent.is_symlink()):
                raise ValueError("helper context directory is a symlink")
            self.cache_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            payload = {
                "schema_version": "mavis.helper-query-context/v1",
                "role": self.role,
                "query": query,
                "citations": citations,
                "expires_at_epoch": expires_at,
                "lease_id": lease_id,
            }
            if self.context_id is not None:
                payload["context_id"] = self.context_id
            write_json(self.cache_path, payload)
        # Synthetic clocks in unit tests are checked by followup_context().
        # Real one-shot CLI runs need a separate worker that survives CLI exit.
        if now is None:
            environment = {key: os.environ[key] for key in ("PYTHONPATH", "PYTHONHOME")
                           if key in os.environ}
            try:
                worker = subprocess.Popen(
                    [sys.executable, "-m", "mavis.helpers", "--expire",
                     str(self.home.resolve()), self.role, lease_id, str(expires_at),
                     self.context_id or "-"],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, start_new_session=True,
                    close_fds=True, env=environment,
                )
                # Keep the Popen handle until it exits so Python does not
                # report an active child as an untracked resource.
                _EXPIRY_WORKERS[:] = [item for item in _EXPIRY_WORKERS if item.poll() is None]
                _EXPIRY_WORKERS.append(worker)
            except OSError as exc:
                self._expire_lease(lease_id, force=True)
                raise RuntimeError("helper context expiry worker could not start") from exc

    def followup_context(self, *, now: float | None = None) -> dict[str, Any] | None:
        observed = time.time() if now is None else now
        with self._locked():
            if not self.cache_path.is_file():
                return None
            try:
                payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.cache_path.unlink(missing_ok=True)
                return None
            if (payload.get("role") != self.role
                    or payload.get("context_id") != self.context_id
                    or observed >= float(payload.get("expires_at_epoch", 0))):
                self.cache_path.unlink(missing_ok=True)
                return None
            return payload

    def _expire_lease(self, lease_id: str, *, force: bool = False) -> None:
        with self._locked(create=False) as present:
            if not present:
                return
            if not self.cache_path.is_file():
                return
            try:
                payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.cache_path.unlink(missing_ok=True)
                return
            if payload.get("lease_id") == lease_id and (force or time.time() >= payload.get("expires_at_epoch", 0)):
                self.cache_path.unlink(missing_ok=True)

    def clear(self) -> None:
        with self._locked():
            self.cache_path.unlink(missing_ok=True)

    def clear_for_compute_pressure(self) -> None:
        with self._locked():
            contexts = self.root / "contexts"
            if contexts.is_symlink():
                raise ValueError("helper context directory is a symlink")
            paths = [self.root / "query-context.json", *contexts.glob("*/query-context.json")]
            for path in paths:
                if path.is_symlink() or path.parent.is_symlink():
                    raise ValueError("helper context file is a symlink")
                path.unlink(missing_ok=True)


def _expiry_worker(argv: list[str]) -> int:
    if len(argv) not in (5, 6) or argv[0] != "--expire":
        raise ValueError("invalid helper expiry worker invocation")
    _, home, role, lease_id, deadline = argv[:5]
    context_id = argv[5] if len(argv) == 6 and argv[5] != "-" else None
    if len(lease_id) != 32 or any(character not in "0123456789abcdef" for character in lease_id):
        raise ValueError("invalid helper context lease")
    expires_at = float(deadline)
    if not -120 <= expires_at - time.time() <= 120:
        raise ValueError("helper expiry deadline is outside the bounded window")
    time.sleep(max(0, expires_at - time.time()))
    HelperSession(Path(home), role, context_id=context_id)._expire_lease(lease_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(_expiry_worker(sys.argv[1:]))
