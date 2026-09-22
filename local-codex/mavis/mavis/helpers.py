"""Isolated helper identities and short-lived follow-up context."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from .storage import write_json


HELPER_ROLES = {"librarian", "output-reader"}


class HelperSession:
    def __init__(self, home: Path, role: str, ttl_seconds: int = 60):
        if role not in HELPER_ROLES:
            raise ValueError(f"unknown helper role: {role}")
        self.role = role
        self.root = Path(home) / "helpers" / role
        self.ttl_seconds = ttl_seconds
        self.cache_path = self.root / "query-context.json"

    def store_query_context(self, query: str, citations: list[dict[str, Any]], *, now: float | None = None) -> None:
        observed = time.time() if now is None else now
        write_json(
            self.cache_path,
            {
                "schema_version": "mavis.helper-query-context/v1",
                "role": self.role,
                "query": query,
                "citations": citations,
                "expires_at_epoch": observed + self.ttl_seconds,
            },
        )

    def followup_context(self, *, now: float | None = None) -> dict[str, Any] | None:
        observed = time.time() if now is None else now
        if not self.cache_path.is_file():
            return None
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.clear()
            return None
        if payload.get("role") != self.role or observed >= float(payload.get("expires_at_epoch", 0)):
            self.clear()
            return None
        return payload

    def clear(self) -> None:
        self.cache_path.unlink(missing_ok=True)

    def clear_for_compute_pressure(self) -> None:
        self.clear()
