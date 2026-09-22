"""Complete transcript segments, compaction handoffs, and exact archive search."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import uuid
from typing import Any

from .storage import read_json, require_safe_id, sha256_file, write_json


HANDOFF_FIELDS = (
    "goals",
    "accepted_decisions",
    "completed_requirements",
    "current_changes",
    "recent_work",
    "unresolved_failures",
    "evidence_links",
)


class TranscriptArchive:
    def __init__(self, home: Path, conversation_id: str):
        self.conversation_id = require_safe_id(conversation_id, "conversation id")
        self.root = Path(home) / "transcripts" / self.conversation_id
        self.manifest_path = self.root / "manifest.json"

    def append_segment(
        self, messages: list[dict[str, Any]], source: dict[str, Any] | None = None
    ) -> Path:
        if not messages:
            raise ValueError("cannot archive an empty transcript segment")
        segment_id = uuid.uuid4().hex
        path = self.root / "segments" / f"{segment_id}.jsonl"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for message in messages:
                handle.write(
                    json.dumps(message, sort_keys=True, ensure_ascii=False) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        manifest = (
            read_json(self.manifest_path)
            if self.manifest_path.exists()
            else {
                "schema_version": "mavis.transcript-manifest/v1",
                "conversation_id": self.conversation_id,
                "segments": [],
            }
        )
        manifest["segments"].append(
            {
                "segment_id": segment_id,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "messages": len(messages),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                **({"source": source} if source else {}),
            }
        )
        write_json(self.manifest_path, manifest)
        return path

    def import_rollout(self, rollout_path: Path) -> Path | None:
        """Archive only complete JSONL records added since the last import."""
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        lock_path = self.root / "import.lock"
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        with os.fdopen(descriptor, "rb") as lock:
            os.fchmod(lock.fileno(), 0o600)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            return self._import_rollout_locked(rollout_path)

    def _import_rollout_locked(self, rollout_path: Path) -> Path | None:
        rollout_path = Path(rollout_path).resolve(strict=True)
        manifest = (
            read_json(self.manifest_path)
            if self.manifest_path.exists()
            else {"segments": []}
        )
        previous = [
            entry["source"] for entry in manifest["segments"] if "source" in entry
        ]
        if previous and any(item["path"] != str(rollout_path) for item in previous):
            raise ValueError("conversation archive is bound to a different rollout")
        data = rollout_path.read_bytes()
        offset = previous[-1]["end"] if previous else 0
        if len(data) < offset or (
            previous
            and hashlib.sha256(data[:offset]).hexdigest()
            != previous[-1]["prefix_sha256"]
        ):
            raise ValueError("rollout changed before the archived offset")
        added = data[offset:]
        if not added:
            return None
        if not added.endswith(b"\n"):
            raise ValueError("rollout ends with an incomplete JSONL record")
        messages = []
        for line in added.splitlines():
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("rollout record must be an object")
            messages.append(item)
        return self.append_segment(
            messages,
            {
                "path": str(rollout_path),
                "start": offset,
                "end": len(data),
                "prefix_sha256": hashlib.sha256(data).hexdigest(),
            },
        )

    def write_handoff(self, payload: dict[str, Any]) -> Path:
        missing = [field for field in HANDOFF_FIELDS if field not in payload]
        if missing:
            raise ValueError(f"compaction handoff is missing: {', '.join(missing)}")
        if not self.manifest_path.exists():
            raise ValueError("archive at least one complete transcript segment first")
        handoff = dict(payload)
        handoff["schema_version"] = "mavis.compaction-handoff/v1"
        handoff["conversation_id"] = self.conversation_id
        handoff["manifest"] = str(self.manifest_path.resolve())
        handoff["written_at"] = datetime.now(timezone.utc).isoformat()
        path = self.root / "handoffs" / f"{uuid.uuid4().hex}.json"
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        write_json(path, handoff)
        return path

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query cannot be empty")
        if not self.manifest_path.exists():
            return []
        results: list[dict[str, Any]] = []
        manifest = read_json(self.manifest_path)
        for segment in manifest["segments"]:
            path = Path(segment["path"])
            if not path.is_file() or sha256_file(path) != segment["sha256"]:
                raise ValueError(f"transcript segment is missing or changed: {path}")
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if query.casefold() in line.casefold():
                    results.append(
                        {
                            "path": str(path),
                            "line": line_number,
                            "text": line,
                            "sha256": segment["sha256"],
                        }
                    )
                    if len(results) >= limit:
                        return results
        return results
