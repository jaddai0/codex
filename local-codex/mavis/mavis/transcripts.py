"""Complete transcript segments, compaction handoffs, and exact archive search."""

from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import gzip
import hashlib
import fcntl
import json
import os
from pathlib import Path
from typing import Iterator
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

    @contextmanager
    def _locked(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        descriptor = os.open(self.root / "import.lock", os.O_CREAT | os.O_RDWR, 0o600)
        with os.fdopen(descriptor, "rb") as lock:
            os.fchmod(lock.fileno(), 0o600)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            yield

    def append_segment(
        self, messages: list[dict[str, Any]], source: dict[str, Any] | None = None
    ) -> Path:
        with self._locked():
            return self._append_segment_locked(messages, source)

    def _append_segment_locked(
        self, messages: list[dict[str, Any]], source: dict[str, Any] | None = None
    ) -> Path:
        if not messages:
            raise ValueError("cannot archive an empty transcript segment")
        from .archive_hygiene import prepare_evidence_write
        prepare_evidence_write(self.root.parent.parent)
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
        with self._locked():
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
        return self._append_segment_locked(
            messages,
            {
                "path": str(rollout_path),
                "start": offset,
                "end": len(data),
                "prefix_sha256": hashlib.sha256(data).hexdigest(),
            },
        )

    def write_handoff(self, payload: dict[str, Any]) -> Path:
        with self._locked():
            return self._write_handoff_locked(payload)

    def _write_handoff_locked(self, payload: dict[str, Any]) -> Path:
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

    def resolve_segment_file(self, declared_path: str | Path) -> Path:
        """Resolve and validate a manifest segment path.

        A segment must be a regular ``.jsonl`` file located directly inside
        this archive's ``segments`` directory. Symlinks in the archive-owned
        transcript, conversation, or segments directories or the file, and any
        path that resolves outside the segments directory are rejected, so a
        forged manifest cannot expose an external file even when its hash
        matches.
        """
        path = Path(declared_path)
        if not (path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz")):
            raise ValueError(f"segment must be a .jsonl or .jsonl.gz file: {path}")
        for directory in (self.root.parent, self.root, self.root / "segments"):
            if directory.is_symlink():
                raise ValueError(f"archive directory escapes the archive: {directory}")
        if path.is_symlink():
            raise ValueError(f"segment symlink escapes the archive: {path}")
        resolved = path.resolve()
        if resolved.parent != (self.root / "segments").resolve():
            raise ValueError(f"segment escapes the archive: {resolved}")
        if not resolved.is_file():
            raise ValueError(f"segment is missing: {resolved}")
        return resolved

    def verify_segment(self, segment: dict[str, Any]) -> Path:
        """Verify stored bytes and the original transcript before exposing lines."""
        path = self.resolve_segment_file(segment["path"])
        if segment.get("compression") == "gzip":
            if not path.name.endswith(".jsonl.gz") or sha256_file(path) != segment.get("compressed_sha256"):
                raise ValueError(f"compressed transcript segment is missing or changed: {path}")
            digest = hashlib.sha256()
            try:
                with gzip.open(path, "rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
            except (OSError, EOFError) as exc:
                raise ValueError(f"compressed transcript segment cannot be reconstructed: {path}") from exc
            if digest.hexdigest() != segment["sha256"]:
                raise ValueError(f"reconstructed transcript segment changed: {path}")
        elif path.name.endswith(".jsonl") and sha256_file(path) == segment["sha256"]:
            pass
        else:
            raise ValueError(f"transcript segment is missing or changed: {path}")
        return path

    def segment_lines(self, segment: dict[str, Any]) -> Iterator[str]:
        path = self.verify_segment(segment)
        opener = gzip.open if segment.get("compression") == "gzip" else open
        with opener(path, "rt", encoding="utf-8") as source:
            yield from source

    def search(
        self, query: str, limit: int = 20, offset: int = 0
    ) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query cannot be empty")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be nonnegative")
        if type(limit) is not int or not 1 <= limit <= 200:
            raise ValueError("limit must be 1..200")
        if not self.manifest_path.exists():
            return []
        results: list[dict[str, Any]] = []
        matched = 0
        manifest = read_json(self.manifest_path)
        for segment in manifest["segments"]:
            path = self.verify_segment(segment)
            if len(results) >= limit:
                continue
            for line_number, line in enumerate(self.segment_lines(segment), 1):
                line = line.rstrip("\n")
                if query.casefold() in line.casefold():
                    if matched >= offset:
                        results.append(
                            {
                                "path": str(path),
                                "line": line_number,
                                "text": line,
                                "sha256": segment["sha256"],
                            }
                        )
                    matched += 1
                    if len(results) >= limit:
                        break
        return results
