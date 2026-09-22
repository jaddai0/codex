"""Complete transcript segments, compaction handoffs, and exact archive search."""

from __future__ import annotations

from datetime import datetime, timezone
import json
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

    def append_segment(self, messages: list[dict[str, Any]]) -> Path:
        if not messages:
            raise ValueError("cannot archive an empty transcript segment")
        segment_id = uuid.uuid4().hex
        path = self.root / "segments" / f"{segment_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            for message in messages:
                handle.write(json.dumps(message, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
        manifest = (
            read_json(self.manifest_path)
            if self.manifest_path.exists()
            else {"schema_version": "mavis.transcript-manifest/v1", "conversation_id": self.conversation_id, "segments": []}
        )
        manifest["segments"].append(
            {
                "segment_id": segment_id,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "messages": len(messages),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(self.manifest_path, manifest)
        return path

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
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if query.casefold() in line.casefold():
                    results.append({"path": str(path), "line": line_number, "text": line, "sha256": segment["sha256"]})
                    if len(results) >= limit:
                        return results
        return results
