"""Closed-project transcript retention and conservative storage-pressure actions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import warnings

from .storage import read_json, require_safe_id, sha256_file, write_json
from .transcripts import TranscriptArchive


RETENTION_DAYS = 30
PRESSURE_FREE_BYTES = 20 * 1024**3
PRESSURE_FREE_FRACTION = 0.10


def _stamp(instant: datetime) -> str:
    if instant.tzinfo is None:
        raise ValueError("retention clock must include a timezone")
    return instant.astimezone(timezone.utc).isoformat()


def _parse(stamp: str) -> datetime:
    parsed = datetime.fromisoformat(stamp)
    if parsed.tzinfo is None:
        raise ValueError("archive activity timestamp lacks a timezone")
    return parsed.astimezone(timezone.utc)


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


class ArchiveRetention:
    """Explicit project closure is required before any raw duplicate is removed."""

    def __init__(self, home: Path):
        self.home = Path(home).resolve()
        self.root = self.home / "archive-projects"

    def _path(self, project_id: str) -> Path:
        path = self.root / f"{require_safe_id(project_id, 'project id')}.json"
        if self.root.is_symlink() or path.is_symlink():
            raise ValueError("archive project record is a symlink")
        return path

    def register(
        self, project_id: str, conversation_ids: list[str], objective_ids: list[str]
    ) -> dict[str, Any]:
        if not conversation_ids or len(set(conversation_ids)) != len(conversation_ids):
            raise ValueError("register distinct project conversations")
        for item in conversation_ids:
            require_safe_id(item, "conversation id")
        for item in objective_ids:
            require_safe_id(item, "objective id")
        path = self._path(project_id)
        if path.exists():
            raise FileExistsError(path)
        record = {
            "schema_version": "mavis.archive-project/v1",
            "project_id": project_id,
            "conversation_ids": conversation_ids,
            "objective_ids": objective_ids,
            "state": "open",
            "closed_at": None,
            "protected_paths": [],
            "segments_at_close": {},
        }
        write_json(path, record)
        return record

    def close(
        self, project_id: str, *, protected_paths: list[str] | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        record = read_json(self._path(project_id))
        if record.get("state") != "open":
            raise ValueError("project is already closed")
        self._check_objectives(record)
        for conversation_id in record["conversation_ids"]:
            archive = TranscriptArchive(self.home, conversation_id)
            if not archive.manifest_path.is_file():
                raise ValueError(f"conversation has no archive: {conversation_id}")
            self._check_binding(conversation_id, record)
            self._safe_manifest(archive)
        paths = protected_paths or []
        if any(not isinstance(path, str) or not Path(path).is_absolute() for path in paths):
            raise ValueError("protected evidence paths must be absolute")
        record["protected_paths"] = paths
        record["segments_at_close"] = {
            conversation_id: {
                item["segment_id"]: item["sha256"]
                for item in self._safe_manifest(TranscriptArchive(self.home, conversation_id))["segments"]
            }
            for conversation_id in record["conversation_ids"]
        }
        record["closed_at"] = _stamp(now or datetime.now(timezone.utc))
        record["state"] = "closed"
        write_json(self._path(project_id), record)
        return record

    def _check_binding(self, conversation_id: str, record: dict[str, Any]) -> None:
        path = self.home / "objective_sessions" / f"{conversation_id}.json"
        if path.is_file():
            binding = read_json(path)
            if (binding.get("session_id") != conversation_id
                or binding.get("objective_id") not in record["objective_ids"]):
                raise ValueError("conversation has an objective outside the closed project")

    def _check_objectives(self, record: dict[str, Any]) -> None:
        for objective_id in record["objective_ids"]:
            path = self.home / "objectives" / f"{objective_id}.json"
            objective = read_json(path)
            if objective.get("objective_id") != objective_id or objective.get("state") not in {"accepted", "cancelled"}:
                raise ValueError(f"objective is still active: {objective_id}")

    def _safe_manifest(self, archive: TranscriptArchive) -> dict[str, Any]:
        if archive.root.parent.is_symlink() or archive.root.is_symlink() or archive.manifest_path.is_symlink():
            raise ValueError("archive manifest escapes the archive")
        manifest = read_json(archive.manifest_path)
        if manifest.get("conversation_id") != archive.conversation_id:
            raise ValueError("archive manifest conversation mismatch")
        for segment in manifest.get("segments", []):
            archive.verify_segment(segment)
        return manifest

    def _activity_and_references(
        self, record: dict[str, Any], archive: TranscriptArchive,
        manifest: dict[str, Any], closed_at: datetime,
    ) -> set[str]:
        references = set(record["protected_paths"])
        files = []
        for segment in manifest["segments"]:
            if segment.get("compression") != "gzip":
                files.append(archive.resolve_segment_file(segment["path"]))
            stamped = _parse(segment["recorded_at"])
            if stamped > closed_at:
                raise ValueError("conversation changed after project closure")
        handoffs = archive.root / "handoffs"
        if handoffs.is_dir():
            for path in handoffs.glob("*.json"):
                if path.is_symlink():
                    raise ValueError("handoff symlink is unsafe")
                payload = read_json(path)
                if _parse(payload["written_at"]) > closed_at:
                    raise ValueError("handoff changed after project closure")
                references.update(_strings(payload.get("evidence_links", [])))
                files.append(path)
        objective_root = self.home / "objectives"
        for path in objective_root.glob("*.json"):
            if path.is_symlink():
                raise ValueError("objective symlink is unsafe")
            objective = read_json(path)
            if objective.get("objective_id") in record["objective_ids"] and _parse(objective["updated_at"]) > closed_at:
                raise ValueError("objective changed after project closure")
            references.update(_strings(objective))
        # Other conversations may cite this source after the project closes.
        for path in (self.home / "transcripts").glob("*/handoffs/*.json"):
            if path.is_symlink():
                raise ValueError("handoff symlink is unsafe")
            references.update(_strings(read_json(path).get("evidence_links", [])))
        # A file touched after closure means the archive is no longer idle.
        if any(datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) > closed_at
               for path in files):
            raise ValueError("archive file changed after project closure")
        return {str(Path(value).resolve()) for value in references if Path(value).is_absolute()}

    def compact(self, project_id: str, *, now: datetime | None = None) -> dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        record = read_json(self._path(project_id))
        if record.get("state") != "closed":
            raise ValueError("only closed projects can be compressed")
        closed_at = _parse(record["closed_at"])
        if now.astimezone(timezone.utc) - closed_at < timedelta(days=RETENTION_DAYS):
            raise ValueError("closed project has not been idle for 30 days")
        self._check_objectives(record)
        outcomes: list[dict[str, Any]] = []
        for conversation_id in record["conversation_ids"]:
            self._check_binding(conversation_id, record)
            archive = TranscriptArchive(self.home, conversation_id)
            with archive._locked():
                manifest = self._safe_manifest(archive)
                if {item["segment_id"]: item["sha256"] for item in manifest["segments"]} != record["segments_at_close"].get(conversation_id):
                    raise ValueError("archive segment set changed after project closure")
                references = self._activity_and_references(record, archive, manifest, closed_at)
                for segment in manifest["segments"]:
                    original = Path(segment["path"])
                    if segment.get("compression") == "gzip":
                        raw_duplicate = original.with_suffix("")
                        if raw_duplicate.exists():
                            if (raw_duplicate.is_symlink() or raw_duplicate.stat().st_nlink != 1
                                or sha256_file(raw_duplicate) != segment["sha256"]
                                or str(raw_duplicate.resolve()) in references):
                                raise ValueError("unverified or referenced raw duplicate remains")
                            raw_duplicate.unlink()
                        outcomes.append({"segment_id": segment["segment_id"], "status": "already-compressed"})
                        continue
                    if str(original.resolve()) in references:
                        outcomes.append({"segment_id": segment["segment_id"], "status": "referenced"})
                        continue
                    if original.stat().st_nlink != 1:
                        raise ValueError("hard-linked evidence cannot be compressed safely")
                    compressed = original.with_name(original.name + ".gz")
                    if compressed.is_symlink():
                        raise ValueError("compressed path is a symlink")
                    if not compressed.exists():
                        descriptor, temporary = tempfile.mkstemp(prefix=".compress-", dir=original.parent)
                        try:
                            with os.fdopen(descriptor, "wb") as target, original.open("rb") as source:
                                with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as zipper:
                                    shutil.copyfileobj(source, zipper, length=1024 * 1024)
                                target.flush()
                                os.fsync(target.fileno())
                            os.chmod(temporary, 0o600)
                            os.replace(temporary, compressed)
                        finally:
                            Path(temporary).unlink(missing_ok=True)
                    if not compressed.is_file() or compressed.stat().st_nlink != 1:
                        raise ValueError("compressed path is unsafe")
                    digest = hashlib.sha256()
                    try:
                        with gzip.open(compressed, "rb") as restored:
                            for chunk in iter(lambda: restored.read(1024 * 1024), b""):
                                digest.update(chunk)
                    except (OSError, EOFError) as exc:
                        raise ValueError("compressed segment cannot be reconstructed") from exc
                    if digest.hexdigest() != segment["sha256"]:
                        raise ValueError("compressed segment did not reconstruct exactly")
                    prior = dict(segment)
                    segment.update({"path": str(compressed), "compression": "gzip",
                                    "compressed_sha256": sha256_file(compressed)})
                    try:
                        archive.verify_segment(segment)
                        write_json(archive.manifest_path, manifest)
                    except BaseException:
                        segment.clear()
                        segment.update(prior)
                        raise
                    # The verified gzip is now authoritative. Raw bytes remain
                    # only until the manifest transaction succeeds.
                    original.unlink()
                    outcomes.append({"segment_id": segment["segment_id"], "status": "compressed",
                                     "original_sha256": segment["sha256"],
                                     "compressed_sha256": segment["compressed_sha256"]})
        return {"project_id": project_id, "segments": outcomes}

    def restore(self, project_id: str) -> dict[str, Any]:
        """Recreate exact raw files before making a closed archive active again."""
        record = read_json(self._path(project_id))
        restored = []
        for conversation_id in record["conversation_ids"]:
            archive = TranscriptArchive(self.home, conversation_id)
            with archive._locked():
                manifest = self._safe_manifest(archive)
                for segment in manifest["segments"]:
                    if segment.get("compression") != "gzip":
                        continue
                    compressed = Path(segment["path"])
                    raw = compressed.with_suffix("")
                    if raw.is_symlink():
                        raise ValueError("raw restoration path is a symlink")
                    if raw.exists():
                        if raw.stat().st_nlink != 1 or sha256_file(raw) != segment["sha256"]:
                            raise ValueError("raw restoration path contains other evidence")
                    else:
                        descriptor, temporary = tempfile.mkstemp(prefix=".restore-", dir=raw.parent)
                        try:
                            with os.fdopen(descriptor, "wb") as target, gzip.open(compressed, "rb") as source:
                                shutil.copyfileobj(source, target, length=1024 * 1024)
                                target.flush()
                                os.fsync(target.fileno())
                            os.chmod(temporary, 0o600)
                            if sha256_file(temporary) != segment["sha256"]:
                                raise ValueError("restored segment hash does not match")
                            os.replace(temporary, raw)
                        finally:
                            Path(temporary).unlink(missing_ok=True)
                    prior = dict(segment)
                    segment["path"] = str(raw)
                    segment.pop("compression")
                    segment.pop("compressed_sha256")
                    try:
                        archive.verify_segment(segment)
                        write_json(archive.manifest_path, manifest)
                    except BaseException:
                        segment.clear()
                        segment.update(prior)
                        raise
                    compressed.unlink()
                    restored.append(segment["segment_id"])
        return {"project_id": project_id, "restored_segments": restored}

    def reopen(self, project_id: str) -> dict[str, Any]:
        self.restore(project_id)
        record = read_json(self._path(project_id))
        record["state"] = "open"
        record["closed_at"] = None
        record["segments_at_close"] = {}
        write_json(self._path(project_id), record)
        return record


def storage_pressure(home: Path, *, prune: bool = False) -> dict[str, Any]:
    """Report capacity; prune only cache files with registered retained sources."""
    home = Path(home).resolve()
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    usage = shutil.disk_usage(home)
    threshold = max(PRESSURE_FREE_BYTES, int(usage.total * PRESSURE_FREE_FRACTION))
    pressure = usage.free < threshold
    cache = home / "cache" / "reproducible"
    freed = 0
    candidates = []
    if cache.exists():
        if cache.parent.is_symlink() or cache.is_symlink() or not cache.is_dir():
            raise ValueError("reproducible cache directory is unsafe")
        for path in cache.rglob("*"):
            if path.is_symlink():
                raise ValueError("reproducible cache symlink is unsafe")
        registry_path = cache / "manifest.json"
        if registry_path.is_symlink():
            raise ValueError("reproducible cache registry is unsafe")
        registry = read_json(registry_path) if registry_path.is_file() else {"entries": []}
        if not isinstance(registry.get("entries"), list):
            raise ValueError("reproducible cache registry is unsafe")
        for entry in registry["entries"]:
            declared = cache / entry["path"]
            source_declared = Path(entry["source"])
            path = declared.resolve()
            source = source_declared.resolve()
            if (declared.is_symlink() or source_declared.is_symlink()
                or not path.is_relative_to(cache.resolve()) or path == registry_path
                or source.is_relative_to(cache.resolve())
                or not source.is_file()
                or sha256_file(source) != entry["source_sha256"]):
                continue
            if path.is_file():
                if path.is_symlink() or path.stat().st_nlink != 1:
                    raise ValueError("registered cache file is unsafe")
                candidates.append(path)
        if prune and pressure:
            for path in candidates:
                freed += path.stat().st_size
                path.unlink()
            for path in sorted(cache.rglob("*"), reverse=True):
                if path.is_dir() and path != cache:
                    try:
                        path.rmdir()
                    except OSError:
                        pass
    return {"free_bytes": usage.free, "total_bytes": usage.total,
            "pressure": pressure, "threshold_free_bytes": threshold,
            "reproducible_cache_bytes": sum(path.stat().st_size for path in candidates if path.exists()),
            "freed_cache_bytes": freed,
            "action": "cache-first" if pressure else "none",
            "evidence_removed": False}


def register_reproducible_cache(home: Path, cache_file: Path, source_file: Path) -> None:
    """Mark a generated cache as disposable only while its source is retained."""
    cache = Path(home).resolve() / "cache" / "reproducible"
    if cache.parent.is_symlink() or cache.is_symlink():
        raise ValueError("reproducible cache directory is unsafe")
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    declared = Path(cache_file)
    source_declared = Path(source_file)
    path = declared.resolve(strict=True)
    source = source_declared.resolve(strict=True)
    if (not path.is_relative_to(cache) or path.name == "manifest.json"
        or source.is_relative_to(cache) or not path.is_file() or not source.is_file()
        or declared.is_symlink() or source_declared.is_symlink() or path.stat().st_nlink != 1):
        raise ValueError("cache registration needs a safe generated file and retained source")
    registry_path = cache / "manifest.json"
    registry = read_json(registry_path) if registry_path.exists() else {
        "schema_version": "mavis.reproducible-cache/v1", "entries": []}
    entries = [item for item in registry["entries"] if item["path"] != str(path.relative_to(cache))]
    entries.append({"path": str(path.relative_to(cache)), "source": str(source),
                    "source_sha256": sha256_file(source)})
    registry["entries"] = entries
    write_json(registry_path, registry)


def prepare_evidence_write(home: Path) -> dict[str, Any]:
    """Reclaim safe cache space and persist an early capacity warning."""
    report = storage_pressure(home, prune=True)
    if report["pressure"]:
        write_json(Path(home) / "storage-pressure.json", {
            "schema_version": "mavis.storage-pressure/v1",
            "observed_at": _stamp(datetime.now(timezone.utc)),
            **report,
        })
        warnings.warn("Mavis storage pressure: reproducible caches pruned; evidence capacity needs attention", RuntimeWarning)
    return report
