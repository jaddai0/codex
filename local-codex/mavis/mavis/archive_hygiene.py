"""Closed-project transcript retention and conservative storage-pressure actions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time
from typing import Any
import warnings

from .storage import read_json, require_safe_id, sha256_file, write_json
from .helpers import HelperSession
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


class _PinnedSegments:
    """Operate on one verified archive directory, never a later path lookup."""

    def __init__(self, archive: TranscriptArchive, root_fd: int):
        self.archive = archive
        self.root_fd = root_fd
        self.fd = os.open("segments", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                          dir_fd=root_fd)
        self.root_identity = (os.fstat(root_fd).st_dev, os.fstat(root_fd).st_ino)
        self.identity = (os.fstat(self.fd).st_dev, os.fstat(self.fd).st_ino)
        self.assert_attached()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        os.close(self.fd)

    @staticmethod
    def _name(name: str) -> str:
        if not name or name in {".", ".."} or Path(name).name != name or "/" in name:
            raise ValueError("segment filename is unsafe")
        return name

    def assert_attached(self) -> None:
        root = self.archive.root
        if root.parent.is_symlink() or root.is_symlink():
            raise ValueError("archive directory was replaced")
        root_stat = os.stat(root, follow_symlinks=False)
        child = os.stat("segments", dir_fd=self.root_fd, follow_symlinks=False)
        if (not stat.S_ISDIR(root_stat.st_mode) or not stat.S_ISDIR(child.st_mode)
            or (root_stat.st_dev, root_stat.st_ino) != self.root_identity
            or (child.st_dev, child.st_ino) != self.identity):
            raise ValueError("archive segments directory was replaced")

    def _open(self, name: str) -> tuple[int, os.stat_result]:
        descriptor = os.open(self._name(name), os.O_RDONLY | os.O_NOFOLLOW,
                             dir_fd=self.fd)
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            os.close(descriptor)
            raise ValueError("segment file is not a unique regular file")
        return descriptor, observed

    def exists(self, name: str) -> bool:
        try:
            observed = os.stat(self._name(name), dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(observed.st_mode):
            raise ValueError("segment path is not a regular file")
        return True

    @staticmethod
    def segment_name(segment: dict[str, Any]) -> str:
        segment_id = require_safe_id(str(segment["segment_id"]), "segment id")
        suffix = ".jsonl.gz" if segment.get("compression") == "gzip" else ".jsonl"
        expected = segment_id + suffix
        if Path(segment["path"]).name != expected:
            raise ValueError("manifest segment name does not match its identity")
        return expected

    def digest(self, name: str) -> tuple[str, tuple[int, int]]:
        descriptor, observed = self._open(name)
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest(), (observed.st_dev, observed.st_ino)

    def reconstructed_digest(self, name: str) -> str:
        descriptor, _ = self._open(name)
        digest = hashlib.sha256()
        try:
            with os.fdopen(descriptor, "rb") as source:
                with gzip.GzipFile(fileobj=source, mode="rb") as restored:
                    for chunk in iter(lambda: restored.read(1024 * 1024), b""):
                        digest.update(chunk)
        except (OSError, EOFError) as exc:
            raise ValueError("compressed segment cannot be reconstructed") from exc
        return digest.hexdigest()

    def unlink_verified(self, name: str, expected_sha256: str,
                        expected_identity: tuple[int, int]) -> None:
        self.assert_attached()
        digest, identity = self.digest(name)
        observed = os.stat(self._name(name), dir_fd=self.fd, follow_symlinks=False)
        if (digest != expected_sha256 or identity != expected_identity
            or (observed.st_dev, observed.st_ino) != identity):
            raise ValueError("segment changed before duplicate removal")
        os.unlink(name, dir_fd=self.fd)

    def _temporary(self, stem: str) -> tuple[str, int]:
        name = f".{stem}-{os.urandom(16).hex()}"
        descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=self.fd)
        return name, descriptor

    def write_gzip(self, raw_name: str, compressed_name: str) -> None:
        temporary, target_fd = self._temporary("compress")
        try:
            with os.fdopen(target_fd, "wb") as target:
                source_fd, _ = self._open(raw_name)
                with os.fdopen(source_fd, "rb") as source:
                    with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as zipper:
                        shutil.copyfileobj(source, zipper, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            self.assert_attached()
            os.link(temporary, self._name(compressed_name),
                    src_dir_fd=self.fd, dst_dir_fd=self.fd, follow_symlinks=False)
            os.fsync(self.fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=self.fd)
            except FileNotFoundError:
                pass

    def write_raw(self, compressed_name: str, raw_name: str) -> None:
        temporary, target_fd = self._temporary("restore")
        try:
            with os.fdopen(target_fd, "wb") as target:
                source_fd, _ = self._open(compressed_name)
                with os.fdopen(source_fd, "rb") as source:
                    with gzip.GzipFile(fileobj=source, mode="rb") as restored:
                        shutil.copyfileobj(restored, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            self.assert_attached()
            os.link(temporary, self._name(raw_name),
                    src_dir_fd=self.fd, dst_dir_fd=self.fd, follow_symlinks=False)
            os.fsync(self.fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=self.fd)
            except FileNotFoundError:
                pass


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

    @contextmanager
    def _registry_lock(self):
        if self.root.is_symlink():
            raise ValueError("archive project registry is a symlink")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            try:
                descriptor = os.open("registry.lock", os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW,
                                     0o600, dir_fd=directory_fd)
            except FileExistsError:
                descriptor = os.open("registry.lock", os.O_RDWR | os.O_NOFOLLOW,
                                     dir_fd=directory_fd)
            with os.fdopen(descriptor, "rb") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                yield
        finally:
            os.close(directory_fd)

    def _check_exclusive(self, record: dict[str, Any]) -> None:
        owned = set(record["conversation_ids"])
        for path in self.root.glob("*.json"):
            if path.is_symlink():
                raise ValueError("archive project record is a symlink")
            other = read_json(path)
            if other.get("schema_version") != "mavis.archive-project/v1":
                raise ValueError("archive project registry contains an invalid record")
            other_id = other.get("project_id")
            if (not isinstance(other_id, str)
                or path.name != f"{require_safe_id(other_id, 'project id')}.json"
                or not isinstance(other.get("conversation_ids"), list)
                or any(not isinstance(item, str) for item in other["conversation_ids"])):
                raise ValueError("archive project filename or ownership record is invalid")
            if other_id == record["project_id"]:
                continue
            if owned.intersection(other.get("conversation_ids", [])):
                raise ValueError("conversation belongs to another archive project")

    def register(
        self, project_id: str, conversation_ids: list[str], objective_ids: list[str]
    ) -> dict[str, Any]:
        with self._registry_lock():
            return self._register_locked(project_id, conversation_ids, objective_ids)

    def _register_locked(
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
        self._check_exclusive(record)
        write_json(path, record)
        return record

    def close(
        self, project_id: str, *, protected_paths: list[str] | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        with self._registry_lock():
            return self._close_locked(project_id, protected_paths=protected_paths, now=now)

    def _close_locked(
        self, project_id: str, *, protected_paths: list[str] | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        record = read_json(self._path(project_id))
        self._check_exclusive(record)
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
        helper_context = self.home / "helpers" / "librarian" / "query-context.json"
        if helper_context.is_symlink():
            raise ValueError("librarian citation cache is a symlink")
        if helper_context.is_file():
            payload = read_json(helper_context)
            if (payload.get("role") == "librarian"
                and float(payload.get("expires_at_epoch", 0)) > time.time()):
                references.update(_strings(payload.get("citations", [])))
        # A file touched after closure means the archive is no longer idle.
        if any(datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) > closed_at
               for path in files):
            raise ValueError("archive file changed after project closure")
        return {str(Path(value).resolve()) for value in references if Path(value).is_absolute()}

    def compact(self, project_id: str, *, now: datetime | None = None) -> dict[str, Any]:
        with self._registry_lock():
            with HelperSession(self.home, "librarian")._locked():
                return self._compact_locked(project_id, now=now)

    def _compact_locked(self, project_id: str, *, now: datetime | None = None) -> dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        record = read_json(self._path(project_id))
        self._check_exclusive(record)
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
            with archive._locked() as root_fd:
                manifest = self._safe_manifest(archive)
                if {item["segment_id"]: item["sha256"] for item in manifest["segments"]} != record["segments_at_close"].get(conversation_id):
                    raise ValueError("archive segment set changed after project closure")
                references = self._activity_and_references(record, archive, manifest, closed_at)
                with _PinnedSegments(archive, root_fd) as pinned:
                    for segment in manifest["segments"]:
                        pinned.assert_attached()
                        name = pinned.segment_name(segment)
                        source_path = Path(segment["path"])
                        if segment.get("compression") == "gzip":
                            compressed_sha, _ = pinned.digest(name)
                            if (compressed_sha != segment["compressed_sha256"]
                                or pinned.reconstructed_digest(name) != segment["sha256"]):
                                raise ValueError("compressed segment changed")
                            raw_name = name[:-3]
                            if pinned.exists(raw_name):
                                raw_path = source_path.with_suffix("")
                                if str(raw_path) in references:
                                    raise ValueError("referenced raw duplicate remains")
                                raw_sha, raw_identity = pinned.digest(raw_name)
                                if raw_sha != segment["sha256"]:
                                    raise ValueError("unverified raw duplicate remains")
                                pinned.unlink_verified(raw_name, raw_sha, raw_identity)
                                pinned.assert_attached()
                            outcomes.append({"segment_id": segment["segment_id"], "status": "already-compressed"})
                            continue
                        if str(source_path) in references:
                            outcomes.append({"segment_id": segment["segment_id"], "status": "referenced"})
                            continue
                        raw_sha, raw_identity = pinned.digest(name)
                        if raw_sha != segment["sha256"]:
                            raise ValueError("raw segment changed")
                        compressed_name = name + ".gz"
                        if not pinned.exists(compressed_name):
                            pinned.write_gzip(name, compressed_name)
                        compressed_sha, _ = pinned.digest(compressed_name)
                        if pinned.reconstructed_digest(compressed_name) != raw_sha:
                            raise ValueError("compressed segment did not reconstruct exactly")
                        prior = dict(segment)
                        segment.update({"path": str(source_path.with_name(compressed_name)),
                                        "compression": "gzip", "compressed_sha256": compressed_sha})
                        published = False
                        removed = False
                        try:
                            pinned.assert_attached()
                            published = True  # A write may publish before raising.
                            archive._write_manifest_locked(root_fd, manifest)
                            pinned.assert_attached()
                            pinned.unlink_verified(name, raw_sha, raw_identity)
                            removed = True
                            pinned.assert_attached()
                        except BaseException:
                            if not removed:
                                segment.clear()
                                segment.update(prior)
                                if published:
                                    archive._write_manifest_locked(root_fd, manifest)
                            raise
                        outcomes.append({"segment_id": segment["segment_id"], "status": "compressed",
                                         "original_sha256": raw_sha,
                                         "compressed_sha256": compressed_sha})
        return {"project_id": project_id, "segments": outcomes}

    def restore(self, project_id: str) -> dict[str, Any]:
        """Recreate exact raw files before making a closed archive active again."""
        record = read_json(self._path(project_id))
        restored = []
        for conversation_id in record["conversation_ids"]:
            archive = TranscriptArchive(self.home, conversation_id)
            with archive._locked() as root_fd:
                manifest = self._safe_manifest(archive)
                with _PinnedSegments(archive, root_fd) as pinned:
                    for segment in manifest["segments"]:
                        if segment.get("compression") != "gzip":
                            continue
                        pinned.assert_attached()
                        compressed_name = pinned.segment_name(segment)
                        compressed_sha, compressed_identity = pinned.digest(compressed_name)
                        if (compressed_sha != segment["compressed_sha256"]
                            or pinned.reconstructed_digest(compressed_name) != segment["sha256"]):
                            raise ValueError("compressed segment changed")
                        raw_name = compressed_name[:-3]
                        if not pinned.exists(raw_name):
                            pinned.write_raw(compressed_name, raw_name)
                        raw_sha, _ = pinned.digest(raw_name)
                        if raw_sha != segment["sha256"]:
                            raise ValueError("raw restoration path contains other evidence")
                        prior = dict(segment)
                        segment["path"] = str(Path(prior["path"]).with_suffix(""))
                        segment.pop("compression")
                        segment.pop("compressed_sha256")
                        published = False
                        removed = False
                        try:
                            pinned.assert_attached()
                            published = True  # A write may publish before raising.
                            archive._write_manifest_locked(root_fd, manifest)
                            pinned.assert_attached()
                            pinned.unlink_verified(compressed_name, compressed_sha, compressed_identity)
                            removed = True
                            pinned.assert_attached()
                        except BaseException:
                            if not removed:
                                segment.clear()
                                segment.update(prior)
                                if published:
                                    archive._write_manifest_locked(root_fd, manifest)
                            raise
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

    def sweep_due(self, *, now: datetime | None = None) -> dict[str, Any]:
        """At an idle maintenance claim, attempt each due closed project once daily."""
        now = now or datetime.now(timezone.utc)
        results = []
        if not self.root.is_dir():
            return {"checked_at": _stamp(now), "projects": results}
        for path in sorted(self.root.glob("*.json")):
            try:
                if path.is_symlink():
                    raise ValueError("archive project record is a symlink")
                record = read_json(path)
                if record.get("state") != "closed":
                    continue
                if now - _parse(record["closed_at"]) < timedelta(days=RETENTION_DAYS):
                    continue
                last = record.get("last_sweep_at")
                if last and _parse(last).date() >= now.date():
                    continue
                outcome = self.compact(record["project_id"], now=now)
                with self._registry_lock():
                    current = read_json(self._path(record["project_id"]))
                    current["last_sweep_at"] = _stamp(now)
                    write_json(self._path(record["project_id"]), current)
                results.append({"project_id": record["project_id"], "status": "checked",
                                "segments": outcome["segments"]})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                results.append({"project_record": str(path), "status": "failed", "error": str(exc)})
        report = {"schema_version": "mavis.archive-sweep/v1", "checked_at": _stamp(now),
                  "projects": results}
        if results:
            try:
                write_json(self.home / "maintenance" / "archive-sweep-last.json", report)
            except OSError as exc:
                report["report_write_error"] = str(exc)
        return report


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
    cache_warning = None
    if cache.exists() or cache.is_symlink() or cache.parent.is_symlink():
        try:
            if cache.parent.is_symlink() or cache.is_symlink() or not cache.is_dir():
                raise ValueError("reproducible cache directory is unsafe")
            for path in cache.rglob("*"):
                if path.is_symlink():
                    raise ValueError("reproducible cache symlink is unsafe")
            registry_path = cache / "manifest.json"
            if registry_path.is_symlink():
                raise ValueError("reproducible cache registry is unsafe")
            registry = read_json(registry_path) if registry_path.is_file() else {"entries": []}
            if (registry.get("schema_version") not in (None, "mavis.reproducible-cache/v1")
                or not isinstance(registry.get("entries"), list)):
                raise ValueError("reproducible cache registry has invalid shape")
            for entry in registry["entries"]:
                if (not isinstance(entry, dict)
                    or any(not isinstance(entry.get(key), str) for key in ("path", "source", "source_sha256"))):
                    raise ValueError("reproducible cache registry has invalid entry")
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
        except (OSError, ValueError, KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
            candidates = []
            cache_warning = f"registered cache skipped: {exc}"
        eligible_bytes = sum(path.stat().st_size for path in candidates if path.exists())
        if prune and pressure and candidates:
            try:
                for path in candidates:
                    freed += path.stat().st_size
                    path.unlink()
                for path in sorted(cache.rglob("*"), reverse=True):
                    if path.is_dir() and path != cache:
                        try:
                            path.rmdir()
                        except OSError:
                            pass
            except OSError as exc:
                cache_warning = f"registered cache pruning stopped: {exc}"
    else:
        eligible_bytes = 0
    return {"free_bytes": usage.free, "total_bytes": usage.total,
            "pressure": pressure, "threshold_free_bytes": threshold,
            "reproducible_cache_bytes": eligible_bytes,
            "freed_cache_bytes": freed,
            "action": "registered-cache-pruned" if freed else "capacity-warning" if pressure else "none",
            "cache_warning": cache_warning,
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
    if report["pressure"] or report["cache_warning"]:
        try:
            write_json(Path(home) / "storage-pressure.json", {
                "schema_version": "mavis.storage-pressure/v1",
                "observed_at": _stamp(datetime.now(timezone.utc)),
                **report,
            })
        except OSError:
            pass  # The optional warning must not block the evidence write.
        if report["pressure"]:
            warnings.warn("Mavis storage pressure: evidence capacity needs attention", RuntimeWarning)
        if report["cache_warning"]:
            warnings.warn(report["cache_warning"], RuntimeWarning)
    return report
