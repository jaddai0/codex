"""Durable, fail-closed writes for the experiment and profile pointer pair."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .storage import read_json, write_json


def path(home: Path) -> Path:
    return Path(home).resolve() / "experiments" / "profile-transition.json"


def pending(home: Path) -> bool:
    return path(home).exists()


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_optional(target: Path) -> dict[str, Any] | None:
    return read_json(target) if target.exists() else None


def _apply(home: Path, journal: dict[str, Any]) -> None:
    entries = journal.get("entries")
    if (journal.get("schema_version") != "mavis.profile-transition/v1"
            or not isinstance(entries, list) or not entries
            or journal.get("digest") != _digest({key: value for key, value in journal.items() if key != "digest"})):
        raise ValueError("profile transition journal is invalid")
    root = Path(home).resolve()
    seen: set[Path] = set()
    for entry in entries:
        if (not isinstance(entry, dict) or not isinstance(entry.get("path"), str)
                or "old" not in entry or "new" not in entry
                or (entry["old"] is not None and not isinstance(entry["old"], dict))
                or (entry["new"] is not None and not isinstance(entry["new"], dict))):
            raise ValueError("profile transition entry is invalid")
        target = Path(entry["path"])
        if (not target.is_absolute() or target.resolve() != target or root not in target.parents
                or target in seen or target == path(home)
                or _read_optional(target) not in (entry["old"], entry["new"])):
            raise ValueError("profile transition file changed outside the journal")
        seen.add(target)
    for entry in entries:
        target = Path(entry["path"])
        if entry["new"] is None:
            target.unlink(missing_ok=True)
        else:
            write_json(target, entry["new"])
    for directory in {Path(entry["path"]).parent for entry in entries}:
        descriptor = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    path(home).unlink()
    descriptor = os.open(str(path(home).parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def commit(home: Path, *, operation: str, experiment_id: str,
           changes: list[tuple[Path, dict[str, Any] | None]]) -> None:
    journal_path = path(home)
    if journal_path.exists():
        raise ValueError("an interrupted profile transition needs recovery")
    entries = []
    for target, value in changes:
        target = Path(target).absolute()
        if target.is_symlink():
            raise ValueError("profile transition target is a symlink")
        target = target.resolve()
        if Path(home).resolve() not in target.parents:
            raise ValueError("profile transition target is outside Mavis home")
        entries.append({"path": str(target), "old": _read_optional(target), "new": value})
    journal = {"schema_version": "mavis.profile-transition/v1", "operation": operation,
               "experiment_id": experiment_id, "entries": entries}
    journal["digest"] = _digest(journal)
    write_json(journal_path, journal)
    descriptor = os.open(str(journal_path.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _apply(home, journal)


def resume(home: Path, *, operation: str, experiment_id: str) -> None:
    journal = read_json(path(home))
    if journal.get("operation") != operation or journal.get("experiment_id") != experiment_id:
        raise ValueError("a different profile transition needs recovery")
    _apply(home, journal)
