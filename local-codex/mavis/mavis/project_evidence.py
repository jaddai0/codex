"""Project-local evidence paths without changing Mavis's shared service home."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any
from uuid import uuid4

from .storage import read_json, require_safe_id, sha256_file, write_json


def project_root(path: Path) -> Path:
    """Resolve a Git checkout, rejecting paths outside it and unsafe state."""
    candidate = Path(path).expanduser().resolve(strict=True)
    if not candidate.is_dir():
        raise ValueError("project path is not a directory")
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=candidate,
        text=True, capture_output=True, check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise ValueError("project path is not a Git checkout")
    root = Path(result.stdout.strip()).resolve(strict=True)
    if root == Path.home().resolve():
        raise ValueError("the home-directory Git root is not a Mavis project")
    if not candidate.is_relative_to(root):
        raise ValueError("project path escapes its Git checkout")
    return root


def project_home(path: Path, *, create: bool = False) -> Path:
    root = project_root(path)
    state = root / ".mavis"
    if state.is_symlink() or (state.exists() and not state.is_dir()):
        raise ValueError("project .mavis path is unsafe")
    if create:
        state.mkdir(mode=0o700, exist_ok=True)
        os.chmod(state, 0o700)
        ignore = state / ".gitignore"
        if ignore.is_symlink() or (ignore.exists() and not ignore.is_file()):
            raise ValueError("project .mavis ignore file is unsafe")
        if not ignore.exists():
            descriptor = os.open(ignore, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write("*\n")
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", "--", ".mavis/evidence/probe"],
            cwd=root, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if ignored.returncode != 0:
            raise ValueError("project .mavis evidence is not Git-ignored")
    return state


def active_project_home(shared_home: Path, *, create: bool = False,
                        path: Path | None = None) -> Path:
    """Pin a session to its declared project even when hook cwd changes."""
    declared = os.environ.get("MAVIS_PROJECT_ROOT")
    if declared:
        selected = project_root(Path(declared))
        if path is not None:
            try:
                observed = project_root(Path(path))
            except ValueError as error:
                if str(error) not in {"project path is not a Git checkout",
                                      "the home-directory Git root is not a Mavis project"}:
                    raise
            else:
                if observed != selected:
                    raise ValueError("hook or command cwd belongs to a different Mavis project")
        return project_home(selected, create=create)
    if path is None:
        return Path(shared_home)
    try:
        return project_home(Path(path), create=create)
    except ValueError as error:
        # Service-only sessions may not run from a specific Git project.
        if str(error) in {"project path is not a Git checkout",
                          "the home-directory Git root is not a Mavis project"}:
            return Path(shared_home)
        raise


def legacy_home_for_record(shared_home: Path, project_state: Path,
                           collection: str, name: str) -> Path:
    """Keep pre-migration evidence readable without rewriting signed receipts."""
    require_safe_id(name)
    if (project_state / collection / name).exists():
        return project_state
    if (Path(shared_home) / collection / name).exists():
        return Path(shared_home)
    return project_state


def _belongs_to_project(record: dict[str, Any], root: Path) -> bool:
    paths: list[str] = []
    for assignment in record.get("assignments", []):
        checkout = assignment.get("checkout", {}) if isinstance(assignment, dict) else {}
        if isinstance(checkout, dict) and isinstance(checkout.get("path"), str):
            paths.append(checkout["path"])
    for item in record.get("evidence_receipts", []):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            continue
        try:
            receipt = read_json(Path(item["path"]))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(receipt.get("cwd"), str):
            paths.append(receipt["cwd"])
    return any(Path(value).is_absolute() and Path(value).resolve().is_relative_to(root)
               for value in paths)


def objective_home(shared_home: Path, project_state: Path, objective_id: str) -> Path:
    """Resolve one objective without attaching a different project's legacy ID."""
    objective_id = require_safe_id(objective_id, "objective id")
    if Path(project_state) == Path(shared_home):
        return Path(shared_home)
    if (Path(project_state) / "objectives" / f"{objective_id}.json").exists():
        return Path(project_state)
    legacy = Path(shared_home) / "objectives" / f"{objective_id}.json"
    if legacy.exists():
        if _belongs_to_project(read_json(legacy), project_root(Path(project_state).parent)):
            return Path(shared_home)
        raise ValueError("legacy objective is not bound to this project")
    return Path(project_state)


def _evidence_files(directory: Path) -> list[Path]:
    if directory.is_symlink():
        raise ValueError("legacy evidence directory is a symlink")
    files = []
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("legacy evidence contains a symlink")
        if path.is_file():
            files.append(path)
    return files


def migrate_legacy_objective(shared_home: Path, project: Path,
                             objective_id: str) -> Path:
    """Copy an immutable legacy evidence snapshot; never rewrite signed links."""
    objective_id = require_safe_id(objective_id, "objective id")
    shared_home = Path(shared_home).resolve(strict=True)
    root = project_root(project)
    source = shared_home / "objectives" / f"{objective_id}.json"
    record = read_json(source)
    if record.get("objective_id") != objective_id or not _belongs_to_project(record, root):
        raise ValueError("legacy objective is not bound to this project")
    state = project_home(root, create=True)
    parent = state / "legacy-import" / "objectives"
    parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    destination = parent / objective_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    copied: list[dict[str, str]] = []
    paths = [source]
    evidence = shared_home / "evidence" / objective_id
    if evidence.exists():
        paths.extend(_evidence_files(evidence))
    bindings = shared_home / "objective_sessions"
    if bindings.is_dir():
        for binding in bindings.glob("*.json"):
            if binding.is_symlink():
                raise ValueError("legacy session binding is a symlink")
            value = read_json(binding)
            if value.get("objective_id") != objective_id:
                continue
            session_id = require_safe_id(str(value.get("session_id") or ""), "session id")
            paths.append(binding)
            archive = shared_home / "transcripts" / session_id
            if archive.exists():
                paths.extend(_evidence_files(archive))
    staging = parent / f".{objective_id}.{uuid4().hex}.pending"
    staging.mkdir(mode=0o700)
    try:
        for path in paths:
            if path.is_symlink() or not path.resolve().is_relative_to(shared_home):
                raise ValueError("legacy evidence contains an unsafe path")
            relative = path.relative_to(shared_home)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copyfile(path, target, follow_symlinks=False)
            os.chmod(target, 0o600)
            digest = sha256_file(path)
            if sha256_file(target) != digest:
                raise ValueError("legacy evidence copy changed")
            copied.append({"path": relative.as_posix(), "sha256": digest})
        write_json(staging / "manifest.json", {
            "schema_version": "mavis.legacy-project-evidence/v1",
            "project_root": str(root), "objective_id": objective_id,
            "legacy_home": str(shared_home), "files": copied,
            "snapshot_sha256": hashlib.sha256(
                json.dumps(copied, sort_keys=True).encode()).hexdigest(),
        })
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    manifest = destination / "manifest.json"
    return manifest
