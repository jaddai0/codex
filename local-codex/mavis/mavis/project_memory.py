"""Branch-bound, source-linked project knowledge kept outside Git."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any

from .storage import require_safe_id, sha256_file


KINDS = {"decision", "failure", "tool-recipe", "history"}
VERSION_RE = re.compile(r"v([1-9][0-9]*)\.json$")
SHA256_RE = re.compile(r"[0-9a-f]{64}$")
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_CLAIM_CHARS = 4000
MAX_SOURCES = 16
MAX_RECORD_BYTES = 64 * 1024


class ProjectMemory:
    """Append project records and return only records backed by current files."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve(strict=True)
        self.root = self.project_root / ".mavis" / "memory" / "records"

    def _git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.project_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip() or "project Git state is unavailable"
            )
        return result.stdout.strip()

    def branch(self) -> str:
        result = subprocess.run(
            ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
            cwd=self.project_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        branch = result.stdout.strip() if result.returncode == 0 else ""
        return branch or f"detached-{self._git('rev-parse', 'HEAD')}"

    def _branch_root(self, branch: str) -> Path:
        return self.root / hashlib.sha256(branch.encode()).hexdigest()

    def _safe_storage(self) -> None:
        for path in (self.project_root / ".mavis", self.root.parent, self.root):
            if path.is_symlink():
                raise ValueError("project memory directory is a symlink")

    def _prepare_storage(self) -> None:
        self._safe_storage()
        state = self.project_root / ".mavis"
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        ignore = state / ".gitignore"
        if ignore.is_symlink():
            raise ValueError("project memory ignore file is a symlink")
        if not ignore.exists():
            descriptor = os.open(ignore, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write("*\n")
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", "--", ".mavis/memory/records/probe"],
            cwd=self.project_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if ignored.returncode != 0:
            raise ValueError("project .mavis directory is not Git-ignored")

    def _source(self, declared: str) -> tuple[str, str] | None:
        path = PurePosixPath(declared)
        if (
            not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or path.parts[0] in {".mavis", ".git"}
            or str(path) != declared
        ):
            return None
        ignore_status = subprocess.run(
            ["git", "check-ignore", "-q", "--", declared],
            cwd=self.project_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        if ignore_status != 1:
            return None
        candidate = self.project_root.joinpath(*path.parts)
        if any(
            parent.is_symlink()
            for parent in (candidate, *candidate.parents)
            if parent != self.project_root and parent.is_relative_to(self.project_root)
        ):
            return None
        try:
            resolved = candidate.resolve(strict=True)
            if (
                not resolved.is_relative_to(self.project_root)
                or not resolved.is_file()
                or resolved.stat().st_size > MAX_SOURCE_BYTES
            ):
                return None
            return declared, sha256_file(resolved)
        except (OSError, RuntimeError):
            return None

    def add(
        self, record_id: str, kind: str, claim: str, sources: list[str]
    ) -> dict[str, Any]:
        require_safe_id(record_id, "memory record id")
        if (
            kind not in KINDS
            or not isinstance(claim, str)
            or not claim.strip()
            or len(claim) > MAX_CLAIM_CHARS
        ):
            raise ValueError("project memory needs a kind and nonempty claim")
        if (
            not isinstance(sources, list)
            or not sources
            or len(sources) > MAX_SOURCES
            or any(not isinstance(s, str) for s in sources)
        ):
            raise ValueError("project memory needs source paths")
        self._prepare_storage()
        references = []
        for source in dict.fromkeys(sources):
            verified = self._source(source)
            if verified is None:
                raise ValueError(
                    f"project memory source is unavailable or unsafe: {source}"
                )
            references.append({"path": verified[0], "sha256": verified[1]})
        branch = self.branch()
        folder = self._branch_root(branch) / record_id
        if folder.parent.is_symlink() or folder.is_symlink():
            raise ValueError("project memory record directory is a symlink")
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        versions = [
            int(match.group(1))
            for path in folder.iterdir()
            if (match := VERSION_RE.fullmatch(path.name))
        ]
        version = max(versions, default=0) + 1
        record = {
            "schema_version": "mavis.memory-record/v1",
            "record_id": record_id,
            "scope": "project",
            "kind": kind,
            "claim": claim.strip(),
            "source_references": references,
            "verification_state": "unverified",
            "freshness": {"observed_at": datetime.now(timezone.utc).isoformat()},
            "branch": branch,
            "revision": self._git("rev-parse", "HEAD"),
            "record_version": version,
        }
        target = folder / f"v{version}.json"
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return record

    def _valid_record(
        self, path: Path, branch: str, version: int
    ) -> dict[str, Any] | None:
        if path.is_symlink():
            return None
        try:
            if path.stat().st_size > MAX_RECORD_BYTES:
                return None
            record = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(record, dict)
                or set(record)
                != {
                    "schema_version",
                    "record_id",
                    "scope",
                    "kind",
                    "claim",
                    "source_references",
                    "verification_state",
                    "freshness",
                    "branch",
                    "revision",
                    "record_version",
                }
                or record.get("schema_version") != "mavis.memory-record/v1"
                or record.get("scope") != "project"
                or record.get("record_id") != path.parent.name
                or type(record.get("record_version")) is not int
                or record["record_version"] != version
                or record.get("branch") != branch
                or record.get("kind") not in KINDS
                or record.get("verification_state") != "unverified"
                or not isinstance(record.get("claim"), str)
                or not record["claim"].strip()
                or len(record["claim"]) > MAX_CLAIM_CHARS
                or not isinstance(record.get("revision"), str)
                or not re.fullmatch(r"[0-9a-f]{40}", record["revision"])
            ):
                return None
            if (
                subprocess.run(
                    ["git", "merge-base", "--is-ancestor", record["revision"], "HEAD"],
                    cwd=self.project_root,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                ).returncode
                != 0
            ):
                return None
            freshness = record.get("freshness")
            if (
                not isinstance(freshness, dict)
                or not isinstance(freshness.get("observed_at"), str)
                or not freshness["observed_at"]
                or set(freshness) != {"observed_at"}
            ):
                return None
            references = record.get("source_references")
            if (
                not isinstance(references, list)
                or not 1 <= len(references) <= MAX_SOURCES
            ):
                return None
            for reference in references:
                if (
                    not isinstance(reference, dict)
                    or set(reference) != {"path", "sha256"}
                    or not isinstance(reference["path"], str)
                    or not isinstance(reference["sha256"], str)
                    or not SHA256_RE.fullmatch(reference["sha256"])
                ):
                    return None
                source = self._source(reference["path"])
                if source is None or source[1] != reference["sha256"]:
                    return None
            return record
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def search(self, query: str) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query must not be empty")
        self._safe_storage()
        branch = self.branch()
        branch_root = self._branch_root(branch)
        if not branch_root.is_dir() or branch_root.is_symlink():
            return []
        hits = []
        needle = query.casefold()
        for folder in sorted(branch_root.iterdir()):
            if not folder.is_dir() or folder.is_symlink():
                continue
            try:
                require_safe_id(folder.name, "memory record id")
            except ValueError:
                continue
            versions = sorted(
                (int(match.group(1)), path)
                for path in folder.iterdir()
                if (match := VERSION_RE.fullmatch(path.name))
            )
            if not versions:
                continue
            version, path = versions[-1]
            record = self._valid_record(path, branch, version)
            if record is None or needle not in record["claim"].casefold():
                continue
            score = 100 if record["claim"].casefold() == needle else 50
            hits.append(
                {
                    "scope": "project",
                    "path": str(path),
                    "line": 1,
                    "text": record["claim"],
                    "score": score,
                    "source": "project-memory",
                    "record_id": record["record_id"],
                    "record_version": version,
                    "kind": record["kind"],
                    "branch": branch,
                    "revision": record["revision"],
                    "verification_state": record["verification_state"],
                    "freshness": record["freshness"],
                    "source_references": record["source_references"],
                }
            )
        return sorted(hits, key=lambda item: (-item["score"], item["record_id"]))
