"""Project-first retrieval with hash tracking and safe exact fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
from typing import Any, Iterator


TEXT_SUFFIXES = {
    ".c", ".cc", ".cpp", ".css", ".go", ".h", ".hpp", ".html", ".java",
    ".js", ".json", ".jsx", ".kt", ".md", ".py", ".rb", ".rs", ".sh",
    ".sql", ".swift", ".toml", ".ts", ".tsx", ".txt", ".yaml", ".yml",
}
MAX_INDEX_BYTES = 2 * 1024 * 1024
SYMBOL_RE = re.compile(
    r"^\s*(?:(?:export\s+(?:default\s+)?)|(?:pub(?:\([^)]*\))?\s+))?"
    r"(?:async\s+)?(?:def|class|fn|func|function|interface|struct|enum|protocol|trait)"
    r"\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)
DEPENDENCY_RE = re.compile(
    r"^\s*(?:import\b[^\n]*?\bfrom\s+['\"]([^'\"]+)|"
    r"(?:import|export)\s+['\"]([^'\"]+)|from\s+([\w.]+)\s+import|"
    r"import\s+([\w.]+)|use\s+([\w:]+)|[^\n]*?\brequire\(['\"]([^'\"]+))",
    re.MULTILINE,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=root, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )


@dataclass(frozen=True)
class SearchHit:
    scope: str
    path: str
    line: int
    text: str
    score: int
    source: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class ProjectIndex:
    """SQLite evidence index stored under the project's ignored .mavis dir."""

    def __init__(self, project_root: Path, shared_home: Path):
        self.project_root = Path(project_root).resolve()
        self.project_state = self.project_root / ".mavis"
        self.shared_home = Path(shared_home).resolve()
        self.database = self.project_state / "index.sqlite3"

    def _connect(self) -> sqlite3.Connection:
        self.project_state.mkdir(parents=True, exist_ok=True)
        ignore = self.project_state / ".gitignore"
        if not ignore.exists():
            ignore.write_text("*\n", encoding="utf-8")
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS files (
              branch TEXT NOT NULL,
              path TEXT NOT NULL,
              sha256 TEXT NOT NULL,
              bytes INTEGER NOT NULL,
              content TEXT NOT NULL,
              symbols TEXT NOT NULL,
              dependencies TEXT NOT NULL,
              indexed_at TEXT NOT NULL,
              PRIMARY KEY (branch, path)
            );
            CREATE TABLE IF NOT EXISTS jobs (
              job_id INTEGER PRIMARY KEY AUTOINCREMENT,
              branch TEXT NOT NULL,
              started_at TEXT NOT NULL,
              finished_at TEXT,
              status TEXT NOT NULL,
              detail TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS embedding_versions (
              scope TEXT PRIMARY KEY,
              model TEXT NOT NULL,
              version TEXT NOT NULL,
              dimensions INTEGER NOT NULL,
              recorded_at TEXT NOT NULL
            );
            """
        )
        return connection

    def branch(self) -> str:
        result = _git(self.project_root, "branch", "--show-current")
        name = result.stdout.strip()
        if result.returncode != 0 or not name:
            revision = _git(self.project_root, "rev-parse", "--short", "HEAD")
            if revision.returncode != 0:
                raise RuntimeError("project root is not a readable Git checkout")
            return f"detached-{revision.stdout.strip()}"
        return name

    def revision(self) -> str:
        result = _git(self.project_root, "rev-parse", "HEAD")
        if result.returncode != 0:
            raise RuntimeError("project root has no readable Git revision")
        return result.stdout.strip()

    def _candidate_paths(self) -> list[str]:
        result = _git(
            self.project_root,
            "ls-files", "-z", "--cached", "--others", "--exclude-standard",
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "git ls-files failed")
        shared_relative = None
        if self.shared_home.is_relative_to(self.project_root):
            shared_relative = self.shared_home.relative_to(self.project_root).as_posix()
        return sorted({
            path for path in result.stdout.split("\0")
            if path and not path.startswith(".mavis/")
            and (not shared_relative or path != shared_relative
                 and not path.startswith(shared_relative + "/"))
        })

    def _read_live(self, relative: str) -> tuple[bytes, str] | None:
        path = self.project_root / relative
        try:
            resolved = path.resolve(strict=True)
            if (path.is_symlink() or not resolved.is_relative_to(self.project_root) or not path.is_file()
                    or path.suffix.lower() not in TEXT_SUFFIXES
                    or path.stat().st_size > MAX_INDEX_BYTES):
                return None
            raw = path.read_bytes()
            if len(raw) > MAX_INDEX_BYTES:
                return None
            return raw, raw.decode("utf-8", errors="replace")
        except (OSError, RuntimeError):
            return None

    def refresh(self) -> dict[str, Any]:
        branch = self.branch()
        head = self.revision()
        connection = self._connect()
        started = _now()
        job = connection.execute(
            "INSERT INTO jobs(branch, started_at, status, detail) VALUES (?, ?, 'running', '')",
            (branch, started),
        ).lastrowid
        connection.commit()
        changed = unchanged = skipped = 0
        seen: set[str] = set()
        try:
            previous = {
                row["path"]: row["sha256"] for row in connection.execute(
                    "SELECT path,sha256 FROM files WHERE branch=?", (branch,)
                )
            }
            current_hashes: dict[str, str] = {}
            for relative in self._candidate_paths():
                path = self.project_root / relative
                if (path.is_symlink() or not path.is_file()
                        or not path.resolve(strict=True).is_relative_to(self.project_root)
                        or path.suffix.lower() not in TEXT_SUFFIXES):
                    skipped += 1
                    continue
                size = path.stat().st_size
                if size > MAX_INDEX_BYTES:
                    skipped += 1
                    continue
                raw = path.read_bytes()
                digest = _digest(raw)
                seen.add(relative)
                current_hashes[relative] = digest
                row = connection.execute(
                    "SELECT sha256 FROM files WHERE branch=? AND path=?",
                    (branch, relative),
                ).fetchone()
                if row and row["sha256"] == digest:
                    unchanged += 1
                    continue
                content = raw.decode("utf-8", errors="replace")
                symbols = sorted(set(SYMBOL_RE.findall(content)))
                dependencies = sorted(
                    {next(item for item in match if item) for match in DEPENDENCY_RE.findall(content)}
                )
                connection.execute(
                    """INSERT INTO files(branch,path,sha256,bytes,content,symbols,dependencies,indexed_at)
                       VALUES (?,?,?,?,?,?,?,?)
                       ON CONFLICT(branch,path) DO UPDATE SET
                         sha256=excluded.sha256,bytes=excluded.bytes,content=excluded.content,
                         symbols=excluded.symbols,dependencies=excluded.dependencies,
                         indexed_at=excluded.indexed_at""",
                    (branch, relative, digest, size, content, json.dumps(symbols), json.dumps(dependencies), _now()),
                )
                changed += 1
            deleted = sorted(previous.keys() - seen)
            added_by_hash = {digest: path for path, digest in current_hashes.items()
                             if path not in previous}
            renamed = sorted(
                ({"from": path, "to": added_by_hash[previous[path]]}
                 for path in deleted if previous[path] in added_by_hash),
                key=lambda item: (item["from"], item["to"]),
            )
            for relative in deleted:
                connection.execute("DELETE FROM files WHERE branch=? AND path=?", (branch, relative))
            detail = json.dumps(
                {"head": head, "changed": changed, "unchanged": unchanged,
                 "deleted": deleted, "renamed": renamed, "skipped": skipped},
                sort_keys=True,
            )
            connection.execute(
                "UPDATE jobs SET finished_at=?, status='complete', detail=? WHERE job_id=?",
                (_now(), detail, job),
            )
            connection.commit()
            return {"branch": branch, "head": head, "changed": changed,
                    "unchanged": unchanged, "deleted": deleted,
                    "renamed": renamed, "skipped": skipped}
        except Exception as exc:
            connection.rollback()
            connection.execute(
                "UPDATE jobs SET finished_at=?, status='failed', detail=? WHERE job_id=?",
                (_now(), f"{type(exc).__name__}: {exc}", job),
            )
            connection.commit()
            raise
        finally:
            connection.close()

    def changed_paths(self) -> list[str]:
        result = _git(self.project_root, "status", "--porcelain=v1", "-z")
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "git status failed")
        entries = result.stdout.split("\0")
        paths: set[str] = set()
        position = 0
        while position < len(entries):
            entry = entries[position]
            position += 1
            if len(entry) < 4:
                continue
            status, relative = entry[:2], entry[3:]
            if relative and not relative.startswith(".mavis/"):
                paths.add(relative)
            if "R" in status or "C" in status:
                if position < len(entries):
                    prior = entries[position]
                    position += 1
                    if prior and not prior.startswith(".mavis/"):
                        paths.add(prior)
        return sorted(paths)

    def status(self) -> dict[str, Any]:
        branch = self.branch()
        head = self.revision()
        connection = self._connect()
        try:
            latest = connection.execute(
                "SELECT job_id,started_at,finished_at,status,detail FROM jobs "
                "WHERE branch=? ORDER BY job_id DESC LIMIT 1", (branch,),
            ).fetchone()
            if latest is None:
                return {"branch": branch, "head": head, "snapshot_current": False,
                        "latest_job": None}
            job = dict(latest)
            try:
                indexed_head = json.loads(job["detail"]).get("head")
            except (ValueError, TypeError, AttributeError):
                indexed_head = None
            return {"branch": branch, "head": head,
                    "snapshot_current": job["status"] == "complete" and indexed_head == head,
                    "latest_job": job}
        finally:
            connection.close()

    @staticmethod
    def _line_hits(content: str, query: str, path: str, scope: str, source: str) -> list[SearchHit]:
        needle = query.casefold()
        hits = []
        for number, line in enumerate(content.splitlines(), 1):
            if needle in line.casefold():
                exact = line.strip().casefold() == needle
                hits.append(SearchHit(scope, path, number, line[:1000], 100 if exact else 50, source))
        return hits

    @staticmethod
    def _page(limit: int, offset: int) -> None:
        if not 1 <= limit <= 200 or offset < 0:
            raise ValueError("limit must be 1..200 and offset must be nonnegative")

    def _project_documents(self) -> Iterator[tuple[str, str, str]]:
        """Yield current files; a stale or failed snapshot never supplies answers."""
        branch = self.branch()
        head = self.revision()
        candidates = set(self._candidate_paths())
        connection = self._connect()
        try:
            latest = connection.execute(
                "SELECT status,detail FROM jobs WHERE branch=? ORDER BY job_id DESC LIMIT 1",
                (branch,),
            ).fetchone()
            try:
                indexed_head = json.loads(latest["detail"]).get("head") if latest else None
            except (ValueError, TypeError):
                indexed_head = None
            if not latest or latest["status"] != "complete" or indexed_head != head:
                rows: dict[str, sqlite3.Row] = {}
                fallback = True
            else:
                rows = {row["path"]: row for row in connection.execute(
                    "SELECT path,sha256,content FROM files WHERE branch=?", (branch,)
                )}
                fallback = False
        finally:
            connection.close()

        changed = set(self.changed_paths())
        for relative in sorted(candidates):
            live = self._read_live(relative)
            if live is None:
                continue
            raw, content = live
            row = rows.get(relative)
            if relative in changed:
                source = "changed-file"
            elif fallback or row is None or row["sha256"] != _digest(raw):
                source = "live-fallback"
            else:
                source = "index"
                content = row["content"]
            yield relative, content, source

    def _verified_global_hits(self, query: str) -> list[SearchHit]:
        directory = self.shared_home / "knowledge"
        if not directory.is_dir():
            return []
        hits: list[SearchHit] = []
        for path in sorted(directory.glob("*.json")):
            if path.is_symlink():
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if (record.get("schema_version") != "mavis.memory-record/v1"
                        or record.get("scope") != "global-coding"
                        or record.get("verification_state") != "verified"):
                    continue
                references = record.get("source_references")
                if not isinstance(references, list) or not references:
                    continue
                if any(
                    not isinstance(item, dict)
                    or not isinstance(item.get("path"), str)
                    or not isinstance(item.get("sha256"), str)
                    or not Path(item["path"]).is_file()
                    or _digest(Path(item["path"]).read_bytes()) != item["sha256"]
                    for item in references
                ):
                    continue
                claim = record.get("claim")
                if isinstance(claim, str):
                    hits.extend(self._line_hits(claim, query, str(path), "global-coding", "verified-global"))
            except (OSError, ValueError, TypeError):
                continue
        return hits

    def search(self, query: str, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query must not be empty")
        self._page(limit, offset)
        hits: list[SearchHit] = []
        for relative, content, source in self._project_documents():
            hits.extend(self._line_hits(content, query, relative, "project", source))
        dedup: dict[tuple[str, str, int, str], SearchHit] = {}
        for hit in hits:
            key = (hit.scope, hit.path, hit.line, hit.text)
            if key not in dedup or hit.score > dedup[key].score:
                dedup[key] = hit
        project_ranked = sorted(dedup.values(), key=lambda item: (-item.score, item.path, item.line))
        global_ranked = sorted(self._verified_global_hits(query),
                               key=lambda item: (-item.score, item.path, item.line))
        return [item.as_dict() for item in (project_ranked + global_ranked)[offset:offset + limit]]

    def symbol(self, name: str, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        self._page(limit, offset)
        results = []
        for path, content, source in self._project_documents():
            definitions = [match for match in SYMBOL_RE.finditer(content) if match.group(1) == name]
            dependencies = [match for match in DEPENDENCY_RE.finditer(content)
                            if name in (item for item in match.groups() if item)]
            if definitions or dependencies:
                match = (definitions or dependencies)[0]
                results.append({"path": path, "line": content.count("\n", 0, match.start()) + 1,
                                "defines": bool(definitions), "depends_on": bool(dependencies),
                                "source": source})
        return results[offset:offset + limit]

    def dependency(self, name: str, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        self._page(limit, offset)
        results = []
        for path, content, source in self._project_documents():
            for match in DEPENDENCY_RE.finditer(content):
                dependency = next((item for item in match.groups() if item), None)
                if dependency == name:
                    results.append({"path": path,
                                    "line": content.count("\n", 0, match.start()) + 1,
                                    "dependency": dependency, "source": source})
        return results[offset:offset + limit]

    def bind_embedding_version(self, scope: str, model: str, version: str, dimensions: int) -> None:
        connection = self._connect()
        try:
            prior = connection.execute(
                "SELECT model,version,dimensions FROM embedding_versions WHERE scope=?", (scope,)
            ).fetchone()
            current = (model, version, dimensions)
            if prior and (prior["model"], prior["version"], prior["dimensions"]) != current:
                raise ValueError("embedding version changed; rebuild the scope before comparing vectors")
            connection.execute(
                "INSERT OR REPLACE INTO embedding_versions(scope,model,version,dimensions,recorded_at) VALUES (?,?,?,?,?)",
                (scope, model, version, dimensions, _now()),
            )
            connection.commit()
        finally:
            connection.close()
