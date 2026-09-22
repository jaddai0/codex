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
from typing import Any, Iterable


TEXT_SUFFIXES = {
    ".c", ".cc", ".cpp", ".css", ".go", ".h", ".hpp", ".html", ".java",
    ".js", ".json", ".jsx", ".kt", ".md", ".py", ".rb", ".rs", ".sh",
    ".sql", ".swift", ".toml", ".ts", ".tsx", ".txt", ".yaml", ".yml",
}
MAX_INDEX_BYTES = 2 * 1024 * 1024
SYMBOL_RE = re.compile(
    r"^\s*(?:async\s+)?(?:def|class|fn|func|function|interface|struct|enum|protocol)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)
DEPENDENCY_RE = re.compile(
    r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w./@-]+)|use\s+([\w:]+)|require\(['\"]([^'\"]+))",
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

    def _candidate_paths(self) -> list[str]:
        result = _git(
            self.project_root,
            "ls-files", "--cached", "--others", "--exclude-standard",
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "git ls-files failed")
        return sorted(
            path for path in result.stdout.splitlines()
            if path and not path.startswith(".mavis/")
        )

    def refresh(self) -> dict[str, Any]:
        branch = self.branch()
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
            for relative in self._candidate_paths():
                path = self.project_root / relative
                if path.is_symlink() or not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                    skipped += 1
                    continue
                size = path.stat().st_size
                if size > MAX_INDEX_BYTES:
                    skipped += 1
                    continue
                raw = path.read_bytes()
                digest = _digest(raw)
                seen.add(relative)
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
            existing = {
                row["path"] for row in connection.execute(
                    "SELECT path FROM files WHERE branch=?", (branch,)
                )
            }
            deleted = sorted(existing - seen)
            for relative in deleted:
                connection.execute("DELETE FROM files WHERE branch=? AND path=?", (branch, relative))
            detail = json.dumps(
                {"changed": changed, "unchanged": unchanged, "deleted": deleted, "skipped": skipped},
                sort_keys=True,
            )
            connection.execute(
                "UPDATE jobs SET finished_at=?, status='complete', detail=? WHERE job_id=?",
                (_now(), detail, job),
            )
            connection.commit()
            return {"branch": branch, "changed": changed, "unchanged": unchanged, "deleted": deleted, "skipped": skipped}
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
            return []
        paths: list[str] = []
        for entry in result.stdout.split("\0"):
            if len(entry) < 4:
                continue
            value = entry[3:]
            if " -> " in value:
                value = value.split(" -> ", 1)[1]
            if value and not value.startswith(".mavis/"):
                paths.append(value)
        return sorted(set(paths))

    @staticmethod
    def _line_hits(content: str, query: str, path: str, scope: str, source: str) -> list[SearchHit]:
        needle = query.casefold()
        hits = []
        for number, line in enumerate(content.splitlines(), 1):
            if needle in line.casefold():
                exact = line.strip().casefold() == needle
                hits.append(SearchHit(scope, path, number, line[:1000], 100 if exact else 50, source))
        return hits

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query must not be empty")
        branch = self.branch()
        hits: list[SearchHit] = []
        changed = set(self.changed_paths())
        for relative in changed:
            path = self.project_root / relative
            if not path.is_symlink() and path.is_file() and path.stat().st_size <= MAX_INDEX_BYTES:
                hits.extend(self._line_hits(path.read_text(errors="replace"), query, relative, "project", "changed-file"))
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT path,content,symbols,dependencies FROM files WHERE branch=? AND lower(content) LIKE ?",
                (branch, f"%{query.casefold()}%"),
            )
            for row in rows:
                if row["path"] in changed:
                    continue
                hits.extend(self._line_hits(row["content"], query, row["path"], "project", "index"))
        finally:
            connection.close()
        dedup: dict[tuple[str, int, str], SearchHit] = {}
        for hit in hits:
            key = (hit.path, hit.line, hit.text)
            if key not in dedup or hit.score > dedup[key].score:
                dedup[key] = hit
        ranked = sorted(dedup.values(), key=lambda item: (-item.score, item.path, item.line))
        return [item.as_dict() for item in ranked[:limit]]

    def symbol(self, name: str) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT path,symbols,dependencies FROM files WHERE branch=?",
                (self.branch(),),
            )
            results = []
            for row in rows:
                symbols = json.loads(row["symbols"])
                dependencies = json.loads(row["dependencies"])
                if name in symbols or name in dependencies:
                    results.append({"path": row["path"], "defines": name in symbols, "depends_on": name in dependencies})
            return results
        finally:
            connection.close()

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
