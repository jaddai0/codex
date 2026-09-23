"""Optional, source-verified vectors for the deterministic project index."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Protocol


@dataclass(frozen=True)
class EmbeddingIdentity:
    model: str
    revision: str
    version: str
    dimensions: int

    def validate(self) -> None:
        if (
            not all(
                isinstance(value, str) and value.strip()
                for value in (self.model, self.revision, self.version)
            )
            or type(self.dimensions) is not int
            or self.dimensions < 1
        ):
            raise ValueError(
                "embedding model, revision, version, and dimensions are required"
            )

    def as_dict(self) -> dict[str, str | int]:
        return {
            "model": self.model,
            "revision": self.revision,
            "version": self.version,
            "dimensions": self.dimensions,
        }


class EmbeddingProvider(Protocol):
    """Injected model adapter; this module never loads or downloads a model."""

    identity: EmbeddingIdentity

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


def _unit(vector: list[float], dimensions: int) -> list[float]:
    if (
        not isinstance(vector, list)
        or len(vector) != dimensions
        or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in vector
        )
    ):
        raise ValueError("embedding vector has incompatible dimensions or values")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("embedding vector has no finite magnitude")
    return [float(value / norm) for value in vector]


class EmbeddingIndex:
    """Manage one branch's vectors in ProjectIndex's ignored SQLite database."""

    def __init__(self, source: Any):
        self.source = source

    def _connect(self):
        connection = self.source._connect()
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(embedding_versions)")
        }
        if "revision" not in columns:
            connection.execute(
                "ALTER TABLE embedding_versions ADD COLUMN revision TEXT NOT NULL DEFAULT ''"
            )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS embedding_passages (
                 branch TEXT NOT NULL,
                 path TEXT NOT NULL,
                 passage_index INTEGER NOT NULL,
                 sha256 TEXT NOT NULL,
                 model TEXT NOT NULL,
                 revision TEXT NOT NULL,
                 version TEXT NOT NULL,
                 dimensions INTEGER NOT NULL,
                 vector TEXT NOT NULL,
                 start_line INTEGER NOT NULL,
                 end_line INTEGER NOT NULL,
                 text TEXT NOT NULL,
                 PRIMARY KEY (branch, path, passage_index)
               )"""
        )
        connection.commit()
        return connection

    @staticmethod
    def _identity(provider: EmbeddingProvider) -> EmbeddingIdentity:
        identity = getattr(provider, "identity", None)
        if not isinstance(identity, EmbeddingIdentity):
            raise ValueError("embedding provider lacks a fixed identity")
        identity.validate()
        return identity

    @staticmethod
    def _row_identity(row: Any) -> tuple[str, str, str, int]:
        return (row["model"], row["revision"], row["version"], row["dimensions"])

    @staticmethod
    def _tuple(identity: EmbeddingIdentity) -> tuple[str, str, str, int]:
        return (
            identity.model,
            identity.revision,
            identity.version,
            identity.dimensions,
        )

    @staticmethod
    def _passages(content: str) -> list[tuple[int, int, str]]:
        """Bound provider input while retaining a checkable source line span."""
        max_chars, max_lines = 8192, 48
        passages: list[tuple[int, int, str]] = []
        parts: list[str] = []
        first_line = last_line = 0
        size = 0

        def flush() -> None:
            nonlocal parts, first_line, last_line, size
            if parts:
                passages.append((first_line, last_line, "".join(parts)))
            parts, first_line, last_line, size = [], 0, 0, 0

        for number, line in enumerate(content.splitlines(keepends=True), 1):
            position = 0
            while position < len(line):
                if parts and (size == max_chars or number - first_line >= max_lines):
                    flush()
                if not parts:
                    first_line = number
                take = min(max_chars - size, len(line) - position)
                parts.append(line[position : position + take])
                position += take
                size += take
                last_line = number
        flush()
        return passages

    def _documents(self) -> dict[str, tuple[str, list[tuple[int, int, str]]]]:
        documents = {}
        for path in self.source._candidate_paths():
            live = self.source._read_live(path)
            if live is None or not live[0]:
                continue
            raw, content = live
            documents[path] = (
                hashlib.sha256(raw).hexdigest(),
                self._passages(content),
            )
        return documents

    def status(self) -> dict[str, Any]:
        """Report the bound branch identity and whether every vector matches live source."""
        branch = self.source.branch()
        connection = self._connect()
        try:
            identity = connection.execute(
                "SELECT model,revision,version,dimensions FROM embedding_versions WHERE scope=?",
                (f"project:{branch}",),
            ).fetchone()
            rows: dict[str, list[Any]] = {}
            for row in connection.execute(
                "SELECT * FROM embedding_passages WHERE branch=? ORDER BY path,passage_index",
                (branch,),
            ):
                rows.setdefault(row["path"], []).append(row)
            documents = self._documents() if identity is not None else {}
            current = identity is not None and set(rows) == set(documents) and all(
                self._matches(rows[path], digest, passages)
                and all(self._row_identity(row) == self._row_identity(identity)
                        for row in rows[path])
                for path, (digest, passages) in documents.items()
            )
            return {
                "identity": dict(identity) if identity is not None else None,
                "passages": sum(len(group) for group in rows.values()),
                "snapshot_current": current,
            }
        finally:
            connection.close()

    def refresh(
        self, provider: EmbeddingProvider, *, rebuild: bool = False
    ) -> dict[str, Any]:
        identity = self._identity(provider)
        branch = self.source.branch()
        scope = f"project:{branch}"
        documents = self._documents()
        connection = self._connect()
        try:
            prior = connection.execute(
                "SELECT model,revision,version,dimensions FROM embedding_versions WHERE scope=?",
                (scope,),
            ).fetchone()
            rows: dict[str, list[Any]] = {}
            for row in connection.execute(
                "SELECT * FROM embedding_passages WHERE branch=? ORDER BY path,passage_index",
                (branch,),
            ):
                rows.setdefault(row["path"], []).append(row)
            if (prior is None and rows) or (
                prior is not None and self._row_identity(prior) != self._tuple(identity)
            ):
                if not rebuild:
                    raise ValueError(
                        "embedding identity changed; rebuild before comparing vectors"
                    )
            if not rebuild and any(
                self._row_identity(row) != self._tuple(identity)
                for group in rows.values() for row in group
            ):
                raise ValueError("stored embedding vector has an incompatible identity")

            reusable = {} if rebuild else rows
            removed = sorted(set(rows) - set(documents))
            removed_by_hash = {}
            if not rebuild:
                for path in removed:
                    removed_by_hash.setdefault(rows[path][0]["sha256"], rows[path])
            pending: list[tuple[str, str, int, int, int, str]] = []
            prepared: list[tuple[str, str, int, int, int, str, list[float]]] = []
            renamed = []
            unchanged = 0
            changed = 0
            for path, (digest, passages) in sorted(documents.items()):
                group = reusable.get(path)
                if group is not None and self._matches(group, digest, passages):
                    unchanged += 1
                    continue
                old = removed_by_hash.get(digest)
                if old is not None and self._matches(old, digest, passages):
                    for row in old:
                        vector = _unit(json.loads(row["vector"]), identity.dimensions)
                        prepared.append((path, digest, row["passage_index"],
                                         row["start_line"], row["end_line"], row["text"], vector))
                    renamed.append({"from": old[0]["path"], "to": path})
                else:
                    changed += 1
                    pending.extend((path, digest, index, start, end, text)
                                   for index, (start, end, text) in enumerate(passages))

            for start in range(0, len(pending), 16):
                batch = pending[start : start + 16]
                vectors = provider.embed_documents([item[5] for item in batch])
                if not isinstance(vectors, list) or len(vectors) != len(batch):
                    raise ValueError(
                        "embedding provider returned the wrong document count"
                    )
                for item, vector in zip(batch, vectors):
                    prepared.append((*item, _unit(vector, identity.dimensions)))

            if {path: item[0] for path, item in self._documents().items()} != {
                path: item[0] for path, item in documents.items()
            } or branch != self.source.branch():
                raise ValueError("project changed while embeddings were prepared")

            connection.execute("BEGIN IMMEDIATE")
            if rebuild:
                connection.execute(
                    "DELETE FROM embedding_passages WHERE branch=?", (branch,)
                )
                connection.execute(
                    "DELETE FROM embedding_versions WHERE scope=?", (scope,)
                )
            connection.execute(
                """INSERT OR REPLACE INTO embedding_versions
                   (scope,model,version,dimensions,recorded_at,revision) VALUES (?,?,?,?,?,?)""",
                (
                    scope,
                    identity.model,
                    identity.version,
                    identity.dimensions,
                    datetime.now(timezone.utc).isoformat(),
                    identity.revision,
                ),
            )
            for path in removed:
                connection.execute(
                    "DELETE FROM embedding_passages WHERE branch=? AND path=?",
                    (branch, path),
                )
            prepared_paths = {item[0] for item in prepared}
            renamed_paths = {item["to"] for item in renamed}
            for path in documents:
                if path in rows and path not in renamed_paths and path not in prepared_paths:
                    continue
                connection.execute(
                    "DELETE FROM embedding_passages WHERE branch=? AND path=?", (branch, path)
                )
            for path, digest, passage_index, start, end, text, vector in prepared:
                connection.execute(
                    """INSERT INTO embedding_passages
                       (branch,path,passage_index,sha256,model,revision,version,
                        dimensions,vector,start_line,end_line,text)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        branch,
                        path,
                        passage_index,
                        digest,
                        identity.model,
                        identity.revision,
                        identity.version,
                        identity.dimensions,
                        json.dumps(vector),
                        start,
                        end,
                        text,
                    ),
                )
            # Older source-only builds stored one vector per file. Retire those
            # rows only after the passage refresh has succeeded.
            connection.execute("DROP TABLE IF EXISTS embedding_documents")
            connection.commit()
            return {
                "branch": branch,
                "identity": identity.as_dict(),
                "changed": changed,
                "renamed": renamed,
                "unchanged": unchanged,
                "deleted": sorted(set(removed) - {item["from"] for item in renamed}),
            }
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _matches(
        group: list[Any], digest: str, passages: list[tuple[int, int, str]]
    ) -> bool:
        return len(group) == len(passages) and all(
            row["sha256"] == digest
            and row["passage_index"] == index
            and (row["start_line"], row["end_line"], row["text"]) == passage
            for index, (row, passage) in enumerate(zip(group, passages))
        )

    def search(self, query: str, provider: EmbeddingProvider) -> list[dict[str, Any]]:
        identity = self._identity(provider)
        branch = self.source.branch()
        connection = self._connect()
        try:
            prior = connection.execute(
                "SELECT model,revision,version,dimensions FROM embedding_versions WHERE scope=?",
                (f"project:{branch}",),
            ).fetchone()
            if prior is None:
                return []
            if self._row_identity(prior) != self._tuple(identity):
                raise ValueError(
                    "embedding identity changed; rebuild before comparing vectors"
                )
            rows = connection.execute(
                "SELECT * FROM embedding_passages WHERE branch=? ORDER BY path,passage_index",
                (branch,),
            ).fetchall()
        finally:
            connection.close()
        valid = []
        candidates = set(self.source._candidate_paths())
        current: dict[str, tuple[str, list[tuple[int, int, str]]] | None] = {}
        for row in rows:
            if self._row_identity(row) != self._tuple(identity):
                raise ValueError("stored embedding vector has an incompatible identity")
            if row["path"] not in candidates:
                continue
            if row["path"] not in current:
                live = self.source._read_live(row["path"])
                current[row["path"]] = (
                    (hashlib.sha256(live[0]).hexdigest(), self._passages(live[1]))
                    if live is not None else None
                )
            document = current[row["path"]]
            if document is None or document[0] != row["sha256"]:
                continue
            index = row["passage_index"]
            if index < 0 or index >= len(document[1]) or (
                row["start_line"], row["end_line"], row["text"]
            ) != document[1][index]:
                continue
            vector = _unit(json.loads(row["vector"]), identity.dimensions)
            valid.append((row, vector))
        if not valid:
            return []
        query_vector = _unit(provider.embed_query(query), identity.dimensions)
        hits = []
        for row, vector in valid:
            similarity = sum(a * b for a, b in zip(query_vector, vector))
            hits.append(
                {
                    "scope": "project",
                    "path": row["path"],
                    "line": row["start_line"],
                    "end_line": row["end_line"],
                    "text": row["text"],
                    "score": similarity,
                    "source": "embedding",
                    "sha256": row["sha256"],
                    "embedding_identity": identity.as_dict(),
                    "evidence_kind": "passage-candidate",
                }
            )
        return sorted(hits, key=lambda item: (-item["score"], item["path"], item["line"]))
