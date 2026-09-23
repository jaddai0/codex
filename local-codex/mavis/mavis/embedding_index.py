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
            """CREATE TABLE IF NOT EXISTS embedding_documents (
                 branch TEXT NOT NULL,
                 path TEXT NOT NULL,
                 sha256 TEXT NOT NULL,
                 model TEXT NOT NULL,
                 revision TEXT NOT NULL,
                 version TEXT NOT NULL,
                 dimensions INTEGER NOT NULL,
                 vector TEXT NOT NULL,
                 first_line TEXT NOT NULL,
                 PRIMARY KEY (branch, path)
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

    def _documents(self) -> dict[str, tuple[str, str, str]]:
        documents = {}
        for path in self.source._candidate_paths():
            live = self.source._read_live(path)
            if live is None or not live[0]:
                continue
            raw, content = live
            lines = content.splitlines()
            documents[path] = (
                hashlib.sha256(raw).hexdigest(),
                content,
                lines[0] if lines else "",
            )
        return documents

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
            rows = {
                row["path"]: row
                for row in connection.execute(
                    "SELECT * FROM embedding_documents WHERE branch=?", (branch,)
                )
            }
            if (prior is None and rows) or (
                prior is not None and self._row_identity(prior) != self._tuple(identity)
            ):
                if not rebuild:
                    raise ValueError(
                        "embedding identity changed; rebuild before comparing vectors"
                    )
            if not rebuild and any(
                self._row_identity(row) != self._tuple(identity)
                for row in rows.values()
            ):
                raise ValueError("stored embedding vector has an incompatible identity")

            reusable = {} if rebuild else rows
            removed = sorted(set(rows) - set(documents))
            removed_by_hash = {}
            if not rebuild:
                for path in removed:
                    removed_by_hash.setdefault(rows[path]["sha256"], rows[path])
            pending = []
            prepared = []
            renamed = []
            unchanged = 0
            for path, (digest, content, first_line) in sorted(documents.items()):
                row = reusable.get(path)
                if row is not None and row["sha256"] == digest:
                    unchanged += 1
                    continue
                old = removed_by_hash.get(digest)
                if old is not None:
                    vector = _unit(json.loads(old["vector"]), identity.dimensions)
                    prepared.append((path, digest, vector, first_line))
                    renamed.append({"from": old["path"], "to": path})
                else:
                    pending.append((path, digest, content, first_line))

            for start in range(0, len(pending), 16):
                batch = pending[start : start + 16]
                vectors = provider.embed_documents([item[2] for item in batch])
                if not isinstance(vectors, list) or len(vectors) != len(batch):
                    raise ValueError(
                        "embedding provider returned the wrong document count"
                    )
                for item, vector in zip(batch, vectors):
                    prepared.append(
                        (item[0], item[1], _unit(vector, identity.dimensions), item[3])
                    )

            if {path: item[0] for path, item in self._documents().items()} != {
                path: item[0] for path, item in documents.items()
            } or branch != self.source.branch():
                raise ValueError("project changed while embeddings were prepared")

            connection.execute("BEGIN IMMEDIATE")
            if rebuild:
                connection.execute(
                    "DELETE FROM embedding_documents WHERE branch=?", (branch,)
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
                    "DELETE FROM embedding_documents WHERE branch=? AND path=?",
                    (branch, path),
                )
            for path, digest, vector, first_line in prepared:
                connection.execute(
                    """INSERT OR REPLACE INTO embedding_documents
                       (branch,path,sha256,model,revision,version,dimensions,vector,first_line)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        branch,
                        path,
                        digest,
                        identity.model,
                        identity.revision,
                        identity.version,
                        identity.dimensions,
                        json.dumps(vector),
                        first_line,
                    ),
                )
            connection.commit()
            return {
                "branch": branch,
                "identity": identity.as_dict(),
                "changed": len(pending),
                "renamed": renamed,
                "unchanged": unchanged,
                "deleted": sorted(set(removed) - {item["from"] for item in renamed}),
            }
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

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
                "SELECT * FROM embedding_documents WHERE branch=?", (branch,)
            ).fetchall()
        finally:
            connection.close()
        valid = []
        candidates = set(self.source._candidate_paths())
        for row in rows:
            if self._row_identity(row) != self._tuple(identity):
                raise ValueError("stored embedding vector has an incompatible identity")
            if row["path"] not in candidates:
                continue
            live = self.source._read_live(row["path"])
            if live is None or hashlib.sha256(live[0]).hexdigest() != row["sha256"]:
                continue
            lines = live[1].splitlines()
            if not lines or lines[0] != row["first_line"]:
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
                    "line": 1,
                    "text": row["first_line"],
                    "score": similarity,
                    "source": "embedding",
                    "sha256": row["sha256"],
                    "embedding_identity": identity.as_dict(),
                }
            )
        return sorted(hits, key=lambda item: (-item["score"], item["path"]))
