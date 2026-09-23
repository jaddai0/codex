"""Source-only tests for the optional, injected project vector index."""

import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import unittest

from mavis.embedding_index import EmbeddingIdentity
from mavis.retrieval import ProjectIndex


class FakeProvider:
    def __init__(self, identity=None):
        self.identity = identity or EmbeddingIdentity("fake", "revision-1", "v1", 2)
        self.documents = []
        self.queries = []
        self.fail = False

    def embed_documents(self, texts):
        self.documents.extend(texts)
        if self.fail:
            raise RuntimeError("provider unavailable")
        return [[1.0, 0.0] if "apple" in text else [0.0, 1.0] for text in texts]

    def embed_query(self, query):
        self.queries.append(query)
        return [1.0, 0.0]


class EmbeddingIndexTests(unittest.TestCase):
    def make_index(self, root):
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.invalid"],
            cwd=root,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Mavis Test"], cwd=root, check=True
        )
        (root / ".gitignore").write_text(".mavis/\nignored.md\n")
        (root / "apple.md").write_text("apple source\n")
        (root / "banana.md").write_text("banana source\n")
        subprocess.run(
            ["git", "add", ".gitignore", "apple.md", "banana.md"], cwd=root, check=True
        )
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
        return ProjectIndex(root, root / "shared")

    def test_incremental_vectors_identity_and_explicit_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = self.make_index(root)
            provider = FakeProvider()
            self.assertEqual(index.refresh_embeddings(provider)["changed"], 2)
            self.assertEqual(index.refresh_embeddings(provider)["unchanged"], 2)
            self.assertEqual(len(provider.documents), 2)
            (root / "banana.md").write_text("banana changed\n")
            self.assertEqual(index.refresh_embeddings(provider)["changed"], 1)
            self.assertEqual(len(provider.documents), 3)
            for identity in (
                EmbeddingIdentity("other", "revision-1", "v1", 2),
                EmbeddingIdentity("fake", "revision-2", "v1", 2),
                EmbeddingIdentity("fake", "revision-1", "v2", 2),
                EmbeddingIdentity("fake", "revision-1", "v1", 3),
            ):
                with self.assertRaisesRegex(ValueError, "rebuild"):
                    index.refresh_embeddings(FakeProvider(identity))
                with self.assertRaisesRegex(ValueError, "rebuild"):
                    index.search("concept", embedding_provider=FakeProvider(identity))
            replacement = FakeProvider(EmbeddingIdentity("new", "revision-3", "v2", 2))
            self.assertEqual(
                index.refresh_embeddings(replacement, rebuild=True)["changed"], 2
            )
            self.assertEqual(len(replacement.documents), 2)
            self.assertEqual(
                index.search("concept", embedding_provider=replacement)[0]["path"],
                "apple.md",
            )

    def test_stale_exact_fallback_delete_rename_ignored_and_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = self.make_index(root)
            provider = FakeProvider()
            index.refresh_embeddings(provider)
            (root / "apple.md").write_text("changed exact fact\n")
            hits = index.search("changed exact fact", embedding_provider=provider)
            self.assertEqual(hits[0]["source"], "changed-file")
            self.assertNotIn(
                "apple.md",
                [
                    hit["path"]
                    for hit in index.search("concept", embedding_provider=provider)
                ],
            )
            self.assertEqual(index.refresh_embeddings(provider)["changed"], 1)
            subprocess.run(
                ["git", "mv", "apple.md", "renamed.md"], cwd=root, check=True
            )
            self.assertNotIn(
                "apple.md",
                [
                    hit["path"]
                    for hit in index.search("concept", embedding_provider=provider)
                ],
            )
            count = len(provider.documents)
            result = index.refresh_embeddings(provider)
            self.assertEqual(
                result["renamed"], [{"from": "apple.md", "to": "renamed.md"}]
            )
            self.assertEqual(len(provider.documents), count)
            self.assertIn(
                "renamed.md",
                [
                    hit["path"]
                    for hit in index.search("concept", embedding_provider=provider)
                ],
            )
            (root / "renamed.md").unlink()
            self.assertNotIn(
                "renamed.md",
                [
                    hit["path"]
                    for hit in index.search("concept", embedding_provider=provider)
                ],
            )
            (root / "ignored.md").write_text("apple ignored\n")
            self.assertNotIn(
                "ignored.md",
                [
                    hit["path"]
                    for hit in index.search("concept", embedding_provider=provider)
                ],
            )
            subprocess.run(["git", "checkout", "-qb", "other"], cwd=root, check=True)
            self.assertEqual(index.search("concept", embedding_provider=provider), [])
            self.assertEqual(index.refresh_embeddings(provider)["changed"], 1)
            self.assertTrue(index.search("concept", embedding_provider=provider))

    def test_opt_in_ranking_paging_exact_and_global_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            root.mkdir()
            index = self.make_index(root)
            provider = FakeProvider()
            index.refresh_embeddings(provider)
            shared = root / "shared"
            evidence = Path(directory) / "evidence.txt"
            evidence.write_text("verified\n")
            knowledge = shared / "knowledge"
            knowledge.mkdir(parents=True)
            (knowledge / "fact.json").write_text(
                json.dumps(
                    {
                        "schema_version": "mavis.memory-record/v1",
                        "scope": "global-coding",
                        "verification_state": "verified",
                        "claim": "concept global",
                        "source_references": [
                            {
                                "path": str(evidence),
                                "sha256": hashlib.sha256(
                                    evidence.read_bytes()
                                ).hexdigest(),
                            }
                        ],
                    }
                )
            )
            self.assertEqual(
                [hit["source"] for hit in index.search("concept")], ["verified-global"]
            )
            hits = index.search("concept", embedding_provider=provider)
            self.assertEqual(
                [hit["source"] for hit in hits],
                ["embedding", "embedding", "verified-global"],
            )
            self.assertEqual(
                [hit["path"] for hit in hits[:2]], ["apple.md", "banana.md"]
            )
            self.assertEqual(
                index.search("concept", limit=1, offset=1, embedding_provider=provider),
                hits[1:2],
            )
            for hit in hits[:2]:
                self.assertEqual(
                    (root / hit["path"]).read_text().splitlines()[hit["line"] - 1],
                    hit["text"],
                )
            self.assertEqual(
                index.search("apple source", embedding_provider=provider)[0]["source"],
                "live-fallback",
            )

    def test_disabled_preserves_vectors_and_provider_failure_does_not_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = self.make_index(root)
            provider = FakeProvider()
            index.refresh_embeddings(provider)
            connection = sqlite3.connect(index.database)
            before = connection.execute(
                "SELECT path,sha256,vector FROM embedding_documents ORDER BY path"
            ).fetchall()
            connection.close()
            (root / "banana.md").write_text("banana revised\n")
            self.assertEqual(
                index.search("banana revised")[0]["source"], "changed-file"
            )
            connection = sqlite3.connect(index.database)
            self.assertEqual(
                before,
                connection.execute(
                    "SELECT path,sha256,vector FROM embedding_documents ORDER BY path"
                ).fetchall(),
            )
            connection.close()
            provider.fail = True
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                index.refresh_embeddings(provider)
            connection = sqlite3.connect(index.database)
            self.assertEqual(
                before,
                connection.execute(
                    "SELECT path,sha256,vector FROM embedding_documents ORDER BY path"
                ).fetchall(),
            )
            connection.close()

    def test_bad_vectors_rejected_before_storage_or_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = self.make_index(root)
            provider = FakeProvider()
            provider.embed_documents = lambda texts: [[1.0] for _ in texts]
            with self.assertRaisesRegex(ValueError, "dimensions"):
                index.refresh_embeddings(provider)
            connection = sqlite3.connect(index.database)
            self.assertEqual(
                connection.execute(
                    "SELECT count(*) FROM embedding_documents"
                ).fetchone()[0],
                0,
            )
            connection.close()
            provider = FakeProvider()
            index.refresh_embeddings(provider)
            connection = sqlite3.connect(index.database)
            connection.execute(
                "UPDATE embedding_documents SET vector='[0, 0]' WHERE path='apple.md'"
            )
            connection.commit()
            connection.close()
            with self.assertRaisesRegex(ValueError, "magnitude"):
                index.search("concept", embedding_provider=provider)


if __name__ == "__main__":
    unittest.main()
