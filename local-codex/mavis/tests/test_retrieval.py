from pathlib import Path
import hashlib
import json
import subprocess
import tempfile
import unittest
from unittest import mock

from mavis.retrieval import ProjectIndex
from mavis.cli import build_parser


class RetrievalTests(unittest.TestCase):
    def make_repo(self, root: Path):
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Mavis Test"], cwd=root, check=True)
        (root / ".gitignore").write_text(".mavis/\n")
        (root / "alpha.py").write_text("def useful_symbol():\n    return 'early fact'\n")
        subprocess.run(["git", "add", ".gitignore", "alpha.py"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)

    def test_hash_refresh_search_changed_fallback_delete_and_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repo(root)
            index = ProjectIndex(root, root / "shared")
            first = index.refresh()
            self.assertEqual(first["changed"], 1)
            self.assertEqual(index.search("early fact")[0]["source"], "index")
            self.assertTrue(index.symbol("useful_symbol")[0]["defines"])

            (root / "alpha.py").write_text("def useful_symbol():\n    return 'changed fact'\n")
            self.assertEqual(index.search("changed fact")[0]["source"], "changed-file")
            index.refresh()
            (root / "alpha.py").unlink()
            result = index.refresh()
            self.assertEqual(result["deleted"], ["alpha.py"])
            self.assertEqual(index.search("changed fact"), [])

            subprocess.run(["git", "checkout", "-qb", "other"], cwd=root, check=True)
            (root / "branch.txt").write_text("branch only")
            self.assertEqual(index.refresh()["branch"], "other")
            self.assertEqual(index.search("branch only")[0]["path"], "branch.txt")

    def test_incompatible_embedding_versions_require_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repo(root)
            index = ProjectIndex(root, root / "shared")
            index.bind_embedding_version("project", "embed-a", "1", 32)
            index.bind_embedding_version("project", "embed-a", "1", 32)
            with self.assertRaisesRegex(ValueError, "rebuild"):
                index.bind_embedding_version("project", "embed-b", "1", 64)

    def test_index_does_not_follow_project_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            root.mkdir()
            self.make_repo(root)
            outside = Path(directory) / "outside.py"
            outside.write_text("secret outside project")
            (root / "outside.py").symlink_to(outside)
            index = ProjectIndex(root, root / "shared")
            index.refresh()
            self.assertEqual(index.search("secret outside project"), [])

    def test_project_state_is_ignored_without_project_gitignore_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repo(root)
            (root / ".gitignore").unlink()
            index = ProjectIndex(root, root / "shared")
            index.refresh()
            ignored = subprocess.run(
                ["git", "check-ignore", "-q", ".mavis/index.sqlite3"], cwd=root, check=False
            )
            self.assertEqual(ignored.returncode, 0)

    def test_committed_change_rename_and_delete_never_return_stale_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repo(root)
            index = ProjectIndex(root, root / "shared")
            index.refresh()
            (root / "alpha.py").write_text("def later_symbol():\n    return 'later fact'\n")
            subprocess.run(["git", "commit", "-qam", "changed"], cwd=root, check=True)
            self.assertEqual(index.search("early fact"), [])
            self.assertEqual(index.symbol("useful_symbol"), [])
            self.assertEqual(index.search("later fact")[0]["source"], "live-fallback")
            self.assertFalse(index.status()["snapshot_current"])

            index.refresh()
            subprocess.run(["git", "mv", "alpha.py", "renamed.py"], cwd=root, check=True)
            self.assertEqual(index.search("later fact")[0]["path"], "renamed.py")
            renamed = index.refresh()["renamed"]
            self.assertEqual(renamed, [{"from": "alpha.py", "to": "renamed.py"}])
            (root / "renamed.py").unlink()
            self.assertEqual(index.search("later fact"), [])
            self.assertEqual(index.symbol("later_symbol"), [])

    def test_failed_refresh_exposes_failure_and_searches_live(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repo(root)
            index = ProjectIndex(root, root / "shared")
            index.refresh()
            (root / "alpha.py").write_text("def fresh_symbol():\n    return 'fresh fact'\n")
            with mock.patch.object(index, "_candidate_paths", side_effect=RuntimeError("failed job")):
                with self.assertRaisesRegex(RuntimeError, "failed job"):
                    index.refresh()
            status = index.status()
            self.assertEqual(status["latest_job"]["status"], "failed")
            self.assertFalse(status["snapshot_current"])
            self.assertEqual(index.search("early fact"), [])
            self.assertEqual(index.search("fresh fact")[0]["source"], "changed-file")

    def test_exact_symbol_dependency_paging_and_verified_global(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            root.mkdir()
            self.make_repo(root)
            shared = Path(directory) / "shared"
            index = ProjectIndex(root, shared)
            (root / "mod.ts").write_text("import { X } from 'alpha-lib'\nexport function found() {}\n")
            (root / "other.py").write_text("from alpha_lib import X\ndef found():\n    pass\n")
            (root / "many.txt").write_text("\n".join(f"needle {i}" for i in range(45)))
            index.refresh()
            self.assertEqual([hit["line"] for hit in index.symbol("found")], [2, 2])
            self.assertEqual(index.dependency("alpha-lib")[0]["line"], 1)
            self.assertEqual(index.dependency("alpha_lib")[0]["path"], "other.py")
            self.assertEqual(len(index.search("needle", limit=200)), 45)
            self.assertEqual(len(index.search("needle", limit=20, offset=40)), 5)
            self.assertEqual(index.symbol("found", limit=1, offset=1)[0]["path"], "other.py")
            with self.assertRaises(ValueError):
                index.search("needle", limit=201)

            source = Path(directory) / "evidence.txt"
            source.write_text("verified source")
            knowledge = shared / "knowledge"
            knowledge.mkdir(parents=True)
            record = {"schema_version": "mavis.memory-record/v1", "scope": "global-coding",
                      "verification_state": "verified", "claim": "needle global fact",
                      "source_references": [{"path": str(source),
                                             "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}]}
            (knowledge / "fact.json").write_text(json.dumps(record))
            hits = index.search("needle", limit=200)
            self.assertEqual(hits[0]["scope"], "project")
            self.assertEqual(hits[-1]["source"], "verified-global")
            source.write_text("now stale")
            self.assertTrue(all(hit["scope"] == "project" for hit in index.search("needle", limit=200)))

    def test_internal_symlink_is_not_searched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repo(root)
            (root / "alias.py").symlink_to(root / "alpha.py")
            index = ProjectIndex(root, root / "shared")
            self.assertEqual([hit["path"] for hit in index.search("early fact")], ["alpha.py"])

    def test_cli_exposes_status_dependency_and_pages(self):
        parser = build_parser()
        args = parser.parse_args(["project-index", "--project", "/tmp/project",
                                  "search", "needle", "--limit", "100", "--offset", "20"])
        self.assertEqual((args.limit, args.offset), (100, 20))
        args = parser.parse_args(["project-index", "--project", "/tmp/project",
                                  "dependency", "alpha-lib", "--offset", "1"])
        self.assertEqual((args.index_command, args.offset), ("dependency", 1))
        args = parser.parse_args(["project-index", "--project", "/tmp/project", "status"])
        self.assertEqual(args.index_command, "status")


if __name__ == "__main__":
    unittest.main()
