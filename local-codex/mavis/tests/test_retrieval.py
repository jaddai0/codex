from pathlib import Path
import subprocess
import tempfile
import unittest

from mavis.retrieval import ProjectIndex


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


if __name__ == "__main__":
    unittest.main()
