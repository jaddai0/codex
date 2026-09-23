"""Project-root binding must match the checkout Codex will actually open."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1]))

from launch_core import bind_project_root


class LaunchProjectBindingTests(unittest.TestCase):
    def test_cd_selects_nested_checkout_over_shell_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            subprocess.run(["git", "init", "-q", str(home)], check=True)
            first = home / "first"
            second = home / "second"
            for root in (first, second):
                root.mkdir()
                subprocess.run(["git", "init", "-q", str(root)], check=True)
            with patch.dict(os.environ, {"HOME": str(home),
                                              "GIT_DIR": str(first / ".git"),
                                              "GIT_WORK_TREE": str(first)}), patch(
                "os.getcwd", return_value=str(first)
            ):
                os.environ.pop("MAVIS_PROJECT_ROOT", None)
                os.environ.pop("MAVIS_PROJECT_DIR", None)
                self.assertEqual(
                    bind_project_root(["core", "exec", "-C", str(second)]),
                    second.resolve(),
                )
                self.assertEqual(os.environ["MAVIS_PROJECT_ROOT"], str(second.resolve()))

    def test_stale_root_and_conflicting_explicit_anchor_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            first = base / "first"
            second = base / "second"
            for root in (first, second):
                root.mkdir()
                subprocess.run(["git", "init", "-q", str(root)], check=True)
            with patch.dict(os.environ, {"MAVIS_PROJECT_ROOT": str(first)}):
                os.environ.pop("MAVIS_PROJECT_DIR", None)
                with self.assertRaisesRegex(ValueError, "differs from the effective checkout"):
                    bind_project_root(["core", "-C", str(second)])
            with patch.dict(os.environ, {"MAVIS_PROJECT_DIR": str(first)}):
                os.environ.pop("MAVIS_PROJECT_ROOT", None)
                with self.assertRaisesRegex(ValueError, "differs from -C"):
                    bind_project_root(["core", "--cd", str(second)])

    def test_desktop_anchor_and_home_git_root(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            subprocess.run(["git", "init", "-q", str(home)], check=True)
            project = home / "project"
            project.mkdir()
            subprocess.run(["git", "init", "-q", str(project)], check=True)
            with patch.dict(os.environ, {"HOME": str(home),
                                              "MAVIS_PROJECT_DIR": str(project)}), patch(
                "os.getcwd", return_value=str(home)
            ):
                os.environ.pop("MAVIS_PROJECT_ROOT", None)
                self.assertEqual(bind_project_root(["core"]), project.resolve())
                self.assertEqual(os.environ["MAVIS_PROJECT_ROOT"], str(project.resolve()))
            with patch.dict(os.environ, {"HOME": str(home)}), patch(
                "os.getcwd", return_value=str(home)
            ):
                os.environ.pop("MAVIS_PROJECT_DIR", None)
                os.environ.pop("MAVIS_PROJECT_ROOT", None)
                self.assertIsNone(bind_project_root(["core"]))
                self.assertNotIn("MAVIS_PROJECT_ROOT", os.environ)


if __name__ == "__main__":
    unittest.main()
