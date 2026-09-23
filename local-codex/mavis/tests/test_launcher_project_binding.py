"""Exercise the installed shell launcher's project selection without a server."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


LAUNCHER = Path(__file__).resolve().parents[2] / "bin" / "local-codex"


class LauncherProjectBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.home = self.root / "home"
        self.broad = self.home / "Dev-Projects"
        self.broad.mkdir(parents=True)
        self._git(self.home)
        self.a = self.broad / "a"
        self.b = self.broad / "b"
        for project in (self.a, self.b):
            project.mkdir()
            self._git(project)
        shim = self.root / "shim"
        shim.mkdir()
        fake_python = shim / "python3"
        fake_python.write_text(
            '#!/bin/sh\nprintf "PROJECT=%s\\n" "${MAVIS_PROJECT_ROOT-UNSET}"\n',
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        fake_core = self.root / "core"
        fake_core.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_core.chmod(0o755)
        self.environment = os.environ.copy()
        for name in ("MAVIS_PROJECT_ROOT", "MAVIS_PROJECT_DIR", "MAVIS_GATEWAY_ENV_FILE"):
            self.environment.pop(name, None)
        self.environment.update({
            "HOME": str(self.home), "PATH": str(shim) + os.pathsep + os.environ["PATH"],
            "LOCAL_CODEX_BIN": str(fake_core), "LOCAL_CODEX_SHARE_DIR": str(self.root / "share"),
        })

    @staticmethod
    def _git(path: Path) -> None:
        subprocess.run(["git", "init", "-q", str(path)], check=True)

    def launch(self, cwd: Path, *args: str, **environment: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["zsh", str(LAUNCHER), *args], cwd=cwd,
            env={**self.environment, **environment}, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )

    def test_broad_home_git_root_has_no_project_binding(self):
        result = self.launch(self.broad)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PROJECT=UNSET", result.stdout)

    def test_nested_checkout_uses_its_real_root(self):
        result = self.launch(self.a, "objective", "show", "example")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"PROJECT={self.a}", result.stdout)

    def test_explicit_desktop_directory_overrides_broad_cwd(self):
        result = self.launch(self.broad, MAVIS_PROJECT_DIR=str(self.a))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"PROJECT={self.a}", result.stdout)

    def test_cd_option_overrides_cwd_and_desktop_directory(self):
        for option in (("-C", str(self.b)), ("--cd", str(self.b)),
                       (f"--cd={self.b}",), (f"-C{self.b}",)):
            with self.subTest(option=option):
                result = self.launch(self.a, *option, MAVIS_PROJECT_DIR=str(self.a))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"PROJECT={self.b}", result.stdout)

    def test_stale_inherited_root_is_rejected(self):
        for cwd, options in ((self.b, ()), (self.broad, ())):
            with self.subTest(cwd=cwd):
                result = self.launch(cwd, *options, MAVIS_PROJECT_ROOT=str(self.a))
                self.assertEqual(result.returncode, 2)
                self.assertIn("inherited project root differs", result.stderr)

    def test_matching_inherited_root_is_accepted(self):
        result = self.launch(self.broad, MAVIS_PROJECT_ROOT=str(self.a),
                             MAVIS_PROJECT_DIR=str(self.a))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"PROJECT={self.a}", result.stdout)

    def test_missing_cd_value_is_rejected(self):
        for option in ("-C", "--cd="):
            with self.subTest(option=option):
                result = self.launch(self.a, option)
                self.assertEqual(result.returncode, 2)
                self.assertIn("needs an existing directory", result.stderr)


if __name__ == "__main__":
    unittest.main()
