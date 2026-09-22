import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from mavis.cli import main
from mavis.objectives import ObjectiveStore


class ObjectiveCliTests(unittest.TestCase):
    def test_run_attaches_host_receipt_to_existing_objective(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Mavis Test"], cwd=repo, check=True)
            (repo / "file.txt").write_text("fixture", encoding="utf-8")
            subprocess.run(["git", "add", "file.txt"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
            home = root / "home"
            ObjectiveStore(home).create({
                "schema_version": "mavis.objective/v1",
                "objective_id": "task-1",
                "blueprint": "Run one check",
                "requirements": [{"id": "r1", "text": "check fixture"}],
                "dependencies": [], "scope": {},
                "acceptance_checks": [{"id": "c1"}],
                "unresolved_decisions": [],
            })
            with patch.dict(os.environ, {"MAVIS_HOME": str(home)}), contextlib.redirect_stdout(io.StringIO()):
                status = main(["run", "--cwd", str(repo), "--check-id", "c1", "task-1", "--", "python3", "-c", "print('1 passed')"])
            self.assertEqual(status, 0)
            retained = ObjectiveStore(home).load("task-1")["evidence_receipts"]
            self.assertEqual(len(retained), 1)
            receipt = json.loads(Path(retained[0]["path"]).read_text(encoding="utf-8"))
            self.assertEqual(receipt["verdict"], "pass")


if __name__ == "__main__":
    unittest.main()
