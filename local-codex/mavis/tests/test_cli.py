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
    def test_pre_compact_imports_matching_rollout_and_writes_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            codex_home = root / "codex"
            codex_home.mkdir()
            rollout = codex_home / "rollout.jsonl"
            rollout.write_text(
                json.dumps({"type": "session_meta", "payload": {"id": "session-1"}})
                + "\n"
            )
            hook = {
                "hook_event_name": "PreCompact",
                "session_id": "session-1",
                "turn_id": "turn-1",
                "trigger": "manual",
                "transcript_path": str(rollout),
            }
            output = io.StringIO()
            with (
                patch.dict(
                    os.environ,
                    {"MAVIS_HOME": str(root / "mavis"), "CODEX_HOME": str(codex_home)},
                ),
                patch("sys.stdin", io.StringIO(json.dumps(hook))),
                contextlib.redirect_stdout(output),
            ):
                self.assertEqual(main(["pre-compact"]), 0)
            self.assertEqual(json.loads(output.getvalue()), {"continue": True})
            handoff_path = next(
                (root / "mavis" / "transcripts" / "session-1" / "handoffs").glob(
                    "*.json"
                )
            )
            handoff = json.loads(handoff_path.read_text())
            self.assertEqual(handoff["conversation_id"], "session-1")
            self.assertTrue(handoff["unresolved_failures"])
            hook["session_id"] = "another-session"
            with (
                patch.dict(
                    os.environ,
                    {"MAVIS_HOME": str(root / "mavis"), "CODEX_HOME": str(codex_home)},
                ),
                patch("sys.stdin", io.StringIO(json.dumps(hook))),
            ):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    main(["pre-compact"])

    def test_pre_compact_includes_bound_objective_without_claiming_unverified_work(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home = root / "mavis"
            codex_home = root / "codex"
            codex_home.mkdir()
            rollout = codex_home / "rollout.jsonl"
            rollout.write_text(
                json.dumps({"type": "session_meta", "payload": {"id": "session-2"}})
                + "\n"
            )
            store = ObjectiveStore(home)
            store.create(
                {
                    "schema_version": "mavis.objective/v1",
                    "objective_id": "task-2",
                    "blueprint": "Repair the parser",
                    "requirements": [{"id": "r1", "text": "keep failures"}],
                    "dependencies": [],
                    "scope": {},
                    "acceptance_checks": [{"id": "c1"}],
                    "unresolved_decisions": ["Choose input format"],
                }
            )
            with (
                patch.dict(os.environ, {"MAVIS_HOME": str(home)}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    main(["objective", "bind-session", "task-2", "session-2"]), 0
                )
            hook = {
                "hook_event_name": "PreCompact",
                "session_id": "session-2",
                "turn_id": "turn-2",
                "trigger": "manual",
                "transcript_path": str(rollout),
            }
            with (
                patch.dict(
                    os.environ, {"MAVIS_HOME": str(home), "CODEX_HOME": str(codex_home)}
                ),
                patch("sys.stdin", io.StringIO(json.dumps(hook))),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(main(["pre-compact"]), 0)
            handoff_path = next(
                (home / "transcripts" / "session-2" / "handoffs").glob("*.json")
            )
            handoff = json.loads(handoff_path.read_text())
            self.assertEqual(handoff["objective_id"], "task-2")
            self.assertEqual(handoff["goals"], ["Repair the parser"])
            self.assertEqual(handoff["completed_requirements"], [])
            self.assertIn("Choose input format", handoff["unresolved_failures"])
            self.assertIn("accepted_decisions", handoff["unknown_fields"])

    def test_run_attaches_host_receipt_to_existing_objective(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.invalid"],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Mavis Test"], cwd=repo, check=True
            )
            (repo / "file.txt").write_text("fixture", encoding="utf-8")
            subprocess.run(["git", "add", "file.txt"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
            home = root / "home"
            ObjectiveStore(home).create(
                {
                    "schema_version": "mavis.objective/v1",
                    "objective_id": "task-1",
                    "blueprint": "Run one check",
                    "requirements": [{"id": "r1", "text": "check fixture"}],
                    "dependencies": [],
                    "scope": {},
                    "acceptance_checks": [{"id": "c1"}],
                    "unresolved_decisions": [],
                }
            )
            with (
                patch.dict(os.environ, {"MAVIS_HOME": str(home)}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                status = main(
                    [
                        "run",
                        "--cwd",
                        str(repo),
                        "--check-id",
                        "c1",
                        "task-1",
                        "--",
                        "python3",
                        "-c",
                        "print('1 passed')",
                    ]
                )
            self.assertEqual(status, 0)
            retained = ObjectiveStore(home).load("task-1")["evidence_receipts"]
            self.assertEqual(len(retained), 1)
            receipt = json.loads(Path(retained[0]["path"]).read_text(encoding="utf-8"))
            self.assertEqual(receipt["verdict"], "pass")


if __name__ == "__main__":
    unittest.main()
