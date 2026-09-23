import contextlib
import io
import json
import os
import time
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from mavis.cli import main, build_parser
from mavis.maintenance import MaintenanceQueue
from mavis.objectives import ObjectiveStore
from mavis.transcripts import TranscriptArchive


class ArchiveSearchCliTests(unittest.TestCase):
    def test_archive_search_cli_returns_requested_page(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            archive = TranscriptArchive(home, "conv-1")
            archive.append_segment([
                {"role": "user", "content": "decision one"},
                {"role": "user", "content": "decision two"},
            ])
            archive.append_segment([
                {"role": "user", "content": "decision three"},
                {"role": "user", "content": "decision four"},
            ])
            output = io.StringIO()
            with patch.dict(os.environ, {"MAVIS_HOME": str(home)}), contextlib.redirect_stdout(output):
                self.assertEqual(main(["archive-search", "conv-1", "decision",
                                       "--limit", "2", "--offset", "2"]), 0)
            hits = json.loads(output.getvalue())
            self.assertEqual(len(hits), 2)
            self.assertIn("decision three", hits[0]["text"])
            self.assertIn("decision four", hits[1]["text"])
            self.assertTrue(all(hit["sha256"] for hit in hits))

    def test_archive_search_cli_accepts_offset_argument(self):
        parser = build_parser()
        args = parser.parse_args(
            ["archive-search", "conv-1", "decision", "--offset", "10"]
        )
        self.assertEqual(args.offset, 10)
        self.assertEqual(args.limit, 20)

    def test_archive_search_cli_accepts_limit_and_offset_together(self):
        parser = build_parser()
        args = parser.parse_args(
            ["archive-search", "conv-1", "decision", "--limit", "5", "--offset", "20"]
        )
        self.assertEqual(args.limit, 5)
        self.assertEqual(args.offset, 20)

    def test_archive_search_cli_defaults_offset_to_zero(self):
        parser = build_parser()
        args = parser.parse_args(["archive-search", "conv-1", "decision"])
        self.assertEqual(args.offset, 0)


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


class MaintenanceCliTests(unittest.TestCase):
    def test_tick_cli_calls_host_runner_and_returns_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            output = io.StringIO()
            with (patch.dict(os.environ, {"MAVIS_HOME": str(home)}),
                  patch("mavis.cli.maintenance_tick", return_value={
                      "schema_version": "mavis.maintenance-tick/v1",
                      "status": "deferred", "receipt_path": str(home / "receipt.json"),
                  }) as runner,
                  contextlib.redirect_stdout(output)):
                self.assertEqual(main(["maintenance", "tick"]), 0)
            runner.assert_called_once_with(home)
            self.assertEqual(json.loads(output.getvalue())["status"], "deferred")

    def test_cancel_cli_returns_cancelled_job(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            queue = MaintenanceQueue(home)
            job = queue.enqueue("daily", "review", {"key": "value"})
            time.sleep(0.01)
            output = io.StringIO()
            with (
                patch.dict(os.environ, {"MAVIS_HOME": str(home)}),
                contextlib.redirect_stdout(output),
            ):
                self.assertEqual(
                    main(
                        [
                            "maintenance",
                            "cancel",
                            job["job_id"],
                            "--reason",
                            "obsolete",
                        ]
                    ),
                    0,
                )
            cancelled = json.loads(output.getvalue())
            self.assertEqual(cancelled["state"], "cancelled")
            self.assertEqual(cancelled["detail"], "obsolete")
            self.assertEqual(cancelled["payload"], {"key": "value"})
            self.assertNotEqual(cancelled["updated_at"], job["created_at"])
            self.assertEqual(queue.get(job["job_id"])["state"], "cancelled")

    def test_cancel_cli_parser_accepts_job_id_and_reason(self):
        parser = build_parser()
        args = parser.parse_args(
            ["maintenance", "cancel", "job-1", "--reason", "obsolete"]
        )
        self.assertEqual(args.job_id, "job-1")
        self.assertEqual(args.reason, "obsolete")

    def test_cancel_cli_requires_reason(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["maintenance", "cancel", "job-1"])

    def test_cancel_cli_rejects_running_job(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            queue = MaintenanceQueue(home)
            job = queue.enqueue("weekly", "review", {})
            self.assertIsNotNone(queue.claim_next(foreground_active=False))
            with (
                patch.dict(os.environ, {"MAVIS_HOME": str(home)}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(ValueError, "cannot cancel"):
                    main(
                        [
                            "maintenance",
                            "cancel",
                            job["job_id"],
                            "--reason",
                            "too late",
                        ]
                    )

    def test_cancel_cli_rejects_unknown_job(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            MaintenanceQueue(home)
            with (
                patch.dict(os.environ, {"MAVIS_HOME": str(home)}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with self.assertRaises(KeyError):
                    main(
                        [
                            "maintenance",
                            "cancel",
                            "missing",
                            "--reason",
                            "nobody",
                        ]
                    )


class LauncherRoutingTests(unittest.TestCase):
    def test_service_commands_reach_python_without_loading_a_model(self):
        root = Path(__file__).resolve().parents[3]
        launcher = root / "local-codex" / "bin" / "local-codex"
        share = root / "local-codex" / "mavis"
        with tempfile.TemporaryDirectory() as directory:
            env = {**os.environ, "LOCAL_CODEX_SHARE_DIR": str(share),
                   "MAVIS_HOME": str(Path(directory) / "service"),
                   "CODEX_HOME": str(Path(directory) / "codex")}
            for command in ("objective", "archive-retention", "maintenance", "e1"):
                with self.subTest(command=command):
                    run = subprocess.run([str(launcher), command, "--help"],
                                         env=env, text=True, capture_output=True, timeout=15)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    self.assertIn("usage: mavis-service", run.stdout)
                    if command == "maintenance":
                        self.assertIn("tick", run.stdout)


if __name__ == "__main__":
    unittest.main()
