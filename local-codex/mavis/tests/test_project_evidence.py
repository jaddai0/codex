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
from mavis.evidence import run_command
from mavis.objectives import ObjectiveStore
from mavis.project_evidence import migrate_legacy_objective, project_home, project_root
from mavis.storage import read_json, sha256_file
from mavis.transcripts import TranscriptArchive


def git_project(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Mavis Test"], check=True)
    (path / "source.txt").write_text("source\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "source.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)
    return path


def objective(identifier: str) -> dict:
    return {
        "schema_version": "mavis.objective/v1", "objective_id": identifier,
        "blueprint": "check project", "requirements": [{"id": "r1", "text": "pass"}],
        "scope": {"paths": ["source.txt"]},
        "acceptance_checks": [{"id": "c1", "command": ["python3", "-c", "print('1 passed')"]}],
        "state": "queued",
    }


class ProjectEvidenceTests(unittest.TestCase):
    def test_new_project_objective_receipt_output_and_transcript_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = git_project(root / "repo")
            shared = root / "shared"
            shared.mkdir()
            payload = root / "objective.json"
            payload.write_text(json.dumps(objective("job-1")), encoding="utf-8")
            environment = {"MAVIS_HOME": str(shared), "MAVIS_PROJECT_ROOT": str(project)}
            with patch.dict(os.environ, environment), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["objective", "create", str(payload)]), 0)
                self.assertEqual(main(["run", "--cwd", str(project),
                                       "--check-id", "c1", "job-1", "--", "python3", "-c",
                                       "print('1 passed')"]), 0)
            evidence_home = project.resolve() / ".mavis"
            record = ObjectiveStore(evidence_home).load("job-1")
            receipt = Path(record["evidence_receipts"][0]["path"])
            self.assertTrue(receipt.is_relative_to(evidence_home))
            self.assertFalse((shared / "objectives").exists())
            self.assertFalse((shared / "evidence").exists())
            output = io.StringIO()
            with patch.dict(os.environ, environment), contextlib.redirect_stdout(output):
                self.assertEqual(main(["output", "inspect", str(receipt)]), 0)
            self.assertEqual(json.loads(output.getvalue())["host"]["verdict"], "pass")
            ignored = subprocess.run(["git", "check-ignore", "-q", ".mavis/evidence/job-1/probe"], cwd=project)
            self.assertEqual(ignored.returncode, 0)
            status = subprocess.run(["git", "status", "--short", "--untracked-files=all"],
                                    cwd=project, capture_output=True, text=True)
            self.assertEqual(status.stdout, "")

            codex = root / "codex"
            codex.mkdir()
            rollout = codex / "rollout.jsonl"
            rollout.write_text(json.dumps({"type": "session_meta", "payload": {"id": "session-1"}}) + "\n")
            hook = {"hook_event_name": "PreCompact", "session_id": "session-1",
                    "turn_id": "turn-1", "trigger": "manual", "transcript_path": str(rollout),
                    "cwd": str(project)}
            with patch.dict(os.environ, {**environment, "CODEX_HOME": str(codex)}), \
                    patch("sys.stdin", io.StringIO(json.dumps(hook))), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["pre-compact"]), 0)
            self.assertTrue((evidence_home / "transcripts" / "session-1" / "manifest.json").is_file())
            self.assertFalse((shared / "transcripts").exists())
            resumed = io.StringIO()
            with patch.dict(os.environ, {**environment, "CODEX_HOME": str(codex)}), \
                    patch("sys.stdin", io.StringIO(json.dumps({
                        "hook_event_name": "SessionStart", "session_id": "session-1",
                        "source": "compact", "cwd": str(project),
                    }))), contextlib.redirect_stdout(resumed):
                self.assertEqual(main(["compaction-handoff"]), 0)
            self.assertTrue(json.loads(resumed.getvalue())["continue"])
            self.assertIn("Verified Mavis compaction handoff", resumed.getvalue())
            other = git_project(root / "other")
            hook["cwd"] = str(other)
            with patch.dict(os.environ, {**environment, "CODEX_HOME": str(codex)}), \
                    patch("sys.stdin", io.StringIO(json.dumps(hook))):
                with self.assertRaisesRegex(ValueError, "different Mavis project"):
                    main(["pre-compact"])
            self.assertFalse((other / ".mavis").exists())

    def test_legacy_import_copies_bytes_without_rewriting_receipt_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = git_project(root / "repo")
            other = git_project(root / "other")
            shared = root / "shared"
            store = ObjectiveStore(shared)
            store.create(objective("legacy-1"))
            receipt = run_command(shared, "legacy-1", ["python3", "-c", "print('1 passed')"],
                                  project, acceptance_check_ids=["c1"])
            store.add_receipt("legacy-1", receipt)
            store.bind_session("legacy-1", "session-legacy")
            TranscriptArchive(shared, "session-legacy").append_segment([
                {"role": "user", "content": "keep this history"},
            ])
            old_digest = sha256_file(receipt)
            with self.assertRaisesRegex(ValueError, "not bound"):
                migrate_legacy_objective(shared, other, "legacy-1")
            manifest_path = migrate_legacy_objective(shared, project, "legacy-1")
            manifest = read_json(manifest_path)
            copied = manifest_path.parent / receipt.relative_to(shared)
            self.assertEqual(sha256_file(copied), old_digest)
            self.assertEqual(sha256_file(receipt), old_digest)
            self.assertEqual(read_json(copied)["raw_output"]["path"],
                             read_json(receipt)["raw_output"]["path"])
            self.assertEqual(read_json(shared / "objectives" / "legacy-1.json"),
                             read_json(manifest_path.parent / "objectives" / "legacy-1.json"))
            with patch.dict(os.environ, {"MAVIS_HOME": str(shared),
                                              "MAVIS_PROJECT_ROOT": str(project)}), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["objective", "show", "legacy-1"]), 0)
            with patch.dict(os.environ, {"MAVIS_HOME": str(shared),
                                              "MAVIS_PROJECT_ROOT": str(other)}):
                with self.assertRaisesRegex(ValueError, "not bound"):
                    main(["objective", "show", "legacy-1"])
            paths = {item["path"] for item in manifest["files"]}
            self.assertGreaterEqual(paths, {
                "objectives/legacy-1.json",
                f"evidence/legacy-1/{receipt.parent.name}/receipt.json",
                f"evidence/legacy-1/{receipt.parent.name}/stdout.log",
                f"evidence/legacy-1/{receipt.parent.name}/stderr.log",
                "objective_sessions/session-legacy.json",
                "transcripts/session-legacy/manifest.json",
            })
            self.assertTrue(any(path.startswith("transcripts/session-legacy/segments/") for path in paths))
            with self.assertRaises(FileExistsError):
                migrate_legacy_objective(shared, project, "legacy-1")

    def test_rejects_symlinked_project_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = git_project(root / "repo")
            (project / ".mavis").symlink_to(root)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                project_home(project, create=True)

    def test_rejects_home_directory_git_root(self):
        with tempfile.TemporaryDirectory() as directory:
            project = git_project(Path(directory) / "repo")
            with patch("mavis.project_evidence.Path.home", return_value=project.resolve()):
                with self.assertRaisesRegex(ValueError, "home-directory Git root"):
                    project_root(project)


if __name__ == "__main__":
    unittest.main()
