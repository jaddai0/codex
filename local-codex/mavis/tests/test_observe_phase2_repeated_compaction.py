"""Evidence checks for the installed two-compaction observer."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from mavis.transcripts import HANDOFF_FIELDS, TranscriptArchive


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "observe_phase2_repeated_compaction.py"
spec = importlib.util.spec_from_file_location("phase2_repeated_compaction", SCRIPT)
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


class RepeatedCompactionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.workspace = Path("/private/tmp/fixture")
        self.prompts = ("remember fact", "continue", "what fact")
        self.answers = ("ACK1", "ACK2", "fact")
        session = "session-1"
        self.rows = [{"type": "session_meta", "payload": {"id": session,
                                                            "cwd": str(self.workspace)}}]
        for index, (prompt, answer) in enumerate(zip(self.prompts, self.answers)):
            turn = f"turn-{index}"
            self.rows.extend([
                {"type": "event_msg", "payload": {"type": "task_started", "turn_id": turn}},
                {"type": "event_msg", "payload": {"type": "item_completed",
                                                  "thread_id": session, "turn_id": turn,
                                                  "item": {"type": "UserMessage", "content": [
                                                      {"type": "text", "text": prompt}]} }},
                {"type": "event_msg", "payload": {"type": "task_complete",
                                                  "turn_id": turn,
                                                  "last_agent_message": answer}},
            ])
            if index < 2:
                self.rows.append({"type": "compacted", "payload": {}})
        self.boundaries = (5, 9, 12)
        first = {"path": "/archive/h1.json", "sha256": "a", "source_turn_id": "compact-1"}
        second = {"path": "/archive/h2.json", "sha256": "b", "source_turn_id": "compact-2"}
        self.checkpoints = (
            {"handoffs": [first], "latest_source_turn_id": "compact-1",
             "segment_paths": ["/archive/s1.jsonl"]},
            {"handoffs": [first, second], "latest_source_turn_id": "compact-2",
             "segment_paths": ["/archive/s1.jsonl", "/archive/s2.jsonl"]},
        )

    def verify(self):
        return observer.verify_repeated_compaction(
            self.rows, workspace=self.workspace, prompts=self.prompts,
            answers=self.answers, boundaries=self.boundaries,
            checkpoints=self.checkpoints)

    def test_accepts_exact_two_compactions_across_three_processes(self):
        proof = self.verify()
        self.assertEqual(proof["compact_event_indexes"], [4, 8])
        self.assertEqual(proof["answers"], list(self.answers))

    def test_rejects_second_compaction_missing(self):
        self.rows.pop(8)
        self.boundaries = (5, 8, 11)
        with self.assertRaisesRegex(ValueError, "two real compactions"):
            self.verify()

    def test_rejects_wrong_final_answer(self):
        self.rows[-1]["payload"]["last_agent_message"] = "almost fact"
        with self.assertRaisesRegex(ValueError, "exact expected answer"):
            self.verify()

    def test_rejects_compaction_in_wrong_process(self):
        self.boundaries = (4, 9, 12)
        with self.assertRaisesRegex(ValueError, "two real compactions"):
            self.verify()

    def test_rejects_reused_handoff(self):
        self.checkpoints[1]["latest_source_turn_id"] = "compact-1"
        with self.assertRaisesRegex(ValueError, "handoffs did not advance"):
            self.verify()

    def test_rejects_unfinished_native_turn(self):
        self.rows.pop(3)
        self.boundaries = (4, 8, 11)
        with self.assertRaisesRegex(ValueError, "did not complete"):
            self.verify()

    def test_rollout_parser_ignores_only_partial_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.jsonl"
            path.write_text(json.dumps(self.rows[0]) + "\n" + '{"partial":', encoding="utf-8")
            self.assertEqual(observer.events(path), [self.rows[0]])
            path.write_text(json.dumps(self.rows[0]) + "\n" + '{"broken":}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "malformed rollout row 2"):
                observer.events(path)

    def test_closed_rollout_requires_complete_last_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.jsonl"
            path.write_text(json.dumps(self.rows[0]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete record"):
                observer.completed_events(path)

    def test_archive_checkpoints_verify_both_handoffs_and_early_fact(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = TranscriptArchive(Path(directory), "session-1")
            archive.append_segment([{"text": "early fact"}])
            handoff = {field: [] for field in HANDOFF_FIELDS}
            handoff["unknown_fields"] = []
            handoff["source_turn_id"] = "compact-1"
            archive.write_handoff(handoff)
            first = observer._archive_checkpoint(archive, 1, "early fact")
            archive.append_segment([{"text": "later turn"}])
            handoff["source_turn_id"] = "compact-2"
            archive.write_handoff(handoff)
            second = observer._archive_checkpoint(archive, 2, "early fact")
            self.assertEqual(len(first["handoffs"]), 1)
            self.assertEqual(len(second["handoffs"]), 2)
            self.assertEqual(len(second["segment_paths"]), 2)

    def test_archive_checkpoint_rejects_missing_early_fact(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = TranscriptArchive(Path(directory), "session-1")
            archive.append_segment([{"text": "different fact"}])
            handoff = {field: [] for field in HANDOFF_FIELDS}
            handoff["unknown_fields"] = []
            handoff["source_turn_id"] = "compact-1"
            archive.write_handoff(handoff)
            with self.assertRaisesRegex(ValueError, "early decision is absent"):
                observer._archive_checkpoint(archive, 1, "early fact")

    def test_private_pty_drives_compact_and_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "project"
            workspace.mkdir()
            sessions = root / "sessions"
            sessions.mkdir()
            rollout = sessions / "rollout-fixture.jsonl"
            script = root / "fake_tui.py"
            script.write_text(
                "import json, pathlib, sys\n"
                "path = pathlib.Path(sys.argv[1])\n"
                "workspace, prompt = sys.argv[2:]\n"
                "sys.stdout.write('Trust\\x1b[5;9Hthis\\x1b[5;14Hfolder?\\n1. Trust and continue\\n')\n"
                "sys.stdout.flush()\n"
                "if sys.stdin.readline().strip(): sys.exit(4)\n"
                "rows = [\n"
                " {'type':'session_meta','payload':{'id':'session-1','cwd':workspace}},\n"
                " {'type':'event_msg','payload':{'type':'task_started','turn_id':'turn-1'}},\n"
                " {'type':'event_msg','payload':{'type':'item_completed','thread_id':'session-1','turn_id':'turn-1','item':{'type':'UserMessage','content':[{'type':'text','text':prompt}]} }},\n"
                " {'type':'event_msg','payload':{'type':'task_complete','turn_id':'turn-1','last_agent_message':'ACK1'}}]\n"
                "with path.open('w') as output:\n"
                " for row in rows: output.write(json.dumps(row) + '\\n')\n"
                "if sys.stdin.readline().strip() != '/compact': sys.exit(2)\n"
                "with path.open('a') as output: output.write(json.dumps({'type':'compacted','payload':{}}) + '\\n')\n"
                "if sys.stdin.readline().strip() != '/exit': sys.exit(3)\n",
                encoding="utf-8",
            )
            lease_fd = os.open(os.devnull, os.O_RDONLY)
            try:
                with patch.object(observer, "handoff_lease_fd", return_value=lease_fd):
                    pid, observed, rows, code = observer._run_stage(
                        [sys.executable, str(script), str(rollout), str(workspace), "remember fact"],
                        workspace=workspace, env=os.environ.copy(),
                        log=root / "terminal.log", sessions=sessions,
                        prompt="remember fact", expected_answer="ACK1",
                        expected_compactions=1, rollout=None,
                        launch_started=time.time(), stage_timeout=8)
            finally:
                os.close(lease_fd)
            self.assertGreater(pid, 0)
            self.assertEqual(observed, rollout)
            self.assertEqual(code, 0)
            self.assertEqual(rows[-1]["type"], "compacted")


if __name__ == "__main__":
    unittest.main()
