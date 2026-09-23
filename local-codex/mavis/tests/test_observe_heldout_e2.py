"""The held-out observer drives its own terminal after a real completed turn."""

from pathlib import Path
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "observe_heldout_e2.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("observe_heldout_e2", SCRIPT)
assert spec and spec.loader
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)

FAKE_TUI = r"""
import json,sys,termios,time
from pathlib import Path
rollout=Path(sys.argv[1]);repo=sys.argv[2];prompt=sys.argv[3]
rollout.parent.mkdir(parents=True,exist_ok=True)
sys.stdout.write('Trust\x1b[5;9Hthis\x1b[5;14Hfolder?\n1. Trust and continue\n')
sys.stdout.flush()
time.sleep(0.4)
termios.tcflush(sys.stdin.fileno(),termios.TCIFLUSH)
if sys.stdin.readline().strip():raise SystemExit('trust was not confirmed')
rows=[
 {"type":"session_meta","payload":{"id":"session-test","cwd":repo}},
 {"type":"event_msg","payload":{"type":"task_started","turn_id":"turn-test"}},
 {"type":"event_msg","payload":{"type":"item_completed","thread_id":"session-test","turn_id":"turn-test","item":{"type":"UserMessage","content":[{"type":"text","text":prompt}]}}},
 {"type":"event_msg","payload":{"type":"task_complete","turn_id":"turn-test"}},
]
with rollout.open('w') as f:
 for row in rows:f.write(json.dumps(row)+'\n')
 f.flush()
command=sys.stdin.readline().strip()
if command=='/compact':
 with rollout.open('a') as f:f.write(json.dumps({"type":"compacted"})+'\n')
 command=sys.stdin.readline().strip()
if command!='/exit':raise SystemExit('unexpected terminal command '+repr(command))
if sys.stdin.readline().strip():raise SystemExit('retry Enter had content')
"""


class HeldoutPrivateTerminalTests(unittest.TestCase):
    def test_completed_turn_then_compact_and_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            sessions = root / "sessions"
            rollout = sessions / "rollout-test.jsonl"
            fake = root / "fake_tui.py"
            fake.write_text(FAKE_TUI)
            prompt = "Catalog task nonce-123"
            lease_fd = os.open(os.devnull, os.O_RDONLY)
            try:
                with patch.object(observer, "handoff_lease_fd", return_value=lease_fd):
                    pid, code = observer._run_tui(
                        [sys.executable, str(fake), str(rollout), str(repo), prompt],
                        repo=repo,
                        env=os.environ.copy(),
                        prompt=prompt,
                        session_root=sessions,
                        log_path=root / "terminal.log",
                        compact=True,
                    )
            finally:
                os.close(lease_fd)
            self.assertGreater(pid, 0)
            self.assertEqual(code, 0)
            self.assertTrue((root / "terminal.log").is_file())
            self.assertEqual(
                sum(
                    row.get("type") == "compacted" for row in observer._records(rollout)
                ),
                1,
            )

    def test_completed_turn_exits_without_compaction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            sessions = root / "sessions"
            rollout = sessions / "rollout-test.jsonl"
            fake = root / "fake_tui.py"
            fake.write_text(FAKE_TUI)
            prompt = "Resume task nonce-456"
            lease_fd = os.open(os.devnull, os.O_RDONLY)
            try:
                with patch.object(observer, "handoff_lease_fd", return_value=lease_fd):
                    _pid, code = observer._run_tui(
                        [sys.executable, str(fake), str(rollout), str(repo), prompt],
                        repo=repo,
                        env=os.environ.copy(),
                        prompt=prompt,
                        session_root=sessions,
                        log_path=root / "terminal.log",
                        rollout=rollout,
                    )
            finally:
                os.close(lease_fd)
            self.assertEqual(code, 0)
            self.assertFalse(
                any(
                    row.get("type") == "compacted" for row in observer._records(rollout)
                )
            )


if __name__ == "__main__":
    unittest.main()
