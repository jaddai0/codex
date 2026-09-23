"""Tests for the events() parser in observe_compaction_restart.py."""

from pathlib import Path
import importlib.util
import json
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "observe_compaction_restart.py"


def load_events():
    spec = importlib.util.spec_from_file_location("_compaction_canary", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.events


class CompactionObserverTests(unittest.TestCase):
    def setUp(self):
        self.events = load_events()

    def test_complete_records_parsed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.jsonl"
            records = [
                {"type": "session_meta", "payload": {"id": "abc", "cwd": "/tmp/work"}},
                {"type": "event_msg", "payload": {"text": "hello"}},
            ]
            path.write_text(
                "\n".join(json.dumps(r) for r in records) + "\n",
                encoding="utf-8",
            )
            result = self.events(path)
            self.assertEqual(result, records)

    def test_incomplete_trailing_record_tolerated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.jsonl"
            first = {"type": "session_meta", "payload": {"id": "abc"}}
            path.write_text(
                json.dumps(first) + "\n" + '{"partial": tru',
                encoding="utf-8",
            )
            self.assertEqual(self.events(path), [first])

    def test_complete_trailing_record_without_newline_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.jsonl"
            first = {"type": "session_meta", "payload": {"id": "abc"}}
            last = {"type": "event_msg", "payload": {"last_agent_message": "ACK"}}
            path.write_text(json.dumps(first) + "\n" + json.dumps(last))
            self.assertEqual(self.events(path), [first, last])

    def test_incomplete_utf8_trailing_record_tolerated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.jsonl"
            first = {"type": "session_meta", "payload": {"id": "abc"}}
            path.write_bytes((json.dumps(first) + "\n").encode() + b'{"text":"\xe2\x82')
            self.assertEqual(self.events(path), [first])

    def test_malformed_completed_record_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.jsonl"
            path.write_text(
                "not json at all\n" + '{"answer": 1}' + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "malformed JSONL.*line 1"):
                self.events(path)


if __name__ == "__main__":
    unittest.main()
