from pathlib import Path
import tempfile
import unittest

from mavis.evidence import parse_test_output, run_command
from mavis.storage import read_json


class EvidenceTests(unittest.TestCase):
    def test_parser_never_turns_failure_or_unknown_into_success(self):
        self.assertEqual(parse_test_output("10 passed, 1 failed", 0), "fail")
        self.assertEqual(parse_test_output("all done", 0), "uncertain")
        self.assertEqual(parse_test_output("OK", 1), "fail")
        self.assertEqual(parse_test_output("OK", 0, timed_out=True), "incomplete")

    def test_command_captures_complete_stdout_and_stderr(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_path = run_command(
                root,
                "obj-1",
                ["python3", "-c", "import sys; print('OK'); print('warning', file=sys.stderr)"],
                root,
            )
            receipt = read_json(receipt_path)
            output_root = Path(receipt["raw_output"]["path"])
            self.assertEqual(receipt["verdict"], "pass")
            self.assertEqual((output_root / "stdout.log").read_text().strip(), "OK")
            self.assertEqual((output_root / "stderr.log").read_text().strip(), "warning")
            self.assertGreater(receipt["raw_output"]["bytes"], 0)


if __name__ == "__main__":
    unittest.main()
