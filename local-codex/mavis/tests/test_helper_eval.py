import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mavis.evidence import run_command
from mavis.helper_eval import SUITE_VERSION, _completion, evaluate, main
from mavis.helpers import HelperSession
from mavis.transcripts import TranscriptArchive


BASE_URL = "http://127.0.0.1:8001/v1"
MODEL = "Qwen3.5-0.8B"


class HelperEvaluationTests(unittest.TestCase):
    def test_librarian_accepts_exact_citation_and_expires_context(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            archive = TranscriptArchive(home, "history-one")
            archive.append_segment([{"role": "user", "content": "Decision: retain the blue engine after the first review."}])
            packet = archive.search("first review")
            citation = {key: packet[0][key] for key in ("path", "line", "sha256")}
            calls = []

            def ask(base_url, model_id, messages, timeout):
                calls.append(messages)
                return {"answer": "Keep the blue engine.", "uncertainty": "The reason is not recorded.",
                        "citations": [citation]}

            suite = {"schema_version": SUITE_VERSION, "role": "librarian", "cases": [
                {"id": "decision", "conversation_id": "history-one", "question": "What engine was retained?",
                 "search_terms": ["first review"], "expected_answer_terms": ["blue engine"],
                 "expected_source_terms": ["Decision: retain the blue engine"]}
            ]}
            result = evaluate(home, suite, MODEL, BASE_URL, ask=ask)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(len(calls), 1)
            self.assertNotIn("expected_answer_terms", calls[0][1]["content"])
            session = HelperSession(home, "librarian")
            expiry = result["cases"][0]["result"]["context_expires_at_epoch"]
            self.assertIsNotNone(session.followup_context(now=expiry - 0.01))
            self.assertIsNone(session.followup_context(now=expiry))
            self.assertFalse((home / "helpers" / "output-reader").exists())

    def test_librarian_rejects_citation_that_does_not_support_case(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            archive = TranscriptArchive(home, "history-one")
            archive.append_segment([{"content": "Decision: blue engine"},
                                    {"content": "Decision: red engine"}])
            packet = archive.search("Decision:")
            wrong = {key: packet[1][key] for key in ("path", "line", "sha256")}
            suite = {"schema_version": SUITE_VERSION, "role": "librarian", "cases": [
                {"id": "decision", "conversation_id": "history-one", "question": "Which engine?",
                 "search_terms": ["Decision:"], "expected_answer_terms": ["blue engine"],
                 "expected_source_terms": ["blue engine"]}
            ]}
            result = evaluate(home, suite, MODEL, BASE_URL, ask=lambda *args: {
                "answer": "blue engine", "uncertainty": "No reason", "citations": [wrong]})
            self.assertEqual(result["status"], "fail")
            self.assertIn("cited source", result["cases"][0]["error"])

    def test_followup_never_reuses_another_conversations_history(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            first = TranscriptArchive(home, "first")
            first.append_segment([{"content": "old decision"}])
            old = first.search("old decision")[0]
            HelperSession(home, "librarian").store_query_context("old question", [
                {key: old[key] for key in ("path", "line", "sha256")}
            ])
            second = TranscriptArchive(home, "second")
            second.append_segment([{"content": "new decision"}])
            current = second.search("new decision")[0]
            citation = {key: current[key] for key in ("path", "line", "sha256")}
            suite = {"schema_version": SUITE_VERSION, "role": "librarian", "cases": [
                {"id": "followup", "conversation_id": "second", "question": "What decision?",
                 "search_terms": ["new decision"], "followup": True,
                 "expected_answer_terms": ["new decision"],
                 "expected_source_terms": ["new decision"]}
            ]}

            def ask(*args):
                user = json.loads(args[2][1]["content"])
                self.assertIsNone(user["followup"])
                return {"answer": "new decision", "uncertainty": "No reason recorded",
                        "citations": [citation]}

            self.assertEqual(evaluate(home, suite, MODEL, BASE_URL, ask=ask)["status"], "pass")

    def test_output_known_format_uses_host_parser_before_model(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            receipt = run_command(home, "reader", ["python3", "-c", "print('2 passed')"], home)
            suite = {"schema_version": SUITE_VERSION, "role": "output-reader", "cases": [
                {"id": "pytest", "receipt_path": str(receipt), "expected_verdict": "pass"}
            ]}
            result = evaluate(home, suite, MODEL, BASE_URL,
                              ask=lambda *args: self.fail("model called for known output"))
            self.assertEqual(result["status"], "inconclusive")
            self.assertFalse(result["cases"][0]["result"]["model_called"])

    def test_output_irregular_model_observation_must_be_exact_line(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            receipt = run_command(home, "reader", ["python3", "-c", "print('compile complete')"], home)
            suite = {"schema_version": SUITE_VERSION, "role": "output-reader", "cases": [
                {"id": "irregular", "receipt_path": str(receipt), "expected_verdict": "uncertain",
                 "expected_observation_terms": ["compile complete"]}
            ]}
            valid = {"verdict": "uncertain", "summary": "", "observations": [
                {"line": 1, "text": "compile complete"}]}
            accepted = evaluate(home, suite, MODEL, BASE_URL, ask=lambda *args: valid)
            self.assertEqual(accepted["status"], "pass")
            forged = {**valid, "observations": [{"line": 1, "text": "all tests passed"}]}
            rejected = evaluate(home, suite, MODEL, BASE_URL, ask=lambda *args: forged)
            self.assertEqual(rejected["status"], "fail")
            promoted = {**valid, "verdict": "pass", "summary": "all tests passed"}
            rejected = evaluate(home, suite, MODEL, BASE_URL, ask=lambda *args: promoted)
            self.assertEqual(rejected["status"], "fail")

    def test_large_buried_failure_uses_full_host_log_without_model_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            code = "print('ordinary output\\n' * 1500 + 'FAILED: hidden test\\n' + 'ordinary output\\n' * 1500)"
            receipt = run_command(home, "reader", ["python3", "-c", code], home)
            suite = {"schema_version": SUITE_VERSION, "role": "output-reader", "cases": [
                {"id": "buried", "receipt_path": str(receipt), "expected_verdict": "fail"}
            ]}
            result = evaluate(home, suite, MODEL, BASE_URL,
                              ask=lambda *args: self.fail("model called for known buried failure"))
            self.assertEqual(result["status"], "inconclusive")
            failure_lines = result["cases"][0]["result"]["host"]["failure_lines"]
            self.assertEqual([item["text"] for item in failure_lines], ["FAILED: hidden test"])

    def test_output_failure_and_timeout_cannot_become_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            receipt = run_command(home, "reader", ["python3", "-c",
                                                   "print('hidden fault'); raise SystemExit(7)"], home)
            suite = {"schema_version": SUITE_VERSION, "role": "output-reader", "cases": [
                {"id": "failed", "receipt_path": str(receipt), "expected_verdict": "fail",
                 "force_model": True, "expected_observation_terms": ["hidden fault"]}
            ]}
            result = evaluate(home, suite, MODEL, BASE_URL, ask=lambda *args: {
                "verdict": "pass", "summary": "success", "observations": [{"line": 1, "text": "hidden fault"}]})
            self.assertEqual(result["status"], "fail")
            self.assertEqual(result["cases"][0]["proposed"]["summary"], "success")
            result = evaluate(home, suite, MODEL, BASE_URL, ask=lambda *args: {
                "verdict": "fail", "summary": "success", "observations": [{"line": 1, "text": "hidden fault"}]})
            self.assertEqual(result["status"], "fail")

    def test_never_calls_iris_port(self):
        suite = {"schema_version": SUITE_VERSION, "role": "librarian", "cases": [{}]}
        with self.assertRaisesRegex(ValueError, "8001"):
            evaluate(Path("/tmp"), suite, MODEL, "http://127.0.0.1:8000/v1")

    def test_completion_requires_exact_model_identity_and_complete_json(self):
        class Response:
            status = 200

            def __init__(self, body):
                self.body = io.BytesIO(json.dumps(body).encode())

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self, size):
                return self.body.read(size)

        answer = {"answer": "blue"}
        envelope = {"model": MODEL, "choices": [{"finish_reason": "stop",
                     "message": {"content": json.dumps(answer)}}]}
        with patch("mavis.helper_eval.request.urlopen", return_value=Response(envelope)) as urlopen:
            self.assertEqual(_completion(BASE_URL, MODEL, [], 1), answer)
            self.assertIn("8001", urlopen.call_args.args[0].full_url)
        with patch("mavis.helper_eval.request.urlopen", return_value=Response({**envelope, "model": "other"})):
            with self.assertRaisesRegex(ValueError, "identity"):
                _completion(BASE_URL, MODEL, [], 1)
        with patch("mavis.helper_eval.request.urlopen", return_value=Response({
            **envelope, "choices": [{"finish_reason": "length", "message": {"content": json.dumps(answer)}}]
        })):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                _completion(BASE_URL, MODEL, [], 1)

    def test_service_model_path_and_bytes_are_bound_to_result(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            model = home / "model"
            model.mkdir()
            (model / "config.json").write_text('{"model_type":"qwen"}')
            weights = model / "model.safetensors"
            weights.write_bytes(b"frozen weights")
            receipt = run_command(home, "reader", ["python3", "-c", "print('2 passed')"], home)
            suite = {"schema_version": SUITE_VERSION, "role": "output-reader", "cases": [
                {"id": "pytest", "receipt_path": str(receipt), "expected_verdict": "pass"}
            ]}
            inventory = lambda _: [{"id": MODEL, "model_path": str(model)}]
            result = evaluate(home, suite, MODEL, BASE_URL, model_path=model,
                              inventory_reader=inventory)
            self.assertEqual(result["status"], "inconclusive")
            self.assertIn("model.safetensors", result["model_binding"]["files"])
            with self.assertRaisesRegex(ValueError, "differs"):
                evaluate(home, suite, MODEL, BASE_URL, model_path=model,
                         inventory_reader=lambda _: [{"id": MODEL, "model_path": str(home)}])

    def test_preflight_failure_keeps_private_failure_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            suite = home / "suite.json"
            suite.write_text(json.dumps({"schema_version": SUITE_VERSION,
                                         "role": "librarian", "cases": [{"id": "case"}]}))
            with patch("mavis.helper_eval._model_binding", side_effect=ValueError("wrong model")):
                exit_status = main(["--home", str(home), "--suite", str(suite),
                                    "--model-id", MODEL, "--model-path", str(home)])
            self.assertEqual(exit_status, 1)
            receipts = list((home / "helpers" / "librarian" / "evaluations").glob("*.json"))
            self.assertEqual(len(receipts), 1)
            self.assertEqual(json.loads(receipts[0].read_text())["cases"][0]["error"], "wrong model")


if __name__ == "__main__":
    unittest.main()
