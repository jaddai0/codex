from pathlib import Path
import json
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from mavis.evaluations import (E0_CASES, E0Evaluator, _safe_buried_inspection,
                               installed_candidate_fingerprint, native_review_completed)
from mavis.e0_tasks import prepare_small_repository, small_repository_review_prompt
from mavis.e2_tasks import terra_review_command
from mavis.runtime import RuntimeConfig
from mavis.storage import sha256_file


class E0EvaluationTests(unittest.TestCase):
    def test_buried_inspection_allows_stdin_sort_unique_without_file_output(self):
        self.assertTrue(_safe_buried_inspection("grep -o MAVIS_E0_FAILURE_ output.raw | sort -u | head"))
        self.assertFalse(_safe_buried_inspection("grep -o MAVIS_E0_FAILURE_ output.raw | sort -o stolen.log"))
        self.assertFalse(_safe_buried_inspection("grep -o MAVIS_E0_FAILURE_ output.raw | sort private.txt"))

    def test_installed_candidate_tracks_launcher_and_profile_code(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            files = (
                ".local/share/local-codex/local-codex-core",
                ".local/share/local-codex/prepare_runtime.py",
                ".local/share/local-codex/launch_core.py",
                ".local/share/local-codex/base-instructions.md",
                ".local/share/local-codex/persona.toml",
                ".local/share/local-codex/mavis/__init__.py",
                ".local/bin/mavis",
                "Desktop/Mavis.command",
            )
            for name in files:
                path = home / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
            with patch("mavis.evaluations.Path.home", return_value=home), patch.dict(os.environ, {}, clear=True):
                baseline = installed_candidate_fingerprint()
                for name in files:
                    path = home / name
                    path.write_text(name + " changed")
                    changed = installed_candidate_fingerprint()
                    self.assertNotEqual(baseline, changed, name)
                    path.write_text(name)
                for name in ("MAVIS_BIN", "LOCAL_CODEX_SHARE_DIR", "LOCAL_CODEX_BIN", "LOCAL_CODEX_MODEL"):
                    with patch.dict(os.environ, {name: str(home / "alternate")}):
                        with self.assertRaisesRegex(ValueError, name):
                            installed_candidate_fingerprint()
                with patch.dict(os.environ, {"MAVIS_HOME": str(home / "alternate")}):
                    with self.assertRaisesRegex(ValueError, "MAVIS_HOME"):
                        installed_candidate_fingerprint()
                with patch.dict(os.environ, {"PYTHONPATH": str(home / "alternate")}):
                    with self.assertRaisesRegex(ValueError, "Python package path"):
                        installed_candidate_fingerprint()
                with patch.dict(os.environ, {"PYTHONPATH": str(home / ".local/share/local-codex")}):
                    self.assertEqual(installed_candidate_fingerprint(), baseline)

    def test_native_review_accepts_completed_structured_cli_receipt_only(self):
        verdict = "ACCEPT\nExact tests passed and protected file hash matched."
        result = {
            "sessionId": "sess_review",
            "turnId": "turn_review",
            "response": verdict,
            "eventCount": 12,
            "projection": {"status": "idle"},
        }
        log = "ZCode Built-in skipped (not-due)\n" + json.dumps(result, indent=2)
        self.assertTrue(native_review_completed(log, verdict))
        bold_verdict = "**ACCEPT**\nExact tests passed and protected file hash matched."
        result["response"] = bold_verdict
        self.assertTrue(native_review_completed("ZCode Built-in skipped (not-due)\n" + json.dumps(result), bold_verdict))
        result["response"] = "ACCEPTABLE\nThis is not an acceptance heading."
        self.assertFalse(native_review_completed(json.dumps(result), result["response"]))
        result["response"] = verdict
        self.assertFalse(native_review_completed(log, "ACCEPT\nDifferent finding"))
        result["projection"]["status"] = "running"
        self.assertFalse(native_review_completed(json.dumps(result), verdict))
        self.assertFalse(native_review_completed("ACCEPT\nNo session receipt", verdict))

    def test_small_repository_fixture_has_failing_baseline_and_protected_dirty_note(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            manifest = json.loads(prepare_small_repository(Path(directory)).read_text())
            repo = Path(manifest["repo"])
            self.assertNotEqual(manifest["baseline_exit_status"], 0)
            self.assertEqual(
                sha256_file(Path(manifest["protected_dirty_file"])),
                manifest["protected_dirty_sha256"],
            )
            self.assertIn(
                "user-notes.txt",
                subprocess.check_output(
                    ["git", "-C", str(repo), "status", "--porcelain"], text=True
                ),
            )
            result = subprocess.run(
                manifest["test_command"], cwd=repo, capture_output=True, text=True
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("test_discount_reduces_price", result.stderr)

    def test_small_repository_requires_bound_native_terra_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            evaluator = E0Evaluator(home, RuntimeConfig(home=home))
            manifest_path = prepare_small_repository(home)
            task = manifest_path.parent
            repo = (task / "repo").resolve()
            (repo / "package" / "pricing.py").write_text(
                "def apply_discount(subtotal: int, discount: int) -> int:\n"
                "    return subtotal - discount\n")
            mavis_log = task / "installed-mavis-repair.jsonl"
            mavis_log.write_text(json.dumps({"type": "turn.completed"}) + "\n" + json.dumps({
                "type": "item.completed", "item": {"type": "file_change", "changes": [
                    {"path": str(repo / "package" / "pricing.py")}]}}) + "\n")
            verdict = "ACCEPT: both tests pass and user notes are intact."
            events = [
                {"type": "thread.started", "thread_id": "native-terra"},
                {"type": "turn.started"},
                {"type": "item.completed", "item": {"type": "agent_message", "text": verdict}},
                {"type": "turn.completed"},
            ]
            terra_log = task / "terra-review.jsonl"
            terra_log.write_text("".join(json.dumps(event) + "\n" for event in events))
            stderr = task / "terra-review.stderr.log"
            stderr.write_text("")
            review_text = task / "terra-review.txt"
            review_text.write_text(verdict)
            observed = {
                "candidate": {"core_sha256": "a" * 64},
                "mavis_log_sha256": sha256_file(mavis_log),
                "terra_log_sha256": sha256_file(terra_log),
                "terra_stderr_sha256": sha256_file(stderr),
                "terra_text_sha256": sha256_file(review_text),
                "review_argv": terra_review_command(
                    repo, review_text, small_repository_review_prompt(manifest_path)),
                "review_exit": 0, "review_thread_id": "native-terra",
            }
            (task / "installed-run.json").write_text(json.dumps(observed))
            with patch("mavis.evaluations.installed_candidate_fingerprint",
                       return_value=observed["candidate"]):
                self.assertEqual(evaluator._small_repository("small-repository")["status"], "pass")
                observed["review_argv"][3] = "another-model"
                (task / "installed-run.json").write_text(json.dumps(observed))
                with self.assertRaisesRegex(ValueError, "command changed"):
                    evaluator._small_repository("small-repository")

    def test_full_e0_never_passes_when_real_worker_cases_are_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RuntimeConfig(home=Path(directory) / "runtime")
            evaluator = E0Evaluator(Path(directory), config)
            with (
                patch("mavis.evaluations.endpoint_alive", return_value=False),
                patch("mavis.evaluations._listener_pids", return_value=set()),
            ):
                result = evaluator.run()
            self.assertEqual(result["status"], "reject")
            self.assertEqual(result["mandatory_cases"], list(E0_CASES))
            statuses = {item["case"]: item["status"] for item in result["results"]}
            self.assertEqual(statuses["small-repository"], "blocked")
            self.assertEqual(statuses["external-harness"], "blocked")

    def test_passing_e0_summary_binds_installed_candidate_model_and_case_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            evaluator = E0Evaluator(home, RuntimeConfig(home=home, model="model-a"))
            def passing_case(case):
                return evaluator._receipt(case, "pass", ["retained host evidence"])
            with (
                patch.object(evaluator, "run_case", side_effect=passing_case),
                patch("mavis.evaluations.installed_candidate_fingerprint",
                      return_value={"core_sha256": "a" * 64}),
            ):
                summary = evaluator.run()
            self.assertEqual(summary["model_id"], "model-a")
            self.assertEqual(summary["installed_candidate"], {"core_sha256": "a" * 64})
            self.assertEqual(set(summary["case_receipts"]), set(E0_CASES))
            for case in E0_CASES:
                self.assertEqual(summary["case_receipts"][case],
                                 sha256_file(evaluator.root / f"{case}.json"))

    def test_buried_failure_and_compaction_require_live_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = E0Evaluator(
                Path(directory), RuntimeConfig(home=Path(directory))
            )
            self.assertEqual(evaluator.run_case("buried-failure")["status"], "blocked")
            self.assertEqual(
                evaluator.run_case("compaction-restart")["status"], "blocked"
            )

    def test_buried_failure_rejects_incomplete_raw_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            evaluator = E0Evaluator(home, RuntimeConfig(home=home))
            task = home / "evaluations" / "e0" / "buried-live-test"
            repo = task / "repo"
            repo.mkdir(parents=True)
            (repo / "produce_log.py").write_text("print('fixture')\n")
            raw = home / "tool-output" / "burst.raw"
            raw.parent.mkdir()
            marker = "MAVIS_E0_FAILURE_" + "a" * 24
            complete = ("A" * 800000 + "\n" + marker + "\n" + "Z" * 800000 + "\n").encode()
            raw.write_bytes(complete[:-100])
            output = f"Process exited with code 1\nComplete raw output: {raw} ({len(complete)} bytes; closed)\n"
            rollout = task / "rollout.jsonl"
            events = [
                {"type": "session_meta", "payload": {"cwd": str(repo)}},
                {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command", "call_id": "run", "arguments": '{"cmd":"python3 produce_log.py"}'}},
                {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "run", "output": output}},
                {"type": "event_msg", "payload": {"type": "task_complete", "last_agent_message": f"The command exited with code 1: {marker}"}},
            ]
            rollout.write_text("".join(json.dumps(event) + "\n" for event in events))
            (task / "result.json").write_text(json.dumps({
                "candidate": {"core_sha256": "test"},
                "candidate_after": {"core_sha256": "test"}, "rollout": str(rollout),
                "repo": str(repo), "mavis_exit": 0, "iris_loaded": True,
                "mavis_loaded": False,
            }))
            with patch("mavis.evaluations.installed_candidate_fingerprint", return_value={"core_sha256": "test"}):
                self.assertEqual(evaluator.run_case("buried-failure")["status"], "blocked")
                raw.write_bytes(complete)
                self.assertEqual(evaluator.run_case("buried-failure")["status"], "pass")
                events[2]["payload"]["call_id"] = "other"
                rollout.write_text("".join(json.dumps(event) + "\n" for event in events))
                self.assertEqual(evaluator.run_case("buried-failure")["status"], "blocked")
                events[2]["payload"]["call_id"] = "run"
                events[1]["payload"]["arguments"] = json.dumps({"cmd": "python3 produce_log.py", "workdir": str(home)})
                rollout.write_text("".join(json.dumps(event) + "\n" for event in events))
                self.assertEqual(evaluator.run_case("buried-failure")["status"], "blocked")
                events[1]["payload"]["arguments"] = '{"cmd":"python3 produce_log.py"}'
                events[-1]["payload"]["last_agent_message"] = f"The command failed with exit code 1: {marker}"
                rollout.write_text("".join(json.dumps(event) + "\n" for event in events))
                self.assertEqual(evaluator.run_case("buried-failure")["status"], "pass")
                events[-1]["payload"]["last_agent_message"] = f"The command **failed** (non-zero exit code **1**): {marker}"
                rollout.write_text("".join(json.dumps(event) + "\n" for event in events))
                self.assertEqual(evaluator.run_case("buried-failure")["status"], "pass")
                events[-1]["payload"]["last_agent_message"] = f"The command succeeded: {marker}"
                rollout.write_text("".join(json.dumps(event) + "\n" for event in events))
                self.assertEqual(evaluator.run_case("buried-failure")["status"], "blocked")
                events[-1]["payload"]["last_agent_message"] = f"The command succeeded. An earlier unrelated test failed; exit code 1 is in the transcript. {marker}"
                rollout.write_text("".join(json.dumps(event) + "\n" for event in events))
                self.assertEqual(evaluator.run_case("buried-failure")["status"], "blocked")
                events[-1]["payload"]["last_agent_message"] = f"The command failed with exit code 0. An earlier process had exit code 1. {marker}"
                rollout.write_text("".join(json.dumps(event) + "\n" for event in events))
                self.assertEqual(evaluator.run_case("buried-failure")["status"], "blocked")
                events[-1]["payload"]["last_agent_message"] = f"The command failed with exit code 11, though code 1 was also mentioned. {marker}"
                rollout.write_text("".join(json.dumps(event) + "\n" for event in events))
                self.assertEqual(evaluator.run_case("buried-failure")["status"], "blocked")
                events[-1]["payload"]["last_agent_message"] = f"The command failed with exit code 1. The command also exited with code 11. {marker}"
                rollout.write_text("".join(json.dumps(event) + "\n" for event in events))
                self.assertEqual(evaluator.run_case("buried-failure")["status"], "blocked")
                events[-1]["payload"]["last_agent_message"] = f"The command failed with exit code 1: {marker}"
                events.insert(1, {"type": "response_item", "payload": {
                    "type": "function_call", "name": "exec_command", "call_id": "preflight",
                    "arguments": json.dumps({"cmd": "ls produce_log.py; rg marker output.raw | cut -c1-120"}),
                }})
                rollout.write_text("".join(json.dumps(event) + "\n" for event in events))
                self.assertEqual(evaluator.run_case("buried-failure")["status"], "pass")
                events.insert(2, {"type": "response_item", "payload": {
                    "type": "function_call", "name": "exec_command", "call_id": "second",
                    "arguments": '{"cmd":"python3 produce_log.py"}',
                }})
                rollout.write_text("".join(json.dumps(event) + "\n" for event in events))
                self.assertEqual(evaluator.run_case("buried-failure")["status"], "blocked")
                events[2] = {"type": "response_item", "payload": {
                    "type": "function_call", "name": "exec_command", "call_id": "second",
                    "arguments": json.dumps({"cmd": "python3 ./produce_log.py"}),
                }}
                rollout.write_text("".join(json.dumps(event) + "\n" for event in events))
                self.assertEqual(evaluator.run_case("buried-failure")["status"], "blocked")
                events[2]["payload"]["arguments"] = json.dumps({"cmd": "python3 -m produce_log"})
                rollout.write_text("".join(json.dumps(event) + "\n" for event in events))
                self.assertEqual(evaluator.run_case("buried-failure")["status"], "blocked")

    def test_fabricated_success_case_checks_an_existing_objective(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = E0Evaluator(
                Path(directory), RuntimeConfig(home=Path(directory))
            )
            result = evaluator.run_case("fabricated-success-rejection")
            self.assertEqual(result["status"], "pass")
            self.assertTrue(
                (
                    Path(directory)
                    / "e0-fixtures"
                    / "fabricated"
                    / "objectives"
                    / "e0-fabricated.json"
                ).is_file()
            )

    def test_tool_roundtrip_fails_closed_without_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = E0Evaluator(
                Path(directory), RuntimeConfig(home=Path(directory))
            )
            with patch("mavis.evaluations.endpoint_alive", return_value=False):
                result = evaluator.run_case("tool-roundtrip")
            self.assertEqual(result["status"], "blocked")

    def test_tool_roundtrip_does_not_implicitly_load_model(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = E0Evaluator(
                Path(directory), RuntimeConfig(home=Path(directory))
            )
            with (
                patch("mavis.evaluations.endpoint_alive", return_value=True),
                patch(
                    "mavis.evaluations.inventory",
                    return_value=[{"id": evaluator.config.model, "loaded": False}],
                ),
                patch("mavis.evaluations._post_json") as request,
            ):
                result = evaluator.run_case("tool-roundtrip")
            self.assertEqual(result["status"], "blocked")
            request.assert_not_called()

    def test_one_case_does_not_replace_full_suite_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = E0Evaluator(
                Path(directory), RuntimeConfig(home=Path(directory))
            )
            with (
                patch("mavis.evaluations.endpoint_alive", return_value=False),
                patch("mavis.evaluations._listener_pids", return_value=set()),
            ):
                evaluator.run()
            summary = evaluator.root / "summary.json"
            before = summary.read_bytes()
            evaluator.run("fabricated-success-rejection")
            self.assertEqual(summary.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
