"""E2 fixture integrity and host evidence gate tests; no model is used."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from mavis.e2_tasks import (CATALOG_TEST, FULL_TEST, codex_terra_review_completed,
                            fixture_state, prepare_heldout, run_host_check,
                            terra_review_command, terra_review_prompt, verify_heldout)
from mavis.runtime import DEFAULT_MODEL
from mavis.storage import sha256_file, write_json


class E2HeldoutTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        manifest = prepare_heldout(root)
        task = manifest.parent
        repo = task / "repo"
        return manifest, task, repo

    def _catalog_repair(self, repo: Path) -> None:
        path = repo / "catalog" / "price.py"
        source = path.read_text().replace(
            "    return unit_cents * quantity + discount_cents\n",
            "    subtotal = unit_cents * quantity\n"
            "    if discount_cents > subtotal:\n"
            "        raise ValueError('discount exceeds subtotal')\n"
            "    return subtotal - discount_cents\n",
        )
        path.write_text(source)

    def _checkout_repair(self, repo: Path) -> None:
        path = repo / "checkout" / "invoice.py"
        path.write_text(path.read_text().replace(
            "((subtotal + shipping_cents) * tax_percent // 100)",
            "(subtotal * tax_percent // 100)",
        ))

    def _observed(self, root: Path) -> tuple[Path, Path, Path, dict]:
        manifest, task, repo = self._fixture(root)
        self._catalog_repair(repo)
        stage1 = fixture_state(manifest, stage="catalog")
        first_check = run_host_check(task, "catalog", CATALOG_TEST)
        self._checkout_repair(repo)
        final = fixture_state(manifest, stage="complete")
        second_check = run_host_check(task, "complete", FULL_TEST)
        events = [
            {"type": "session_meta", "payload": {"cwd": str(repo), "id": "session-e2"}},
            {"type": "event_msg", "payload": {"type": "task_complete", "last_agent_message": "Catalog fixed"}},
            {"type": "compacted", "payload": {}},
        ]
        prefix = b"".join(json.dumps(event).encode() + b"\n" for event in events)
        events.append({"type": "event_msg", "payload": {"type": "task_complete", "last_agent_message": "Checkout fixed"}})
        rollout = task / "rollout.jsonl"
        rollout.write_bytes(b"".join(json.dumps(event).encode() + b"\n" for event in events))
        review_log = task / "terra-review.jsonl"
        review_events = [
            {"type": "thread.started", "thread_id": "terra-thread-e2"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "ACCEPT: verified both packages and files"}},
            {"type": "turn.completed"},
        ]
        review_log.write_text("".join(json.dumps(row) + "\n" for row in review_events))
        review_stderr = task / "terra-review.stderr.log"
        review_stderr.write_text("")
        review_text = task / "terra-review.txt"
        review_text.write_text("ACCEPT: verified both packages and files")
        e0_summary = task / "e0-summary.json"
        write_json(e0_summary, {"model_id": DEFAULT_MODEL})
        observer = Path(__file__).resolve().parent.parent / "scripts" / "observe_heldout_e2.py"
        result = {
            "schema_version": "mavis.e2-installed-observation/v1",
            "manifest_sha256": sha256_file(manifest), "candidate": {"core_sha256": "candidate"},
            "candidate_after": {"core_sha256": "candidate"},
            "model_id": DEFAULT_MODEL, "e0_summary_sha256": sha256_file(e0_summary),
            "observer_path": str(observer), "observer_sha256": sha256_file(observer),
            "first_pid": 1001, "resume_pid": 1002, "first_exit": 0, "resume_exit": 0,
            "iris_loaded": True, "mavis_loaded": False,
            "stage1_state": stage1, "final_state": final,
            "catalog_check": str(first_check), "complete_check": str(second_check),
            "rollout": str(rollout), "session_id": "session-e2",
            "first_rollout_bytes": len(prefix), "first_rollout_events": 3,
            "first_rollout_sha256": hashlib.sha256(prefix).hexdigest(),
            "review_exit": 0,
            "review_argv": terra_review_command(repo, review_text, terra_review_prompt(manifest, rollout)),
            "review_thread_id": "terra-thread-e2",
            "review_log_sha256": sha256_file(review_log),
            "review_stderr_sha256": sha256_file(review_stderr),
            "review_text_sha256": sha256_file(review_text),
        }
        write_json(task / "result.json", result)
        return manifest, task, repo, result

    def test_fixture_has_failing_baseline_and_preserved_dirty_files(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, task, repo = self._fixture(Path(directory))
            self.assertEqual(fixture_state(manifest, stage="baseline")["repo"], str(repo))
            self.assertIn("FAILED", (task / "baseline-test.log").read_text())
            self.assertIn("orchid-47", (repo / "user-notes.txt").read_text())
            self.assertIn("violet-29", (repo / "private-draft.txt").read_text())

    def test_checkpoint_requires_one_package_then_both_and_host_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, task, repo = self._fixture(Path(directory))
            self._catalog_repair(repo)
            fixture_state(manifest, stage="catalog")
            self.assertEqual(json.loads(run_host_check(task, "catalog", CATALOG_TEST).read_text())["exit_status"], 0)
            with self.assertRaises(ValueError):
                fixture_state(manifest, stage="complete")
            self._checkout_repair(repo)
            fixture_state(manifest, stage="complete")
            self.assertEqual(json.loads(run_host_check(task, "complete", FULL_TEST).read_text())["exit_status"], 0)

    def test_protected_blueprint_dirty_and_untracked_files_are_checked(self):
        for path, text in (("BLUEPRINT.md", "changed"), ("user-notes.txt", "changed"),
                           ("private-draft.txt", "changed")):
            with self.subTest(path=path), tempfile.TemporaryDirectory() as directory:
                manifest, _task, repo = self._fixture(Path(directory))
                (repo / path).write_text(text)
                with self.assertRaises(ValueError):
                    fixture_state(manifest, stage="baseline")

    def test_staged_changes_to_tests_are_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, _task, repo = self._fixture(Path(directory))
            test_file = repo / "tests" / "test_catalog.py"
            test_file.write_text(test_file.read_text() + "\n# unauthorized change\n")
            subprocess.run(["git", "-C", str(repo), "add", "tests/test_catalog.py"], check=True)
            with self.assertRaises(ValueError):
                fixture_state(manifest, stage="baseline")

    def test_verifier_replays_raw_checks_and_compaction_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, task, repo, result = self._observed(Path(directory))
            with patch("mavis.e2_tasks.installed_candidate_fingerprint", return_value={"core_sha256": "candidate"}), patch(
                    "mavis.e2_tasks.current_e0_summary", return_value=(task / "e0-summary.json", {"model_id": DEFAULT_MODEL})):
                self.assertEqual(verify_heldout(manifest)["status"], "pass")
                (task / "host-checks" / "complete" / "stderr.log").write_text("Ran 6 tests\nFAILED")
                with self.assertRaises(ValueError):
                    verify_heldout(manifest)

    def test_verifier_rejects_missing_restart_or_compaction(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, task, _repo, result = self._observed(Path(directory))
            with patch("mavis.e2_tasks.installed_candidate_fingerprint", return_value={"core_sha256": "candidate"}), patch(
                    "mavis.e2_tasks.current_e0_summary", return_value=(task / "e0-summary.json", {"model_id": DEFAULT_MODEL})):
                result["resume_pid"] = result["first_pid"]
                write_json(task / "result.json", result)
                with self.assertRaises(ValueError):
                    verify_heldout(manifest)
                result["resume_pid"] = 1002
                result["first_rollout_sha256"] = "0" * 64
                write_json(task / "result.json", result)
                with self.assertRaises(ValueError):
                    verify_heldout(manifest)

    def test_verifier_rejects_final_source_drift_and_reviewer_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, task, repo, result = self._observed(Path(directory))
            with patch("mavis.e2_tasks.installed_candidate_fingerprint", return_value={"core_sha256": "candidate"}), patch(
                    "mavis.e2_tasks.current_e0_summary", return_value=(task / "e0-summary.json", {"model_id": DEFAULT_MODEL})):
                (repo / "catalog" / "price.py").write_text("# changed after review\n")
                with self.assertRaises(ValueError):
                    verify_heldout(manifest)
        with tempfile.TemporaryDirectory() as directory:
            manifest, task, _repo, result = self._observed(Path(directory))
            (task / "terra-review.txt").write_text("REJECT: failed review")
            result["review_text_sha256"] = sha256_file(task / "terra-review.txt")
            write_json(task / "result.json", result)
            with patch("mavis.e2_tasks.installed_candidate_fingerprint", return_value={"core_sha256": "candidate"}), patch(
                    "mavis.e2_tasks.current_e0_summary", return_value=(task / "e0-summary.json", {"model_id": DEFAULT_MODEL})):
                with self.assertRaises(ValueError):
                    verify_heldout(manifest)

    def test_native_terra_review_requires_exact_final_message_and_completion(self):
        events = [
            {"type": "thread.started", "thread_id": "terra-thread-e2"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "ACCEPT: checked"}},
            {"type": "turn.completed"},
        ]
        log = "".join(json.dumps(row) + "\n" for row in events)
        self.assertEqual(codex_terra_review_completed(log, "ACCEPT: checked"), "terra-thread-e2")
        with self.assertRaises(ValueError):
            codex_terra_review_completed(log, "ACCEPT: different")
        with self.assertRaises(ValueError):
            codex_terra_review_completed(log.removesuffix(json.dumps(events[-1]) + "\n"), "ACCEPT: checked")
        with self.assertRaises(ValueError):
            codex_terra_review_completed(log, "REJECT: checked")


if __name__ == "__main__":
    unittest.main()
