"""Supervise an installed two-package task across compact and process restart.

First terminal: wait for Mavis to finish the catalog slice, enter /compact,
wait for completion, then /exit. Second terminal: wait for Mavis to finish
checkout and tests, then /exit. The observer retains host and reviewer output.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from urllib.parse import quote

from mavis.e2_tasks import (CATALOG_TEST, FULL_TEST, _completed_prompt_turn,
                            _user_turn, codex_terra_review_completed, first_task_prompt,
                            fixture_state, matching_first_rollout, prepare_heldout,
                            resume_task_prompt, run_host_check, task_launch_commands,
                            terra_review_command, terra_review_prompt, verify_heldout)
from mavis.e1_bootstrap import _summary as current_e0_summary
from mavis.evaluations import installed_candidate_fingerprint
from mavis.runtime import (RuntimeConfig, endpoint_alive, ensure_runtime, inventory,
                           load_model, request_json, stop_server)
from mavis.storage import sha256_file, write_json


def _records(path: Path) -> list[dict]:
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        raise ValueError("installed rollout ended with an incomplete event")
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def _run_tui(command: list[str], *, repo: Path, env: dict[str, str]) -> tuple[int, int]:
    process = subprocess.Popen(command, cwd=repo, env=env)
    try:
        return process.pid, process.wait(timeout=1800)
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise


def main() -> int:
    service = Path.home() / ".local-codex" / "mavis-service"
    config = RuntimeConfig(home=service)
    model_path = f"/v1/models/{quote(config.model, safe='')}"
    manifest_path = prepare_heldout(service)
    task = manifest_path.parent
    repo = task / "repo"
    fixture_state(manifest_path, stage="baseline")

    def loaded(endpoint: str) -> bool:
        return any(row.get("id") == config.model and row.get("loaded")
                   for row in inventory(endpoint))

    if not loaded(config.iris_endpoint) or loaded(config.endpoint):
        raise RuntimeError("IRIS must own the main model and Mavis must be unloaded")
    candidate = installed_candidate_fingerprint()
    e0_path, e0 = current_e0_summary(service)
    if e0["model_id"] != config.model:
        raise RuntimeError("installed E0 summary has a different model")
    result: dict[str, object] = {
        "schema_version": "mavis.e2-installed-observation/v1",
        "manifest_sha256": sha256_file(manifest_path),
        "candidate": candidate, "manifest": str(manifest_path),
        "model_id": config.model, "e0_summary_sha256": sha256_file(e0_path),
        "observer_path": str(Path(__file__).resolve()),
        "observer_sha256": sha256_file(Path(__file__).resolve()),
    }

    def interrupt(number: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"signal {number}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        request_json(config.iris_endpoint, model_path + "/unload", method="POST", timeout=180)
        if loaded(config.iris_endpoint):
            raise RuntimeError("IRIS main model did not unload")
        load_model(config)
        env = {**os.environ, "MAVIS_PROJECT_DIR": str(repo), "PYTHONDONTWRITEBYTECODE": "1"}
        nonce = uuid.uuid4().hex
        result["prompt_nonce"] = nonce
        first_argv, _unused_resume = task_launch_commands(repo, "pending", nonce)
        result["first_argv"] = first_argv
        print(f"Mavis catalog slice in {repo}: after its answer, enter /compact, then /exit", flush=True)
        started = time.time()
        result["first_pid"], result["first_exit"] = _run_tui(first_argv, repo=repo, env=env)
        if result["first_exit"] != 0:
            raise RuntimeError("first installed Mavis process failed")
        result["stage1_state"] = fixture_state(manifest_path, stage="catalog")
        result["catalog_check"] = str(run_host_check(task, "catalog", CATALOG_TEST))
        if json.loads(Path(result["catalog_check"]).read_text())["exit_status"] != 0:
            raise RuntimeError("catalog host check failed")
        session_root = Path.home() / ".local-codex" / "sessions"
        rollout, session_id, first_turn = matching_first_rollout(
            session_root, repo, first_task_prompt(nonce), started)
        first_data = rollout.read_bytes()
        first_rows = _records(rollout)
        _first_user, first_complete = _completed_prompt_turn(
            first_rows, first_task_prompt(nonce), session_id, first_turn)
        result.update({
            "rollout": str(rollout),
            "session_id": session_id,
            "first_turn_id": first_turn,
            "first_rollout_bytes": len(first_data),
            "first_rollout_events": len(first_rows),
            "first_rollout_sha256": hashlib.sha256(first_data).hexdigest(),
        })
        if not any(index > first_complete and row.get("type") == "compacted"
                   for index, row in enumerate(first_rows)):
            raise RuntimeError("first process did not complete work and compact")
        _first_command, resume_argv = task_launch_commands(repo, session_id, nonce)
        result["resume_argv"] = resume_argv
        print("Mavis resumed checkout slice: after its answer, enter /exit", flush=True)
        result["resume_pid"], result["resume_exit"] = _run_tui(resume_argv, repo=repo, env=env)
        if result["resume_exit"] != 0:
            raise RuntimeError("resumed installed Mavis process failed")
        resumed_rows = _records(rollout)[len(first_rows):]
        _resume_user, resume_turn = _user_turn(resumed_rows, resume_task_prompt(nonce), session_id)
        _completed_prompt_turn(resumed_rows, resume_task_prompt(nonce), session_id, resume_turn)
        result["resume_turn_id"] = resume_turn
        result["final_state"] = fixture_state(manifest_path, stage="complete")
        result["complete_check"] = str(run_host_check(task, "complete", FULL_TEST))
        if json.loads(Path(result["complete_check"]).read_text())["exit_status"] != 0:
            raise RuntimeError("complete host check failed")
    except BaseException as exc:
        result["error"] = repr(exc)
    finally:
        try:
            if loaded(config.endpoint):
                request_json(config.endpoint, model_path + "/unload", method="POST", timeout=180)
            if loaded(config.endpoint):
                raise RuntimeError("Mavis main model remained loaded")
        except BaseException as exc:
            result["mavis_unload_error"] = repr(exc)
            try:
                if endpoint_alive(config.endpoint):
                    stop_server(config)
                    for _ in range(100):
                        if not endpoint_alive(config.endpoint):
                            break
                        time.sleep(0.2)
                    ensure_runtime(config, load=False)
            except BaseException as recovery_exc:
                result["mavis_stop_error"] = repr(recovery_exc)
        try:
            if endpoint_alive(config.endpoint) and loaded(config.endpoint):
                raise RuntimeError("Mavis still holds the model")
            if not loaded(config.iris_endpoint):
                request_json(config.iris_endpoint, model_path + "/load", method="POST", timeout=900)
            result["iris_loaded"] = loaded(config.iris_endpoint)
            result["mavis_loaded"] = loaded(config.endpoint)
            result["candidate_after"] = installed_candidate_fingerprint()
        except BaseException as exc:
            result["iris_restore_error"] = repr(exc)
        write_json(task / "result.json", result)
    if result.get("error") or result.get("iris_loaded") is not True or result.get("mavis_loaded") is not False:
        print(task / "result.json", flush=True)
        return 1
    review_text = task / "terra-review.txt"
    review_argv = terra_review_command(
        repo, review_text, terra_review_prompt(manifest_path, Path(result["rollout"])))
    before_review = fixture_state(manifest_path, stage="complete")
    with (task / "terra-review.jsonl").open("wb") as log, (task / "terra-review.stderr.log").open("wb") as err:
        review = subprocess.run(review_argv, cwd=repo, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=err,
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=900)
    if fixture_state(manifest_path, stage="complete") != before_review:
        raise RuntimeError("Terra review changed the fixture")
    thread_id = None
    try:
        thread_id = codex_terra_review_completed(
            (task / "terra-review.jsonl").read_text(), review_text.read_text())
    except (ValueError, FileNotFoundError) as exc:
        result["review_parse_error"] = repr(exc)
    result.update({"review_exit": review.returncode,
                   "review_argv": review_argv, "review_thread_id": thread_id,
                   "review_log_sha256": sha256_file(task / "terra-review.jsonl"),
                   "review_stderr_sha256": sha256_file(task / "terra-review.stderr.log"),
                   "review_text_sha256": sha256_file(review_text) if review_text.is_file() else None})
    write_json(task / "result.json", result)
    try:
        verified = verify_heldout(manifest_path)
        write_json(task / "verification.json", verified)
        print(task / "verification.json", flush=True)
        return 0
    except Exception as exc:
        result["verification_error"] = repr(exc)
        write_json(task / "result.json", result)
        print(task / "result.json", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
