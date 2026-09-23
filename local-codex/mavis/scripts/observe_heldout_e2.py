"""Supervise an installed two-package task across compact and process restart.

The observer drives both TUI processes through a private pseudo-terminal and
retains their terminal, host-check, and independent reviewer output.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import pty
import select
import signal
import struct
import subprocess
import sys
import termios
import time
import uuid

from mavis.e2_tasks import (CATALOG_TEST, FULL_TEST, _completed_prompt_turn,
                            _user_turn, codex_terra_review_completed, first_task_prompt,
                            fixture_state, matching_first_rollout, prepare_heldout,
                            resume_task_prompt, run_host_check, task_launch_commands,
                            terra_review_command, terra_review_prompt, verify_heldout)
from mavis.e1_bootstrap import _summary as current_e0_summary
from mavis.evaluations import installed_candidate_fingerprint
from mavis.runtime import (RuntimeConfig, endpoint_alive, handoff_lease_fd,
                           inventory, loaded_generation_models,
                           require_installed_selected_model,
                           with_mavis_handoff_lease)
from mavis.storage import sha256_file, write_json
from shared_gpu_observation import renew_gpu_lease, shared_mavis_model


def _records(path: Path) -> list[dict]:
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        raise ValueError("installed rollout ended with an incomplete event")
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def _live_records(path: Path) -> list[dict]:
    data = path.read_bytes()
    parts = data.split(b"\n")
    complete = parts[:-1] if not data.endswith(b"\n") else parts
    return [json.loads(part) for part in complete if part.strip()]


def _live_rollout(session_root: Path, repo: Path, prompt: str, started: float) -> Path | None:
    matches: list[Path] = []
    for path in session_root.glob("**/rollout-*.jsonl"):
        if path.stat().st_mtime < started - 2 or prompt.encode() not in path.read_bytes():
            continue
        rows = _live_records(path)
        if (rows and rows[0].get("type") == "session_meta"
                and rows[0].get("payload", {}).get("cwd") == str(repo)):
            matches.append(path)
    if len(matches) > 1:
        raise RuntimeError("multiple installed rollouts matched the held-out task")
    return matches[0] if matches else None


def _run_tui(command: list[str], *, repo: Path, env: dict[str, str],
             prompt: str, session_root: Path, log_path: Path,
             rollout: Path | None = None, compact: bool = False) -> tuple[int, int]:
    """Send /compact and /exit only after the exact native turn completes."""
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 140, 0, 0))
    lease_fd = handoff_lease_fd()
    process = None
    try:
        process = subprocess.Popen(
            command, cwd=repo,
            env={**env, "TERM": "xterm-256color",
                   "MAVIS_GENERATION_LEASE_FD": str(lease_fd)},
            stdin=slave, stdout=slave, stderr=slave, pass_fds=(lease_fd,),
            start_new_session=True,
            preexec_fn=lambda: fcntl.ioctl(slave, termios.TIOCSCTTY, 0),
        )
        os.close(slave)
        slave = -1
        started = time.time()
        stage = "answer"
        trust_sent = False
        terminal_tail = b""
        deadline = time.monotonic() + 1800
        completed_at: int | None = None
        with log_path.open("wb") as log:
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"installed TUI timed out waiting for {stage}")
                readable, _, _ = select.select([master], [], [], 0.25)
                if readable:
                    try:
                        chunk = os.read(master, 65536)
                        log.write(chunk)
                        log.flush()
                        terminal_tail = (terminal_tail + chunk)[-16384:]
                        if (not trust_sent and b"folder?" in terminal_tail
                                and b"Trust and continue" in terminal_tail):
                            os.write(master, b"\r")
                            trust_sent = True
                    except OSError:
                        pass
                if rollout is None:
                    rollout = _live_rollout(session_root, repo, prompt, started)
                rows = _live_records(rollout) if rollout and rollout.exists() else []
                if stage == "answer" and rows:
                    session = rows[0].get("payload", {}).get("id")
                    if isinstance(session, str):
                        try:
                            _index, turn = _user_turn(rows, prompt, session)
                            _user, completed_at = _completed_prompt_turn(
                                rows, prompt, session, turn)
                        except ValueError:
                            pass
                        else:
                            time.sleep(0.5)
                            os.write(master, b"/compact\r" if compact else b"/exit\r")
                            stage = "compact" if compact else "exit"
                            deadline = time.monotonic() + (600 if compact else 60)
                elif (stage == "compact" and completed_at is not None
                      and any(row.get("type") == "compacted"
                              for row in rows[completed_at + 1:])):
                    time.sleep(0.5)
                    os.write(master, b"/exit\r")
                    stage = "exit"
                    deadline = time.monotonic() + 60
                code = process.poll()
                if code is not None:
                    if stage != "exit" or code != 0:
                        raise RuntimeError(f"installed TUI exited during {stage} with status {code}")
                    return process.pid, code
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        os.close(master)
        if slave != -1:
            os.close(slave)


@with_mavis_handoff_lease("observe_heldout_e2")
def main() -> int:
    service = Path.home() / ".local-codex" / "mavis-service"
    config = RuntimeConfig(home=service, allow_concurrent_local=True)
    manifest_path = prepare_heldout(service)
    task = manifest_path.parent
    repo = task / "repo"
    fixture_state(manifest_path, stage="baseline")

    def loaded(endpoint: str) -> bool:
        return any(row.get("id") == config.model and row.get("loaded")
                   for row in inventory(endpoint))

    require_installed_selected_model(config)
    if (not loaded(config.iris_endpoint)
            or (endpoint_alive(config.endpoint) and loaded_generation_models(config.endpoint))):
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
    shared: dict[str, object] | None = None
    try:
        with shared_mavis_model(config, "installed Mavis held-out E2") as shared:
            env = {**os.environ, "MAVIS_PROJECT_DIR": str(repo), "PYTHONDONTWRITEBYTECODE": "1"}
            nonce = uuid.uuid4().hex
            result["prompt_nonce"] = nonce
            first_argv, _unused_resume = task_launch_commands(repo, "pending", nonce)
            result["first_argv"] = first_argv
            print(f"Mavis catalog slice in private terminal: {repo}", flush=True)
            renew_gpu_lease()
            started = time.time()
            result["first_pid"], result["first_exit"] = _run_tui(
                first_argv, repo=repo, env=env, prompt=first_task_prompt(nonce),
                session_root=Path.home() / ".local-codex" / "sessions",
                log_path=task / "terminal-first.log", compact=True)
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
            print("Mavis checkout slice resumed in private terminal", flush=True)
            renew_gpu_lease()
            result["resume_pid"], result["resume_exit"] = _run_tui(
                resume_argv, repo=repo, env=env, prompt=resume_task_prompt(nonce),
                session_root=session_root, log_path=task / "terminal-resume.log",
                rollout=rollout)
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
        if shared is not None:
            result.update(shared)
            result["iris_loaded"] = shared.get("iris_stayed_loaded")
        result["candidate_after"] = installed_candidate_fingerprint()
        write_json(task / "result.json", result)
    if (result.get("error") or result.get("iris_loaded") is not True
            or result.get("mavis_loaded") is not False
            or result.get("gpu_lease_released") is not True):
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
