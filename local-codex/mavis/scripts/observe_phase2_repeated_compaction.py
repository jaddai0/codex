"""Observe installed Mavis recovering an early decision after two real compactions.

This owns a private pseudo-terminal; it never takes over the user's terminal.
The result stays local under Mavis's evaluation home. IRIS keeps its generation
model loaded while Mavis observes the installed session under the GPU lease.
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
import tempfile
import termios
import time
import uuid

from mavis.evaluations import installed_candidate_fingerprint
from mavis.runtime import (RuntimeConfig, handoff_lease_fd,
                           require_installed_selected_model,
                           with_mavis_handoff_lease)
from mavis.storage import sha256_file, write_json
from mavis.transcripts import TranscriptArchive
from shared_gpu_observation import renew_gpu_lease, shared_mavis_model


def events(path: Path) -> list[dict]:
    """Read completed rollout rows, rejecting damaged newline-terminated rows."""
    data = path.read_bytes()
    rows = []
    pieces = data.split(b"\n")
    for number, piece in enumerate(pieces, 1):
        if not piece.strip():
            continue
        try:
            row = json.loads(piece)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if number == len(pieces) and not data.endswith(b"\n"):
                break
            raise ValueError(f"malformed rollout row {number}: {path}") from error
        if not isinstance(row, dict):
            raise ValueError(f"non-object rollout row {number}: {path}")
        rows.append(row)
    return rows


def completed_events(path: Path) -> list[dict]:
    """Require a closed process to leave a complete, newline-terminated rollout."""
    if not path.read_bytes().endswith(b"\n"):
        raise ValueError("installed rollout ended with an incomplete record")
    return events(path)


def _prompt_turn(rows: list[dict], prompt: str, session: str) -> tuple[int, int, str, str]:
    """Require exactly one native user turn and its completed answer."""
    matches = []
    for index, row in enumerate(rows):
        payload = row.get("payload")
        if row.get("type") != "event_msg" or not isinstance(payload, dict):
            continue
        item = payload.get("item")
        if (payload.get("type") != "item_completed" or payload.get("thread_id") != session
                or not isinstance(item, dict) or item.get("type") != "UserMessage"):
            continue
        content = item.get("content")
        if (isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict)
                and content[0].get("type") == "text" and content[0].get("text") == prompt):
            matches.append((index, payload.get("turn_id")))
    if len(matches) != 1 or not isinstance(matches[0][1], str) or not matches[0][1]:
        raise ValueError("prompt did not identify exactly one native user turn")
    user_index, turn = matches[0]
    started = [i for i, row in enumerate(rows) if row.get("type") == "event_msg"
               and row.get("payload", {}).get("type") == "task_started"
               and row.get("payload", {}).get("turn_id") == turn]
    completed = [(i, row.get("payload", {}).get("last_agent_message"))
                 for i, row in enumerate(rows) if row.get("type") == "event_msg"
                 and row.get("payload", {}).get("type") == "task_complete"
                 and row.get("payload", {}).get("turn_id") == turn]
    if (len(started) != 1 or len(completed) != 1
            or not started[0] < user_index < completed[0][0]
            or not isinstance(completed[0][1], str)):
        raise ValueError("native prompt turn did not complete")
    return user_index, completed[0][0], turn, completed[0][1]


def _find_rollout(sessions: Path, workspace: Path, prompt: str, started: float) -> Path | None:
    matches = []
    for path in sessions.glob("**/rollout-*.jsonl"):
        if path.stat().st_mtime < started - 2:
            continue
        rows = events(path)
        if (rows and rows[0].get("type") == "session_meta"
                and rows[0].get("payload", {}).get("cwd") == str(workspace)
                and prompt.encode() in path.read_bytes()):
            matches.append(path)
    if len(matches) > 1:
        raise ValueError("launch prompt matched multiple rollouts")
    return matches[0] if matches else None


def _archive_checkpoint(archive: TranscriptArchive, expected_handoffs: int,
                        fact: str) -> dict:
    root = archive.root
    paths = sorted((root / "handoffs").glob("*.json"))
    if len(paths) != expected_handoffs:
        raise ValueError(f"expected {expected_handoffs} handoffs, found {len(paths)}")
    manifest = json.loads(archive.manifest_path.read_text())
    segments = manifest.get("segments")
    if (manifest.get("conversation_id") != archive.conversation_id
            or not isinstance(segments, list) or len(segments) < expected_handoffs):
        raise ValueError("archive manifest lacks the compaction segments")
    for segment in segments:
        archive.verify_segment(segment)
    first = archive.resolve_segment_file(segments[0]["path"])
    if fact not in first.read_text(encoding="utf-8"):
        raise ValueError("early decision is absent from the first archived segment")
    latest = archive.load_latest_handoff()
    if latest is None or latest.get("manifest_sha256") != sha256_file(archive.manifest_path):
        raise ValueError("latest handoff does not verify against the archive")
    return {
        "manifest": str(archive.manifest_path),
        "manifest_sha256": sha256_file(archive.manifest_path),
        "segment_paths": [item["path"] for item in segments],
        "segment_sha256": [item["sha256"] for item in segments],
        "handoffs": [{"path": str(path), "sha256": sha256_file(path),
                      "source_turn_id": json.loads(path.read_text())["source_turn_id"]}
                     for path in paths],
        "latest_source_turn_id": latest["source_turn_id"],
    }


def verify_repeated_compaction(rows: list[dict], *, workspace: Path,
                               prompts: tuple[str, str, str], answers: tuple[str, str, str],
                               boundaries: tuple[int, int, int],
                               checkpoints: tuple[dict, dict]) -> dict:
    """Prove turn and compaction order across three distinct process exits."""
    if not rows or rows[0].get("type") != "session_meta":
        raise ValueError("rollout lacks session metadata")
    meta = rows[0].get("payload", {})
    session = meta.get("id")
    if not isinstance(session, str) or not session or meta.get("cwd") != str(workspace):
        raise ValueError("rollout belongs to a different session or workspace")
    if not (0 < boundaries[0] < boundaries[1] < boundaries[2] == len(rows)):
        raise ValueError("process boundaries do not cover the final rollout")
    turns = [_prompt_turn(rows, prompt, session) for prompt in prompts]
    for index, (_, complete, _, answer) in enumerate(turns):
        if answer != answers[index]:
            raise ValueError(f"turn {index + 1} did not return the exact expected answer")
        lower = 0 if index == 0 else boundaries[index - 1]
        if not lower <= complete < boundaries[index]:
            raise ValueError(f"turn {index + 1} completed outside its process")
    compacted = [i for i, row in enumerate(rows) if row.get("type") == "compacted"]
    if (len(compacted) != 2
            or not turns[0][1] < compacted[0] < boundaries[0]
            or not turns[1][1] < compacted[1] < boundaries[1]
            or not boundaries[0] <= turns[1][0]
            or not boundaries[1] <= turns[2][0]):
        raise ValueError("two real compactions did not follow the two completed turns")
    if (len(checkpoints[0]["handoffs"]) != 1
            or len(checkpoints[1]["handoffs"]) != 2
            or checkpoints[0]["latest_source_turn_id"] == checkpoints[1]["latest_source_turn_id"]
            or checkpoints[0]["handoffs"][0] not in checkpoints[1]["handoffs"]
            or len(checkpoints[1]["segment_paths"]) < 2):
        raise ValueError("archive handoffs did not advance across compactions")
    return {"session_id": session, "turn_ids": [turn[2] for turn in turns],
            "compact_event_indexes": compacted,
            "process_event_boundaries": list(boundaries),
            "answers": [turn[3] for turn in turns]}


def _run_stage(command: list[str], *, workspace: Path, env: dict[str, str],
               log: Path, sessions: Path, prompt: str, expected_answer: str,
               expected_compactions: int, rollout: Path | None,
               launch_started: float, stage_timeout: float = 900) -> tuple[int, Path, list[dict], int]:
    """Drive one installed TUI process through its own private PTY."""
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 140, 0, 0))
    lease_fd = handoff_lease_fd()
    process = None
    try:
        process = subprocess.Popen(
            command, cwd=workspace, env={**env, "TERM": "xterm-256color",
                                         "MAVIS_GENERATION_LEASE_FD": str(lease_fd)},
            stdin=slave, stdout=slave, stderr=slave,
            pass_fds=(lease_fd,), start_new_session=True,
            preexec_fn=lambda: fcntl.ioctl(slave, termios.TIOCSCTTY, 0),
        )
        os.close(slave)
        slave = -1
        stage = "answer"
        trust_sent = False
        terminal_tail = b""
        deadline = time.monotonic() + stage_timeout
        with log.open("wb") as output:
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"installed TUI timed out waiting for {stage}")
                readable, _, _ = select.select([master], [], [], 0.25)
                if readable:
                    try:
                        chunk = os.read(master, 65536)
                        output.write(chunk)
                        output.flush()
                        terminal_tail = (terminal_tail + chunk)[-16384:]
                        if (not trust_sent and b"folder?" in terminal_tail
                                and b"Trust and continue" in terminal_tail):
                            os.write(master, b"\r")
                            trust_sent = True
                    except OSError:
                        pass
                if rollout is None:
                    rollout = _find_rollout(sessions, workspace, prompt, launch_started)
                rows = events(rollout) if rollout and rollout.exists() else []
                if stage == "answer" and rows:
                    session = rows[0].get("payload", {}).get("id")
                    try:
                        _user, _done, _turn, answer = _prompt_turn(rows, prompt, session)
                    except ValueError:
                        pass
                    else:
                        if answer != expected_answer:
                            raise ValueError("installed Mavis gave a different answer")
                        time.sleep(0.5)
                        os.write(master, b"/compact\r" if expected_compactions else b"/exit\r")
                        stage = "compact" if expected_compactions else "exit"
                        deadline = time.monotonic() + min(
                            stage_timeout, 600 if expected_compactions else 60)
                elif stage == "compact" and sum(row.get("type") == "compacted" for row in rows) == expected_compactions:
                    time.sleep(1)
                    os.write(master, b"/exit\r")
                    stage = "exit"
                    deadline = time.monotonic() + min(stage_timeout, 60)
                code = process.poll()
                if code is not None:
                    if stage != "exit":
                        raise RuntimeError(f"installed TUI exited before {stage}: {code}")
                    if code != 0:
                        raise RuntimeError(f"installed TUI exited with status {code}")
                    if rollout is None:
                        raise RuntimeError("installed TUI created no matching rollout")
                    return process.pid, rollout, completed_events(rollout), code
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


@with_mavis_handoff_lease("observe_phase2_repeated_compaction")
def main() -> int:
    # Imported here because this observer is integrated after the project
    # evidence layout; source-only verifier tests remain independent of it.
    from mavis.project_evidence import project_home

    home = Path.home()
    service = home / ".local-codex" / "mavis-service"
    config = RuntimeConfig(home=service, allow_concurrent_local=True)
    task = service / "evaluations" / "phase2" / f"repeated-compaction-{uuid.uuid4().hex}"
    task.mkdir(parents=True)
    workspace = Path(tempfile.mkdtemp(prefix="mavis-p2-compact-", dir="/private/tmp")).resolve()
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(["git", "-C", str(workspace), "-c", "user.name=Mavis Phase 2",
                    "-c", "user.email=mavis@local.invalid", "commit", "--allow-empty",
                    "-qm", "fixture"], check=True)
    fact = f"P2_{uuid.uuid4().hex}=sqlite-before-postgres"
    prompts = (
        f"Record this early design decision for project P2: {fact}. Reply only ACK1.",
        "Continue this project after the first compaction. Reply only ACK2.",
        "What was the exact early design decision for project P2? Reply with only its full value.",
    )
    answers = ("ACK1", "ACK2", fact)
    sessions = home / ".local-codex" / "sessions"
    launcher = str(home / "Desktop" / "Mavis.command")
    candidate = installed_candidate_fingerprint()
    result: dict[str, object] = {
        "schema_version": "mavis.phase2-repeated-compaction-observation/v1",
        "candidate": candidate, "observer_path": str(Path(__file__).resolve()),
        "observer_sha256": sha256_file(Path(__file__).resolve()),
        "workspace": str(workspace), "fact": fact, "prompts": prompts,
        "task_root": str(task), "model_id": config.model,
    }
    shared: dict[str, object] | None = None

    def on_signal(number: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"signal {number}")

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        require_installed_selected_model(config)
        with shared_mavis_model(config, "installed Mavis repeated compaction") as shared:
            env = {**os.environ, "CODEX_HOME": str(home / ".local-codex"),
                   "MAVIS_PROJECT_DIR": str(workspace),
                   "MAVIS_PROJECT_ROOT": str(workspace),
                   "PYTHONDONTWRITEBYTECODE": "1"}
            commands = [
                [launcher, "--no-daemon", "--no-alt-screen", "-C", str(workspace), prompts[0]],
            ]
            checkpoints = []
            boundaries = []
            rollout = None
            for index in range(3):
                renew_gpu_lease()
                if index:
                    commands.append([launcher, "resume", "--no-daemon", "--no-alt-screen",
                                     "-C", str(workspace), result["session_id"], prompts[index]])
                started = time.time()
                pid, rollout, rows, code = _run_stage(
                    commands[index], workspace=workspace, env=env,
                    log=task / f"terminal-{index + 1}.log", sessions=sessions,
                    prompt=prompts[index], expected_answer=answers[index],
                    expected_compactions=index + 1 if index < 2 else 0,
                    rollout=rollout, launch_started=started)
                if index == 0:
                    session = rows[0].get("payload", {}).get("id")
                    if not isinstance(session, str) or not session:
                        raise ValueError("first rollout lacks a session ID")
                    result["session_id"] = session
                    result["rollout"] = str(rollout)
                boundaries.append(len(rows))
                rollout_bytes = rollout.read_bytes()
                result[f"process_{index + 1}"] = {
                    "pid": pid, "exit_status": code, "command": commands[index],
                    "rollout_events": len(rows), "rollout_bytes": len(rollout_bytes),
                    "rollout_sha256": hashlib.sha256(rollout_bytes).hexdigest(),
                    "terminal_log": str(task / f"terminal-{index + 1}.log"),
                    "terminal_log_sha256": sha256_file(task / f"terminal-{index + 1}.log"),
                }
                if index < 2:
                    archive_home = project_home(workspace, create=False)
                    if archive_home != workspace / ".mavis" or not archive_home.is_dir():
                        raise ValueError("installed hook did not create project-local evidence")
                    checkpoints.append(_archive_checkpoint(
                        TranscriptArchive(archive_home, result["session_id"]), index + 1, fact))
            result["archive_checkpoints"] = checkpoints
            final_bytes = rollout.read_bytes()
            for index in range(2):
                snapshot = result[f"process_{index + 1}"]
                if hashlib.sha256(final_bytes[:snapshot["rollout_bytes"]]).hexdigest() != snapshot["rollout_sha256"]:
                    raise ValueError(f"rollout prefix changed after process {index + 1}")
            result["verification"] = verify_repeated_compaction(
                completed_events(rollout), workspace=workspace, prompts=prompts, answers=answers,
                boundaries=tuple(boundaries), checkpoints=tuple(checkpoints))
            result["rollout_final_sha256"] = sha256_file(rollout)
    except BaseException as exc:
        result["error"] = repr(exc)
    finally:
        if shared is not None:
            result.update(shared)
            result["iris_loaded"] = shared.get("iris_stayed_loaded")
        result["candidate_after"] = installed_candidate_fingerprint()
        write_json(task / "result.json", result)
        print(task / "result.json", flush=True)
    return 0 if ("verification" in result and not result.get("error")
                 and result.get("iris_loaded") is True
                 and result.get("mavis_loaded") is False
                 and result.get("gpu_lease_released") is True
                 and result.get("candidate_after") == candidate) else 1


if __name__ == "__main__":
    sys.exit(main())
