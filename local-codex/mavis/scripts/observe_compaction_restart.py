"""Drive a real installed TUI compact/resume canary from a terminal.

After the first ACK, enter /compact, wait for its completion, then /exit.
After the resumed answer, enter /exit. Each command needs Return.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from urllib.parse import quote

from mavis.evaluations import installed_candidate_fingerprint
from mavis.runtime import (RuntimeConfig, ensure_runtime, inventory,
                           load_model, loaded_generation_models, port_in_use,
                           request_json, require_idle_iris_handoff,
                           require_installed_selected_model, stop_server)
from mavis.storage import write_json


def events(path: Path) -> list[dict]:
    """Parse a rollout JSONL file, tolerating only an incomplete trailing line.

    A live rollout can end in a partially written record while its writer is
    still running. That single trailing fragment is dropped. Any malformed
    newline-terminated line is a real integrity failure and is reported
    clearly rather than surfacing a bare JSONDecodeError.
    """
    records: list[dict] = []
    data = path.read_bytes()
    lines = data.split(b"\n")
    for line_number, raw_line in enumerate(lines, 1):
        if not raw_line.strip():
            continue
        trailing = line_number == len(lines) and not data.endswith(b"\n")
        try:
            parsed = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if trailing:
                break
            raise ValueError(
                f"malformed JSONL record at line {line_number} of {path}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                f"non-object JSONL record at line {line_number} of {path}"
            )
        records.append(parsed)
    return records


def main() -> int:
    home = Path.home()
    service = home / ".local-codex" / "mavis-service"
    config = RuntimeConfig(home=service)
    model_path = f"/v1/models/{quote(config.model, safe='')}"
    task = service / "evaluations" / "e0" / f"compaction-live-{uuid.uuid4().hex}"
    task.mkdir(parents=True)
    workspace = Path(tempfile.mkdtemp(prefix="mavis-e0-compact-", dir="/private/tmp")).resolve()
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(["git", "-C", str(workspace), "-c", "user.name=Mavis E0",
                    "-c", "user.email=mavis@local.invalid", "commit", "--allow-empty",
                    "-qm", "fixture"], check=True)
    fact = f"MAVIS_E0_COMPACT_{uuid.uuid4().hex}=orchid-lantern-47"

    def loaded(endpoint: str) -> bool:
        return any(item.get("id") == config.model and item.get("loaded")
                   for item in inventory(endpoint))

    def on_signal(number: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"signal {number}")

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    require_installed_selected_model(config)
    if not loaded(config.iris_endpoint) or loaded_generation_models(config.endpoint):
        raise RuntimeError("IRIS must own the model and Mavis must be unloaded")
    candidate = installed_candidate_fingerprint()
    result: dict[str, object] = {"candidate": candidate,
                                 "workspace": str(workspace),
                                 "fact": fact, "task_root": str(task)}
    try:
        require_idle_iris_handoff(config)
        request_json(config.iris_endpoint, model_path + "/unload", method="POST", timeout=180)
        if loaded(config.iris_endpoint):
            raise RuntimeError("IRIS model did not unload")
        load_model(config)
        env = {**os.environ, "MAVIS_PROJECT_DIR": str(workspace),
               "PYTHONDONTWRITEBYTECODE": "1"}
        launcher = str(home / "Desktop" / "Mavis.command")
        started = time.time()
        print(f"First Mavis TUI in {workspace}; wait for ACK, then /compact and /exit", flush=True)
        first = subprocess.run([launcher, "--no-daemon", "--no-alt-screen", "-C",
                                str(workspace), f"Remember this exact fact: {fact}. Reply only ACK."],
                               cwd=workspace, env=env, timeout=900)
        result["first_exit"] = first.returncode
        session_root = home / ".local-codex" / "sessions"
        transcript: list[dict] = []
        rollout: Path | None = None
        for path in sorted(session_root.glob("**/rollout-*.jsonl"),
                           key=lambda item: item.stat().st_mtime, reverse=True):
            if path.stat().st_mtime < started - 2:
                continue
            records = events(path)
            if records and records[0].get("payload", {}).get("cwd") == str(workspace):
                rollout = path
                transcript = records
                break
        if rollout is None:
            raise RuntimeError("first TUI did not create a rollout in the fixture")
        result["rollout"] = str(rollout)
        session = transcript[0]["payload"]["id"]
        result["session_id"] = session
        result["first_answer"] = next((item.get("payload", {}).get("last_agent_message")
                                       for item in transcript if item.get("type") == "event_msg"
                                       and item.get("payload", {}).get("type") == "task_complete"
                                       and item.get("payload", {}).get("last_agent_message") == "ACK"), None)
        result["compacted_event"] = any(item.get("type") == "compacted" for item in transcript)
        handoff_dir = service / "transcripts" / session / "handoffs"
        result["handoffs"] = [str(path) for path in handoff_dir.glob("*.json")]
        if not (first.returncode == 0 and result["first_answer"] == "ACK"
                and result["compacted_event"] and result["handoffs"]):
            raise RuntimeError("first TUI or real compaction did not complete")
        print("Resuming the same Mavis session; wait for the answer, then /exit", flush=True)
        second = subprocess.run([launcher, "resume", "--no-daemon", "--no-alt-screen",
                                 "-C", str(workspace), session,
                                 "What exact fact did I give before compaction? Reply with only the fact."],
                                cwd=workspace, env=env, timeout=900)
        result["resume_exit"] = second.returncode
        transcript = events(rollout)
        answers = [item.get("payload", {}).get("last_agent_message")
                   for item in transcript if item.get("type") == "event_msg"
                   and item.get("payload", {}).get("type") == "task_complete"]
        result["resumed_answer"] = answers[-1] if answers else None
        result["exact_recovery"] = result["resumed_answer"] == fact
    except BaseException as exc:
        result["error"] = repr(exc)
    finally:
        try:
            if loaded_generation_models(config.endpoint):
                request_json(config.endpoint, model_path + "/unload", method="POST", timeout=180)
            if loaded_generation_models(config.endpoint):
                raise RuntimeError("Mavis model remained loaded")
        except BaseException as exc:
            result["mavis_unload_error"] = repr(exc)
            try:
                stop_server(config)
                for _ in range(100):
                    if not port_in_use(config.endpoint):
                        break
                    time.sleep(0.2)
                if port_in_use(config.endpoint):
                    raise RuntimeError("Mavis listener remained after stop")
                ensure_runtime(config, load=False)
            except BaseException as recovery_exc:
                result["mavis_stop_error"] = repr(recovery_exc)
        try:
            if loaded_generation_models(config.endpoint):
                raise RuntimeError("cannot restore IRIS while Mavis holds the model")
            if not loaded(config.iris_endpoint):
                request_json(config.iris_endpoint, model_path + "/load", method="POST", timeout=900)
            result["iris_loaded"] = loaded(config.iris_endpoint)
            result["mavis_loaded"] = bool(loaded_generation_models(config.endpoint))
            result["candidate_after"] = installed_candidate_fingerprint()
        except BaseException as exc:
            result["iris_restore_error"] = repr(exc)
        write_json(task / "result.json", result)
        print(task / "result.json", flush=True)
    return 0 if (result.get("first_exit") == result.get("resume_exit") == 0
                 and result.get("exact_recovery") is True
                 and result.get("iris_loaded") is True
                 and result.get("mavis_loaded") is False
                 and result.get("candidate_after") == candidate) else 1


if __name__ == "__main__":
    sys.exit(main())
