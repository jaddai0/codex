"""Drive real installed TUI compaction and resume in a private terminal."""

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

from mavis.evaluations import installed_candidate_fingerprint
from mavis.project_evidence import project_home
from mavis.runtime import RuntimeConfig, require_installed_selected_model, with_mavis_handoff_lease
from mavis.storage import write_json
sys.path.insert(0, str(Path(__file__).resolve().parent))
from observe_phase2_repeated_compaction import _run_stage
from shared_gpu_observation import renew_gpu_lease, shared_mavis_model


def events(path: Path) -> list[dict]:
    """Parse rollout JSONL, ignoring only an incomplete trailing record."""
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


@with_mavis_handoff_lease("observe_compaction_restart")
def main() -> int:
    home = Path.home()
    service = home / ".local-codex" / "mavis-service"
    config = RuntimeConfig(home=service, allow_concurrent_local=True)
    task = service / "evaluations" / "e0" / f"compaction-live-{uuid.uuid4().hex}"
    task.mkdir(parents=True)
    workspace = Path(tempfile.mkdtemp(prefix="mavis-e0-compact-", dir="/private/tmp")).resolve()
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(["git", "-C", str(workspace), "-c", "user.name=Mavis E0",
                    "-c", "user.email=mavis@local.invalid", "commit", "--allow-empty",
                    "-qm", "fixture"], check=True)
    fact = f"MAVIS_E0_COMPACT_{uuid.uuid4().hex}=orchid-lantern-47"
    prompts = (f"Remember this exact fact: {fact}. Reply only ACK.",
               "What exact fact did I give before compaction? Reply with only the fact.")
    sessions = home / ".local-codex" / "sessions"
    launcher = str(home / "Desktop" / "Mavis.command")

    def on_signal(number: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"signal {number}")

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    require_installed_selected_model(config)
    candidate = installed_candidate_fingerprint()
    result: dict[str, object] = {"candidate": candidate,
                                 "workspace": str(workspace),
                                 "fact": fact, "task_root": str(task)}
    shared: dict[str, object] | None = None
    try:
        with shared_mavis_model(config, "installed Mavis compaction canary") as shared:
            env = {"CODEX_HOME": str(home / ".local-codex"),
                   "MAVIS_PROJECT_DIR": str(workspace),
                   "MAVIS_PROJECT_ROOT": str(workspace),
                   "PYTHONDONTWRITEBYTECODE": "1"}
            env = {**os.environ, **env}
            _pid, rollout, rows, first_exit = _run_stage(
                [launcher, "--no-daemon", "--no-alt-screen", "-C", str(workspace), prompts[0]],
                workspace=workspace, env=env, log=task / "terminal-first.log",
                sessions=sessions, prompt=prompts[0], expected_answer="ACK",
                expected_compactions=1, rollout=None, launch_started=time.time())
            session = rows[0].get("payload", {}).get("id")
            if not isinstance(session, str) or not session:
                raise RuntimeError("first TUI did not create a session")
            result.update({"rollout": str(rollout), "session_id": session,
                           "first_exit": first_exit, "first_answer": "ACK",
                           "compacted_event": any(row.get("type") == "compacted" for row in rows)})
            handoff_dir = project_home(workspace, create=False) / "transcripts" / session / "handoffs"
            result["handoffs"] = [str(path) for path in handoff_dir.glob("*.json")]
            if not result["compacted_event"] or not result["handoffs"]:
                raise RuntimeError("first TUI did not persist its compaction handoff")
            renew_gpu_lease()
            _pid, resumed_rollout, _resumed_rows, resume_exit = _run_stage(
                [launcher, "resume", "--no-daemon", "--no-alt-screen",
                 "-C", str(workspace), session, prompts[1]],
                workspace=workspace, env=env, log=task / "terminal-resume.log",
                sessions=sessions, prompt=prompts[1], expected_answer=fact,
                expected_compactions=0, rollout=rollout, launch_started=time.time())
            if resumed_rollout != rollout:
                raise RuntimeError("resumed TUI changed the rollout")
            result["resume_exit"] = resume_exit
            transcript = events(rollout)
            answers = [item.get("payload", {}).get("last_agent_message")
                       for item in transcript if item.get("type") == "event_msg"
                       and item.get("payload", {}).get("type") == "task_complete"]
            result["resumed_answer"] = answers[-1] if answers else None
            result["exact_recovery"] = result["resumed_answer"] == fact
    except BaseException as exc:
        result["error"] = repr(exc)
    finally:
        if shared is not None:
            result.update(shared)
            result["iris_loaded"] = shared.get("iris_stayed_loaded")
        result["candidate_after"] = installed_candidate_fingerprint()
        write_json(task / "result.json", result)
        print(task / "result.json", flush=True)
    return 0 if (result.get("first_exit") == result.get("resume_exit") == 0
                 and result.get("exact_recovery") is True
                 and result.get("iris_loaded") is True
                 and result.get("mavis_loaded") is False
                 and result.get("gpu_lease_released") is True
                 and result.get("candidate_after") == candidate) else 1


if __name__ == "__main__":
    sys.exit(main())
