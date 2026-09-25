"""Run the installed E0 large-output canary with the shared GPU lease."""

from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import uuid

from mavis.evaluations import E0Evaluator, installed_candidate_fingerprint
from mavis.runtime import (RuntimeConfig, endpoint_alive, handoff_lease_fd,
                           inventory, loaded_generation_models,
                           require_installed_selected_model, with_mavis_handoff_lease)
from mavis.storage import write_json
from shared_gpu_observation import shared_mavis_model


@with_mavis_handoff_lease("observe_buried_failure")
def main() -> int:
    home = Path.home()
    service = home / ".local-codex" / "mavis-service"
    config = RuntimeConfig(home=service, allow_concurrent_local=True,
                           api_key=secrets.token_urlsafe(48))
    task = service / "evaluations" / "e0" / f"buried-live-{uuid.uuid4().hex}"
    repo = task / "repo"
    repo.mkdir(parents=True)
    (repo / "produce_log.py").write_text(
        "import secrets, sys\n"
        "print('A' * 800000)\n"
        "print('MAVIS_E0_FAILURE_' + secrets.token_hex(12))\n"
        "print('Z' * 800000)\n"
        "sys.exit(1)\n"
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Mavis E0",
                    "-c", "user.email=mavis@local.invalid", "commit", "--allow-empty",
                    "-qm", "fixture"], check=True)

    def loaded(endpoint: str) -> bool:
        return any(item.get("id") == config.model and item.get("loaded")
                   for item in inventory(endpoint))

    require_installed_selected_model(config)
    if (not loaded(config.iris_endpoint)
            or (endpoint_alive(config.endpoint) and loaded_generation_models(config.endpoint))):
        raise RuntimeError("IRIS must own the loaded model and Mavis must be unloaded")
    candidate = installed_candidate_fingerprint()
    result: dict[str, object] = {"candidate": candidate, "repo": str(repo),
                                 "task_root": str(task)}
    shared: dict[str, object] | None = None
    try:
        with shared_mavis_model(config, "installed Mavis buried-output canary") as shared:
            prompt = (
                "Run python3 produce_log.py exactly once, without a pipe or redirect. "
                "Call exec_command with cmd and, if needed, workdir only; do not set "
                "justification or sandbox_permissions. "
                "Its exit code 1 is intentional. The full output is too long to display. "
                "Use the Complete raw output reference from that command to find the "
                "exact MAVIS_E0_FAILURE_ identifier in the middle. Report that exact "
                "identifier and include this single sentence verbatim in the final answer: "
                "'The command python3 produce_log.py failed with exit code 1.' "
                "Use only read-only "
                "inspection commands for the captured .raw file, and answer in English. "
                "Do not edit files."
            )
            command = [str(home / "Desktop" / "Mavis.command"), "exec", "--json",
                       "-C", str(repo), "-s", "read-only", "-o",
                       str(task / "mavis-last-message.txt"), prompt]
            lease_fd = handoff_lease_fd()
            with (task / "installed-mavis-buried.jsonl").open("wb") as log:
                run = subprocess.run(command, cwd=repo, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     env={**os.environ, "MAVIS_PROJECT_DIR": str(repo),
                                          "PYTHONDONTWRITEBYTECODE": "1",
                                          "MAVIS_E0_TRIAL_API_KEY": config.api_key,
                                          "MAVIS_GENERATION_LEASE_FD": str(lease_fd)},
                                     pass_fds=(lease_fd,), timeout=900)
            result["mavis_exit"] = run.returncode
        for line in (task / "installed-mavis-buried.jsonl").read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "thread.started":
                result["session_id"] = event.get("thread_id")
                break
        session = result.get("session_id")
        if session:
            matches = list((home / ".local-codex" / "sessions").glob(f"**/*{session}.jsonl"))
            if len(matches) == 1:
                result["rollout"] = str(matches[0])
    except BaseException as exc:
        result["error"] = repr(exc)
    finally:
        if shared is not None:
            result.update(shared)
            result["iris_loaded"] = shared.get("iris_stayed_loaded")
        result["candidate_after"] = installed_candidate_fingerprint()
        write_json(task / "result.json", result)
    try:
        verdict = E0Evaluator(service, config)._buried_failure(
            task / "result.json", record=False)
        result["host_case_status"] = verdict["status"]
    except BaseException as exc:
        result["host_case_error"] = repr(exc)
    write_json(task / "result.json", result)
    print(task / "result.json")
    return 0 if (result.get("mavis_exit") == 0 and result.get("rollout")
                 and result.get("host_case_status") == "pass"
                 and result.get("iris_loaded") is True
                 and result.get("mavis_loaded") is False
                 and result.get("gpu_lease_released") is True
                 and result.get("candidate_after") == candidate) else 1


if __name__ == "__main__":
    sys.exit(main())
