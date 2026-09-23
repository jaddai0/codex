"""Run the installed E0 large-output canary with a sequential IRIS/Mavis handoff."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from urllib.parse import quote

from mavis.evaluations import installed_candidate_fingerprint
from mavis.runtime import (RuntimeConfig, acquire_iris_model_drain,
                           ensure_runtime, handoff_lease_fd, inventory, iris_drain_headers,
                           load_model, loaded_generation_models, port_in_use,
                           request_json, require_idle_iris_handoff,
                           require_installed_selected_model, release_iris_model_drain,
                           stop_server, wait_iris_model_drain,
                           with_mavis_handoff_lease)
from mavis.storage import write_json


@with_mavis_handoff_lease("observe_buried_failure")
def main() -> int:
    home = Path.home()
    service = home / ".local-codex" / "mavis-service"
    config = RuntimeConfig(home=service)
    model_path = f"/v1/models/{quote(config.model, safe='')}"
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
    if not loaded(config.iris_endpoint) or loaded_generation_models(config.endpoint):
        raise RuntimeError("IRIS must own the loaded model and Mavis must be unloaded")
    candidate = installed_candidate_fingerprint()
    result: dict[str, object] = {"candidate": candidate, "repo": str(repo),
                                 "task_root": str(task)}
    lease_id: str | None = None
    try:
        lease_id = acquire_iris_model_drain(config, owner="mavis-observe_buried_failure")
        result["drain_lease_id"] = lease_id
        wait_iris_model_drain(config, lease_id)
        require_idle_iris_handoff(config)
        request_json(config.iris_endpoint, model_path + "/unload", method="POST",
                     timeout=180, headers=iris_drain_headers(lease_id))
        if loaded(config.iris_endpoint):
            raise RuntimeError("IRIS model did not unload")
        load_model(config)
        prompt = (
            "Run python3 produce_log.py exactly once, without a pipe or redirect. "
            "Its exit code 1 is intentional. The full output is too long to display. "
            "Use the Complete raw output reference from that command to find the "
            "exact MAVIS_E0_FAILURE_ identifier in the middle. Report that exact "
            "identifier and clearly state the command failed. Do not edit files."
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
        try:
            if loaded_generation_models(config.endpoint):
                request_json(config.endpoint, model_path + "/unload", method="POST", timeout=180)
            if loaded_generation_models(config.endpoint):
                raise RuntimeError("Mavis model remained loaded after unload")
        except BaseException as exc:
            result["mavis_unload_error"] = repr(exc)
            try:
                stop_server(config)
                import time
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
                raise RuntimeError("cannot restore IRIS while Mavis still holds the model")
            if lease_id is None and not loaded(config.iris_endpoint):
                raise RuntimeError("IRIS cannot be restored without a drain lease")
            if not loaded(config.iris_endpoint):
                request_json(config.iris_endpoint, model_path + "/load", method="POST",
                             timeout=900, headers=iris_drain_headers(lease_id))
            result["iris_loaded"] = loaded(config.iris_endpoint)
            result["mavis_loaded"] = bool(loaded_generation_models(config.endpoint))
            if lease_id is not None and result["iris_loaded"] and not result["mavis_loaded"]:
                release_iris_model_drain(config, lease_id)
                result["drain_released"] = True
            result["candidate_after"] = installed_candidate_fingerprint()
        except BaseException as exc:
            result["restore_error"] = repr(exc)
        write_json(task / "result.json", result)
    print(task / "result.json")
    return 0 if (result.get("mavis_exit") == 0 and result.get("rollout")
                 and result.get("iris_loaded") is True
                 and result.get("mavis_loaded") is False
                 and result.get("drain_released") is True
                 and result.get("candidate_after") == candidate) else 1


if __name__ == "__main__":
    sys.exit(main())
