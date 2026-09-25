"""Run the installed E0 repair with the shared GPU, then a native GLM 5.3 review."""

from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

from mavis.e0_tasks import prepare_small_repository, small_repository_review_prompt
from mavis.e2_tasks import codex_review_completed, independent_review_command
from mavis.gateway import glm_review_environment, glm_review_launcher
from mavis.evaluations import installed_candidate_fingerprint
from mavis.runtime import (RuntimeConfig, endpoint_alive, handoff_lease_fd,
                           inventory, loaded_generation_models,
                           require_installed_selected_model, with_mavis_handoff_lease)
from mavis.storage import sha256_file, write_json
from shared_gpu_observation import shared_mavis_model


@with_mavis_handoff_lease("observe_small_repository")
def main() -> int:
    # The GLM review runs after the GPU repair; fail before spending the GPU turn.
    glm_review_launcher()
    glm_review_environment()
    home = Path.home()
    service = home / ".local-codex" / "mavis-service"
    config = RuntimeConfig(home=service, allow_concurrent_local=True,
                           api_key=secrets.token_urlsafe(48))
    manifest_path = prepare_small_repository(service)
    manifest = json.loads(manifest_path.read_text())
    repo = Path(manifest["repo"]).resolve()
    task = manifest_path.parent

    def loaded(endpoint: str) -> bool:
        return any(item.get("id") == config.model and item.get("loaded")
                   for item in inventory(endpoint))

    require_installed_selected_model(config)
    if (not loaded(config.iris_endpoint)
            or (endpoint_alive(config.endpoint) and loaded_generation_models(config.endpoint))):
        raise RuntimeError("IRIS must own the model and Mavis must be unloaded")
    candidate = installed_candidate_fingerprint()
    result: dict[str, object] = {"candidate": candidate, "manifest": str(manifest_path)}
    shared: dict[str, object] | None = None
    try:
        with shared_mavis_model(config, "installed Mavis small repair canary") as shared:
            prompt = (
                "Fix the failing discount test in this repository. You may edit only "
                "package/pricing.py. Preserve the already dirty user-notes.txt exactly. "
                "Run python3 -m unittest discover -s tests -q, inspect the result, "
                "and report the actual outcome. Do not commit the fixture."
            )
            command = [str(home / "Desktop" / "Mavis.command"), "exec", "--json",
                       "-C", str(repo), "-s", "workspace-write", prompt]
            lease_fd = handoff_lease_fd()
            with (task / "installed-mavis-repair.jsonl").open("wb") as log:
                run = subprocess.run(command, cwd=repo, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     env={**os.environ, "MAVIS_PROJECT_DIR": str(repo),
                                          "PYTHONDONTWRITEBYTECODE": "1",
                                          "MAVIS_E0_TRIAL_API_KEY": config.api_key,
                                          "MAVIS_GENERATION_LEASE_FD": str(lease_fd)},
                                     pass_fds=(lease_fd,), timeout=900)
            result["mavis_exit"] = run.returncode
    except BaseException as exc:
        result["mavis_error"] = repr(exc)
    finally:
        if shared is not None:
            result.update(shared)
            result["iris_loaded"] = shared.get("iris_stayed_loaded")
        write_json(task / "handoff-summary.json", result)
    if not (result.get("mavis_exit") == 0 and result.get("iris_loaded") is True
            and result.get("mavis_loaded") is False
            and result.get("gpu_lease_released") is True):
        print(task / "handoff-summary.json")
        return 1

    review_text = task / "glm-review.txt"
    review_argv = independent_review_command(
        repo, review_text, small_repository_review_prompt(manifest_path))
    with (task / "glm-review.jsonl").open("wb") as log, \
            (task / "glm-review.stderr.log").open("wb") as stderr:
        review = subprocess.run(review_argv, cwd=repo, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=stderr,
                                env={**glm_review_environment(), "PYTHONDONTWRITEBYTECODE": "1"},
                                timeout=900)
    review_log = (task / "glm-review.jsonl").read_text()
    response = review_text.read_text() if review_text.is_file() else ""
    try:
        review_thread_id = codex_review_completed(review_log, response)
    except ValueError:
        review_thread_id = None
    write_json(task / "installed-run.json", {
        "schema_version": "mavis.e0-installed-run/v2", "candidate": candidate,
        "mavis_log_sha256": sha256_file(task / "installed-mavis-repair.jsonl"),
        "review_log_sha256": sha256_file(task / "glm-review.jsonl"),
        "review_stderr_sha256": sha256_file(task / "glm-review.stderr.log"),
        "review_text_sha256": sha256_file(review_text) if review_text.is_file() else None,
        "review_argv": review_argv, "review_exit": review.returncode,
        "review_thread_id": review_thread_id,
    })
    print(manifest_path)
    return 0 if review.returncode == 0 and review_thread_id else 1


if __name__ == "__main__":
    sys.exit(main())
