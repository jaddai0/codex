"""Run the installed E0 repair, then a separate read-only native ZCode review."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import quote

from mavis.e0_tasks import prepare_small_repository
from mavis.evaluations import installed_candidate_fingerprint, native_review_completed
from mavis.runtime import (RuntimeConfig, endpoint_alive, ensure_runtime, inventory,
                           load_model, request_json, stop_server)
from mavis.storage import sha256_file, write_json


def main() -> int:
    home = Path.home()
    service = home / ".local-codex" / "mavis-service"
    config = RuntimeConfig(home=service)
    model_path = f"/v1/models/{quote(config.model, safe='')}"
    manifest_path = prepare_small_repository(service)
    manifest = json.loads(manifest_path.read_text())
    repo = Path(manifest["repo"])
    task = manifest_path.parent

    def loaded(endpoint: str) -> bool:
        return any(item.get("id") == config.model and item.get("loaded")
                   for item in inventory(endpoint))

    if not loaded(config.iris_endpoint) or loaded(config.endpoint):
        raise RuntimeError("IRIS must own the model and Mavis must be unloaded")
    candidate = installed_candidate_fingerprint()
    result: dict[str, object] = {"candidate": candidate, "manifest": str(manifest_path)}
    try:
        request_json(config.iris_endpoint, model_path + "/unload", method="POST", timeout=180)
        if loaded(config.iris_endpoint):
            raise RuntimeError("IRIS model did not unload")
        load_model(config)
        prompt = (
            "Fix the failing discount test in this repository. You may edit only "
            "package/pricing.py. Preserve the already dirty user-notes.txt exactly. "
            "Run python3 -m unittest discover -s tests -q, inspect the result, "
            "and report the actual outcome. Do not commit the fixture."
        )
        command = [str(home / "Desktop" / "Mavis.command"), "exec", "--json",
                   "-C", str(repo), "-s", "workspace-write", prompt]
        with (task / "installed-mavis-repair.jsonl").open("wb") as log:
            run = subprocess.run(command, cwd=repo, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT,
                                 env={**os.environ, "MAVIS_PROJECT_DIR": str(repo),
                                      "PYTHONDONTWRITEBYTECODE": "1"}, timeout=900)
        result["mavis_exit"] = run.returncode
    except BaseException as exc:
        result["mavis_error"] = repr(exc)
    finally:
        try:
            if loaded(config.endpoint):
                request_json(config.endpoint, model_path + "/unload", method="POST", timeout=180)
            if loaded(config.endpoint):
                raise RuntimeError("Mavis model remained loaded")
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
                raise RuntimeError("cannot restore IRIS while Mavis holds the model")
            if not loaded(config.iris_endpoint):
                request_json(config.iris_endpoint, model_path + "/load", method="POST", timeout=900)
            result["iris_loaded"] = loaded(config.iris_endpoint)
            result["mavis_loaded"] = loaded(config.endpoint)
        except BaseException as exc:
            result["iris_restore_error"] = repr(exc)
        write_json(task / "handoff-summary.json", result)
    if not (result.get("mavis_exit") == 0 and result.get("iris_loaded") is True
            and result.get("mavis_loaded") is False):
        print(task / "handoff-summary.json")
        return 1

    review_prompt = (
        "Read-only independent review of the installed Mavis repair in this repository. "
        f"Read {manifest_path} and its baseline log. Check git status and diff, run "
        "python3 -m unittest discover -s tests -q, verify user-notes.txt still matches "
        "the manifest hash, and assess whether package/pricing.py correctly fixes the "
        "seeded failure without hiding it. Do not edit files. Begin your final response "
        "with ACCEPT or REJECT, followed by concise evidence."
    )
    with (task / "terra-review.jsonl").open("wb") as log:
        review = subprocess.run(["zcode", "--json", "--mode", "yolo",
                                 "--disallowed-tools", "Edit,Write", "--cwd", str(repo),
                                 "--prompt", review_prompt], cwd=repo,
                                stdout=log, stderr=subprocess.STDOUT, timeout=900)
    review_log = (task / "terra-review.jsonl").read_text()
    start = review_log.find("{")
    try:
        response = json.loads(review_log[start:])["response"] if start >= 0 else ""
    except (json.JSONDecodeError, KeyError):
        response = ""
    (task / "terra-review.txt").write_text(response)
    write_json(task / "installed-run.json", {
        "schema_version": "mavis.e0-installed-run/v1", "candidate": candidate,
        "mavis_log_sha256": sha256_file(task / "installed-mavis-repair.jsonl"),
        "terra_log_sha256": sha256_file(task / "terra-review.jsonl"),
    })
    print(manifest_path)
    return 0 if review.returncode == 0 and native_review_completed(review_log, response) else 1


if __name__ == "__main__":
    sys.exit(main())
