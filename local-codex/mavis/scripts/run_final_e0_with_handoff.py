#!/usr/bin/env python3
"""Run the installed E0 gate with a reversible IRIS/Mavis model handoff."""

from __future__ import annotations

import json
import inspect
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from urllib.parse import quote

from mavis.evaluations import E0_CASES, E0Evaluator, installed_candidate_fingerprint
from mavis.runtime import (RuntimeConfig, acquire_iris_model_drain,
                           endpoint_alive, ensure_runtime, inventory,
                           iris_drain_headers,
                           load_model, loaded_generation_models, park_mavis_server,
                           request_json, require_idle_iris_handoff,
                           require_installed_selected_model, release_iris_model_drain,
                           wait_iris_model_drain,
                           with_mavis_handoff_lease)
from mavis.storage import sha256_file, write_json


@with_mavis_handoff_lease("final-e0")
def main() -> int:
    home = Path.home()
    service = home / ".local-codex" / "mavis-service"
    share = home / ".local" / "share" / "local-codex"
    output = service / "evaluations" / "e0" / "final-installed-handoff.json"
    log = output.with_suffix(".log")
    live_log = output.with_name("tool-roundtrip-live.log")
    config = RuntimeConfig(home=service)
    model_path = f"/v1/models/{quote(config.model, safe='')}"

    def loaded(endpoint: str) -> bool:
        return any(row.get("id") == config.model and row.get("loaded") is True
                   for row in inventory(endpoint))

    require_installed_selected_model(config)
    installed_package = share / "mavis"
    if (Path(sys.modules["mavis"].__file__).resolve().parent != installed_package.resolve()
            or not (installed_package / "e2_tasks.py").is_file()
            or not all(token in inspect.getsource(E0Evaluator.run)
                       for token in ("MAVIS_E0_RUN_ID", "case_receipts", "installed_candidate"))):
        raise RuntimeError("final E0 requires the complete installed Mavis package")
    if not endpoint_alive(config.iris_endpoint) or not loaded(config.iris_endpoint):
        raise RuntimeError("IRIS must own the original model before E0")
    if endpoint_alive(config.endpoint) and loaded_generation_models(config.endpoint):
        raise RuntimeError("Mavis generation models must be unloaded before handoff")
    before = installed_candidate_fingerprint()
    run_id = uuid.uuid4().hex
    started_at_epoch = time.time()
    result: dict[str, object] = {"candidate_before": before, "log": str(log),
                                 "live_log": str(live_log), "run_id": run_id}
    env = os.environ.copy()
    for key in ("MAVIS_BIN", "LOCAL_CODEX_SHARE_DIR", "LOCAL_CODEX_BIN", "LOCAL_CODEX_HOME",
                "LOCAL_CODEX_MODEL", "OMLX_BASE_URL", "IRIS_OMLX_BASE_URL", "MAVIS_MODEL_DIR",
                "MAVIS_OMLX_BIN", "MAVIS_GATEWAY_ROOT", "MAVIS_GATEWAY_ENV_FILE", "PYTHONHOME"):
        env.pop(key, None)
    env["PYTHONPATH"] = str(share)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CODEX_HOME"] = str(home / ".local-codex")
    env["MAVIS_E0_RUN_ID"] = run_id
    lease_id: str | None = None
    reservation = None
    try:
        reservation = park_mavis_server(config)
        lease_id = acquire_iris_model_drain(config, owner="mavis-final-e0")
        result["drain_lease_id"] = lease_id
        wait_iris_model_drain(config, lease_id)
        require_idle_iris_handoff(config)
        request_json(config.iris_endpoint, model_path + "/unload", method="POST",
                     timeout=180, headers=iris_drain_headers(lease_id))
        if loaded(config.iris_endpoint):
            raise RuntimeError("IRIS model did not unload")
        reservation.close()
        reservation = None
        ensure_runtime(config, load=False)
        load_model(config)
        with live_log.open("wb") as stream:
            run = subprocess.run([sys.executable, "-m", "mavis", "eval", "e0",
                                  "--case", "tool-roundtrip"],
                                 cwd=home, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
        result["tool_roundtrip_exit"] = run.returncode
    except BaseException as error:
        result["error"] = repr(error)
    finally:
        try:
            if reservation is None:
                if loaded_generation_models(config.endpoint):
                    request_json(config.endpoint, model_path + "/unload", method="POST", timeout=180)
                if loaded_generation_models(config.endpoint):
                    raise RuntimeError("Mavis model remained loaded")
                reservation = park_mavis_server(config)
        except BaseException as error:
            result["mavis_unload_error"] = repr(error)
        try:
            if reservation is None or endpoint_alive(config.endpoint):
                raise RuntimeError("cannot restore IRIS without an exclusive Mavis port reservation")
            if lease_id is None and not loaded(config.iris_endpoint):
                raise RuntimeError("IRIS cannot be restored without a drain lease")
            if not loaded(config.iris_endpoint):
                request_json(config.iris_endpoint, model_path + "/load", method="POST",
                             timeout=900, headers=iris_drain_headers(lease_id))
            result["iris_loaded"] = loaded(config.iris_endpoint)
            result["mavis_loaded"] = False
            if lease_id is not None and result["iris_loaded"] and not result["mavis_loaded"]:
                release_iris_model_drain(config, lease_id)
                result["drain_released"] = True
            result["candidate_after"] = installed_candidate_fingerprint()
        except BaseException as error:
            result["iris_restore_error"] = repr(error)
        write_json(output, result)
    if (result.get("tool_roundtrip_exit") == 0 and result.get("iris_loaded") is True
            and result.get("mavis_loaded") is False
            and result.get("drain_released") is True):
        try:
            with log.open("wb") as stream:
                env["MAVIS_E0_PARK_OWNER_PID"] = str(os.getpid())
                run = subprocess.run([sys.executable, "-m", "mavis", "eval", "e0"],
                                     cwd=home, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
            result["e0_exit"] = run.returncode
            summary = service / "evaluations" / "e0" / "runs" / run_id / "summary.json"
            if summary.is_file() and summary.stat().st_mtime >= started_at_epoch:
                result["summary"] = json.loads(summary.read_text())
                result["case_receipts_observed"] = {
                    case: sha256_file(summary.parent / f"{case}.json")
                    for case in E0_CASES if (summary.parent / f"{case}.json").is_file()
                }
            else:
                result["summary_error"] = "E0 did not write a fresh run summary"
        except BaseException as error:
            result["e0_error"] = repr(error)
        write_json(output, result)
    if reservation is not None:
        reservation.close()
    print(output)
    return 0 if (result.get("e0_exit") == 0 and result.get("iris_loaded") is True
                 and result.get("mavis_loaded") is False
                 and result.get("drain_released") is True
                 and result.get("candidate_after") == before
                 and isinstance(result.get("summary"), dict)
                 and result["summary"].get("status") == "pass"
                 and result["summary"].get("run_id") == run_id
                 and result["summary"].get("installed_candidate") == before
                 and result["summary"].get("model_id") == config.model
                 and result["summary"].get("case_receipts") ==
                 result.get("case_receipts_observed")
                 and set(result["summary"].get("case_receipts", {})) == set(E0_CASES)) else 1


if __name__ == "__main__":
    sys.exit(main())
