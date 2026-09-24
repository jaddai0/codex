#!/usr/bin/env python3
"""Run installed E0 while IRIS stays loaded and Mavis uses the shared GPU lease."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from mavis.evaluations import E0_CASES, E0Evaluator, installed_candidate_fingerprint
from mavis.runtime import RuntimeConfig, reserve_empty_mavis_port, require_installed_selected_model, with_mavis_handoff_lease
from mavis.storage import sha256_file, write_json
from shared_gpu_observation import run_monitored_observation, shared_mavis_model


@with_mavis_handoff_lease("final-e0")
def main() -> int:
    home = Path.home()
    service = home / ".local-codex" / "mavis-service"
    share = home / ".local" / "share" / "local-codex"
    output = service / "evaluations" / "e0" / "final-installed-shared.json"
    log = output.with_suffix(".log")
    live_log = output.with_name("tool-roundtrip-live.log")
    config = RuntimeConfig(home=service, allow_concurrent_local=True)

    require_installed_selected_model(config)
    installed_package = share / "mavis"
    if (Path(sys.modules["mavis"].__file__).resolve().parent != installed_package.resolve()
            or not (installed_package / "e2_tasks.py").is_file()
            or not all(token in inspect.getsource(E0Evaluator.run)
                       for token in ("MAVIS_E0_RUN_ID", "case_receipts", "installed_candidate"))):
        raise RuntimeError("final E0 requires the complete installed Mavis package")
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
    shared: dict[str, object] | None = None
    try:
        with shared_mavis_model(config, "installed Mavis E0 tool roundtrip") as shared:
            env["MAVIS_E0_SHARED_GPU_LEASE"] = "codex-mavis"
            with live_log.open("wb") as stream:
                result["tool_roundtrip_exit"] = run_monitored_observation(
                    config,
                    [sys.executable, "-m", "mavis", "eval", "e0", "--case", "tool-roundtrip"],
                    cwd=home, env=env, stdout=stream, shared=shared, timeout=1800,
                )
    except BaseException as error:
        result["error"] = repr(error)
    finally:
        if shared is not None:
            result.update(shared)
        result["candidate_after"] = installed_candidate_fingerprint()
        write_json(output, result)
    if (result.get("tool_roundtrip_exit") == 0
            and result.get("iris_stayed_loaded") is True
            and result.get("mavis_loaded") is False
            and result.get("gpu_lease_released") is True
            and result.get("candidate_after") == before):
        reservation = None
        try:
            # The shared context closed its own server. A different session may
            # have taken this port in the gap; reserve only if it is still empty.
            reservation = reserve_empty_mavis_port(config)
            env.pop("MAVIS_E0_SHARED_GPU_LEASE", None)
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
        finally:
            if reservation is not None:
                reservation.close()
            write_json(output, result)
    print(output)
    return 0 if (result.get("e0_exit") == 0
                 and result.get("iris_stayed_loaded") is True
                 and result.get("mavis_loaded") is False
                 and result.get("gpu_lease_released") is True
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
