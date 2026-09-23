#!/usr/bin/env python3
"""Observe Mavis service recovery while IRIS keeps its model and process."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from mavis.evaluations import installed_candidate_fingerprint
from mavis.runtime import (
    RuntimeConfig, _listener_pids, endpoint_alive, ensure_runtime, inventory,
    loaded_generation_models, omlx_live_process_binding,
    omlx_runtime_fingerprint, owns_running_server, park_mavis_server,
    require_installed_selected_model, with_mavis_handoff_lease,
)
from mavis.storage import write_json


@with_mavis_handoff_lease("observe_isolation_recovery")
def main() -> int:
    home = Path.home() / ".local-codex" / "mavis-service"
    config = RuntimeConfig(home=home)
    root = home / "evaluations" / "e0"
    result_path = root / "isolation-recovery-shared.json"

    def iris_loaded() -> bool:
        return any(row.get("id") == config.model and row.get("loaded") is True
                   for row in inventory(config.iris_endpoint))

    require_installed_selected_model(config)
    candidate = installed_candidate_fingerprint()
    runtime = omlx_runtime_fingerprint(config)
    if not endpoint_alive(config.iris_endpoint) or not iris_loaded():
        raise RuntimeError("IRIS must keep its model for service recovery")
    if endpoint_alive(config.endpoint) and loaded_generation_models(config.endpoint):
        raise RuntimeError("Mavis must not own a model before service recovery")
    iris_pids = sorted(_listener_pids(8000))
    if not iris_pids:
        raise RuntimeError("IRIS listener process is missing")
    iris_process = omlx_live_process_binding(config, config.iris_endpoint, runtime)

    reservation = None
    before_mavis: list[int] = []
    after_mavis: list[int] = []
    mavis_recovered_process: dict[str, object] | None = None
    outage_observed = False
    result: dict[str, object] = {"candidate": candidate, "omlx_runtime": runtime,
                                 "iris_pids": iris_pids}
    try:
        reservation = park_mavis_server(config)
        reservation.close()
        reservation = None
        ensure_runtime(config, load=False)
        if loaded_generation_models(config.endpoint) or not owns_running_server(config):
            raise RuntimeError("Mavis recovery baseline is not owned and unloaded")
        before_mavis = sorted(_listener_pids(8001))
        if not before_mavis:
            raise RuntimeError("Mavis recovery baseline listener is missing")

        reservation = park_mavis_server(config)
        outage_observed = not endpoint_alive(config.endpoint)
        if (not outage_observed or not endpoint_alive(config.iris_endpoint)
                or not iris_loaded() or sorted(_listener_pids(8000)) != iris_pids):
            raise RuntimeError("IRIS changed or Mavis outage was not observed")
        reservation.close()
        reservation = None
        ensure_runtime(config, load=False)
        after_mavis = sorted(_listener_pids(8001))
        if (not after_mavis or after_mavis == before_mavis
                or not owns_running_server(config)
                or loaded_generation_models(config.endpoint)
                or not iris_loaded() or sorted(_listener_pids(8000)) != iris_pids):
            raise RuntimeError("Mavis did not recover while IRIS stayed loaded")
        mavis_recovered_process = omlx_live_process_binding(config, config.endpoint, runtime)
    except BaseException as error:
        result["observation_error"] = repr(error)
    finally:
        try:
            if reservation is None:
                reservation = park_mavis_server(config)
            if (endpoint_alive(config.endpoint) or not iris_loaded()
                    or sorted(_listener_pids(8000)) != iris_pids
                    or omlx_live_process_binding(config, config.iris_endpoint, runtime)
                    != iris_process):
                raise RuntimeError("Mavis did not park or IRIS model/listener changed")
            result["iris_stayed_loaded"] = True
        except BaseException as error:
            result["cleanup_error"] = repr(error)
        if reservation is not None:
            reservation.close()
        result.update({"before_mavis_pids": before_mavis,
                       "after_mavis_pids": after_mavis,
                       "outage_observed": outage_observed})
        write_json(result_path, result)

    if (result.get("observation_error") or result.get("cleanup_error")
            or result.get("iris_stayed_loaded") is not True or not outage_observed
            or not before_mavis or not after_mavis or before_mavis == after_mavis):
        print(result_path)
        return 1
    proof = {
        "schema_version": "mavis.e0-isolation-recovery/v1",
        "candidate": candidate,
        "omlx_runtime": runtime,
        "iris_process": iris_process,
        "mavis_recovered_process": mavis_recovered_process,
        "recovery_while_iris_loaded": True,
        "before": {"iris_pids": iris_pids, "iris_model_loaded": True,
                   "mavis_pids": before_mavis},
        "after": {"iris_pids": iris_pids, "iris_model_loaded": True,
                  "mavis_pids": after_mavis},
        "outage_observed": True,
        "shared_receipt": str(result_path),
    }
    path = root / "isolation-recovery-live.json"
    write_json(path, proof)
    print(json.dumps({"receipt": str(path), "before": proof["before"],
                      "after": proof["after"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
