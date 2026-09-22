#!/usr/bin/env python3
"""Record a real, installed Mavis service outage without touching IRIS."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

from mavis.evaluations import installed_candidate_fingerprint
from mavis.runtime import (
    RuntimeConfig,
    _listener_pids,
    endpoint_alive,
    ensure_runtime,
    inventory,
    owns_running_server,
    stop_server,
)
from mavis.storage import write_json


def _loaded_iris_model(config: RuntimeConfig) -> bool:
    return any(
        item.get("id") == config.model and item.get("loaded") is True
        for item in inventory(config.iris_endpoint)
    )


def _wait_for(endpoint: str, alive: bool, seconds: float = 30) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if endpoint_alive(endpoint) is alive:
            return
        time.sleep(0.2)
    raise TimeoutError(f"endpoint did not become {'available' if alive else 'unavailable'}")


def main() -> int:
    home = Path.home() / ".local-codex" / "mavis-service"
    config = RuntimeConfig(home=home)
    if not (endpoint_alive(config.endpoint) and owns_running_server(config)):
        raise RuntimeError("Mavis service is not alive and owned")
    if any(item.get("loaded") for item in inventory(config.endpoint)):
        raise RuntimeError("Mavis model must be unloaded before service interruption")
    if not endpoint_alive(config.iris_endpoint) or not _loaded_iris_model(config):
        raise RuntimeError("IRIS model must be loaded before the outage")
    before = {
        "iris_pids": sorted(_listener_pids(8000)),
        "iris_model_loaded": True,
        "mavis_pids": sorted(_listener_pids(8001)),
    }
    if not before["iris_pids"] or not before["mavis_pids"]:
        raise RuntimeError("expected both listener processes")
    outage_observed = False
    try:
        stop_server(config)
        _wait_for(config.endpoint, False)
        outage_observed = True
        if (
            not endpoint_alive(config.iris_endpoint)
            or not _loaded_iris_model(config)
            or sorted(_listener_pids(8000)) != before["iris_pids"]
        ):
            raise RuntimeError("IRIS became unavailable during Mavis outage")
    finally:
        # Restore Mavis even when the observation or IRIS check fails.
        if not endpoint_alive(config.endpoint):
            ensure_runtime(config, load=False)
            _wait_for(config.endpoint, True)
    after = {
        "iris_pids": sorted(_listener_pids(8000)),
        "iris_model_loaded": _loaded_iris_model(config),
        "mavis_pids": sorted(_listener_pids(8001)),
    }
    if (
        not outage_observed
        or before["iris_pids"] != after["iris_pids"]
        or not after["iris_model_loaded"]
        or before["mavis_pids"] == after["mavis_pids"]
        or not owns_running_server(config)
    ):
        raise RuntimeError("outage or recovery did not satisfy isolation checks")
    proof = {
        "schema_version": "mavis.e0-isolation-recovery/v1",
        "candidate": installed_candidate_fingerprint(),
        "before": before,
        "after": after,
        "outage_observed": True,
    }
    path = home / "evaluations" / "e0" / "isolation-recovery-live.json"
    write_json(path, proof)
    print(json.dumps({"receipt": str(path), "before": before, "after": after}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
