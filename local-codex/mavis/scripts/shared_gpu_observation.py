"""Run one installed Mavis observation while IRIS keeps its model loaded."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import subprocess
from typing import Iterator
from urllib.parse import quote

from mavis.runtime import (
    RuntimeConfig,
    endpoint_alive,
    ensure_runtime,
    inventory,
    load_model,
    loaded_generation_models,
    park_mavis_server,
    request_json,
    require_idle_iris_handoff,
)


LEASE = Path.home() / ".local/bin/gpu-lease"
HOLDER = "codex-mavis"


def _lease_command(*args: str) -> str:
    run = subprocess.run(
        [str(LEASE), *args], capture_output=True, text=True, timeout=15, check=False
    )
    if run.returncode:
        raise RuntimeError(
            f"GPU lease {args[0]} failed: {(run.stderr or run.stdout).strip()}"
        )
    return run.stdout


def renew_gpu_lease() -> None:
    """Extend a long installed observation before the current lease expires."""
    _lease_command("renew", HOLDER, "90")


def _iris_loaded(config: RuntimeConfig) -> bool:
    return any(
        row.get("id") == config.model and row.get("loaded") is True
        for row in inventory(config.iris_endpoint)
    )


@contextmanager
def shared_mavis_model(
    config: RuntimeConfig, purpose: str
) -> Iterator[dict[str, object]]:
    """Hold the GPU lease, load only Mavis's model, then return it to idle."""
    if not config.allow_concurrent_local:
        raise ValueError(
            "shared Mavis observation requires explicit concurrent admission"
        )
    status = _lease_command("status")
    if (
        not status.startswith("the GPU is free\n")
        or "IRIS: a game is live" in status
        or "IRIS: not answering" in status
    ):
        raise RuntimeError("GPU lease is occupied or IRIS game state is unsafe")
    _lease_command("acquire", HOLDER, purpose, "90")
    result: dict[str, object] = {
        "gpu_lease_holder": HOLDER,
        "iris_stayed_loaded": False,
        "mavis_loaded": False,
    }
    reservation = None
    mavis_parked = False
    mavis_started = False
    try:
        status = _lease_command("status")
        if (
            not status.startswith(f"{HOLDER} has the GPU:")
            or "IRIS: a game is live" in status
            or "IRIS: not answering" in status
        ):
            raise RuntimeError(
                "GPU lease or IRIS game state changed before the observation"
            )
        require_idle_iris_handoff(config)
        if not _iris_loaded(config):
            raise RuntimeError("IRIS's original generation model is missing")
        if endpoint_alive(config.endpoint) and loaded_generation_models(
            config.endpoint
        ):
            raise RuntimeError("Mavis already owns a generation model")
        reservation = park_mavis_server(config)
        reservation.close()
        reservation = None
        mavis_started = True
        ensure_runtime(config, load=False)
        load_model(config)
        if not _iris_loaded(config):
            raise RuntimeError("IRIS changed while Mavis loaded")
        result["mavis_loaded"] = True
        yield result
    finally:
        try:
            if mavis_started and reservation is None:
                if endpoint_alive(config.endpoint) and loaded_generation_models(
                    config.endpoint
                ):
                    request_json(
                        config.endpoint,
                        f"/v1/models/{quote(config.model, safe='')}/unload",
                        method="POST",
                        timeout=180,
                    )
                reservation = park_mavis_server(config)
            if mavis_started and endpoint_alive(config.endpoint):
                raise RuntimeError("Mavis did not park after the observation")
            result["mavis_loaded"] = False
            mavis_parked = True
            if not _iris_loaded(config):
                raise RuntimeError("IRIS lost its model during the observation")
            result["iris_stayed_loaded"] = True
        finally:
            if reservation is not None:
                reservation.close()
            if mavis_parked:
                _lease_command("release", HOLDER)
                result["gpu_lease_released"] = True
