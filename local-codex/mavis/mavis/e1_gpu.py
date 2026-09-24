"""Shared GPU admission for an installed native E1 trial."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import subprocess
import time
from typing import Callable, Iterator
from urllib.parse import quote
from uuid import uuid4

from .runtime import (
    RuntimeConfig,
    endpoint_alive,
    ensure_runtime,
    inventory,
    load_model,
    loaded_generation_models,
    owns_running_server,
    request_json,
    require_idle_iris_handoff,
)
from .storage import write_json


LEASE = Path.home() / ".local/bin/gpu-lease"
HOLDER = "codex-mavis"
LEASE_MINUTES = "90"


def _lease_command(*args: str) -> str:
    result = subprocess.run([str(LEASE), *args], capture_output=True, text=True,
                            timeout=15, check=False)
    if result.returncode:
        raise RuntimeError(f"GPU lease {args[0]} failed: {(result.stderr or result.stdout).strip()}")
    return result.stdout


def _owned_status(status: str, purpose: str) -> None:
    first = f"{HOLDER} has the GPU: {purpose} ("
    first_line = status.splitlines()[0] if status.splitlines() else ""
    if not first_line.startswith(first):
        raise RuntimeError("E1 GPU lease purpose changed")


def _safe_status(status: str, *, purpose: str | None = None) -> None:
    if purpose is None:
        if not status.startswith("the GPU is free\n"):
            raise RuntimeError("GPU lease is occupied")
    else:
        _owned_status(status, purpose)
    if "IRIS: a game is live" in status or "IRIS: not answering" in status:
        raise RuntimeError("IRIS game state is unsafe for E1")


class _LeaseHeartbeat:
    def __init__(self, purpose: str):
        self.purpose = purpose
        self.iris_endpoint: str | None = None
        self.iris_models: list[str] | None = None
        self.next_status = 0.0
        self.next_renew = time.monotonic() + 300

    def __call__(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if force or now >= self.next_status:
            _safe_status(_lease_command("status"), purpose=self.purpose)
            if self.iris_endpoint is not None and loaded_generation_models(self.iris_endpoint) != self.iris_models:
                raise RuntimeError("IRIS generation model inventory changed during E1")
            self.next_status = now + 15
        if now >= self.next_renew:
            _lease_command("renew", HOLDER, LEASE_MINUTES)
            _safe_status(_lease_command("status"), purpose=self.purpose)
            self.next_renew = now + 300


def _idle_mavis_server(config: RuntimeConfig) -> None:
    status = request_json(config.endpoint, "/api/status")
    if (not isinstance(status, dict) or status.get("status") != "ok"
            or any(type(status.get(key)) is not int or status[key] != 0 for key in
                   ("active_requests", "waiting_requests", "models_loading"))):
        raise RuntimeError("Mavis server has active, waiting, or loading work")


def _cleanup_failure(config: RuntimeConfig, purpose: str, error: BaseException) -> Path:
    def observed(endpoint: str) -> object:
        try:
            return inventory(endpoint)
        except BaseException as failure:
            return {"inventory_error": str(failure)}

    path = config.home / "e1" / "admission-failures" / f"{uuid4().hex}.json"
    write_json(path, {
        "schema_version": "mavis.e1-gpu-cleanup-failure/v1",
        "at_epoch": time.time(), "gpu_lease_holder": HOLDER,
        "gpu_lease_purpose": purpose, "error": str(error),
        "iris_endpoint": config.iris_endpoint,
        "iris_inventory": observed(config.iris_endpoint),
        "mavis_endpoint": config.endpoint,
        "mavis_inventory": observed(config.endpoint),
    })
    return path


@contextmanager
def admitted_e1_model(config: RuntimeConfig, purpose: str) -> Iterator[Callable[[], None]]:
    """Keep IRIS loaded, lease the GPU, and unload only this trial's Mavis model."""
    config = replace(config, allow_concurrent_local=True, idle_seconds=900)
    purpose = f"{purpose}:{uuid4().hex}"
    _safe_status(_lease_command("status"))
    _lease_command("acquire", HOLDER, purpose, LEASE_MINUTES)
    heartbeat = _LeaseHeartbeat(purpose)
    load_attempted = False
    iris_models: list[str] | None = None
    try:
        heartbeat(force=True)
        iris_models = loaded_generation_models(config.iris_endpoint)
        if not iris_models:
            raise RuntimeError("IRIS has no loaded generation model to preserve")
        require_idle_iris_handoff(config, expected_models=iris_models)
        heartbeat.iris_endpoint = config.iris_endpoint
        heartbeat.iris_models = iris_models
        if endpoint_alive(config.endpoint):
            if not owns_running_server(config):
                raise RuntimeError("E1 will not reuse a server Mavis does not own")
            if loaded_generation_models(config.endpoint):
                raise RuntimeError("Mavis server already has a loaded generation model")
            _idle_mavis_server(config)
        ensure_runtime(config, load=False)
        if loaded_generation_models(config.endpoint):
            raise RuntimeError("Mavis model loaded outside this E1 trial")
        _idle_mavis_server(config)
        heartbeat(force=True)
        load_attempted = True
        load_model(config)
        heartbeat(force=True)
        if loaded_generation_models(config.iris_endpoint) != iris_models:
            raise RuntimeError("IRIS changed while Mavis loaded")
        yield heartbeat
    finally:
        owned = False
        cleanup_error: BaseException | None = None
        failure_receipt: Path | None = None
        try:
            _owned_status(_lease_command("status"), purpose)
            owned = True
            if load_attempted:
                loaded = loaded_generation_models(config.endpoint)
                if loaded:
                    if loaded != [config.model] or not owns_running_server(config):
                        raise RuntimeError("Mavis model inventory or server ownership changed during E1")
                    _idle_mavis_server(config)
                    for attempt in range(2):
                        try:
                            request_json(config.endpoint,
                                         f"/v1/models/{quote(config.model, safe='')}/unload",
                                         method="POST", timeout=180)
                        except (OSError, RuntimeError) as error:
                            if not loaded_generation_models(config.endpoint):
                                break  # response failed after the model actually unloaded
                            if attempt == 1:
                                raise RuntimeError("Mavis E1 model unload failed twice") from error
                        if not loaded_generation_models(config.endpoint):
                            break
                    else:
                        raise RuntimeError("Mavis model remained loaded after E1 cleanup")
                if loaded_generation_models(config.iris_endpoint) != iris_models:
                    raise RuntimeError("IRIS lost its original generation model during E1")
        except BaseException as error:
            cleanup_error = error
            failure_receipt = _cleanup_failure(config, purpose, error)
        finally:
            if owned:
                try:
                    _owned_status(_lease_command("status"), purpose)
                    _lease_command("release", HOLDER)
                except BaseException as error:
                    failure_receipt = _cleanup_failure(config, purpose, error)
                    cleanup_error = error
        if cleanup_error is not None:
            raise RuntimeError(
                f"E1 GPU cleanup failed: {cleanup_error}; inventory receipt: {failure_receipt}"
            ) from cleanup_error
