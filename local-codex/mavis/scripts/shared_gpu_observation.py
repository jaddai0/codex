"""Run one installed Mavis observation while IRIS keeps its model loaded."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import subprocess
import threading
import time
from typing import Iterator
from urllib.parse import quote
from uuid import uuid4

from mavis.runtime import (
    RuntimeConfig,
    endpoint_alive,
    inventory,
    loaded_generation_models,
    load_model,
    park_mavis_server,
    read_json,
    request_json,
    require_idle_iris_handoff,
    reserve_empty_mavis_port,
    start_server,
    stop_trial_server,
)
from mavis.storage import write_json


LEASE = Path.home() / ".local/bin/gpu-lease"
HOLDER = "codex-mavis"
_PURPOSE: ContextVar[str | None] = ContextVar("mavis_observation_lease_purpose", default=None)


def _lease_command(*args: str) -> str:
    run = subprocess.run(
        [str(LEASE), *args], capture_output=True, text=True, timeout=15, check=False
    )
    if run.returncode:
        raise RuntimeError(
            f"GPU lease {args[0]} failed: {(run.stderr or run.stdout).strip()}"
        )
    return run.stdout


def _owned_lease(purpose: str) -> None:
    status = _lease_command("status")
    first = status.splitlines()[0] if status.splitlines() else ""
    if not first.startswith(f"{HOLDER} has the GPU: {purpose} ("):
        raise RuntimeError("Mavis observation GPU lease purpose changed")


def _safe_game_state(status: str) -> None:
    if "IRIS: a game is live" in status or "IRIS: not answering" in status:
        raise RuntimeError("IRIS game state is unsafe for Mavis observation")


def renew_gpu_lease() -> None:
    """Extend only this observation's still-owned lease."""
    purpose = _PURPOSE.get()
    if purpose is None:
        raise RuntimeError("no active Mavis observation lease to renew")
    _owned_lease(purpose)
    _safe_game_state(_lease_command("status"))
    _lease_command("renew", HOLDER, "90")
    _owned_lease(purpose)


def _idle_mavis_server(config: RuntimeConfig) -> None:
    status = request_json(config.endpoint, "/api/status")
    if (not isinstance(status, dict) or status.get("status") != "ok"
            or any(type(status.get(key)) is not int or status[key] != 0 for key in
                   ("active_requests", "waiting_requests", "models_loading"))):
        raise RuntimeError("Mavis server has active, waiting, or loading work")


def _failure_receipt(config: RuntimeConfig, purpose: str,
                     error: BaseException) -> Path:
    def observed(endpoint: str) -> object:
        try:
            return inventory(endpoint)
        except BaseException as failure:
            return {"inventory_error": str(failure)}
    path = config.home / "evaluations" / "shared-gpu-failures" / f"{uuid4().hex}.json"
    write_json(path, {
        "schema_version": "mavis.shared-gpu-cleanup-failure/v1",
        "at_epoch": time.time(), "gpu_lease_holder": HOLDER,
        "gpu_lease_purpose": purpose, "error": str(error),
        "iris_inventory": observed(config.iris_endpoint),
        "mavis_inventory": observed(config.endpoint),
    })
    return path


def _record_failure_receipt(result: dict[str, object], config: RuntimeConfig,
                            purpose: str, error: BaseException) -> None:
    if "cleanup_failure_receipt" in result:
        return
    try:
        result["cleanup_failure_receipt"] = str(_failure_receipt(config, purpose, error))
    except BaseException as receipt_error:
        result["cleanup_failure_receipt_error"] = str(receipt_error)


def _monitored_load(config: RuntimeConfig, purpose: str,
                    server_state: dict[str, object], stopped: dict[str, bool],
                    iris_models: list[str]) -> None:
    """Watch the game while oMLX's synchronous load request is in flight."""
    done = threading.Event()
    failures: list[BaseException] = []
    def worker() -> None:
        try:
            load_model(config)
        except BaseException as error:
            failures.append(error)
        finally:
            done.set()
    threading.Thread(target=worker, name="mavis-observation-load", daemon=True).start()
    next_renew = time.monotonic() + 300
    while not done.wait(5):
        try:
            _owned_lease(purpose)
            _safe_game_state(_lease_command("status"))
            if loaded_generation_models(config.iris_endpoint) != iris_models:
                raise RuntimeError("IRIS generation inventory changed during Mavis load")
            if time.monotonic() >= next_renew:
                renew_gpu_lease()
                next_renew = time.monotonic() + 300
        except BaseException:
            stop_trial_server(config, server_state)
            stopped["yes"] = True
            done.wait(10)
            raise
    try:
        _owned_lease(purpose)
        _safe_game_state(_lease_command("status"))
    except BaseException:
        stop_trial_server(config, server_state)
        stopped["yes"] = True
        raise
    if failures:
        raise failures[0]


@contextmanager
def shared_mavis_model(
    config: RuntimeConfig, purpose: str
) -> Iterator[dict[str, object]]:
    """Load Mavis on a newly started server, then park only that server."""
    if not config.allow_concurrent_local:
        raise ValueError("shared Mavis observation requires explicit concurrent admission")
    status = _lease_command("status")
    if not status.startswith("the GPU is free\n"):
        raise RuntimeError("GPU lease is occupied")
    _safe_game_state(status)
    purpose = f"{purpose}:{uuid4().hex}"
    _lease_command("acquire", HOLDER, purpose, "90")
    token = _PURPOSE.set(purpose)
    result: dict[str, object] = {
        "gpu_lease_holder": HOLDER,
        "iris_stayed_loaded": False,
        "mavis_loaded": False,
        "gpu_lease_released": False,
    }
    server_state: dict[str, object] | None = None
    load_attempted = False
    stopped = {"yes": False}
    unsafe_startup = False
    iris_models: list[str] | None = None
    primary_error: BaseException | None = None
    try:
        _owned_lease(purpose)
        _safe_game_state(_lease_command("status"))
        iris_models = loaded_generation_models(config.iris_endpoint)
        if not iris_models:
            raise RuntimeError("IRIS has no loaded generation model to preserve")
        require_idle_iris_handoff(config, expected_models=iris_models)
        _owned_lease(purpose)
        _safe_game_state(_lease_command("status"))
        # This fails on a preexisting server, even one with Mavis settings.
        # The socket also detects a competing listener at bind time.
        reservation = reserve_empty_mavis_port(config)
        reservation.close()
        try:
            server_state = start_server(config, require_new=True)
        except BaseException as error:
            unsafe_startup = bool(getattr(error, "unsafe_gpu_work", False))
            raise
        server_pid = int(server_state["pid"])
        if read_json(config.state_path) != server_state:
            raise RuntimeError("Mavis server state changed during startup")
        if not any(row.get("id") == config.model for row in inventory(config.endpoint)):
            raise RuntimeError("Mavis selected model is absent from the new server")
        if loaded_generation_models(config.endpoint):
            raise RuntimeError("new Mavis server already has a generation model")
        _idle_mavis_server(config)
        _owned_lease(purpose)
        _safe_game_state(_lease_command("status"))
        load_attempted = True
        _monitored_load(config, purpose, server_state, stopped, iris_models)
        if loaded_generation_models(config.iris_endpoint) != iris_models:
            raise RuntimeError("IRIS changed while Mavis loaded")
        result["mavis_loaded"] = True
        yield result
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_error: BaseException | None = None
        gpu_stopped = (server_state is None and not unsafe_startup) or stopped["yes"]
        try:
            _owned_lease(purpose)
            _safe_game_state(_lease_command("status"))
            if server_state is not None and not stopped["yes"]:
                if read_json(config.state_path) != server_state:
                    raise RuntimeError("Mavis server identity changed before cleanup")
                if load_attempted and endpoint_alive(config.endpoint):
                    status = request_json(config.endpoint, "/api/status")
                    if (not isinstance(status, dict) or
                            type(status.get("models_loading")) is not int):
                        raise RuntimeError("Mavis model-loading status is untrusted")
                    if status["models_loading"]:
                        stop_trial_server(config, server_state)
                        stopped["yes"] = True
                    loaded = [] if stopped["yes"] else loaded_generation_models(config.endpoint)
                    if loaded:
                        if loaded != [config.model]:
                            raise RuntimeError("Mavis generation inventory changed before cleanup")
                        _idle_mavis_server(config)
                        request_json(
                            config.endpoint,
                            f"/v1/models/{quote(config.model, safe='')}/unload",
                            method="POST", timeout=180,
                        )
                        if loaded_generation_models(config.endpoint):
                            raise RuntimeError("Mavis model remained loaded after unload")
                if not stopped["yes"]:
                    reservation = park_mavis_server(
                        config, expected_pid=server_pid, expected_state=server_state
                    )
                    reservation.close()
                    if endpoint_alive(config.endpoint):
                        raise RuntimeError("Mavis did not park after the observation")
            gpu_stopped = not unsafe_startup
            result["mavis_loaded"] = False
            if iris_models is not None and loaded_generation_models(config.iris_endpoint) != iris_models:
                raise RuntimeError("IRIS lost its generation model during the observation")
            result["iris_stayed_loaded"] = iris_models is not None
        except BaseException as error:
            cleanup_error = error
            if server_state is not None and not stopped["yes"]:
                try:
                    stop_trial_server(config, server_state)
                    stopped["yes"] = True
                    gpu_stopped = True
                    result["mavis_loaded"] = False
                except BaseException as stop_error:
                    cleanup_error = RuntimeError(f"{error}; trial server abort failed: {stop_error}")
        finally:
            try:
                if gpu_stopped:
                    # A different Codex session may have acquired this holder name.
                    _owned_lease(purpose)
                    _lease_command("release", HOLDER)
                    result["gpu_lease_released"] = True
                else:
                    failure = cleanup_error or primary_error or RuntimeError(
                        "trial GPU work may still be active"
                    )
                    _record_failure_receipt(result, config, purpose, failure)
            except BaseException as error:
                cleanup_error = error if cleanup_error is None else RuntimeError(
                    f"{cleanup_error}; GPU lease release also failed: {error}"
                )
                _record_failure_receipt(result, config, purpose, cleanup_error)
            finally:
                _PURPOSE.reset(token)
        if not gpu_stopped:
            raise RuntimeError(
                f"Mavis observation may still hold GPU work ({cleanup_error or primary_error}); lease retained; "
                f"inventory receipt: {result.get('cleanup_failure_receipt')}"
            ) from cleanup_error
        if cleanup_error is not None:
            raise RuntimeError(
                f"Mavis observation cleanup failed: {cleanup_error}; "
                f"inventory receipt: {result.get('cleanup_failure_receipt')}"
            ) from cleanup_error
