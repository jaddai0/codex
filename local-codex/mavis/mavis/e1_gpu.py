"""Shared GPU admission for an installed native E1 trial."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, Callable, Iterator
from urllib.parse import quote
from uuid import uuid4

from .runtime import (
    RuntimeConfig,
    endpoint_alive,
    inventory,
    load_model,
    loaded_generation_models,
    park_mavis_server,
    request_json,
    require_idle_iris_handoff,
    reserve_empty_mavis_port,
    start_server,
    stop_trial_server,
)
from .storage import read_json, write_json


LEASE = Path.home() / ".local/bin/gpu-lease"
HOLDER = "codex-mavis"
LEASE_MINUTES = "90"


def _lease_command(*args: str) -> str:
    result = subprocess.run([str(LEASE), *args], capture_output=True, text=True,
                            timeout=8 if args[0] == "status" else 15,
                            check=False)
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
        self.abort_idle_server: Callable[[], None] | None = None

    def __call__(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if force or now >= self.next_status:
            _safe_status(_lease_command("status"), purpose=self.purpose)
            if (self.iris_endpoint is not None and
                    loaded_generation_models(self.iris_endpoint, timeout=0.5)
                    != self.iris_models):
                raise RuntimeError("IRIS generation model inventory changed during E1")
            self.next_status = time.monotonic() + 0.25
        if now >= self.next_renew:
            _owned_status(_lease_command("status"), self.purpose)
            _lease_command("renew", HOLDER, LEASE_MINUTES)
            _safe_status(_lease_command("status"), purpose=self.purpose)
            self.next_renew = now + 300

    def monitor_read_only(self, work: Callable[[], Any]) -> Any:
        """Keep the loaded idle server guarded during a long file hash."""
        self(force=True)
        if self.abort_idle_server is None:
            raise RuntimeError("E1 blocking-work server abort is unavailable")
        done = threading.Event()
        unsafe: list[BaseException] = []

        def watch() -> None:
            while not done.wait(0.2):
                try:
                    self(force=True)
                except BaseException as error:
                    try:
                        self.abort_idle_server()
                    except BaseException as stop_error:
                        unsafe.append(RuntimeError(
                            f"{error}; E1 idle server abort failed: {stop_error}"
                        ))
                    else:
                        unsafe.append(error)
                    return

        watcher = threading.Thread(target=watch, name="mavis-e1-binding-watch", daemon=True)
        watcher.start()
        result: Any = None
        work_error: BaseException | None = None
        try:
            result = work()
        except BaseException as error:
            work_error = error
        finally:
            done.set()
            watcher.join()
        if unsafe:
            raise unsafe[0] from work_error
        self(force=True)
        if work_error is not None:
            raise work_error
        return result


def _mavis_status(config: RuntimeConfig) -> dict[str, Any]:
    status = request_json(config.endpoint, "/api/status", timeout=1)
    if (not isinstance(status, dict) or status.get("status") != "ok"
            or any(type(status.get(key)) is not int or status[key] < 0 for key in
                   ("active_requests", "waiting_requests", "models_loading"))):
        raise RuntimeError("Mavis server work status is untrusted")
    return status


def _idle_mavis_server(config: RuntimeConfig) -> None:
    status = _mavis_status(config)
    if any(status[key] != 0 for key in
           ("active_requests", "waiting_requests", "models_loading")):
        raise RuntimeError("Mavis server has active, waiting, or loading work")


def _abort_trial(config: RuntimeConfig, state: dict[str, Any], *,
                 own_loading: bool = False) -> None:
    if endpoint_alive(config.endpoint, timeout=1):
        deadline = time.monotonic() + 2.5
        first_zero: float | None = None
        while True:
            status = _mavis_status(config)
            now = time.monotonic()
            if status["active_requests"] == status["waiting_requests"] == 0:
                if first_zero is not None and now - first_zero >= 0.2:
                    break
                first_zero = now
            else:
                first_zero = None
            if now >= deadline:
                raise RuntimeError(
                    "Mavis active or waiting requests did not drain after native child disconnect; "
                    "foreign work cannot be excluded"
                )
            time.sleep(min(0.2, max(0, deadline - now)))
        if status["models_loading"] and not own_loading:
            raise RuntimeError("Mavis has an unowned model load")
    stop_trial_server(config, state)


def _cleanup_failure(config: RuntimeConfig, purpose: str, error: BaseException,
                     *, gpu_work_stopped: bool, lease_released: bool) -> Path:
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
        "gpu_work_stopped": gpu_work_stopped, "lease_released": lease_released,
        "iris_endpoint": config.iris_endpoint,
        "iris_inventory": observed(config.iris_endpoint),
        "mavis_endpoint": config.endpoint,
        "mavis_inventory": observed(config.endpoint),
    })
    return path


def _monitored_load(config: RuntimeConfig, heartbeat: _LeaseHeartbeat,
                    server_state: dict[str, Any], stopped: dict[str, bool]) -> None:
    """Abort this trial's process group if IRIS becomes live during oMLX load."""
    done = threading.Event()
    failures: list[BaseException] = []
    def worker() -> None:
        try:
            load_model(config)
        except BaseException as error:
            failures.append(error)
        finally:
            done.set()
    threading.Thread(target=worker, name="mavis-e1-load", daemon=True).start()
    while not done.wait(0.2):
        try:
            heartbeat(force=True)
        except BaseException:
            _abort_trial(config, server_state, own_loading=True)
            stopped["yes"] = True
            done.wait(10)
            raise
    try:
        heartbeat(force=True)
    except BaseException:
        _abort_trial(config, server_state)
        stopped["yes"] = True
        raise
    if failures:
        raise failures[0]


@contextmanager
def admitted_e1_model(config: RuntimeConfig, purpose: str) -> Iterator[Callable[[], None]]:
    """Load only on a new trial-owned server; keep IRIS's model untouched."""
    config = replace(config, allow_concurrent_local=True, idle_seconds=900)
    purpose = f"{purpose}:{uuid4().hex}"
    _safe_status(_lease_command("status"))
    _lease_command("acquire", HOLDER, purpose, LEASE_MINUTES)
    heartbeat = _LeaseHeartbeat(purpose)
    iris_models: list[str] | None = None
    server_state: dict[str, Any] | None = None
    stopped = {"yes": False}
    load_attempted = False
    unsafe_startup = False
    primary_error: BaseException | None = None
    try:
        heartbeat(force=True)
        iris_models = loaded_generation_models(config.iris_endpoint)
        if not iris_models:
            raise RuntimeError("IRIS has no loaded generation model to preserve")
        require_idle_iris_handoff(config, expected_models=iris_models)
        heartbeat.iris_endpoint = config.iris_endpoint
        heartbeat.iris_models = iris_models
        heartbeat(force=True)
        # Never reuse a preexisting server, even one with Mavis's base path.
        reservation = reserve_empty_mavis_port(config)
        reservation.close()
        try:
            server_state = start_server(config, require_new=True)
        except BaseException as error:
            unsafe_startup = bool(getattr(error, "unsafe_gpu_work", False))
            raise
        if read_json(config.state_path) != server_state:
            raise RuntimeError("Mavis trial launch record changed during startup")
        if not any(row.get("id") == config.model for row in inventory(config.endpoint)):
            raise RuntimeError("selected Mavis model is absent from the new server")
        if loaded_generation_models(config.endpoint):
            raise RuntimeError("new Mavis server already has a generation model")
        _idle_mavis_server(config)
        heartbeat(force=True)
        load_attempted = True
        _monitored_load(config, heartbeat, server_state, stopped)
        if loaded_generation_models(config.iris_endpoint) != iris_models:
            raise RuntimeError("IRIS changed while Mavis loaded")
        def abort_idle_server() -> None:
            if stopped["yes"]:
                return
            _abort_trial(config, server_state)
            stopped["yes"] = True
        heartbeat.abort_idle_server = abort_idle_server
        yield heartbeat
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_error: BaseException | None = None
        gpu_stopped = (server_state is None and not unsafe_startup) or stopped["yes"]
        lease_released = False
        receipt: Path | None = None
        try:
            _safe_status(_lease_command("status"), purpose=purpose)
            if server_state is not None and not stopped["yes"]:
                if read_json(config.state_path) != server_state:
                    raise RuntimeError("Mavis trial launch record changed before cleanup")
                if load_attempted and endpoint_alive(config.endpoint):
                    status = _mavis_status(config)
                    if status["models_loading"]:
                        _abort_trial(config, server_state, own_loading=True)
                        stopped["yes"] = True
                    if not stopped["yes"]:
                        loaded = loaded_generation_models(config.endpoint)
                        if loaded:
                            if loaded != [config.model]:
                                raise RuntimeError("Mavis generation inventory changed during E1")
                            _idle_mavis_server(config)
                            for attempt in range(2):
                                try:
                                    request_json(config.endpoint,
                                                 f"/v1/models/{quote(config.model, safe='')}/unload",
                                                 method="POST", timeout=180)
                                except (OSError, RuntimeError) as error:
                                    if not loaded_generation_models(config.endpoint):
                                        break
                                    if attempt == 1:
                                        raise RuntimeError("Mavis E1 model unload failed twice") from error
                                if not loaded_generation_models(config.endpoint):
                                    break
                            else:
                                raise RuntimeError("Mavis model remained loaded after E1 cleanup")
                if not stopped["yes"]:
                    reservation = park_mavis_server(
                        config, expected_pid=int(server_state["pid"]),
                        expected_state=server_state,
                    )
                    reservation.close()
                    if endpoint_alive(config.endpoint):
                        raise RuntimeError("Mavis server remained live after E1 cleanup")
            gpu_stopped = not unsafe_startup
            if iris_models is not None and loaded_generation_models(config.iris_endpoint) != iris_models:
                raise RuntimeError("IRIS lost its original generation model during E1")
        except BaseException as error:
            cleanup_error = error
            if server_state is not None and not stopped["yes"]:
                try:
                    _abort_trial(config, server_state, own_loading=load_attempted and
                                 endpoint_alive(config.endpoint) and
                                 _mavis_status(config)["models_loading"] > 0)
                    stopped["yes"] = True
                    gpu_stopped = True
                except BaseException as stop_error:
                    cleanup_error = RuntimeError(f"{error}; trial server abort failed: {stop_error}")
        finally:
            if gpu_stopped:
                try:
                    _owned_status(_lease_command("status"), purpose)
                    _lease_command("release", HOLDER)
                    lease_released = True
                except BaseException as error:
                    cleanup_error = error if cleanup_error is None else RuntimeError(
                        f"{cleanup_error}; GPU lease release also failed: {error}"
                    )
            if cleanup_error is not None or not gpu_stopped or not lease_released:
                failure = cleanup_error or primary_error or RuntimeError(
                    "E1 trial GPU work may still be active"
                )
                try:
                    receipt = _cleanup_failure(config, purpose, failure,
                                               gpu_work_stopped=gpu_stopped,
                                               lease_released=lease_released)
                except BaseException as error:
                    cleanup_error = RuntimeError(f"{failure}; cleanup receipt failed: {error}")
        if not gpu_stopped:
            raise RuntimeError(
                f"E1 trial GPU work may still be active ({cleanup_error or primary_error}); "
                f"lease retained; inventory receipt: {receipt}"
            ) from cleanup_error
        if cleanup_error is not None:
            raise RuntimeError(
                f"E1 GPU cleanup failed: {cleanup_error}; inventory receipt: {receipt}"
            ) from cleanup_error
