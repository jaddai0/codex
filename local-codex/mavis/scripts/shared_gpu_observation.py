"""Run one installed Mavis observation while IRIS keeps its model loaded."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Callable, Iterator
from urllib.error import HTTPError
from urllib.parse import quote
from uuid import uuid4

from mavis.runtime import (
    RuntimeConfig,
    _own_api,
    _own_headers,
    _child_exit_unreaped,
    _process_group_workers,
    _stop_spawned_process_group,
    clear_trial_auth_settings,
    endpoint_alive,
    inventory,
    loaded_generation_models,
    load_model,
    park_mavis_server,
    port_in_use,
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
_EMERGENCY_ABORT: ContextVar[Callable[[], None] | None] = ContextVar(
    "mavis_observation_emergency_abort", default=None,
)


class UnsafeSharedGPU(RuntimeError):
    """The trial can no longer safely continue shared GPU work."""


def _lease_command(*args: str) -> str:
    run = subprocess.run(
        [str(LEASE), *args], capture_output=True, text=True,
        timeout=0.5 if args[0] == "status" else 15, check=False,
    )
    if run.returncode:
        raise RuntimeError(
            f"GPU lease {args[0]} failed: {(run.stderr or run.stdout).strip()}"
        )
    return run.stdout


def _owned_lease(purpose: str) -> str:
    status = _lease_command("status")
    first = status.splitlines()[0] if status.splitlines() else ""
    if not first.startswith(f"{HOLDER} has the GPU: {purpose} ("):
        raise RuntimeError("Mavis observation GPU lease purpose changed")
    return status


def _safe_game_state(status: str) -> None:
    if "IRIS: a game is live" in status or "IRIS: not answering" in status:
        raise RuntimeError("IRIS game state is unsafe for Mavis observation")


def _heartbeat(config: RuntimeConfig, purpose: str,
               iris_models: list[str] | None = None) -> None:
    try:
        _safe_game_state(_owned_lease(purpose))
        if (iris_models is not None and
                loaded_generation_models(config.iris_endpoint, timeout=0.5) != iris_models):
            raise RuntimeError("IRIS generation inventory changed during Mavis observation")
    except BaseException as error:
        raise UnsafeSharedGPU(str(error)) from error


def renew_gpu_lease() -> None:
    """Extend only this observation's still-owned lease."""
    purpose = _PURPOSE.get()
    if purpose is None:
        raise RuntimeError("no active Mavis observation lease to renew")
    try:
        _safe_game_state(_owned_lease(purpose))
        _lease_command("renew", HOLDER, "90")
        _owned_lease(purpose)
    except BaseException as error:
        raise UnsafeSharedGPU(str(error)) from error


def _idle_mavis_server(config: RuntimeConfig) -> None:
    status = request_json(config.endpoint, "/api/status", timeout=0.5,
                          **_own_headers(config))
    if (not isinstance(status, dict) or status.get("status") != "ok"
            or any(type(status.get(key)) is not int or status[key] != 0 for key in
                   ("active_requests", "waiting_requests", "models_loading"))):
        raise RuntimeError("Mavis server has active, waiting, or loading work")


def _drained_request_status(config: RuntimeConfig, *, wait_seconds: float) -> dict[str, object]:
    """Give a disconnected client a bounded chance to cancel its own request."""
    deadline = time.monotonic() + wait_seconds
    first_zero: float | None = None
    while True:
        purpose = _PURPOSE.get()
        if purpose is None:
            raise RuntimeError("Mavis observation lease is absent while requests drain")
        _heartbeat(config, purpose)
        status = request_json(config.endpoint, "/api/status", timeout=0.5,
                              **_own_headers(config))
        _heartbeat(config, purpose)
        if (not isinstance(status, dict) or status.get("status") != "ok"
                or any(type(status.get(key)) is not int or status[key] < 0 for key in
                       ("active_requests", "waiting_requests", "models_loading"))):
            raise RuntimeError("Mavis server work status is untrusted before abort")
        now = time.monotonic()
        if status["active_requests"] == status["waiting_requests"] == 0:
            if wait_seconds == 0 or (first_zero is not None and now - first_zero >= 0.2):
                return status
            first_zero = now
        else:
            first_zero = None
        if now >= deadline:
            raise RuntimeError(
                "Mavis active or waiting requests did not drain after observation child disconnect; "
                "foreign work cannot be excluded"
            )
        time.sleep(min(0.2, max(0, deadline - time.monotonic())))


def _abort_observation_server(config: RuntimeConfig, state: dict[str, object], *,
                              own_loading: bool = False,
                              own_generation: bool = False,
                              own_unload: bool = False,
                              emergency: bool = False,
                              stopped: dict[str, bool] | None = None) -> None:
    if config.api_key:
        settings = json.loads((config.base_path / "settings.json").read_text())
        auth = settings.get("auth") if isinstance(settings, dict) else None
        if (not isinstance(auth, dict) or auth.get("api_key") != config.api_key or
                auth.get("skip_api_key_verification") is not False):
            raise RuntimeError("Mavis trial authentication proof changed")
    if emergency and not config.api_key:
        raise RuntimeError("emergency abort requires a private trial server")
    if not emergency and endpoint_alive(config.endpoint, timeout=1, **_own_api(config)):
        status = _drained_request_status(
            config, wait_seconds=2.5 if (own_generation or own_unload) else 0,
        )
        if status["models_loading"] > (1 if own_loading else 0):
            raise RuntimeError("Mavis has a model load outside this observation")
    elif not emergency and port_in_use(config.endpoint):
        raise RuntimeError("Mavis occupied endpoint failed authenticated status check")
    stop_trial_server(config, state)
    if stopped is not None:
        stopped["yes"] = True
    if config.api_key:
        reservation = reserve_empty_mavis_port(config)
        try:
            clear_trial_auth_settings(config)
        finally:
            reservation.close()


def _prove_trial_auth(config: RuntimeConfig) -> None:
    if not config.api_key:
        return
    try:
        request_json(config.endpoint, "/api/status", timeout=0.5)
    except HTTPError as error:
        if error.code != 401:
            raise RuntimeError("Mavis trial unauthenticated probe had unexpected status") from error
    else:
        raise RuntimeError("Mavis trial server accepted a request without its key")
    status = request_json(config.endpoint, "/api/status", timeout=0.5,
                          **_own_headers(config))
    if not isinstance(status, dict) or status.get("status") != "ok":
        raise RuntimeError("Mavis trial authenticated status is untrusted")


def _failure_receipt(config: RuntimeConfig, purpose: str,
                     error: BaseException, result: dict[str, object] | None = None) -> Path:
    def observed(endpoint: str) -> object:
        try:
            return inventory(endpoint, timeout=0.5,
                             **(_own_api(config) if endpoint == config.endpoint else {}))
        except BaseException as failure:
            return {"inventory_error": str(failure)}
    path = config.home / "evaluations" / "shared-gpu-failures" / f"{uuid4().hex}.json"
    write_json(path, {
        "schema_version": "mavis.shared-gpu-cleanup-failure/v1",
        "at_epoch": time.time(), "gpu_lease_holder": HOLDER,
        "gpu_lease_purpose": purpose, "error": str(error),
        "observation_child": {
            "pid": result.get("observation_child_pid"),
            "safe": result.get("observation_child_safe"),
            "group_remained": result.get("observation_child_group_remained", False),
            "cleanup_error": result.get("observation_child_cleanup_error"),
        } if result is not None else None,
        "iris_inventory": observed(config.iris_endpoint),
        "mavis_inventory": observed(config.endpoint),
    })
    return path


def _record_failure_receipt(result: dict[str, object], config: RuntimeConfig,
                            purpose: str, error: BaseException) -> None:
    if "cleanup_failure_receipt" in result:
        return
    try:
        result["cleanup_failure_receipt"] = str(_failure_receipt(config, purpose, error, result))
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
    while not done.wait(0.2):
        try:
            _heartbeat(config, purpose, iris_models)
            if time.monotonic() >= next_renew:
                renew_gpu_lease()
                next_renew = time.monotonic() + 300
        except BaseException:
            _abort_observation_server(config, server_state, emergency=True,
                                      stopped=stopped)
            done.wait(10)
            raise
    try:
        _heartbeat(config, purpose, iris_models)
    except BaseException:
        _abort_observation_server(config, server_state, emergency=True, stopped=stopped)
        raise
    if failures:
        raise failures[0]


def _monitored_unload(config: RuntimeConfig, purpose: str,
                      server_state: dict[str, object], stopped: dict[str, bool],
                      iris_models: list[str], *, own_generation: bool) -> None:
    """Watch oMLX's shielded teardown until the model is gone or our server stops."""
    done = threading.Event()
    failures: list[BaseException] = []
    def worker() -> None:
        try:
            request_json(
                config.endpoint,
                f"/v1/models/{quote(config.model, safe='')}/unload",
                method="POST", timeout=180, **_own_headers(config),
            )
        except BaseException as error:
            failures.append(error)
        finally:
            done.set()
    threading.Thread(target=worker, name="mavis-observation-unload", daemon=True).start()
    next_renew = time.monotonic() + 300
    while not done.wait(0.2):
        try:
            _heartbeat(config, purpose, iris_models)
            if time.monotonic() >= next_renew:
                renew_gpu_lease()
                next_renew = time.monotonic() + 300
        except BaseException:
            _abort_observation_server(config, server_state, emergency=True, stopped=stopped)
            done.wait(10)
            raise
    try:
        _heartbeat(config, purpose, iris_models)
    except BaseException:
        _abort_observation_server(config, server_state, emergency=True, stopped=stopped)
        raise
    if failures:
        raise failures[0]


def run_monitored_observation(config: RuntimeConfig, command: list[str], *,
                              cwd: Path, env: dict[str, str], stdout: object,
                              shared: dict[str, object], timeout: float = 1800) -> int:
    """Watch an exact E0 child while its model requests can hold the GPU."""
    purpose = _PURPOSE.get()
    if purpose is None or shared.get("iris_generation_models") is None:
        raise RuntimeError("E0 child has no active observation lease and IRIS snapshot")
    expected_iris = shared["iris_generation_models"]
    if not isinstance(expected_iris, list):
        raise RuntimeError("E0 IRIS generation snapshot is invalid")
    _heartbeat(config, purpose, expected_iris)
    process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                               stdout=stdout, stderr=subprocess.STDOUT,
                               start_new_session=True)
    shared["observation_child_pid"] = process.pid
    deadline = time.monotonic() + timeout
    next_renew = time.monotonic() + 300
    try:
        while True:
            time.sleep(min(0.2, max(0.01, deadline - time.monotonic())))
            exited = _child_exit_unreaped(process)
            _heartbeat(config, purpose, expected_iris)
            if time.monotonic() >= next_renew:
                renew_gpu_lease()
                next_renew = time.monotonic() + 300
            if exited is not None:
                # The tool roundtrip must not leave a child in its dedicated
                # process group after the observed command exits.
                if _process_group_workers(process.pid):
                    shared["observation_child_group_remained"] = True
                    raise RuntimeError("E0 observation child group remained after command exit")
                return process.wait(timeout=0)
            if time.monotonic() >= deadline:
                raise TimeoutError("E0 observation child exceeded its deadline")
    except BaseException as error:
        if isinstance(error, UnsafeSharedGPU):
            abort = _EMERGENCY_ABORT.get()
            if abort is not None:
                try:
                    abort()
                except BaseException as abort_error:
                    shared["emergency_abort_error"] = str(abort_error)
        if process.returncode is None:
            try:
                _stop_spawned_process_group(process)
            except BaseException as cleanup_error:
                shared["observation_child_safe"] = False
                shared["observation_child_cleanup_error"] = str(cleanup_error)
                raise
        raise


@contextmanager
def shared_mavis_model(
    config: RuntimeConfig, purpose: str
) -> Iterator[dict[str, object]]:
    """Load Mavis on a newly started server, then park only that server."""
    if not config.allow_concurrent_local:
        raise ValueError("shared Mavis observation requires explicit concurrent admission")
    if not config.api_key:
        raise ValueError("shared Mavis observation requires a private trial key")
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
        "observation_child_safe": True,
    }
    server_state: dict[str, object] | None = None
    load_attempted = False
    stopped = {"yes": False}
    server_stopped = False
    unsafe_startup = False
    residual_auth = False
    start_attempted = False
    iris_models: list[str] | None = None
    primary_error: BaseException | None = None
    try:
        _heartbeat(config, purpose)
        iris_models = loaded_generation_models(config.iris_endpoint)
        if not iris_models:
            raise RuntimeError("IRIS has no loaded generation model to preserve")
        require_idle_iris_handoff(config, expected_models=iris_models)
        _heartbeat(config, purpose, iris_models)
        # This fails on a preexisting server, even one with Mavis settings.
        # The socket also detects a competing listener at bind time.
        reservation = reserve_empty_mavis_port(config)
        reservation.close()
        try:
            start_attempted = True
            server_state = start_server(
                config, require_new=True,
                heartbeat=lambda: _heartbeat(config, purpose, iris_models),
            )
        except BaseException as error:
            unsafe_startup = bool(getattr(error, "unsafe_gpu_work", False))
            residual_auth = bool(getattr(error, "residual_trial_auth", False))
            raise
        server_pid = int(server_state["pid"])
        if read_json(config.state_path) != server_state:
            raise RuntimeError("Mavis server state changed during startup")
        _heartbeat(config, purpose, iris_models)
        _prove_trial_auth(config)
        _heartbeat(config, purpose, iris_models)
        if not any(row.get("id") == config.model for row in
                   inventory(config.endpoint, timeout=0.5, **_own_api(config))):
            raise RuntimeError("Mavis selected model is absent from the new server")
        _heartbeat(config, purpose, iris_models)
        if loaded_generation_models(config.endpoint, timeout=0.5, **_own_api(config)):
            raise RuntimeError("new Mavis server already has a generation model")
        _heartbeat(config, purpose, iris_models)
        _idle_mavis_server(config)
        _heartbeat(config, purpose, iris_models)
        load_attempted = True
        _monitored_load(config, purpose, server_state, stopped, iris_models)
        _heartbeat(config, purpose, iris_models)
        result["iris_generation_models"] = iris_models
        result["mavis_loaded"] = True
        abort_token = _EMERGENCY_ABORT.set(
            lambda: _abort_observation_server(
                config, server_state, emergency=True, stopped=stopped,
            )
        )
        try:
            yield result
        finally:
            _EMERGENCY_ABORT.reset(abort_token)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_error: BaseException | None = None
        gpu_stopped = ((server_state is None and not unsafe_startup) or stopped["yes"])
        try:
            if server_state is not None and not stopped["yes"] and isinstance(
                primary_error, UnsafeSharedGPU
            ):
                _abort_observation_server(
                    config, server_state, emergency=True, stopped=stopped,
                )
            if (server_state is None and start_attempted and not unsafe_startup and
                    not residual_auth and config.api_key):
                reservation = reserve_empty_mavis_port(config)
                try:
                    clear_trial_auth_settings(config)
                finally:
                    reservation.close()
            _heartbeat(config, purpose, iris_models)
            if server_state is not None and not stopped["yes"]:
                if read_json(config.state_path) != server_state:
                    raise RuntimeError("Mavis server identity changed before cleanup")
                if load_attempted and endpoint_alive(config.endpoint, timeout=0.5,
                                                     **_own_api(config)):
                    _heartbeat(config, purpose, iris_models)
                    status = request_json(config.endpoint, "/api/status", timeout=0.5,
                                          **_own_headers(config))
                    _heartbeat(config, purpose, iris_models)
                    if (not isinstance(status, dict) or
                            type(status.get("models_loading")) is not int):
                        raise RuntimeError("Mavis model-loading status is untrusted")
                    if status["models_loading"]:
                        _abort_observation_server(config, server_state, own_loading=True,
                                                  stopped=stopped)
                    loaded = [] if stopped["yes"] else loaded_generation_models(
                        config.endpoint, timeout=0.5, **_own_api(config))
                    _heartbeat(config, purpose, iris_models)
                    if loaded:
                        if loaded != [config.model]:
                            raise RuntimeError("Mavis generation inventory changed before cleanup")
                        _idle_mavis_server(config)
                        _heartbeat(config, purpose, iris_models)
                        _monitored_unload(
                            config, purpose, server_state, stopped, iris_models or [],
                            own_generation=(config.api_key is not None and
                                            result.get("observation_child_pid") is not None and
                                            result.get("observation_child_safe") is True),
                        )
                        if loaded_generation_models(config.endpoint, timeout=0.5,
                                                    **_own_api(config)):
                            raise RuntimeError("Mavis model remained loaded after unload")
                        _heartbeat(config, purpose, iris_models)
                if not stopped["yes"]:
                    _heartbeat(config, purpose, iris_models)
                    park_unsafe: list[UnsafeSharedGPU] = []
                    def park_heartbeat() -> bool:
                        try:
                            _heartbeat(config, purpose, iris_models)
                        except UnsafeSharedGPU as error:
                            park_unsafe.append(error)
                            return False
                        return True
                    reservation = park_mavis_server(
                        config, expected_pid=server_pid, expected_state=server_state,
                        heartbeat=park_heartbeat,
                    )
                    server_stopped = True
                    try:
                        if config.api_key:
                            clear_trial_auth_settings(config)
                    finally:
                        reservation.close()
                    if park_unsafe:
                        raise park_unsafe[0]
                    if endpoint_alive(config.endpoint, timeout=0.5,
                                      **_own_api(config)):
                        raise RuntimeError("Mavis did not park after the observation")
            gpu_stopped = not unsafe_startup and result["observation_child_safe"] is True
            result["mavis_loaded"] = False
            if iris_models is not None and loaded_generation_models(
                config.iris_endpoint, timeout=0.5,
            ) != iris_models:
                raise RuntimeError("IRIS lost its generation model during the observation")
            result["iris_stayed_loaded"] = iris_models is not None
        except BaseException as error:
            cleanup_error = error
            if getattr(error, "gpu_work_stopped", False):
                server_stopped = True
            if server_stopped or stopped["yes"]:
                gpu_stopped = result["observation_child_safe"] is True
                result["mavis_loaded"] = False
            if server_state is not None and not stopped["yes"] and not server_stopped:
                try:
                    _abort_observation_server(
                        config, server_state,
                        emergency=(isinstance(error, UnsafeSharedGPU) or
                                   isinstance(primary_error, UnsafeSharedGPU)),
                        own_generation=(config.api_key is not None and
                                        result.get("observation_child_pid") is not None and
                                        result.get("observation_child_safe") is True),
                        stopped=stopped,
                    )
                    gpu_stopped = result["observation_child_safe"] is True
                    result["mavis_loaded"] = False
                except BaseException as stop_error:
                    if isinstance(stop_error, UnsafeSharedGPU):
                        try:
                            _abort_observation_server(
                                config, server_state, emergency=True, stopped=stopped,
                            )
                        except BaseException as emergency_error:
                            stop_error = RuntimeError(
                                f"{stop_error}; emergency stop failed: {emergency_error}"
                            )
                        else:
                            gpu_stopped = result["observation_child_safe"] is True
                            result["mavis_loaded"] = False
                    detail = "authentication cleanup" if stopped["yes"] else "abort"
                    cleanup_error = RuntimeError(f"{error}; trial server {detail} failed: {stop_error}")
                    if stopped["yes"]:
                        gpu_stopped = result["observation_child_safe"] is True
                        result["mavis_loaded"] = False
        finally:
            try:
                if cleanup_error is not None:
                    _record_failure_receipt(result, config, purpose, cleanup_error)
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
