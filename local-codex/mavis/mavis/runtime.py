"""Isolated oMLX lifecycle and model admission for Mavis."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
import fcntl
from functools import wraps
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import time
from typing import Any, Callable
import uuid
from urllib.error import URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from .storage import read_json, write_json


DEFAULT_MODEL = "Qwen3.8-Flash-Next-Abliterated-MLX-4bit"
_HANDOFF_LEASE_FD: ContextVar[int | None] = ContextVar("mavis_handoff_lease_fd", default=None)


@dataclass(frozen=True)
class RuntimeConfig:
    home: Path
    endpoint: str = "http://127.0.0.1:8001/v1"
    iris_endpoint: str = "http://127.0.0.1:8000/v1"
    model_dir: Path = Path("/Users/dustinpainter/models/vlms")
    omlx_binary: Path = Path("/Users/dustinpainter/.venvs/omlx-dev/bin/omlx")
    model: str = DEFAULT_MODEL
    idle_seconds: int = 900
    reserve_bytes: int = 64 * 1024**3
    allow_concurrent_local: bool = False

    @property
    def base_path(self) -> Path:
        return self.home / "omlx"

    @property
    def state_path(self) -> Path:
        return self.home / "runtime.json"


def _origin(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Mavis oMLX endpoint must be loopback HTTP")
    if parsed.path.rstrip("/") != "/v1":
        raise ValueError("Mavis oMLX endpoint must end in /v1")
    return endpoint.rstrip("/").removesuffix("/v1")


def request_json(
    endpoint: str, path: str, method: str = "GET", timeout: float = 10,
    *, headers: dict[str, str] | None = None, payload: dict[str, Any] | None = None,
) -> Any:
    request_headers = {"Accept": "application/json", **(headers or {})}
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    request = Request(_origin(endpoint) + path, data=body, method=method,
                      headers=request_headers)
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def inventory(endpoint: str) -> list[dict[str, Any]]:
    payload = request_json(endpoint, "/admin/api/models")
    records = payload.get("models", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise RuntimeError("oMLX returned an invalid model inventory")
    return records


def loaded_generation_models(endpoint: str) -> list[str]:
    """Require a readable inventory and name every loaded non-embedding model."""
    models = []
    for row in inventory(endpoint):
        if row.get("loaded") is not True:
            continue
        if row.get("engine_type") == "embedding" or row.get("model_type") == "embedding":
            continue
        model_id = row.get("id")
        if not isinstance(model_id, str) or not model_id:
            raise RuntimeError("loaded generation model lacks an ID")
        models.append(model_id)
    return models


def endpoint_alive(endpoint: str) -> bool:
    try:
        inventory(endpoint)
        return True
    except (OSError, URLError, RuntimeError, ValueError, json.JSONDecodeError):
        return False


def require_idle_iris_handoff(config: RuntimeConfig, *, interval_seconds: float = 1.0) -> None:
    """Refuse a model handoff while IRIS has active or queued generation."""
    for sample in range(2):
        status = request_json(config.iris_endpoint, "/api/status")
        if (not isinstance(status, dict) or status.get("status") != "ok"
            or not isinstance(status.get("loaded_models"), list)
            or config.model not in status["loaded_models"]
            or type(status.get("models_loading")) is not int
            or status["models_loading"] != 0
            or type(status.get("active_requests")) is not int
            or type(status.get("waiting_requests")) is not int
            or status["active_requests"] != 0
            or status["waiting_requests"] != 0):
            raise RuntimeError("IRIS has active or waiting work; model handoff refused")
        if sample == 0:
            time.sleep(interval_seconds)
    if loaded_generation_models(config.iris_endpoint) != [config.model]:
        raise RuntimeError("IRIS generation model inventory changed before handoff")


def require_installed_selected_model(config: RuntimeConfig) -> None:
    """Prove the installed launcher will select the model being handed off."""
    prepare = Path.home() / ".local" / "share" / "local-codex" / "prepare_runtime.py"
    result = subprocess.run(
        [sys.executable, str(prepare), "--mavis-home", str(config.home), "--resolve-model"],
        text=True, capture_output=True, timeout=15, check=True,
    )
    if result.stdout.strip() != config.model:
        raise RuntimeError("installed launcher selects a different Mavis model")


def _iris_drain_token() -> str:
    path = Path.home() / ".omlx" / "model-drain-token"
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        details = os.fstat(descriptor)
        if (not stat.S_ISREG(details.st_mode) or details.st_uid != os.getuid()
                or stat.S_IMODE(details.st_mode) != 0o600):
            raise RuntimeError("IRIS drain token has unsafe ownership or permissions")
        token = os.read(descriptor, 256).decode("ascii").strip()
        if not token or len(token) > 200:
            raise RuntimeError("IRIS drain token is empty or malformed")
        return token
    finally:
        os.close(descriptor)


def iris_drain_headers(lease_id: str | None = None) -> dict[str, str]:
    headers = {"X-OMLX-Drain-Token": _iris_drain_token()}
    if lease_id is not None:
        uuid.UUID(lease_id)
        headers["X-OMLX-Drain-Lease-Id"] = lease_id
    return headers


def _drain_path(config: RuntimeConfig) -> str:
    return f"/admin/api/model-drains/{quote(config.model, safe='')}"


def acquire_iris_model_drain(config: RuntimeConfig, *, owner: str) -> str:
    if not owner or len(owner) > 80:
        raise ValueError("IRIS drain owner must be a short name")
    result = request_json(config.iris_endpoint, _drain_path(config) + "/acquire",
                          method="POST", headers=iris_drain_headers(),
                          payload={"owner": owner})
    if (not isinstance(result, dict)
            or result.get("schema_version") != "omlx.model-drain/v1"
            or result.get("model_id") != config.model
            or result.get("state") not in {"draining", "drained"}):
        raise RuntimeError("IRIS model drain returned an invalid lease")
    lease_id = result.get("lease_id")
    if not isinstance(lease_id, str):
        raise RuntimeError("IRIS model drain returned no lease ID")
    uuid.UUID(lease_id)
    return lease_id


def wait_iris_model_drain(config: RuntimeConfig, lease_id: str, *, timeout: float = 180) -> None:
    uuid.UUID(lease_id)
    deadline = time.monotonic() + timeout
    path = _drain_path(config) + f"/status?lease_id={quote(lease_id, safe='')}"
    while time.monotonic() < deadline:
        result = request_json(config.iris_endpoint, path, headers=iris_drain_headers())
        if (not isinstance(result, dict)
                or result.get("schema_version") != "omlx.model-drain/v1"
                or result.get("model_id") != config.model
                or result.get("lease_id") != lease_id
                or result.get("state") not in {"draining", "drained"}):
            raise RuntimeError("IRIS model drain status changed unexpectedly")
        if result["state"] == "drained":
            return
        time.sleep(0.2)
    raise TimeoutError("IRIS model did not drain before handoff")


def release_iris_model_drain(config: RuntimeConfig, lease_id: str) -> None:
    uuid.UUID(lease_id)
    result = request_json(config.iris_endpoint, _drain_path(config) + "/release",
                          method="POST", headers=iris_drain_headers(),
                          payload={"lease_id": lease_id})
    if (not isinstance(result, dict)
            or result.get("schema_version") != "omlx.model-drain/v1"
            or result.get("model_id") != config.model
            or result.get("lease_id") != lease_id
            or result.get("state") != "released"):
        raise RuntimeError("IRIS model drain release was not confirmed")


@contextmanager
def mavis_generation_lease(config: RuntimeConfig, *, purpose: str):
    """Hold the same host lock as the installed launcher for a full handoff."""
    config.home.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(config.home / "generation.lock",
                         os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another Mavis local generation owns the host lease") from error
        os.ftruncate(descriptor, 0)
        os.write(descriptor, (json.dumps({"pid": os.getpid(), "purpose": purpose,
                                          "acquired_at_epoch": time.time()}) + "\n").encode())
        os.fsync(descriptor)
        context_token = _HANDOFF_LEASE_FD.set(descriptor)
        try:
            yield
        finally:
            _HANDOFF_LEASE_FD.reset(context_token)
            os.ftruncate(descriptor, 0)
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def handoff_lease_fd() -> int:
    descriptor = _HANDOFF_LEASE_FD.get()
    if descriptor is None:
        raise RuntimeError("Mavis handoff lease is not held")
    return descriptor


def with_mavis_handoff_lease(purpose: str) -> Callable:
    """Protect an installed observer from concurrent Mavis launcher startup."""
    def decorate(function: Callable) -> Callable:
        @wraps(function)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            config = RuntimeConfig(home=Path.home() / ".local-codex" / "mavis-service")
            with mavis_generation_lease(config, purpose=purpose):
                return function(*args, **kwargs)
        return guarded
    return decorate


def _port(endpoint: str) -> int:
    parsed = urlparse(endpoint)
    return parsed.port or 80


def port_in_use(endpoint: str) -> bool:
    parsed = urlparse(endpoint)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.3)
        return probe.connect_ex((parsed.hostname or "127.0.0.1", _port(endpoint))) == 0


def _listener_pids(port: int) -> set[int]:
    result = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fp"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return {
        int(line[1:])
        for line in result.stdout.splitlines()
        if line.startswith("p") and line[1:].isdigit()
    }


def _process_open_paths(pid: int) -> set[Path]:
    result = subprocess.run(
        ["lsof", "-p", str(pid), "-Fn"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return {
        Path(line[1:])
        for line in result.stdout.splitlines()
        if line.startswith("n") and line[1:].startswith("/")
    }


def owns_running_server(config: RuntimeConfig) -> bool:
    if not config.state_path.exists():
        return False
    state = read_json(config.state_path)
    pid = state.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    if pid not in _listener_pids(_port(config.endpoint)):
        return False
    base_path = config.base_path.resolve()
    return any(
        base_path == path.resolve() or base_path in path.resolve().parents
        for path in _process_open_paths(pid)
    )


def ensure_isolated_settings(config: RuntimeConfig) -> None:
    """Pin loopback auth and every mutable path to Mavis's base directory."""
    settings_path = config.base_path / "settings.json"
    if settings_path.exists():
        settings = read_json(settings_path)
    else:
        settings = {"version": "1.0"}
    auth = settings.setdefault("auth", {})
    if not isinstance(auth, dict):
        raise ValueError("Mavis oMLX auth settings must be an object")
    auth["skip_api_key_verification"] = True
    server = settings.setdefault("server", {})
    if not isinstance(server, dict):
        raise ValueError("Mavis oMLX server settings must be an object")
    server.update({"host": "127.0.0.1", "port": _port(config.endpoint)})
    model = settings.setdefault("model", {})
    if not isinstance(model, dict):
        raise ValueError("Mavis oMLX model settings must be an object")
    model.update({"model_dir": str(config.model_dir), "model_dirs": [str(config.model_dir)]})
    cache = settings.setdefault("cache", {})
    if not isinstance(cache, dict):
        raise ValueError("Mavis oMLX cache settings must be an object")
    cache.update({"enabled": True, "ssd_cache_dir": str(config.base_path / "cache")})
    write_json(settings_path, settings)


def start_server(config: RuntimeConfig, wait_seconds: float = 30.0) -> dict[str, Any]:
    if endpoint_alive(config.endpoint):
        if not owns_running_server(config):
            raise RuntimeError("Mavis endpoint is occupied by a server Mavis does not own")
        return read_json(config.state_path)
    if port_in_use(config.endpoint):
        raise RuntimeError("Mavis port is occupied by a non-oMLX process")
    if not config.omlx_binary.is_file():
        raise FileNotFoundError(config.omlx_binary)

    logs = config.home / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    config.base_path.mkdir(parents=True, exist_ok=True)
    ensure_isolated_settings(config)
    command = [
        str(config.omlx_binary),
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(_port(config.endpoint)),
        "--base-path",
        str(config.base_path),
        "--model-dir",
        str(config.model_dir),
        "--paged-ssd-cache-dir",
        str(config.base_path / "cache"),
        "--max-concurrent-requests",
        "1",
        "--memory-guard",
        "safe",
        "--no-hf-cache",
    ]
    stdout = (logs / "omlx.stdout.log").open("ab")
    stderr = (logs / "omlx.stderr.log").open("ab")
    try:
        child_env = os.environ.copy()
        child_env["OMLX_BASE_PATH"] = str(config.base_path)
        isolated_home = config.home / "user-home"
        isolated_home.mkdir(parents=True, exist_ok=True)
        child_env["HOME"] = str(isolated_home)
        child_env["XDG_CACHE_HOME"] = str(config.home / "cache")
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            env=child_env,
        )
    finally:
        stdout.close()
        stderr.close()
    state = {
        "schema_version": "mavis.runtime/v1",
        "pid": process.pid,
        "endpoint": config.endpoint,
        "base_path": str(config.base_path),
        "command": command,
        "started_at_epoch": time.time(),
    }
    write_json(config.state_path, state)
    deadline = time.monotonic() + wait_seconds
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Mavis oMLX exited during startup with status {process.returncode}"
                )
            if endpoint_alive(config.endpoint):
                return state
            time.sleep(0.25)
        raise TimeoutError("Mavis oMLX did not become healthy before the startup deadline")
    except BaseException:
        if process.poll() is None:
            process.terminate()
        raise


def available_memory_bytes() -> int:
    page_size = int(subprocess.check_output(["sysctl", "-n", "hw.pagesize"], text=True).strip())
    output = subprocess.check_output(["vm_stat"], text=True)
    counts: dict[str, int] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        value = raw.strip().rstrip(".")
        if value.isdigit():
            counts[key] = int(value)
    pages = sum(counts.get(key, 0) for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"))
    return pages * page_size


def admission(config: RuntimeConfig, *, now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    mavis_records = inventory(config.endpoint)
    target = next((item for item in mavis_records if item.get("id") == config.model), None)
    if target is None:
        raise RuntimeError(f"Mavis model is not discoverable: {config.model}")
    expected = int(target.get("estimated_size") or 0)
    free = available_memory_bytes()
    reasons: list[str] = []
    if expected <= 0:
        reasons.append("target model size is unknown")
    if free < expected + config.reserve_bytes:
        reasons.append("aggregate available memory does not satisfy model plus reserve")

    if endpoint_alive(config.iris_endpoint):
        for record in inventory(config.iris_endpoint):
            if not record.get("loaded") or record.get("model_type") == "embedding":
                continue
            if not config.allow_concurrent_local:
                reasons.append(
                    "IRIS already owns an active local generation model; concurrent local inference has no accepted safety evidence"
                )
                break
            last_access = record.get("last_access")
            if last_access is None or now - float(last_access) < config.idle_seconds:
                reasons.append("IRIS local generation model is not proven idle for 15 minutes")
                break
    return {
        "allowed": not reasons,
        "reasons": reasons,
        "available_memory_bytes": free,
        "estimated_model_bytes": expected,
        "reserve_bytes": config.reserve_bytes,
    }


def load_model(config: RuntimeConfig) -> dict[str, Any]:
    decision = admission(config)
    if not decision["allowed"]:
        raise RuntimeError("; ".join(decision["reasons"]))
    request_json(
        config.endpoint,
        f"/v1/models/{quote(config.model, safe='')}/load",
        method="POST",
        timeout=900,
    )
    records = inventory(config.endpoint)
    selected = next((item for item in records if item.get("id") == config.model), None)
    if not selected or not selected.get("loaded"):
        raise RuntimeError("oMLX did not report the selected Mavis model as loaded")
    return {"model": config.model, "loaded": True, "admission": decision}


def ensure_runtime(config: RuntimeConfig, *, load: bool = True) -> dict[str, Any]:
    state = start_server(config)
    records = inventory(config.endpoint)
    selected = next((item for item in records if item.get("id") == config.model), None)
    if selected is None:
        raise RuntimeError("Mavis server identity check failed: selected model is absent")
    model_state = {"model": config.model, "loaded": bool(selected.get("loaded"))}
    if load and not model_state["loaded"]:
        model_state = load_model(config)
    return {"runtime": state, "model": model_state}


def stop_server(config: RuntimeConfig) -> None:
    if not owns_running_server(config):
        raise RuntimeError("refusing to stop a process Mavis does not own")
    pid = int(read_json(config.state_path)["pid"])
    os.kill(pid, signal.SIGTERM)
