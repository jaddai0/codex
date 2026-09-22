"""Isolated oMLX lifecycle and model admission for Mavis."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from typing import Any
from urllib.error import URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from .storage import read_json, write_json


DEFAULT_MODEL = "Qwen3.8-Flash-Next-Abliterated-MLX-4bit"


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
    endpoint: str, path: str, method: str = "GET", timeout: float = 10
) -> Any:
    request = Request(_origin(endpoint) + path, method=method, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def inventory(endpoint: str) -> list[dict[str, Any]]:
    payload = request_json(endpoint, "/admin/api/models")
    records = payload.get("models", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise RuntimeError("oMLX returned an invalid model inventory")
    return records


def endpoint_alive(endpoint: str) -> bool:
    try:
        inventory(endpoint)
        return True
    except (OSError, URLError, RuntimeError, ValueError, json.JSONDecodeError):
        return False


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
