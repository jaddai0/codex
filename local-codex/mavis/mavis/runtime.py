"""Isolated oMLX lifecycle and model admission for Mavis."""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
import fcntl
from functools import wraps
import hashlib
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
_TRIAL_PROCESSES: dict[int, subprocess.Popen[Any]] = {}


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
    api_key: str | None = field(default=None, repr=False)

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


def _auth_headers(api_key: str | None) -> dict[str, str] | None:
    return {"Authorization": f"Bearer {api_key}"} if api_key else None


def _own_api(config: RuntimeConfig) -> dict[str, str]:
    return {"api_key": config.api_key} if config.api_key else {}


def _own_headers(config: RuntimeConfig) -> dict[str, dict[str, str]]:
    return {"headers": _auth_headers(config.api_key)} if config.api_key else {}


def inventory(endpoint: str, *, api_key: str | None = None,
              timeout: float = 10) -> list[dict[str, Any]]:
    # The admin route requires a browser session cookie. A trial key uses the
    # public, Bearer-authenticated status route with the same admission fields.
    path = "/v1/models/status" if api_key else "/admin/api/models"
    payload = request_json(endpoint, path, **(
        {"headers": _auth_headers(api_key)} if api_key else {}
    ), timeout=timeout)
    records = payload.get("models", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise RuntimeError("oMLX returned an invalid model inventory")
    return records


def loaded_generation_models(endpoint: str, *, api_key: str | None = None,
                             timeout: float = 10) -> list[str]:
    """Require a readable inventory and name every loaded non-embedding model."""
    models = []
    for row in inventory(endpoint, timeout=timeout,
                         **({"api_key": api_key} if api_key else {})):
        if row.get("loaded") is not True:
            continue
        if row.get("engine_type") == "embedding" or row.get("model_type") == "embedding":
            continue
        model_id = row.get("id")
        if not isinstance(model_id, str) or not model_id:
            raise RuntimeError("loaded generation model lacks an ID")
        models.append(model_id)
    return models


def endpoint_alive(endpoint: str, *, api_key: str | None = None,
                   timeout: float = 10) -> bool:
    try:
        inventory(endpoint, timeout=timeout,
                  **({"api_key": api_key} if api_key else {}))
        return True
    except (OSError, URLError, RuntimeError, ValueError, json.JSONDecodeError):
        return False


def require_idle_iris_handoff(config: RuntimeConfig, *, interval_seconds: float = 1.0,
                              expected_models: list[str] | None = None) -> None:
    """Refuse a model handoff while IRIS has active or queued generation."""
    expected = [config.model] if expected_models is None else expected_models
    if not expected or len(expected) != len(set(expected)):
        raise RuntimeError("IRIS generation model inventory is invalid")
    status_models = None
    for sample in range(2):
        status = request_json(config.iris_endpoint, "/api/status")
        if (not isinstance(status, dict) or status.get("status") != "ok"
            or not isinstance(status.get("loaded_models"), list)
            or not all(model in status["loaded_models"] for model in expected)
            or type(status.get("models_loading")) is not int
            or status["models_loading"] != 0
            or type(status.get("active_requests")) is not int
            or type(status.get("waiting_requests")) is not int
            or status["active_requests"] != 0
            or status["waiting_requests"] != 0):
            raise RuntimeError("IRIS has active or waiting work; model handoff refused")
        if status_models is not None and status["loaded_models"] != status_models:
            raise RuntimeError("IRIS loaded model status changed during idle handoff")
        status_models = status["loaded_models"]
        if sample == 0:
            time.sleep(interval_seconds)
    if loaded_generation_models(config.iris_endpoint) != expected:
        raise RuntimeError("IRIS generation model inventory changed before handoff")


def require_installed_selected_model(config: RuntimeConfig) -> None:
    """Prove the installed launcher will select the model being handed off."""
    prepare = Path.home() / ".local" / "share" / "local-codex" / "prepare_runtime.py"
    result = subprocess.run(
        [sys.executable, str(prepare), "--mavis-home", str(config.home), "--resolve-model"],
        text=True, capture_output=True, timeout=15, check=True,
        env={key: value for key, value in os.environ.items()
             if key != "MAVIS_E0_TRIAL_API_KEY"},
    )
    if result.stdout.strip() != config.model:
        raise RuntimeError("installed launcher selects a different Mavis model")


def omlx_runtime_fingerprint(config: RuntimeConfig) -> dict[str, Any]:
    """Bind a recovery observation to the editable server and model metadata."""
    python = config.omlx_binary.parent / "python"
    modules = ("omlx.server", "omlx.engine_pool", "omlx.model_drain")
    script = ("import importlib.util,json,sys; "
              "print(json.dumps({name:importlib.util.find_spec(name).origin "
              "for name in sys.argv[1:]}))")
    result = subprocess.run([str(python), "-c", script, *modules],
                            text=True, capture_output=True, check=True, timeout=15,
                            env={key: value for key, value in os.environ.items()
                                 if key != "MAVIS_E0_TRIAL_API_KEY"})
    paths = json.loads(result.stdout)
    if set(paths) != set(modules):
        raise RuntimeError("oMLX runtime source mapping is incomplete")
    sources = {}
    for name in modules:
        path = Path(paths[name]).resolve(strict=True)
        if path.name != name.rsplit(".", 1)[1] + ".py":
            raise RuntimeError("oMLX runtime source path changed unexpectedly")
        sources[name] = {"path": str(path),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    package = Path(paths["omlx.server"]).resolve(strict=True).parent
    package_files = sorted(path for path in package.rglob("*.py")
                           if "__pycache__" not in path.parts)
    if len(package_files) < 20:
        raise RuntimeError("oMLX runtime package inventory is incomplete")
    package_digest = hashlib.sha256()
    for path in package_files:
        package_digest.update(str(path.relative_to(package)).encode())
        package_digest.update(bytes.fromhex(hashlib.sha256(path.read_bytes()).hexdigest()))
    model_path = (config.model_dir / config.model).resolve(strict=True)
    metadata = {}
    for name in ("config.json", "tokenizer_config.json", "model.safetensors.index.json",
                 "generation_config.json", "chat_template.jinja"):
        path = model_path / name
        if path.is_file():
            metadata[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if "config.json" not in metadata:
        raise RuntimeError("selected oMLX model metadata is missing")
    shards = sorted((path.name, path.stat().st_size, path.stat().st_mtime_ns)
                    for path in model_path.glob("*.safetensors") if path.is_file())
    if not shards:
        raise RuntimeError("selected oMLX model has no weight shards")
    shard_signature = hashlib.sha256(json.dumps(shards, separators=(",", ":")).encode()).hexdigest()
    return {"binary_sha256": hashlib.sha256(config.omlx_binary.read_bytes()).hexdigest(),
            "sources": sources, "package_path": str(package),
            "package_files": len(package_files), "package_sha256": package_digest.hexdigest(),
            "model_id": config.model, "model_path": str(model_path),
            "model_metadata_sha256": metadata,
            "weight_file_signature_sha256": shard_signature}


def omlx_live_process_binding(config: RuntimeConfig, endpoint: str,
                              runtime: dict[str, Any]) -> dict[str, Any]:
    """Require one live oMLX process launched after the fingerprinted code."""
    pids = _listener_pids(_port(endpoint))
    if len(pids) != 1:
        raise RuntimeError("oMLX listener does not have one process")
    pid = next(iter(pids))
    base_path = (config.base_path if endpoint == config.endpoint
                 else Path.home() / ".omlx").resolve(strict=True)
    if not any(path == base_path or base_path in path.parents
               for path in _process_open_paths(pid)):
        raise RuntimeError("oMLX process is not bound to the expected base path")
    status = request_json(endpoint, "/api/status")
    snapshot = status.get("runtime_source") if isinstance(status, dict) else None
    shebang = config.omlx_binary.open("rb").readline().decode("ascii").strip()
    if not shebang.startswith("#!"):
        raise RuntimeError("oMLX launcher has no Python shebang")
    expected_python = Path(shebang[2:].split()[0])
    expected_prefix = config.omlx_binary.parent.parent.resolve(strict=True)
    if (not expected_python.is_file()
            or expected_python.parent.resolve() != config.omlx_binary.parent.resolve()):
        raise RuntimeError("oMLX launcher uses a Python outside its configured venv")
    if (not isinstance(snapshot, dict) or snapshot.get("process_pid") != pid
            or snapshot.get("package_sha256") != runtime["package_sha256"]
            or snapshot.get("package_files") != runtime["package_files"]
            or snapshot.get("python_executable") != str(expected_python)
            or snapshot.get("python_prefix") != str(expected_prefix)):
        raise RuntimeError("live oMLX process does not report the fingerprinted package and Python")
    models = [row for row in inventory(endpoint) if row.get("id") == runtime["model_id"]]
    if (len(models) != 1 or not isinstance(models[0].get("model_path"), str)
            or Path(models[0]["model_path"]).resolve(strict=True)
            != Path(runtime["model_path"])):
        raise RuntimeError("live oMLX selected model path differs from fingerprinted model")
    result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="],
                            text=True, capture_output=True, check=True, timeout=5,
                            env={**{key: value for key, value in os.environ.items()
                                    if key != "MAVIS_E0_TRIAL_API_KEY"}, "LC_ALL": "C"})
    started = datetime.strptime(result.stdout.strip(), "%a %b %d %H:%M:%S %Y").timestamp()
    package = Path(runtime["package_path"])
    latest_source = max(path.stat().st_mtime for path in package.rglob("*.py")
                        if "__pycache__" not in path.parts)
    latest_source = max(latest_source, config.omlx_binary.stat().st_mtime)
    model = Path(runtime["model_path"])
    latest_source = max(latest_source, *(path.stat().st_mtime for path in
        (model / name for name in runtime["model_metadata_sha256"])) )
    latest_source = max(latest_source, *(path.stat().st_mtime for path in
        model.glob("*.safetensors")))
    if started <= latest_source + 1:
        raise RuntimeError("oMLX process may predate its current source or model files")
    return {"pid": pid, "started_at_epoch": started, "base_path": str(base_path),
            "package_sha256": runtime["package_sha256"],
            "python_executable": str(expected_python),
            "python_prefix": str(expected_prefix)}


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
        timeout=1,
        check=False,
        env={key: value for key, value in os.environ.items()
             if key != "MAVIS_E0_TRIAL_API_KEY"},
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
        timeout=1,
        check=False,
        env={key: value for key, value in os.environ.items()
             if key != "MAVIS_E0_TRIAL_API_KEY"},
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
    if (config.base_path / "cluster" / "deployments.json").exists():
        raise RuntimeError("isolated Mavis runtime refuses a distributed deployment registry")
    settings_path = config.base_path / "settings.json"
    if settings_path.exists():
        settings = read_json(settings_path)
    else:
        settings = {"version": "1.0"}
    auth = settings.setdefault("auth", {})
    if not isinstance(auth, dict):
        raise ValueError("Mavis oMLX auth settings must be an object")
    if config.api_key:
        if auth.get("api_key") not in (None, config.api_key):
            raise RuntimeError("residual authentication from another Mavis trial")
        auth["api_key"] = config.api_key
        auth["skip_api_key_verification"] = False
    else:
        if auth.get("api_key"):
            raise RuntimeError("residual Mavis trial authentication must be cleared after server stop")
        auth["skip_api_key_verification"] = True
    server = settings.setdefault("server", {})
    if not isinstance(server, dict):
        raise ValueError("Mavis oMLX server settings must be an object")
    server.update({"host": "127.0.0.1", "port": _port(config.endpoint),
                   "distributed_inference_enabled": False})
    model = settings.setdefault("model", {})
    if not isinstance(model, dict):
        raise ValueError("Mavis oMLX model settings must be an object")
    model.update({"model_dir": str(config.model_dir), "model_dirs": [str(config.model_dir)]})
    cache = settings.setdefault("cache", {})
    if not isinstance(cache, dict):
        raise ValueError("Mavis oMLX cache settings must be an object")
    cache.update({"enabled": True, "ssd_cache_dir": str(config.base_path / "cache")})
    write_json(settings_path, settings)


def clear_trial_auth_settings(config: RuntimeConfig) -> None:
    """Clear only this trial's key after its server group is proven stopped."""
    if not config.api_key:
        raise ValueError("trial authentication key is missing")
    path = config.base_path / "settings.json"
    if not path.is_file():
        return
    settings = read_json(path)
    auth = settings.get("auth")
    if not isinstance(auth, dict):
        raise RuntimeError("Mavis trial authentication settings are invalid")
    current = auth.get("api_key")
    if current is None:
        return
    if current != config.api_key or auth.get("skip_api_key_verification") is not False:
        raise RuntimeError("Mavis trial authentication identity changed")
    auth.pop("api_key")
    auth["skip_api_key_verification"] = True
    write_json(path, settings)


def reject_persisted_trial_preload(config: RuntimeConfig) -> None:
    """A trial must not inherit oMLX pinned models from an earlier session."""
    path = config.base_path / "model_settings.json"
    if path.is_symlink():
        raise RuntimeError("Mavis trial model settings are a symbolic link")
    if not path.exists():
        return
    document = read_json(path)
    if (not isinstance(document, dict) or document.get("version") != 1
            or not isinstance(document.get("models"), dict)):
        raise RuntimeError("Mavis trial model settings cannot rule out a pinned preload")
    for model_id, settings in document["models"].items():
        if not isinstance(model_id, str) or not isinstance(settings, dict):
            raise RuntimeError("Mavis trial model settings are malformed")
        pinned = settings.get("is_pinned", False)
        if type(pinned) is not bool:
            raise RuntimeError("Mavis trial pinned-model state is untrusted")
        if pinned:
            raise RuntimeError("Mavis trial refuses a persisted pinned-model preload")


def reject_residual_trial_auth(config: RuntimeConfig) -> None:
    """A new trial never adopts a key left by a prior interrupted trial."""
    path = config.base_path / "settings.json"
    if path.is_symlink():
        raise RuntimeError("Mavis trial settings are a symbolic link")
    if not path.exists():
        return
    settings = read_json(path)
    auth = settings.get("auth", {})
    if not isinstance(auth, dict):
        raise RuntimeError("Mavis trial authentication settings are malformed")
    # oMLX persists its unauthenticated default as an explicit JSON null.
    # Only that value with verification skipped is equivalent to no key.
    if "api_key" in auth:
        safe_default = (auth["api_key"] is None and
                        auth.get("skip_api_key_verification") is True)
    else:
        safe_default = auth.get("skip_api_key_verification", True) is True
    if not safe_default:
        error = RuntimeError("residual Mavis trial authentication requires recovery")
        error.residual_trial_auth = True
        raise error


def start_server(config: RuntimeConfig, wait_seconds: float = 30.0, *,
                 require_new: bool = False,
                 heartbeat: Callable[[], None] | None = None) -> dict[str, Any]:
    if heartbeat is not None and (not require_new or not config.api_key):
        raise ValueError("monitored startup requires a new authenticated trial server")
    if endpoint_alive(config.endpoint, **_own_api(config)):
        if require_new:
            raise RuntimeError("Mavis endpoint was already running before this observation")
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
    if heartbeat is not None:
        reject_residual_trial_auth(config)
    ensure_isolated_settings(config)
    if heartbeat is not None:
        reject_persisted_trial_preload(config)
        heartbeat()
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
        child_env.pop("MAVIS_E0_TRIAL_API_KEY", None)
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
    try:
        write_json(config.state_path, state)
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            if heartbeat is not None:
                heartbeat()
            exited = _child_exit_unreaped(process)
            if exited is not None:
                raise RuntimeError(
                    f"Mavis oMLX exited during startup with status {exited.si_status}"
                )
            if endpoint_alive(config.endpoint,
                              **({"timeout": 0.5} if heartbeat is not None else {}),
                              **_own_api(config)):
                if heartbeat is not None:
                    heartbeat()
                if require_new and not owns_running_server(config):
                    raise RuntimeError("Mavis endpoint changed owner during startup")
                _TRIAL_PROCESSES[process.pid] = process
                return state
            time.sleep(0.1 if heartbeat is not None else 0.25)
        raise TimeoutError("Mavis oMLX did not become healthy before the startup deadline")
    except BaseException:
        try:
            _stop_spawned_process_group(process)
        except BaseException as cleanup_error:
            failure = RuntimeError(
                f"Mavis launched server could not be stopped after startup failure: {cleanup_error}"
            )
            failure.unsafe_gpu_work = True
            raise failure from cleanup_error
        # A failed launch must not leave a PID-bearing record that could later
        # alias another process. Preserve a changed record owned by someone else.
        if config.state_path.is_file() and read_json(config.state_path) == state:
            config.state_path.unlink()
        raise


def _child_exit_unreaped(process: subprocess.Popen[Any]) -> os.waitid_result | None:
    """Observe an exact child without freeing its PID for reuse."""
    if process.returncode is not None:
        raise RuntimeError("Mavis child leader was already reaped; group identity is unproven")
    try:
        return os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError as error:
        raise RuntimeError("Mavis child leader is no longer waitable") from error


def _process_group_workers(pid: int) -> set[int]:
    """Inspect other members while the unreaped leader reserves this group ID."""
    probe_env = {key: value for key, value in os.environ.items()
                 if key != "MAVIS_E0_TRIAL_API_KEY"}
    result = subprocess.run(["ps", "-axo", "pid=,pgid=,stat="], capture_output=True,
                            text=True, timeout=1, check=False, env=probe_env)
    if result.returncode:
        raise RuntimeError("Mavis process-group member probe failed")
    workers: set[int] = set()
    for line in result.stdout.splitlines():
        columns = line.split(maxsplit=2)
        if len(columns) != 3 or not columns[0].isdigit() or not columns[1].isdigit():
            raise RuntimeError("Mavis process-group member probe is malformed")
        member, group, state = int(columns[0]), int(columns[1]), columns[2]
        # A zombie has already stopped all GPU and request work. The group
        # leader remains unreaped until after the last possible group signal.
        if group == pid and member != pid and not state.startswith("Z"):
            workers.add(member)
    return workers


def _signal_spawned_group(process: subprocess.Popen[Any], pid: int, sig: int) -> None:
    """Signal this Popen's group; accept only the stopped-group answers.

    macOS answers EPERM, not ESRCH, when the only member left is the exited,
    unreaped leader (measured 2026-09-25). That group has no work left. Any other
    EPERM (leader still running, or a worker remains) is a real refusal.
    """
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        if _child_exit_unreaped(process) is None or _process_group_workers(pid):
            raise


def _stop_spawned_process_group(process: subprocess.Popen[Any], *,
                                timeout: float = 5.0,
                                heartbeat: Callable[[], bool] | None = None) -> None:
    """Signal only while this Popen's unreaped leader reserves its group ID."""
    pid = process.pid
    exited = _child_exit_unreaped(process)
    if exited is None:
        try:
            group = os.getpgid(pid)
        except ProcessLookupError:
            # macOS can stop exposing a zombie leader through getpgid while
            # waitid still observes it without reaping its reserved PID.
            if _child_exit_unreaped(process) is None:
                raise RuntimeError("spawned Mavis child group identity is unproven")
        else:
            if group != pid:
                raise RuntimeError("spawned Mavis child left its dedicated process group")
    _signal_spawned_group(process, pid, signal.SIGTERM)
    forced = False
    for phase in ("term", "kill"):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if heartbeat is not None and not forced and not heartbeat():
                # The leader is still unreaped, so the group number cannot
                # alias another session when an unsafe handoff escalates.
                _child_exit_unreaped(process)
                _signal_spawned_group(process, pid, signal.SIGKILL)
                forced = True
            exited = _child_exit_unreaped(process)
            workers = _process_group_workers(pid)
            if exited is not None and not workers:
                process.wait(timeout=0)
                return
            time.sleep(0.1)
        if phase == "term":
            # The leader is still unreaped, even if it has exited. Its PID
            # cannot alias a foreign process group before this final signal.
            _child_exit_unreaped(process)
            _signal_spawned_group(process, pid, signal.SIGKILL)
    raise RuntimeError("Mavis child group or worker remained; leader kept unreaped")


def stop_trial_server(config: RuntimeConfig, expected_state: dict[str, Any], *,
                      timeout: float = 10.0,
                      heartbeat: Callable[[], bool] | None = None) -> None:
    """Abort only the server this trial launched and prove its group exited."""
    if not config.state_path.is_file() or read_json(config.state_path) != expected_state:
        raise RuntimeError("Mavis trial server launch record changed")
    pid = expected_state.get("pid")
    if type(pid) is not int or pid <= 0 or pid not in _TRIAL_PROCESSES:
        raise RuntimeError("Mavis trial server has no process handle from this launch")
    process = _TRIAL_PROCESSES[pid]
    if process.pid != pid:
        raise RuntimeError("Mavis trial server process handle changed")
    if _child_exit_unreaped(process) is None and not owns_running_server(config):
        raise RuntimeError("Mavis trial server ownership is unproven")
    _stop_spawned_process_group(
        process, timeout=timeout,
        **({"heartbeat": heartbeat} if heartbeat is not None else {}),
    )
    _TRIAL_PROCESSES.pop(pid, None)
    # A stopped trial must not leave a PID-bearing record for a later process
    # to mistake for a server it can still control.
    if config.state_path.is_file() and read_json(config.state_path) == expected_state:
        config.state_path.unlink()


def available_memory_bytes() -> int:
    # The installed E0 evaluator carries a trial-only API key in its own env.
    # OS probes have no need for that credential.
    probe_env = {key: value for key, value in os.environ.items()
                 if key != "MAVIS_E0_TRIAL_API_KEY"}
    page_size = int(subprocess.check_output(["sysctl", "-n", "hw.pagesize"],
                                            text=True, env=probe_env).strip())
    output = subprocess.check_output(["vm_stat"], text=True, env=probe_env)
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
    mavis_records = inventory(config.endpoint, **_own_api(config))
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
        **_own_headers(config),
    )
    records = inventory(config.endpoint, **_own_api(config))
    selected = next((item for item in records if item.get("id") == config.model), None)
    if not selected or not selected.get("loaded"):
        raise RuntimeError("oMLX did not report the selected Mavis model as loaded")
    return {"model": config.model, "loaded": True, "admission": decision}


def ensure_runtime(config: RuntimeConfig, *, load: bool = True) -> dict[str, Any]:
    state = start_server(config)
    records = inventory(config.endpoint, **_own_api(config))
    selected = next((item for item in records if item.get("id") == config.model), None)
    if selected is None:
        raise RuntimeError("Mavis server identity check failed: selected model is absent")
    model_state = {"model": config.model, "loaded": bool(selected.get("loaded"))}
    if load and not model_state["loaded"]:
        model_state = load_model(config)
    return {"runtime": state, "model": model_state}


def stop_server(config: RuntimeConfig) -> None:
    """Stop only an exact server launched by this process, including its workers.

    A separate CLI process has no launch handle, so a persisted PID and open
    base path are not enough to authorize signaling a potentially reused PID.
    """
    if not config.state_path.is_file():
        raise RuntimeError("Mavis server has no launch record")
    state = read_json(config.state_path)
    pid = state.get("pid")
    if type(pid) is not int or pid not in _TRIAL_PROCESSES:
        raise RuntimeError("Mavis server exact launch ownership is unproven")
    if not owns_running_server(config):
        raise RuntimeError("refusing to stop a process Mavis does not own")
    status = request_json(config.endpoint, "/api/status", **_own_headers(config))
    if (not isinstance(status, dict) or status.get("status") != "ok"
            or any(type(status.get(key)) is not int or status[key] != 0 for key in
                   ("active_requests", "waiting_requests", "models_loading"))):
        raise RuntimeError("Mavis server has active, waiting, or loading work")
    stop_trial_server(config, state)
    if config.state_path.is_file() and read_json(config.state_path) == state:
        config.state_path.unlink()


def reserve_empty_mavis_port(config: RuntimeConfig, *,
                             verify_endpoint: bool = True) -> socket.socket:
    """Reserve only an empty port; never stop an existing server."""
    if verify_endpoint and endpoint_alive(config.endpoint, **_own_api(config)):
        raise RuntimeError("Mavis endpoint is occupied; refusing to reserve its port")
    parsed = urlparse(config.endpoint)
    reservation = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        reservation.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            reservation.bind((parsed.hostname or "127.0.0.1", _port(config.endpoint)))
        except OSError as error:
            raise RuntimeError("Mavis endpoint is occupied; refusing to reserve its port") from error
        reservation.listen(1)
        if _listener_pids(_port(config.endpoint)) != {os.getpid()}:
            raise RuntimeError("Mavis port reservation did not exclusively own the listener")
        return reservation
    except BaseException:
        reservation.close()
        raise


def park_mavis_server(config: RuntimeConfig, *, timeout: float = 30,
                      expected_pid: int | None = None,
                      expected_state: dict[str, Any] | None = None,
                      heartbeat: Callable[[], bool] | None = None) -> socket.socket:
    """Stop the owned server or reserve its empty port through IRIS restoration.

    The caller must keep the returned listening socket open until IRIS is
    restored and its drain released. A failed stop never proves safe handoff.
    """
    pid = read_json(config.state_path).get("pid") if config.state_path.is_file() else None
    if pid is not None and (not isinstance(pid, int) or pid <= 0):
        raise RuntimeError("Mavis owned server PID is invalid")
    if expected_pid is not None and pid != expected_pid:
        raise RuntimeError("Mavis server PID changed before parking")
    if expected_state is not None and read_json(config.state_path) != expected_state:
        raise RuntimeError("Mavis server launch record changed before parking")
    if pid is not None and pid not in _TRIAL_PROCESSES:
        raise RuntimeError("Mavis server has no process handle from this launch")
    state = expected_state if expected_state is not None else (
        read_json(config.state_path) if pid is not None else None
    )
    def reserve_after_verified_stop() -> socket.socket:
        try:
            return reserve_empty_mavis_port(config, verify_endpoint=False)
        except BaseException as error:
            failure = RuntimeError(
                f"Mavis trial group stopped, but empty port reservation failed: {error}"
            )
            failure.gpu_work_stopped = True
            raise failure from error

    def emergency_reservation() -> socket.socket:
        if state is None:
            raise RuntimeError("Mavis emergency park lacks exact trial state")
        stop_trial_server(config, state, timeout=1)
        return reserve_after_verified_stop()

    stopped_here = False
    if heartbeat is not None and not heartbeat():
        return emergency_reservation()
    if endpoint_alive(config.endpoint, timeout=0.5 if heartbeat else 10,
                      **_own_api(config)):
        if heartbeat is not None and not heartbeat():
            return emergency_reservation()
        status = request_json(config.endpoint, "/api/status",
                              timeout=0.5 if heartbeat else 10,
                              **_own_headers(config))
        if heartbeat is not None and not heartbeat():
            return emergency_reservation()
        if (not isinstance(status, dict) or status.get("status") != "ok"
                or type(status.get("active_requests")) is not int
                or type(status.get("waiting_requests")) is not int
                or type(status.get("models_loading")) is not int
                or any(status[key] != 0 for key in
                       ("active_requests", "waiting_requests", "models_loading"))):
            raise RuntimeError("Mavis server has active, queued, or loading work")
        if pid is None or state is None:
            raise RuntimeError("Mavis live server lacks owned process state")
        stop_trial_server(config, state, timeout=timeout, heartbeat=heartbeat)
        stopped_here = True
    elif heartbeat is not None and not heartbeat():
        return emergency_reservation()
    elif _listener_pids(_port(config.endpoint)):
        raise RuntimeError("Mavis endpoint is occupied by an unidentified listener")
    elif state is not None:
        # The server may have exited while an owned worker remains. Its
        # unreaped leader still reserves the process-group number.
        stop_trial_server(config, state, timeout=timeout, heartbeat=heartbeat)
        stopped_here = True
    reservation = (reserve_after_verified_stop() if stopped_here else
                   reserve_empty_mavis_port(config, verify_endpoint=False))
    return reservation
