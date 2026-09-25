"""Fail-closed client for the configured local model gateway MCP server."""

from __future__ import annotations

import json
import os
from pathlib import Path
import select
import subprocess
import time
import tomllib
from typing import Any


# The gateway's independent verifier: GLM 5.3 in the Codex CLI harness,
# started through the gateway's bin/glm-codex.sh launcher.
VERIFIER = "glm-codex"
VERIFIER_MODEL = "z-ai/glm-5.3"
VERIFIER_JOB_SUFFIX = "-glm"
VERIFIER_EVIDENCE_SCHEMA = "model-gateway-verifier-evidence/v2"
GLM_LAUNCHER_NAME = "glm-codex.sh"


class GatewayUnavailable(RuntimeError):
    """The trusted gateway cannot provide a status receipt."""


def _configured_gateway(codex_home: str | None = None) -> tuple[list[str], dict[str, str]]:
    codex_home = codex_home or os.environ.get("CODEX_HOME")
    if not codex_home:
        raise GatewayUnavailable("CODEX_HOME is required for the configured gateway")
    config_path = Path(codex_home).expanduser() / "config.toml"
    try:
        with config_path.open("rb") as handle:
            config = tomllib.load(handle)
        gateway = config["mcp_servers"]["model-gateway"]
        command = gateway["command"]
    except (FileNotFoundError, KeyError, OSError, tomllib.TOMLDecodeError) as error:
        raise GatewayUnavailable("configured model gateway is unavailable") from error
    if not isinstance(command, str):
        raise GatewayUnavailable("configured model gateway command is invalid")
    executable = Path(command).expanduser()
    if not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK):
        raise GatewayUnavailable("configured model gateway command is not executable")
    configured_env = gateway.get("env", {})
    if not isinstance(configured_env, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in configured_env.items()
    ):
        raise GatewayUnavailable("configured model gateway environment is invalid")
    return [str(executable)], dict(configured_env)


def _review_codex_home() -> str:
    """CODEX_HOME, else Mavis's own isolated Codex home (as run_final_e0_with_handoff sets)."""
    return os.environ.get("CODEX_HOME") or str(Path.home() / ".local-codex")


def glm_review_launcher() -> Path:
    """The configured gateway's GLM 5.3 reviewer launcher, beside its server launcher."""
    command, _ = _configured_gateway(_review_codex_home())
    launcher = Path(command[0]).parent / GLM_LAUNCHER_NAME
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise GatewayUnavailable("configured gateway has no GLM reviewer launcher")
    return launcher


def glm_review_environment() -> dict[str, str]:
    """Minimal environment for a direct GLM review: the env-file path, never its contents."""
    _, configured_env = _configured_gateway(_review_codex_home())
    environment = {name: os.environ[name] for name in ("PATH", "HOME", "TMPDIR", "LANG", "OPENROUTER_API_KEY")
                   if name in os.environ}
    if "MODEL_GATEWAY_ENV_FILE" in configured_env:
        environment["MODEL_GATEWAY_ENV_FILE"] = configured_env["MODEL_GATEWAY_ENV_FILE"]
    return environment


def _read_response(process: subprocess.Popen[str], request_id: int, deadline: float) -> dict[str, Any]:
    while time.monotonic() < deadline:
        if process.stdout is None:
            break
        readable, _, _ = select.select([process.stdout], [], [], max(0, deadline - time.monotonic()))
        if not readable:
            break
        line = process.stdout.readline()
        if not line:
            break
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("id") == request_id:
            if "error" in message:
                raise GatewayUnavailable("configured model gateway rejected status request")
            result = message.get("result")
            if isinstance(result, dict):
                return result
            break
    raise GatewayUnavailable("configured model gateway did not return a status receipt")


def _send(process: subprocess.Popen[str], payload: dict[str, Any]) -> None:
    if process.stdin is None:
        raise GatewayUnavailable("configured model gateway stdin is unavailable")
    process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
    process.stdin.flush()


def _gateway_tool(name: str, arguments: dict[str, Any], timeout: float,
                  *, require_success: bool = True,
                  environment_overrides: dict[str, str] | None = None) -> dict[str, Any]:
    """Call one tool on the trusted Mavis gateway without a provider fallback."""
    command, configured_env = _configured_gateway()
    environment = os.environ.copy()
    environment.update(configured_env)
    if environment_overrides:
        environment.update(environment_overrides)
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=environment,
        )
    except OSError as error:
        raise GatewayUnavailable("configured model gateway could not start") from error
    try:
        deadline = time.monotonic() + timeout
        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "mavis-objectives", "version": "1"},
                },
            },
        )
        _read_response(process, 1, deadline)
        _send(process, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
        )
        result = _read_response(process, 2, deadline)
    finally:
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        payload = structured
    else:
        content = result.get("content")
        if not isinstance(content, list):
            raise GatewayUnavailable("configured model gateway returned malformed status")
        text = next((item.get("text") for item in content if isinstance(item, dict) and item.get("type") == "text"), None)
        try:
            payload = json.loads(text) if isinstance(text, str) else None
        except json.JSONDecodeError as error:
            raise GatewayUnavailable("configured model gateway returned malformed status") from error
    if not isinstance(payload, dict) or (require_success and payload.get("success") is not True):
        raise GatewayUnavailable(f"configured model gateway returned unsuccessful {name}")
    return payload


def harness_job_status(job_id: str, timeout: float = 10.0) -> dict[str, Any]:
    """Return a status only from the Mavis-configured gateway, or fail closed."""
    payload = _gateway_tool("harness_job_status", {"job_id": job_id}, timeout)
    if not isinstance(payload.get("status"), dict):
        raise GatewayUnavailable("configured model gateway returned malformed status")
    return payload["status"]


def harness_assignment_start(assignment: dict[str, Any], timeout: float = 15.0, *,
                             opencode_policy: str | None = None) -> dict[str, Any]:
    """Start a Mavis-bound native assignment through the configured gateway."""
    return _gateway_tool(
        "harness_assignment_start", assignment, timeout,
        environment_overrides=(
            {"OPENCODE_CONFIG_CONTENT": opencode_policy}
            if opencode_policy is not None else None),
    )


def harness_job_complete(job_id: str, check_results: dict[str, Any], timeout: float = 15.0) -> dict[str, Any]:
    """Record a worker's terminal receipt and exact host check output."""
    return _gateway_tool("harness_job_complete", {
        "job_id": job_id, "summary": "E1 bootstrap evidence review complete",
        "check_results": check_results,
    }, timeout)


def harness_verifier_start(job_id: str, verifier_job_id: str, timeout: float = 15.0) -> dict[str, Any]:
    """Start the gateway's independent GLM 5.3 verifier for one worker."""
    return _gateway_tool("harness_verifier_start", {
        "job_id": job_id, "verifier_job_id": verifier_job_id,
    }, timeout)


def harness_job_verify(job_id: str, verifier_job_id: str, report_sha256: str,
                       timeout: float = 15.0) -> dict[str, Any]:
    """Ask the gateway to bind the GLM verifier's verdict to the exact worker report."""
    return _gateway_tool("harness_job_verify", {
        "job_id": job_id, "verifier": VERIFIER, "verdict": "accepted",
        "evidence_sha256": report_sha256, "verifier_job_id": verifier_job_id,
    }, timeout)


def mavis_jev_decisions(purpose: str, signals: dict[str, Any],
                        estimated_cost_usd: float, timeout: float = 30.0, *,
                        context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the gateway's fixed-template decision or explicit refusal."""
    arguments = {
        "purpose": purpose,
        "signals": signals,
        "estimated_cost_usd": estimated_cost_usd,
    }
    if context is not None:
        arguments["context"] = context
    return _gateway_tool("mavis_jev_decisions", arguments, timeout, require_success=False)
