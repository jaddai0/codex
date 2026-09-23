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


class GatewayUnavailable(RuntimeError):
    """The trusted gateway cannot provide a status receipt."""


def _configured_gateway() -> tuple[list[str], dict[str, str]]:
    codex_home = os.environ.get("CODEX_HOME")
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


def _gateway_tool(name: str, arguments: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Call one tool on the trusted Mavis gateway without a provider fallback."""
    command, configured_env = _configured_gateway()
    environment = os.environ.copy()
    environment.update(configured_env)
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
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise GatewayUnavailable(f"configured model gateway returned unsuccessful {name}")
    return payload


def harness_job_status(job_id: str, timeout: float = 10.0) -> dict[str, Any]:
    """Return a status only from the Mavis-configured gateway, or fail closed."""
    payload = _gateway_tool("harness_job_status", {"job_id": job_id}, timeout)
    if not isinstance(payload.get("status"), dict):
        raise GatewayUnavailable("configured model gateway returned malformed status")
    return payload["status"]


def harness_assignment_start(assignment: dict[str, Any], timeout: float = 15.0) -> dict[str, Any]:
    """Start a Mavis-bound native assignment through the configured gateway."""
    return _gateway_tool("harness_assignment_start", assignment, timeout)
