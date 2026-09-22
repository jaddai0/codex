#!/usr/bin/env python3
"""Prepare an isolated Codex profile from the live local oMLX server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import tomllib
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}
SUPPORTED_MODEL_TYPES = {"llm", "vlm"}
MAVIS_ARCHIVE_COMMAND = "python3 -m mavis pre-compact"
MAVIS_ARCHIVE_STATUS = "Mavis transcript archive v1"


def mavis_archive_hook_hash() -> str:
    # Mirrors Codex's normalized TOML hook identity: absent fields are omitted,
    # and the command timeout defaults to 600 seconds.
    identity = {
        "event_name": "pre_compact",
        "hooks": [
            {
                "type": "command",
                "command": MAVIS_ARCHIVE_COMMAND,
                "timeout": 600,
                "async": False,
                "statusMessage": MAVIS_ARCHIVE_STATUS,
            }
        ],
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def local_base_url(raw: str) -> str:
    value = raw.rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme != "http" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError("oMLX base URL must use plain HTTP on localhost")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "oMLX base URL cannot contain credentials, a query, or a fragment"
        )
    if parsed.path not in {"", "/v1"}:
        raise ValueError("oMLX base URL path must be /v1")
    return f"http://{parsed.netloc}/v1"


def read_json(url: str) -> object:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=10) as response:
        return json.load(response)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def reasoning_levels(record: dict[str, object]) -> list[dict[str, str]]:
    raw = record.get("reasoning_effort_options")
    values = raw if isinstance(raw, list) else []
    allowed = [
        str(value)
        for value in values
        if value in {"low", "medium", "high", "xhigh", "max"}
    ]
    return [
        {"effort": value, "description": f"Use {value} local reasoning effort"}
        for value in allowed
    ]


def model_info(
    record: dict[str, object], priority: int, base_instructions: str
) -> dict[str, object]:
    model_id = str(record["id"])
    context_window = int(record.get("model_context_length") or 32768)
    is_vlm = record.get("model_type") == "vlm" or record.get("engine_type") == "vlm"
    levels = reasoning_levels(record)
    default_effort = record.get("reasoning_effort_default")
    if default_effort not in {item["effort"] for item in levels}:
        default_effort = levels[0]["effort"] if levels else None
    return {
        "slug": model_id,
        "display_name": str(record.get("display_name") or model_id),
        "description": "Local model served by oMLX",
        "default_reasoning_level": default_effort,
        "supported_reasoning_levels": levels,
        "shell_type": "unified_exec",
        "visibility": "list",
        "supported_in_api": True,
        "priority": priority,
        "additional_speed_tiers": [],
        "service_tiers": [],
        "default_service_tier": None,
        "available_access_programs": None,
        "availability_nux": None,
        "upgrade": None,
        "model_messages": None,
        "base_instructions": base_instructions,
        "include_skills_usage_instructions": True,
        "include_plugin_usage_instructions": True,
        "include_apps_usage_instructions": False,
        "supports_reasoning_summary_parameter": False,
        "default_reasoning_summary": "none",
        "support_verbosity": False,
        "default_verbosity": None,
        "apply_patch_tool_type": "function",
        "web_search_tool_type": "text",
        "truncation_policy": {"mode": "tokens", "limit": 10000},
        "supports_image_detail_original": is_vlm,
        "context_window": context_window,
        "max_context_window": context_window,
        "auto_compact_token_limit": int(context_window * 0.85),
        "comp_hash": f"omlx:{model_id}:{context_window}",
        "effective_context_window_percent": 90,
        "experimental_supported_tools": [],
        "input_modalities": ["text", "image"] if is_vlm else ["text"],
        "supports_search_tool": False,
        "supports_experimental_context": False,
        "use_responses_lite": False,
        "supports_reasoning_effort_updates": False,
        "guardian": None,
        "node_repl_auto_review_required": False,
        "node_repl_disabled": True,
        "auto_review_model_override": None,
        "model_specialty": None,
        "tool_mode": "direct",
        "multi_agent_version": "v1",
        "multi_agent_reasoning_effort": None,
    }


def eligible(record: dict[str, object]) -> bool:
    return (
        bool(record.get("id"))
        and not bool(record.get("is_helper"))
        and not bool(record.get("is_hidden"))
        and record.get("model_type") in SUPPORTED_MODEL_TYPES
    )


def prepare_catalog(
    records: list[dict[str, object]], requested: str | None, base_instructions: str
) -> tuple[dict[str, object], str]:
    candidates = [record for record in records if eligible(record)]
    if not candidates:
        raise RuntimeError("oMLX reported no coding-capable local models")
    by_id = {str(record["id"]): record for record in candidates}
    if requested:
        if requested not in by_id:
            raise RuntimeError(
                f"requested model is not available from oMLX: {requested}"
            )
        selected = requested
    else:
        preferred = next(
            (
                record
                for record in candidates
                if record.get("loaded") and record.get("is_default")
            ),
            None,
        )
        if preferred is None:
            preferred = next(
                (record for record in candidates if record.get("loaded")), None
            )
        if preferred is None:
            raise RuntimeError(
                "oMLX has no loaded coding model; load one before starting Local Codex"
            )
        selected = str(preferred["id"])

    ordered = sorted(
        candidates,
        key=lambda item: (
            str(item["id"]) != selected,
            not bool(item.get("loaded")),
            str(item["id"]).lower(),
        ),
    )
    return {
        "models": [
            model_info(record, index + 1, base_instructions)
            for index, record in enumerate(ordered)
        ]
    }, selected


def ensure_persona(home: Path, template: Path) -> None:
    persona_path = home / "persona.toml"
    if not persona_path.exists():
        atomic_write(persona_path, template.read_text(encoding="utf-8"))
    with persona_path.open("rb") as handle:
        persona = tomllib.load(handle)
    name = str(persona.get("name") or "Mavis").strip()
    instructions = str(persona.get("instructions") or "").strip()
    rendered = (
        "# Local coding persona\n\n"
        f"Your current working name is {name}.\n"
        "You are a private local coding agent. Your model runs through the local oMLX server. "
        "You are separate from Iris and must not claim to be Iris.\n"
    )
    if instructions:
        rendered += f"\n{instructions}\n"
    atomic_write(home / "AGENTS.md", rendered)


def write_profile(
    home: Path,
    base_url: str,
    selected: str,
    gateway_root: Path | None = None,
    gateway_env_file: Path | None = None,
) -> None:
    if gateway_env_file is not None and gateway_root is None:
        raise ValueError("gateway environment file requires a trusted gateway")
    catalog_path = home / "omlx-models.json"
    hook_key = f"{(home / 'config.toml').resolve()}:pre_compact:0:0"
    hook_hash = mavis_archive_hook_hash()
    profile = f"""model = {json.dumps(selected)}
model_provider = "omlx"
model_catalog_json = {json.dumps(str(catalog_path))}
check_for_update_on_startup = false

[analytics]
enabled = false

[feedback]
enabled = false

[otel]
exporter = "none"
trace_exporter = "none"
metrics_exporter = "none"

[features]
apps = false
multi_agent = true
plugins = false
remote_plugin = false
shell_snapshot = true
tool_suggest = false
unified_exec = true

[[hooks.PreCompact]]

[[hooks.PreCompact.hooks]]
type = "command"
command = {json.dumps(MAVIS_ARCHIVE_COMMAND)}
statusMessage = {json.dumps(MAVIS_ARCHIVE_STATUS)}

[hooks.state.{json.dumps(hook_key)}]
trusted_hash = {json.dumps(hook_hash)}

[model_providers.omlx]
name = "Local oMLX"
base_url = {json.dumps(base_url)}
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
supports_standalone_web_search = false
"""
    if gateway_root is not None:
        gateway_script = gateway_root.resolve() / "bin" / "mcp-server.sh"
        if not gateway_script.is_file() or not os.access(gateway_script, os.X_OK):
            raise FileNotFoundError(
                f"trusted Mavis gateway launcher is unavailable: {gateway_script}"
            )
        profile += (
            "\n[mcp_servers.model-gateway]\n"
            f"command = {json.dumps(str(gateway_script))}\n"
        )
        if gateway_env_file is not None:
            env_file = gateway_env_file.resolve(strict=True)
            if not env_file.is_file():
                raise ValueError("gateway environment path must be a regular file")
            profile += (
                "\n[mcp_servers.model-gateway.env]\n"
                f"MODEL_GATEWAY_ENV_FILE = {json.dumps(str(env_file))}\n"
            )
    start_marker = "# BEGIN LOCAL CODEX MANAGED CONFIG"
    end_marker = "# END LOCAL CODEX MANAGED CONFIG"
    config_path = home / "config.toml"
    suffix = ""
    if config_path.exists():
        current = config_path.read_text(encoding="utf-8")
        if start_marker in current and end_marker in current:
            suffix = current.split(end_marker, 1)[1].lstrip("\n")
        else:
            preserved_sections = (
                "[projects.",
                "[mcp_servers.",
                "[hooks",
                "[profiles.",
                "[plugins.",
            )
            offsets = [
                current.find(section)
                for section in preserved_sections
                if current.find(section) >= 0
            ]
            if offsets:
                suffix = current[min(offsets) :].lstrip("\n")
    if gateway_root is not None and "[mcp_servers.model-gateway]" in suffix:
        raise ValueError(
            "existing Mavis gateway registration needs review before replacement"
        )
    managed = f"{start_marker}\n{profile}{end_marker}\n"
    if suffix:
        managed += f"\n{suffix}"
    atomic_write(config_path, managed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model")
    parser.add_argument("--persona-template", type=Path, required=True)
    parser.add_argument("--instructions-template", type=Path, required=True)
    parser.add_argument("--gateway-root", type=Path)
    parser.add_argument("--gateway-env-file", type=Path)
    args = parser.parse_args()

    base_url = local_base_url(args.base_url)
    origin = base_url.removesuffix("/v1")
    payload = read_json(f"{origin}/admin/api/models")
    records = payload.get("models", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not all(
        isinstance(item, dict) for item in records
    ):
        raise RuntimeError("oMLX returned an invalid model inventory")
    base_instructions = args.instructions_template.read_text(encoding="utf-8")
    if not base_instructions.strip():
        raise RuntimeError("Codex base instruction template is empty")
    catalog, selected = prepare_catalog(records, args.model, base_instructions)
    args.home.mkdir(parents=True, exist_ok=True)
    atomic_write(args.home / "omlx-models.json", json.dumps(catalog, indent=2) + "\n")
    write_profile(args.home, base_url, selected, args.gateway_root, args.gateway_env_file)
    ensure_persona(args.home, args.persona_template)
    print(selected)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"mavis: {error}", file=sys.stderr)
        raise SystemExit(2) from None
