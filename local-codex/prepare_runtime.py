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
from datetime import datetime, timezone
from uuid import uuid4
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}
SUPPORTED_MODEL_TYPES = {"llm", "vlm"}
MAVIS_ARCHIVE_COMMAND = "python3 -m mavis pre-compact"
MAVIS_ARCHIVE_STATUS = "Mavis transcript archive v1"
MAVIS_HANDOFF_COMMAND = "python3 -m mavis compaction-handoff"
MAVIS_HANDOFF_STATUS = "Mavis verified compaction handoff v1"


def mavis_archive_hook_hash() -> str:
    return mavis_command_hook_hash("pre_compact", MAVIS_ARCHIVE_COMMAND, MAVIS_ARCHIVE_STATUS)


def mavis_command_hook_hash(event_name: str, command: str, status: str) -> str:
    # Mirrors Codex's normalized TOML hook identity: absent fields are omitted,
    # and the command timeout defaults to 600 seconds.
    identity = {
        "event_name": event_name,
        "hooks": [
            {
                "type": "command",
                "command": command,
                "timeout": 600,
                "async": False,
                "statusMessage": status,
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


def accepted_main_profile(mavis_home: Path) -> dict[str, object] | None:
    role_root = mavis_home / "profiles" / "main"
    pointer_path = role_root / "active.json"
    if not pointer_path.exists():
        return None
    from mavis.experiments import ExperimentStore

    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    version = pointer.get("version")
    if type(version) is not int or version < 1:
        raise ValueError("invalid active main profile version")
    profile_path = (role_root / f"v{version}.json").resolve()
    if pointer.get("path") != str(profile_path):
        raise ValueError("active main profile pointer does not match versioned file")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if (profile.get("schema_version"), profile.get("role"), profile.get("version"), profile.get("status")) != (
        "mavis.model-profile/v1", "main", version, "active"
    ):
        raise ValueError("active main profile has an invalid state")
    for name in ("accepted_experiment", "verifier_receipt"):
        retained = profile.get(name)
        if not isinstance(retained, dict) or not retained.get("path") or not retained.get("sha256"):
            raise ValueError(f"active main profile lacks {name} evidence")
        evidence_path = Path(retained["path"]).resolve()
        if mavis_home.resolve() not in evidence_path.parents:
            raise ValueError(f"active main profile {name} is outside Mavis home")
        if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != retained["sha256"]:
            raise ValueError(f"active main profile {name} evidence changed")
    experiment_path = Path(profile["accepted_experiment"]["path"]).resolve()
    verification_path = Path(profile["verifier_receipt"]["path"]).resolve()
    experiment_store = ExperimentStore(mavis_home)
    active_experiment = experiment_store.active("main")
    experiment_id = active_experiment.get("experiment_id")
    if not isinstance(experiment_id, str) or experiment_path != experiment_store._record_path(experiment_id).resolve():
        raise ValueError("active main profile differs from active promoted experiment")
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if (experiment.get("schema_version") != "mavis.experiment-lifecycle/v1"
            or experiment.get("state") != "promoted"
            or experiment.get("scope") != "main"
            or experiment.get("candidate") != active_experiment["configuration"]
            or experiment_id not in profile.get("experiments", [])):
        raise ValueError("active main profile has no matching promoted experiment")
    review = experiment.get("review") or {}
    if (review.get("verdict") != "accepted" or review.get("path") != str(verification_path)
            or review.get("sha256") != profile["verifier_receipt"]["sha256"]
            or verification_path.parent != (mavis_home / "verifications" / "experiments").resolve()):
        raise ValueError("active main profile verifier does not match promoted experiment")
    candidate = experiment_store._read_snapshot(active_experiment["configuration"])
    if candidate != {"prompts": profile.get("prompts"), "tool_settings": profile.get("tool_settings"),
                     "retrieval": (profile.get("context_policy") or {}).get("retrieval", {})}:
        raise ValueError("active main profile configuration differs from promoted candidate")
    identity = profile.get("model_identity")
    if not isinstance(identity, dict) or not isinstance(identity.get("model_id"), str) or not identity["model_id"]:
        raise ValueError("active main profile requires exact model_identity.model_id")
    for key in ("architecture", "weights_fingerprint", "tokenizer_fingerprint", "chat_template_fingerprint", "quantization"):
        if not isinstance(identity.get(key), str) or not identity[key]:
            raise ValueError(f"active main profile lacks model_identity.{key}")
    runtime = profile.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("name") != "omlx":
        raise ValueError("active main profile requires oMLX runtime")
    prompts = profile.get("prompts")
    if not isinstance(prompts, dict) or set(prompts) - {"system"} or not isinstance(prompts.get("system", ""), str):
        raise ValueError("active main profile has unsupported prompt settings")
    if profile.get("tool_settings") != {} or profile.get("context_policy") != {"retrieval": {}}:
        raise ValueError("active main profile has settings the Codex launcher cannot apply")
    profile["_source_path"] = str(profile_path)
    profile["_source_sha256"] = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    return profile


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
    accepted_instructions_path: Path | None = None,
    mavis_home: Path | None = None,
    project_root: Path | None = None,
    model_dir: Path | None = None,
) -> None:
    if gateway_env_file is not None and gateway_root is None:
        raise ValueError("gateway environment file requires a trusted gateway")
    catalog_path = home / "omlx-models.json"
    hook_key = f"{(home / 'config.toml').resolve()}:pre_compact:0:0"
    hook_hash = mavis_archive_hook_hash()
    handoff_hook_key = f"{(home / 'config.toml').resolve()}:session_start:0:0"
    handoff_hook_hash = mavis_command_hook_hash(
        "session_start", MAVIS_HANDOFF_COMMAND, MAVIS_HANDOFF_STATUS
    )
    instructions_line = (f"model_instructions_file = {json.dumps(str(accepted_instructions_path.resolve()))}\n"
                         if accepted_instructions_path else "")
    profile = f"""model = {json.dumps(selected)}
model_provider = "omlx"
model_catalog_json = {json.dumps(str(catalog_path))}
{instructions_line}
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

[[hooks.SessionStart]]

[[hooks.SessionStart.hooks]]
type = "command"
command = {json.dumps(MAVIS_HANDOFF_COMMAND)}
statusMessage = {json.dumps(MAVIS_HANDOFF_STATUS)}

[hooks.state.{json.dumps(handoff_hook_key)}]
trusted_hash = {json.dumps(handoff_hook_hash)}

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
    memory_enabled = os.environ.get("MAVIS_MEMORY_MCP_ENABLED", "1").lower() not in {
        "0", "false", "no"
    }
    if mavis_home is not None and project_root is not None and memory_enabled:
        import importlib.util

        if importlib.util.find_spec("mcp") is None:
            raise RuntimeError("Mavis local memory MCP requires the Python mcp package")
        module_dir = Path(__file__).resolve().parent
        if (module_dir / "mavis" / "mavis" / "__init__.py").is_file():
            python_root = module_dir / "mavis"
        elif (module_dir / "mavis" / "__init__.py").is_file():
            python_root = module_dir
        else:
            raise FileNotFoundError("Mavis local memory MCP package is unavailable")
        profile += (
            "\n[mcp_servers.mavis-memory]\n"
            f"command = {json.dumps(sys.executable)}\n"
            'args = ["-m", "mavis.mcp_server"]\n'
            "\n[mcp_servers.mavis-memory.env]\n"
            f"MAVIS_HOME = {json.dumps(str(mavis_home.resolve()))}\n"
            f"MAVIS_PROJECT_ROOT = {json.dumps(str(project_root.resolve(strict=True)))}\n"
            f"MAVIS_LIBRARIAN_MODEL_PATH = {json.dumps(str(((model_dir or Path.home() / 'models/vlms') / 'Mavis-Qwen3.5-4B-HF-Eval').resolve()))}\n"
            f"PYTHONPATH = {json.dumps(str(python_root.resolve()))}\n"
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
    preserved_servers = tomllib.loads(suffix).get("mcp_servers", {}) if suffix else {}
    if gateway_root is not None and "model-gateway" in preserved_servers:
        raise ValueError(
            "existing Mavis gateway registration needs review before replacement"
        )
    if mavis_home is not None and "mavis-memory" in preserved_servers:
        raise ValueError("existing Mavis memory registration needs review before replacement")
    managed = f"{start_marker}\n{profile}{end_marker}\n"
    if suffix:
        managed += f"\n{suffix}"
    atomic_write(config_path, managed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path)
    parser.add_argument("--mavis-home", type=Path)
    parser.add_argument("--resolve-model", action="store_true")
    parser.add_argument("--print-receipt", action="store_true")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--persona-template", type=Path)
    parser.add_argument("--instructions-template", type=Path)
    parser.add_argument("--gateway-root", type=Path)
    parser.add_argument("--gateway-env-file", type=Path)
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()

    accepted = accepted_main_profile(args.mavis_home) if args.mavis_home else None
    accepted_model = accepted["model_identity"]["model_id"] if accepted else None
    if args.model and accepted_model and args.model != accepted_model:
        raise ValueError("explicit model conflicts with accepted main profile")
    if args.resolve_model:
        print(accepted_model or args.model or "Qwen3.8-Flash-Next-Abliterated-MLX-4bit")
        return 0
    if not all((args.home, args.base_url, args.persona_template, args.instructions_template)):
        parser.error("--home, --base-url, --persona-template, and --instructions-template are required")

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
    if accepted:
        custom_system = accepted["prompts"].get("system", "")
        if custom_system:
            base_instructions = base_instructions.rstrip() + "\n\n" + custom_system + "\n"
    catalog, selected = prepare_catalog(records, accepted_model or args.model, base_instructions)
    args.home.mkdir(parents=True, exist_ok=True)
    accepted_instructions_path = args.home / "accepted-model-instructions.md" if accepted else None
    if accepted_instructions_path is not None:
        atomic_write(accepted_instructions_path, base_instructions)
    atomic_write(args.home / "omlx-models.json", json.dumps(catalog, indent=2) + "\n")
    project = Path(os.environ["MAVIS_PROJECT_ROOT"]) if os.environ.get("MAVIS_PROJECT_ROOT") else None
    write_profile(args.home, base_url, selected, args.gateway_root, args.gateway_env_file,
                  accepted_instructions_path, args.mavis_home, project, args.model_dir)
    ensure_persona(args.home, args.persona_template)
    if accepted:
        config_path = args.home / "config.toml"
        catalog_path = args.home / "omlx-models.json"
        receipt = {
            "schema_version": "mavis.profile-launch/v1",
            "state": "prepared",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "profile_id": accepted["profile_id"],
            "profile_version": accepted["version"],
            "previous_version": accepted.get("previous_version"),
            "profile_path": accepted["_source_path"],
            "profile_sha256": accepted["_source_sha256"],
            "mavis_home": str(args.mavis_home.resolve()),
            "model_identity": accepted["model_identity"],
            "selected_model": selected,
            "effective_system_prompt_sha256": hashlib.sha256(base_instructions.encode()).hexdigest(),
            "instructions_path": str(accepted_instructions_path.resolve()),
            "instructions_sha256": hashlib.sha256(accepted_instructions_path.read_bytes()).hexdigest(),
            "config_path": str(config_path.resolve()),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "catalog_path": str(catalog_path.resolve()),
            "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        }
        receipt_path = args.mavis_home / "launches" / f"{uuid4()}.json"
        atomic_write(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(receipt_path if accepted and args.print_receipt else selected if not args.print_receipt else "")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"mavis: {error}", file=sys.stderr)
        raise SystemExit(2) from None
