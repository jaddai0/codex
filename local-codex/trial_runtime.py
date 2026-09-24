#!/usr/bin/env python3
"""Prepare an isolated Mavis E1 prompt trial without activating a profile."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib

from mavis.e1 import E1Runner, _clean_revision
from mavis.e1_bootstrap import _inventory_model, validate_bootstrap
from mavis.e1_gpu import admitted_e1_model
from mavis.package_provenance import package_tree_sha256
from mavis.runtime import RuntimeConfig, inventory
from mavis.storage import read_json, require_safe_id, sha256_file
from prepare_runtime import (
    accepted_main_profile,
    atomic_write,
    ensure_persona,
    prepare_catalog,
    write_profile,
    local_base_url,
)


def trial_binding(mavis_home: Path, experiment_id: str, arm: str, case_id: str) -> dict:
    experiment_id = require_safe_id(experiment_id, "experiment id")
    case_id = require_safe_id(case_id, "case id")
    if not all(
        re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in (experiment_id, case_id)
    ):
        raise ValueError(
            "E1 trial IDs must use letters, digits, underscores, or hyphens"
        )
    if arm not in {"baseline", "candidate"}:
        raise ValueError("E1 trial arm must be baseline or candidate")
    runner = E1Runner(mavis_home)
    record, manifest = runner._frozen(experiment_id)
    if (
        record["state"] != "candidate"
        or record["scope"] != "main"
        or record["kind"] != "prompts"
    ):
        raise ValueError("E1 trial requires a pending main prompt candidate")
    case = next((item for item in manifest["cases"] if item["id"] == case_id), None)
    if case is None:
        raise ValueError("unknown E1 case")
    if not isinstance(case.get("task"), str) or not case["task"].strip():
        raise ValueError("E1 case needs one frozen task")
    baseline = runner.store._read_snapshot(record["baseline"])
    candidate = runner.store._read_snapshot(record["candidate"])
    if (
        baseline.get("tool_settings") != {}
        or candidate.get("tool_settings") != {}
        or baseline.get("retrieval") != {}
        or candidate.get("retrieval") != {}
        or set(baseline.get("prompts", {})) != {"system"}
        or set(candidate.get("prompts", {})) != {"system"}
        or not isinstance(baseline["prompts"]["system"], str)
        or not isinstance(candidate["prompts"]["system"], str)
        or baseline["prompts"]["system"] == candidate["prompts"]["system"]
    ):
        raise ValueError("E1 trial supports only one changed system prompt")
    active = runner.store.active("main")
    if active["configuration"] != record["baseline"]:
        raise ValueError("E1 baseline no longer matches active configuration")
    profile = accepted_main_profile(mavis_home)
    profile_source = "accepted-main"
    if profile is None:
        profile = validate_bootstrap(mavis_home)
        profile_source = "e0-bootstrap"
    if (
        {"prompts": profile["prompts"], "tool_settings": {}, "retrieval": {}}
        != baseline
    ):
        raise ValueError("E1 baseline no longer matches accepted main profile")
    root = runner._root(experiment_id)
    checkout = root / "checkouts" / arm / case_id
    prepared = read_json(root / f"prepared-{arm}.json")
    entry = prepared.get(case_id)
    if (
        not isinstance(entry, dict)
        or Path(entry.get("path", "")).resolve() != checkout.resolve()
        or entry.get("starting_revision") != case["revision"]
        or _clean_revision(checkout) != case["revision"]
    ):
        raise ValueError(
            "E1 trial checkout differs from clean prepared starting revision"
        )
    return {
        "record": record,
        "manifest": manifest,
        "case": case,
        "profile": profile,
        "profile_source": profile_source,
        "snapshot": record[arm],
        "configuration": baseline if arm == "baseline" else candidate,
        "root": root,
        "checkout": checkout,
        "arm": arm,
        "case_id": case_id,
    }


def prepare_trial(
    binding: dict,
    *,
    mavis_home: Path,
    share: Path,
    core_binary: Path,
    base_url: str,
    records: list[dict],
    gateway_root: Path | None = None,
    gateway_env_file: Path | None = None,
    package_manifest: Path | None = None,
) -> Path:
    """Prepare one arm while holding the same lease used by runtime admission."""
    from generation_lease import generation_lease

    with generation_lease(mavis_home, purpose="e1-preparation"):
        return _prepare_trial_locked(
            binding,
            mavis_home=mavis_home,
            share=share,
            core_binary=core_binary,
            base_url=base_url,
            records=records,
            gateway_root=gateway_root,
            gateway_env_file=gateway_env_file,
            package_manifest=package_manifest,
        )


def _prepare_trial_locked(
    binding: dict,
    *,
    mavis_home: Path,
    share: Path,
    core_binary: Path,
    base_url: str,
    records: list[dict],
    gateway_root: Path | None = None,
    gateway_env_file: Path | None = None,
    package_manifest: Path | None = None,
) -> Path:
    if binding["profile_source"] == "e0-bootstrap":
        refreshed = trial_binding(mavis_home, binding["record"]["experiment_id"],
                                  binding["arm"], binding["case_id"])
        if (refreshed["profile_source"] != "e0-bootstrap"
                or refreshed["profile"]["_source_sha256"] != binding["profile"]["_source_sha256"]):
            raise ValueError("E1 bootstrap changed before trial preparation")
        observed_model_path = _inventory_model(
            records, binding["profile"]["model_identity"]["model_id"]
        ).resolve(strict=True)
        if str(observed_model_path) != binding["profile"]["model_artifacts"]["model_path"]:
            raise ValueError("E1 bootstrap oMLX inventory model_path changed")
    record = binding["record"]
    arm, case_id = binding["arm"], binding["case_id"]
    runtime_home = binding["root"] / "runtime" / arm / case_id
    receipt_path = binding["root"] / "trials" / arm / f"{case_id}.json"
    _recover_incomplete_preparation(
        runtime_home,
        receipt_path,
        experiment_id=record["experiment_id"],
        arm=arm,
        case_id=case_id,
    )
    if runtime_home.exists() or receipt_path.exists():
        raise FileExistsError("E1 trial already prepared")
    if not core_binary.is_file() or not os.access(core_binary, os.X_OK):
        raise FileNotFoundError("installed Mavis core is unavailable")
    package_hash = None
    if package_manifest is not None:
        package = read_json(package_manifest)
        launcher = Path(package.get("launcher", ""))
        if (
            package.get("schema_version") != "mavis.installed-core/v1"
            or Path(package.get("core_binary", "")).resolve() != core_binary.resolve()
            or package.get("core_sha256") != sha256_file(core_binary)
            or package.get("trial_runtime_sha256") != sha256_file(Path(__file__))
            or package.get("launch_core_sha256") != sha256_file(package_manifest.parent / "launch_core.py")
            or package.get("prepare_runtime_sha256") != sha256_file(package_manifest.parent / "prepare_runtime.py")
            or package.get("generation_lease_sha256") != sha256_file(package_manifest.parent / "generation_lease.py")
            or package.get("base_instructions_sha256") != sha256_file(package_manifest.parent / "base-instructions.md")
            or package.get("persona_sha256") != sha256_file(package_manifest.parent / "persona.toml")
            or package.get("mavis_package_sha256") != package_tree_sha256(package_manifest.parent / "mavis")
            or not launcher.is_file()
            or package.get("launcher_sha256") != sha256_file(launcher)
        ):
            raise ValueError("E1 installed core differs from package manifest")
        package_hash = sha256_file(package_manifest)
    template = (share / "base-instructions.md").read_text(encoding="utf-8")
    if not template.strip():
        raise ValueError("Mavis base instructions are empty")
    effective = (
        template.rstrip()
        + "\n\n"
        + binding["configuration"]["prompts"]["system"]
        + "\n"
    )
    model_id = binding["profile"]["model_identity"]["model_id"]
    catalog, selected = prepare_catalog(records, model_id, effective)
    runtime_home.mkdir(parents=True, mode=0o700)
    marker = runtime_home / ".e1-preparing.json"
    try:
        atomic_write(
            marker,
            json.dumps(
                {
                    "schema_version": "mavis.e1-preparing/v1",
                    "experiment_id": record["experiment_id"],
                    "arm": arm,
                    "case_id": case_id,
                }
            )
            + "\n",
        )
        result = _write_trial_home(
            binding,
            mavis_home=mavis_home,
            share=share,
            core_binary=core_binary,
            base_url=base_url,
            catalog=catalog,
            selected=selected,
            effective=effective,
            runtime_home=runtime_home,
            receipt_path=receipt_path,
            gateway_root=gateway_root,
            gateway_env_file=gateway_env_file,
            package_manifest=package_manifest,
            package_hash=package_hash,
        )
        marker.unlink()
        return result
    except BaseException:
        if not receipt_path.exists():
            shutil.rmtree(runtime_home)
        raise


def _recover_incomplete_preparation(
    runtime_home: Path,
    receipt_path: Path,
    *,
    experiment_id: str,
    arm: str,
    case_id: str,
) -> None:
    if not runtime_home.exists() or receipt_path.exists():
        return
    marker_path = runtime_home / ".e1-preparing.json"
    if not marker_path.is_file() or runtime_home.is_symlink():
        return
    marker = read_json(marker_path)
    if marker != {
        "schema_version": "mavis.e1-preparing/v1",
        "experiment_id": experiment_id,
        "arm": arm,
        "case_id": case_id,
    }:
        return
    expected = {
        marker_path.name,
        "accepted-model-instructions.md",
        "omlx-models.json",
        "config.toml",
        "AGENTS.md",
    }
    if any(
        path.name not in expected or path.is_symlink() or not path.is_file()
        for path in runtime_home.iterdir()
    ):
        raise ValueError("E1 incomplete trial home has unexpected content")
    shutil.rmtree(runtime_home)


def _write_trial_home(
    binding: dict,
    *,
    mavis_home: Path,
    share: Path,
    core_binary: Path,
    base_url: str,
    catalog: dict,
    selected: str,
    effective: str,
    runtime_home: Path,
    receipt_path: Path,
    gateway_root: Path | None,
    gateway_env_file: Path | None,
    package_manifest: Path | None,
    package_hash: str | None,
) -> Path:
    record = binding["record"]
    arm, case_id = binding["arm"], binding["case_id"]
    instructions_path = runtime_home / "accepted-model-instructions.md"
    catalog_path = runtime_home / "omlx-models.json"
    atomic_write(instructions_path, effective)
    atomic_write(catalog_path, json.dumps(catalog, indent=2) + "\n")
    write_profile(
        runtime_home,
        local_base_url(base_url),
        selected,
        gateway_root=gateway_root,
        gateway_env_file=gateway_env_file,
        accepted_instructions_path=instructions_path,
    )
    ensure_persona(runtime_home, share / "persona.toml")
    trial_id = f"{record['experiment_id']}-{arm}-{case_id}"
    core_argv = [
        str(core_binary.resolve()),
        "exec",
        "-C",
        str(binding["checkout"].resolve()),
        "--",
        binding["case"]["task"],
    ]
    receipt = {
        "schema_version": "mavis.e1-trial-launch/v1",
        "state": "prepared",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trial_id": trial_id,
        "experiment_id": record["experiment_id"],
        "arm": arm,
        "case_id": case_id,
        "task": binding["case"]["task"],
        "task_sha256": hashlib.sha256(binding["case"]["task"].encode()).hexdigest(),
        "core_argv": core_argv,
        "core_argv_sha256": hashlib.sha256(
            json.dumps(core_argv, separators=(",", ":")).encode()
        ).hexdigest(),
        "manifest_sha256": record["workload"]["manifest_sha256"],
        "snapshot_path": binding["snapshot"]["path"],
        "snapshot_sha256": binding["snapshot"]["sha256"],
        "profile_source": binding["profile_source"],
        "accepted_profile_id": binding["profile"]["profile_id"] if binding["profile_source"] == "accepted-main" else None,
        "accepted_profile_path": binding["profile"]["_source_path"] if binding["profile_source"] == "accepted-main" else None,
        "accepted_profile_sha256": binding["profile"]["_source_sha256"] if binding["profile_source"] == "accepted-main" else None,
        "bootstrap_receipt_path": binding["profile"]["_source_path"] if binding["profile_source"] == "e0-bootstrap" else None,
        "bootstrap_receipt_sha256": binding["profile"]["_source_sha256"] if binding["profile_source"] == "e0-bootstrap" else None,
        "checkout": str(binding["checkout"].resolve()),
        "starting_revision": binding["case"]["revision"],
        "runtime_home": str(runtime_home.resolve()),
        "mavis_home": str(mavis_home.resolve()),
        "core_binary": str(core_binary.resolve()),
        "core_sha256": sha256_file(core_binary),
        "package_manifest": str(package_manifest.resolve())
        if package_manifest
        else None,
        "package_manifest_sha256": package_hash,
        "core_provenance": "installed-package"
        if package_manifest
        else "fixture-unverified",
        "gateway_root": str(gateway_root.resolve()) if gateway_root else None,
        "gateway_launcher_sha256": sha256_file(gateway_root / "bin/mcp-server.sh")
        if gateway_root
        else None,
        "gateway_env_file": str(gateway_env_file.resolve())
        if gateway_env_file
        else None,
        "gateway_env_sha256": sha256_file(gateway_env_file)
        if gateway_env_file
        else None,
        "selected_model": selected,
        "model_provider": "omlx",
        "model_endpoint": local_base_url(base_url),
        "config_path": str((runtime_home / "config.toml").resolve()),
        "config_sha256": sha256_file(runtime_home / "config.toml"),
        "catalog_path": str(catalog_path.resolve()),
        "catalog_sha256": sha256_file(catalog_path),
        "instructions_path": str(instructions_path.resolve()),
        "instructions_sha256": sha256_file(instructions_path),
        "effective_system_prompt_sha256": hashlib.sha256(
            effective.encode()
        ).hexdigest(),
        "observation_path": str((runtime_home / "effective-config.json").resolve()),
        "stdout_path": str((runtime_home / "stdout.log").resolve()),
        "stderr_path": str((runtime_home / "stderr.log").resolve()),
    }
    atomic_write(receipt_path, json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt_path


def validate_trial_receipt(receipt: dict) -> None:
    if (
        receipt.get("schema_version") != "mavis.e1-trial-launch/v1"
        or receipt.get("state") != "prepared"
    ):
        raise ValueError("E1 trial receipt is not prepared")
    home = Path(receipt["mavis_home"])
    binding = trial_binding(
        home, receipt["experiment_id"], receipt["arm"], receipt["case_id"]
    )
    profile = binding["profile"]
    if receipt.get("profile_source", "accepted-main") != binding["profile_source"]:
        raise ValueError("E1 trial profile source changed")
    if binding["profile_source"] == "e0-bootstrap":
        if (receipt.get("accepted_profile_id") is not None
                or receipt.get("accepted_profile_path") is not None
                or receipt.get("accepted_profile_sha256") is not None
                or receipt.get("bootstrap_receipt_path") != profile["_source_path"]
                or receipt.get("bootstrap_receipt_sha256") != profile["_source_sha256"]):
            raise ValueError("E1 trial bootstrap receipt changed")
    elif (receipt.get("accepted_profile_sha256") != profile["_source_sha256"]
          or receipt.get("accepted_profile_id") != profile["profile_id"]):
        raise ValueError("E1 trial accepted profile changed")
    if (
        receipt["manifest_sha256"] != binding["record"]["workload"]["manifest_sha256"]
        or receipt["snapshot_path"] != binding["snapshot"]["path"]
        or receipt["snapshot_sha256"] != binding["snapshot"]["sha256"]
        or receipt["checkout"] != str(binding["checkout"].resolve())
        or receipt["starting_revision"] != binding["case"]["revision"]
        or receipt["selected_model"] != profile["model_identity"]["model_id"]
    ):
        raise ValueError("E1 trial receipt lost its frozen binding")
    expected_argv = [
        receipt["core_binary"],
        "exec",
        "-C",
        receipt["checkout"],
        "--",
        binding["case"]["task"],
    ]
    if (
        receipt.get("task") != binding["case"]["task"]
        or receipt.get("task_sha256")
        != hashlib.sha256(binding["case"]["task"].encode()).hexdigest()
        or receipt.get("core_argv") != expected_argv
        or receipt.get("core_argv_sha256")
        != hashlib.sha256(
            json.dumps(expected_argv, separators=(",", ":")).encode()
        ).hexdigest()
    ):
        raise ValueError("E1 trial task or core command differs from frozen case")
    if (
        Path(receipt["runtime_home"]).resolve()
        != (binding["root"] / "runtime" / receipt["arm"] / receipt["case_id"]).resolve()
    ):
        raise ValueError("E1 trial runtime home changed")
    for name in ("core", "config", "catalog", "instructions"):
        path_key = "core_binary" if name == "core" else f"{name}_path"
        if sha256_file(Path(receipt[path_key])) != receipt[f"{name}_sha256"]:
            raise ValueError(f"E1 trial {name} changed after preparation")
    if binding["profile_source"] == "e0-bootstrap":
        config = tomllib.loads(Path(receipt["config_path"]).read_text(encoding="utf-8"))
        endpoint = config["model_providers"]["omlx"]["base_url"]
        if receipt.get("model_endpoint") != endpoint or local_base_url(endpoint) != endpoint:
            raise ValueError("E1 bootstrap model endpoint changed")
        model_path = _inventory_model(inventory(endpoint), receipt["selected_model"]).resolve(strict=True)
        if str(model_path) != profile["model_artifacts"]["model_path"]:
            raise ValueError("E1 bootstrap oMLX inventory model_path changed before launch")
    if receipt.get("core_provenance") == "installed-package" and not receipt.get(
        "package_manifest"
    ):
        raise ValueError("E1 installed core provenance is missing")
    if receipt.get("package_manifest"):
        package_path = Path(receipt["package_manifest"])
        package = read_json(package_path)
        launcher = Path(package.get("launcher", ""))
        if (
            sha256_file(package_path) != receipt.get("package_manifest_sha256")
            or package.get("schema_version") != "mavis.installed-core/v1"
            or package.get("core_binary") != receipt["core_binary"]
            or package.get("core_sha256") != receipt["core_sha256"]
            or package.get("trial_runtime_sha256") != sha256_file(Path(__file__))
            or package.get("launch_core_sha256") != sha256_file(package_path.parent / "launch_core.py")
            or package.get("prepare_runtime_sha256") != sha256_file(package_path.parent / "prepare_runtime.py")
            or package.get("generation_lease_sha256") != sha256_file(package_path.parent / "generation_lease.py")
            or package.get("base_instructions_sha256") != sha256_file(package_path.parent / "base-instructions.md")
            or package.get("persona_sha256") != sha256_file(package_path.parent / "persona.toml")
            or package.get("mavis_package_sha256") != package_tree_sha256(package_path.parent / "mavis")
            or not launcher.is_file()
            or package.get("launcher_sha256") != sha256_file(launcher)
            or receipt.get("core_provenance") != "installed-package"
        ):
            raise ValueError("E1 installed core provenance changed")
    for key, digest_key in (
        ("gateway_root", "gateway_launcher_sha256"),
        ("gateway_env_file", "gateway_env_sha256"),
    ):
        if receipt.get(key):
            path = (
                Path(receipt[key]) / "bin/mcp-server.sh"
                if key == "gateway_root"
                else Path(receipt[key])
            )
            if sha256_file(path) != receipt[digest_key]:
                raise ValueError("E1 gateway source changed")
    if receipt["instructions_sha256"] != receipt["effective_system_prompt_sha256"]:
        raise ValueError("E1 trial effective instructions changed")


def run_trial(experiment_id: str, arm: str, case_id: str, task: str) -> Path:
    if not task.strip():
        raise ValueError("E1 trial task is required")
    mavis_home = Path(
        os.environ.get("MAVIS_HOME", Path.home() / ".local-codex/mavis-service")
    ).resolve()
    share = Path(
        os.environ.get(
            "LOCAL_CODEX_SHARE_DIR", Path.home() / ".local/share/local-codex"
        )
    ).resolve()
    core_binary = Path(
        os.environ.get("LOCAL_CODEX_BIN", share / "local-codex-core")
    ).resolve()
    endpoint = os.environ.get("OMLX_BASE_URL", "http://127.0.0.1:8001/v1")
    binding = trial_binding(mavis_home, experiment_id, arm, case_id)
    if task != binding["case"]["task"]:
        raise ValueError("E1 task differs from frozen case")
    package_manifest = share / "install-manifest.json"
    if (
        core_binary != (share / "local-codex-core").resolve()
        or not package_manifest.is_file()
    ):
        raise ValueError("E1 trial requires the installed package core")
    package = read_json(package_manifest)
    launcher = Path(package.get("launcher", ""))
    if (
        package.get("schema_version") != "mavis.installed-core/v1"
        or package.get("core_binary") != str(core_binary)
        or package.get("core_sha256") != sha256_file(core_binary)
        or package.get("trial_runtime_sha256") != sha256_file(Path(__file__))
        or package.get("launch_core_sha256") != sha256_file(share / "launch_core.py")
        or package.get("prepare_runtime_sha256") != sha256_file(share / "prepare_runtime.py")
        or package.get("generation_lease_sha256") != sha256_file(share / "generation_lease.py")
        or package.get("base_instructions_sha256") != sha256_file(share / "base-instructions.md")
        or package.get("persona_sha256") != sha256_file(share / "persona.toml")
        or package.get("mavis_package_sha256") != package_tree_sha256(share / "mavis")
        or not launcher.is_file()
        or package.get("launcher_sha256") != sha256_file(launcher)
    ):
        raise ValueError("E1 installed package manifest differs from trial runtime")
    gateway_root = Path(
        os.environ.get(
            "MAVIS_GATEWAY_ROOT",
            Path.home()
            / "Dev-Projects/Mavis/checkouts/gateway/marketplace/plugins/model-gateway",
        )
    ).resolve()
    gateway_launcher = gateway_root / "bin/mcp-server.sh"
    if not gateway_launcher.is_file() or not os.access(gateway_launcher, os.X_OK):
        raise FileNotFoundError("trusted Mavis gateway launcher is unavailable")
    configured_env = os.environ.get("MAVIS_GATEWAY_ENV_FILE")
    default_env = Path.home() / "Dev-Projects/env.env"
    gateway_env_file = (
        Path(configured_env).resolve()
        if configured_env
        else (default_env.resolve() if default_env.is_file() else None)
    )
    if gateway_env_file is not None and not gateway_env_file.is_file():
        raise FileNotFoundError("configured gateway environment file is unavailable")
    if (
        not core_binary.is_file()
        or not os.access(core_binary, os.X_OK)
        or not (share / "base-instructions.md").is_file()
        or not (share / "persona.toml").is_file()
    ):
        raise FileNotFoundError("installed E1 trial launcher inputs are unavailable")
    selected = binding["profile"]["model_identity"]["model_id"]
    runtime = RuntimeConfig(
        home=mavis_home,
        endpoint=endpoint,
        iris_endpoint=os.environ.get("IRIS_OMLX_BASE_URL", "http://127.0.0.1:8000/v1"),
        model=selected,
        model_dir=Path(os.environ.get("MAVIS_MODEL_DIR", Path.home() / "models/vlms")),
        omlx_binary=Path(
            os.environ.get("MAVIS_OMLX_BIN", Path.home() / ".venvs/omlx-dev/bin/omlx")
        ),
    )
    runtime_home = binding["root"] / "runtime" / arm / case_id
    receipt_path = binding["root"] / "trials" / arm / f"{case_id}.json"
    from generation_lease import generation_lease
    from launch_core import launch_trial

    with generation_lease(mavis_home, purpose=f"e1:{experiment_id}:{arm}:{case_id}"):
        _recover_incomplete_preparation(
            runtime_home,
            receipt_path,
            experiment_id=experiment_id,
            arm=arm,
            case_id=case_id,
        )
        if runtime_home.exists() or receipt_path.exists():
            raise FileExistsError("E1 trial already prepared")
        with admitted_e1_model(runtime, f"e1:{experiment_id}:{arm}:{case_id}") as heartbeat:
            refreshed = trial_binding(mavis_home, experiment_id, arm, case_id)
            if (
                refreshed["snapshot"] != binding["snapshot"]
                or refreshed["profile"]["_source_sha256"]
                != binding["profile"]["_source_sha256"]
            ):
                raise ValueError("E1 trial inputs changed during runtime admission")
            binding = refreshed
            receipt = _prepare_trial_locked(
                binding,
                mavis_home=mavis_home,
                share=share,
                core_binary=core_binary,
                base_url=endpoint,
                records=inventory(endpoint),
                gateway_root=gateway_root,
                gateway_env_file=gateway_env_file,
                package_manifest=package_manifest,
            )
            argv = [
                str(core_binary),
                "exec",
                "-C",
                str(binding["checkout"].resolve()),
                "--",
                task,
            ]
            if launch_trial(receipt, argv, lease_held=True, heartbeat=heartbeat) != 0:
                raise RuntimeError(
                    f"E1 trial did not produce a matching effective-config observation: {receipt}"
                )
            return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id")
    parser.add_argument("arm", choices=("baseline", "candidate"))
    parser.add_argument("case_id")
    parser.add_argument("task", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.task[:1] != ["--"] or len(args.task) != 2:
        parser.error("supply exactly one task after --")
    print(run_trial(args.experiment_id, args.arm, args.case_id, args.task[1]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"mavis: {error}", file=sys.stderr)
        raise SystemExit(2) from None
