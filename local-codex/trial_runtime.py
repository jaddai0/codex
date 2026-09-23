#!/usr/bin/env python3
"""Prepare an isolated Mavis E1 prompt trial without activating a profile."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from mavis.e1 import E1Runner, _clean_revision, _git
from mavis.runtime import RuntimeConfig, ensure_runtime, inventory
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
    if (
        profile is None
        or {"prompts": profile["prompts"], "tool_settings": {}, "retrieval": {}}
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
) -> Path:
    record = binding["record"]
    arm, case_id = binding["arm"], binding["case_id"]
    runtime_home = binding["root"] / "runtime" / arm / case_id
    receipt_path = binding["root"] / "trials" / arm / f"{case_id}.json"
    if runtime_home.exists() or receipt_path.exists():
        raise FileExistsError("E1 trial already prepared")
    if not core_binary.is_file() or not os.access(core_binary, os.X_OK):
        raise FileNotFoundError("installed Mavis core is unavailable")
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
    instructions_path = runtime_home / "accepted-model-instructions.md"
    catalog_path = runtime_home / "omlx-models.json"
    atomic_write(instructions_path, effective)
    atomic_write(catalog_path, json.dumps(catalog, indent=2) + "\n")
    write_profile(
        runtime_home,
        local_base_url(base_url),
        selected,
        accepted_instructions_path=instructions_path,
    )
    ensure_persona(runtime_home, share / "persona.toml")
    trial_id = f"{record['experiment_id']}-{arm}-{case_id}"
    receipt = {
        "schema_version": "mavis.e1-trial-launch/v1",
        "state": "prepared",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trial_id": trial_id,
        "experiment_id": record["experiment_id"],
        "arm": arm,
        "case_id": case_id,
        "manifest_sha256": record["workload"]["manifest_sha256"],
        "snapshot_path": binding["snapshot"]["path"],
        "snapshot_sha256": binding["snapshot"]["sha256"],
        "accepted_profile_id": binding["profile"]["profile_id"],
        "accepted_profile_path": binding["profile"]["_source_path"],
        "accepted_profile_sha256": binding["profile"]["_source_sha256"],
        "checkout": str(binding["checkout"].resolve()),
        "starting_revision": binding["case"]["revision"],
        "runtime_home": str(runtime_home.resolve()),
        "mavis_home": str(mavis_home.resolve()),
        "core_binary": str(core_binary.resolve()),
        "core_sha256": sha256_file(core_binary),
        "selected_model": selected,
        "model_provider": "omlx",
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
    if (
        receipt["manifest_sha256"] != binding["record"]["workload"]["manifest_sha256"]
        or receipt["snapshot_path"] != binding["snapshot"]["path"]
        or receipt["snapshot_sha256"] != binding["snapshot"]["sha256"]
        or receipt["accepted_profile_sha256"] != profile["_source_sha256"]
        or receipt["accepted_profile_id"] != profile["profile_id"]
        or receipt["checkout"] != str(binding["checkout"].resolve())
        or receipt["starting_revision"] != binding["case"]["revision"]
        or receipt["selected_model"] != profile["model_identity"]["model_id"]
    ):
        raise ValueError("E1 trial receipt lost its frozen binding")
    if (
        Path(receipt["runtime_home"]).resolve()
        != (binding["root"] / "runtime" / receipt["arm"] / receipt["case_id"]).resolve()
    ):
        raise ValueError("E1 trial runtime home changed")
    for name in ("core", "config", "catalog", "instructions"):
        path_key = "core_binary" if name == "core" else f"{name}_path"
        if sha256_file(Path(receipt[path_key])) != receipt[f"{name}_sha256"]:
            raise ValueError(f"E1 trial {name} changed after preparation")
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
    if (binding["root"] / "runtime" / arm / case_id).exists() or (
        binding["root"] / "trials" / arm / f"{case_id}.json"
    ).exists():
        raise FileExistsError("E1 trial already prepared")
    ensure_runtime(runtime)
    refreshed = trial_binding(mavis_home, experiment_id, arm, case_id)
    if (
        refreshed["snapshot"] != binding["snapshot"]
        or refreshed["profile"]["_source_sha256"]
        != binding["profile"]["_source_sha256"]
    ):
        raise ValueError("E1 trial inputs changed during runtime admission")
    binding = refreshed
    receipt = prepare_trial(
        binding,
        mavis_home=mavis_home,
        share=share,
        core_binary=core_binary,
        base_url=endpoint,
        records=inventory(endpoint),
    )
    from launch_core import launch_trial

    if (
        launch_trial(
            receipt,
            [str(core_binary), "exec", "-C", str(binding["checkout"]), "--", task],
        )
        != 0
    ):
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
