#!/usr/bin/env python3
"""Spawn Codex and retain a host-observed profile launch receipt."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import os
import time

from prepare_runtime import accepted_main_profile, atomic_write
from generation_lease import generation_lease


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stop_process_group(child: subprocess.Popen) -> None:
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def matching_trial_transcript(path: Path, runtime_home: Path, session_id: str) -> Path:
    home = runtime_home.resolve(strict=True)
    transcript = path.resolve(strict=True)
    if home not in transcript.parents or not transcript.is_file():
        raise ValueError("core E1 transcript is outside disposable home")
    with transcript.open(encoding="utf-8") as stream:
        first = json.loads(stream.readline())
    metadata = first.get("payload") if isinstance(first, dict) else None
    if (
        not isinstance(first, dict)
        or first.get("type") != "session_meta"
        or not isinstance(metadata, dict)
        or str(metadata.get("id")) != str(session_id)
    ):
        raise ValueError("core E1 transcript session differs from observation")
    return transcript


def check_profile_cli_arguments(arguments: list[str]) -> None:
    """Refuse Codex flags that can change an accepted profile's effective runtime."""
    long_flags = (
        "--model",
        "--profile",
        "--config",
        "--oss",
        "--local-provider",
        "--enable",
        "--disable",
    )
    for argument in arguments:
        if argument == "--":
            break
        if argument in long_flags or any(
            argument.startswith(flag + "=") for flag in long_flags
        ):
            raise ValueError(f"{argument} can override an accepted main profile")
        if argument.startswith(("-m", "-p", "-c")) and not argument.startswith("--"):
            raise ValueError(f"{argument} can override an accepted main profile")


def launch(receipt_path: Path | None, argv: list[str]) -> int:
    if receipt_path is not None:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        with generation_lease(Path(receipt["mavis_home"]), purpose="accepted-main"):
            return _launch_unlocked(receipt_path, argv)
    return _launch_unlocked(receipt_path, argv)


def _launch_unlocked(receipt_path: Path | None, argv: list[str]) -> int:
    if not argv:
        raise ValueError("core binary is required")
    receipt = None
    launch_argv = argv
    if receipt_path is not None:
        check_profile_cli_arguments(argv[1:])
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("state") != "prepared":
            raise ValueError("profile launch receipt is not prepared")
        for name in ("config", "catalog", "instructions"):
            if digest(Path(receipt[f"{name}_path"])) != receipt[f"{name}_sha256"]:
                raise ValueError(f"profile {name} changed after preparation")
        if receipt["instructions_sha256"] != receipt["effective_system_prompt_sha256"]:
            raise ValueError(
                "accepted instructions differ from recorded effective prompt"
            )
        if digest(Path(receipt["profile_path"])) != receipt["profile_sha256"]:
            raise ValueError("accepted profile changed after preparation")
        active = accepted_main_profile(Path(receipt["mavis_home"]))
        if active is None or active["_source_sha256"] != receipt["profile_sha256"]:
            raise ValueError("accepted main profile is no longer active")
        if receipt["selected_model"] != active["model_identity"]["model_id"]:
            raise ValueError("recorded model differs from accepted main profile")
        bindings = {
            "model": receipt["selected_model"],
            "model_provider": "omlx",
            "model_catalog_json": receipt["catalog_path"],
            "model_instructions_file": receipt["instructions_path"],
        }
        overrides = [f"{key}={json.dumps(value)}" for key, value in bindings.items()]
        launch_argv = [
            argv[0],
            *(item for override in overrides for item in ("-c", override)),
            *argv[1:],
        ]
        receipt["binding_overrides"] = overrides
    child = subprocess.Popen(launch_argv)
    if receipt is not None:
        receipt.update(
            {
                "state": "spawned",
                "spawned_at": datetime.now(timezone.utc).isoformat(),
                "core_binary": str(Path(argv[0]).resolve()),
                "core_pid": child.pid,
            }
        )
        try:
            atomic_write(
                receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n"
            )
        except BaseException:
            child.terminate()
            child.wait()
            raise

    def forward(signum, _frame):
        if child.poll() is None:
            child.send_signal(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, forward)
    return_code = child.wait()
    if receipt is not None:
        receipt.update(
            {
                "state": "exited",
                "exited_at": datetime.now(timezone.utc).isoformat(),
                "core_exit_code": return_code,
            }
        )
        atomic_write(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return return_code if return_code >= 0 else 128 - return_code


def launch_trial(
    receipt_path: Path, argv: list[str], *, lease_held: bool = False
) -> int:
    """Run a frozen E1 arm; process exit alone never establishes trial success."""
    if not lease_held:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        with generation_lease(Path(receipt["mavis_home"]), purpose="e1-trial"):
            return _launch_trial_unlocked(receipt_path, argv)
    return _launch_trial_unlocked(receipt_path, argv)


def _launch_trial_unlocked(receipt_path: Path, argv: list[str]) -> int:
    from trial_runtime import validate_trial_receipt

    if not argv:
        raise ValueError("E1 core command is required")
    check_profile_cli_arguments(argv[1:])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    validate_trial_receipt(receipt)
    if argv != receipt["core_argv"]:
        raise ValueError("E1 core command differs from frozen receipt")
    bindings = {
        "model": receipt["selected_model"],
        "model_provider": "omlx",
        "model_catalog_json": receipt["catalog_path"],
        "model_instructions_file": receipt["instructions_path"],
    }
    overrides = [f"{key}={json.dumps(value)}" for key, value in bindings.items()]
    launch_argv = [
        argv[0],
        *(item for override in overrides for item in ("-c", override)),
        *argv[1:],
    ]
    receipt["binding_overrides"] = overrides
    receipt["effective_argv"] = launch_argv
    receipt["effective_argv_sha256"] = hashlib.sha256(
        json.dumps(launch_argv, separators=(",", ":")).encode()
    ).hexdigest()
    environment = os.environ.copy()
    environment.update(
        {
            "CODEX_HOME": receipt["runtime_home"],
            "MAVIS_HOME": receipt["mavis_home"],
            "MAVIS_E1_TRIAL_ID": receipt["trial_id"],
            "MAVIS_E1_EFFECTIVE_CONFIG_EVENT": receipt["observation_path"],
            "MAVIS_E1_CATALOG_PATH": receipt["catalog_path"],
            "MAVIS_PRECOMPACT_REQUIRED": "1",
            "MAVIS_RAW_OUTPUT_REQUIRED": "1",
        }
    )
    share = str(Path(receipt["core_binary"]).parent)
    environment["PYTHONPATH"] = share + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    checkout = Path(receipt["checkout"])
    with (
        Path(receipt["stdout_path"]).open("xb") as stdout,
        Path(receipt["stderr_path"]).open("xb") as stderr,
    ):
        child = subprocess.Popen(
            launch_argv,
            cwd=checkout,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        receipt.update(
            {
                "state": "spawned",
                "spawned_at": datetime.now(timezone.utc).isoformat(),
                "core_pid": child.pid,
            }
        )
        try:
            atomic_write(
                receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n"
            )
        except BaseException:
            _stop_process_group(child)
            raise
        terminated_signal = None
        terminated_at = None
        previous_handlers = {}

        def forward(signum, _frame):
            nonlocal terminated_signal, terminated_at
            if terminated_signal is None:
                terminated_signal = signum
                terminated_at = time.monotonic()
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, forward)
        try:
            while True:
                try:
                    exit_code = child.wait(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    if (
                        terminated_at is not None
                        and time.monotonic() - terminated_at >= 5
                    ):
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        exit_code = child.wait()
                        break
        except BaseException:
            _stop_process_group(child)
            raise
        finally:
            for signum, old in previous_handlers.items():
                signal.signal(signum, old)
    receipt.update(
        {
            "state": "terminated" if terminated_signal is not None else "exited",
            "exited_at": datetime.now(timezone.utc).isoformat(),
            "core_exit_code": exit_code,
            "termination_signal": terminated_signal,
            "stdout_sha256": digest(Path(receipt["stdout_path"])),
            "stderr_sha256": digest(Path(receipt["stderr_path"])),
        }
    )
    from mavis.e1 import _git

    try:
        receipt["resulting_revision"] = _git(checkout, "rev-parse", "HEAD")
        receipt["checkout_dirty"] = bool(
            _git(
                checkout, "status", "--porcelain", "--untracked-files=all", "--ignored"
            )
        )
    except (OSError, subprocess.CalledProcessError) as error:
        receipt["checkout_observation_error"] = str(error)
    try:
        if terminated_signal is not None:
            raise ValueError("E1 core was interrupted")
        if receipt.get("core_provenance") != "installed-package":
            raise ValueError("E1 core has no installed package provenance")
        event_path = Path(receipt["observation_path"])
        event = json.loads(event_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": "mavis.e1-effective-config/v1",
            "trial_id": receipt["trial_id"],
            "model": receipt["selected_model"],
            "model_provider": "omlx",
            "catalog_sha256": receipt["catalog_sha256"],
            "base_instructions_sha256": receipt["instructions_sha256"],
        }
        if any(
            event.get(key) != value for key, value in expected.items()
        ) or not event.get("session_id"):
            raise ValueError(
                "core effective-config observation differs from frozen E1 arm"
            )
        transcript = matching_trial_transcript(
            Path(event["rollout_path"]),
            Path(receipt["runtime_home"]),
            event["session_id"],
        )
        receipt.update(
            {
                "observation_status": "matched_startup_config",
                "observation_sha256": digest(event_path),
                "transcript_path": str(transcript.resolve()),
                "transcript_sha256": digest(transcript),
            }
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        receipt.update(
            {"observation_status": "inconclusive", "observation_error": str(error)}
        )
    atomic_write(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return (
        exit_code
        if (
            exit_code == 0
            and receipt["observation_status"] == "matched_startup_config"
            and "checkout_observation_error" not in receipt
        )
        else 2
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--check-args", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.check_args:
        check_profile_cli_arguments(command)
        return 0
    return launch(args.receipt, command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"mavis: {error}", file=sys.stderr)
        raise SystemExit(2) from None
