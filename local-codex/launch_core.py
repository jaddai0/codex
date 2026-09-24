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
from typing import Callable

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


class TrialSignalGuard:
    """Catch termination before spawn and keep it bound to the new process group."""

    def __init__(self):
        self.child: subprocess.Popen | None = None
        self.signal: int | None = None
        self.at: float | None = None
        self.previous: dict[int, object] = {}

    def __enter__(self):
        for signum in (signal.SIGTERM, signal.SIGINT):
            self.previous[signum] = signal.signal(signum, self._forward)
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        for signum, previous in self.previous.items():
            signal.signal(signum, previous)

    def register(self, child: subprocess.Popen) -> None:
        self.child = child
        if self.signal is not None:
            self._terminate_child()

    def _forward(self, signum, _frame):
        if self.signal is None:
            self.signal = signum
            self.at = time.monotonic()
        self._terminate_child()

    def _terminate_child(self):
        if self.child is not None:
            try:
                os.killpg(self.child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


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


def bind_project_root(argv: list[str]) -> Path | None:
    """Bind one launch to its effective checkout before any model preparation."""
    core_args = argv[1:]
    selected: list[str] = []
    index = 0
    while index < len(core_args):
        argument = core_args[index]
        if argument == "--":
            break
        if argument in ("-C", "--cd"):
            if index + 1 >= len(core_args) or not core_args[index + 1]:
                raise ValueError("Mavis -C/--cd requires a directory")
            selected.append(core_args[index + 1])
            index += 2
            continue
        elif argument.startswith("--cd="):
            selected.append(argument.removeprefix("--cd="))
        elif argument.startswith("-C="):
            selected.append(argument.removeprefix("-C="))
        elif argument.startswith("-C") and argument != "-C":
            selected.append(argument[2:])
        index += 1
    if len(selected) > 1:
        raise ValueError("Mavis project directory is ambiguous")
    explicit_dir = os.environ.get("MAVIS_PROJECT_DIR")
    anchor = Path(selected[0] if selected else explicit_dir or os.getcwd()).resolve(strict=True)
    if not anchor.is_dir():
        raise ValueError("Mavis project directory is not a directory")
    if explicit_dir and selected and Path(explicit_dir).resolve(strict=True) != anchor:
        raise ValueError("MAVIS_PROJECT_DIR differs from -C/--cd")
    git_env = {name: value for name, value in os.environ.items() if name not in
               {"GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE", "GIT_PREFIX",
                "MAVIS_E0_TRIAL_API_KEY"}}
    git = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=anchor, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=git_env, check=False,
    )
    user_home = Path.home().resolve(strict=True)
    root = Path(git.stdout.strip()).resolve(strict=True) if git.returncode == 0 else None
    if root == user_home:
        root = None
    inherited = os.environ.get("MAVIS_PROJECT_ROOT")
    if inherited and (root is None or Path(inherited).resolve(strict=True) != root):
        raise ValueError("MAVIS_PROJECT_ROOT differs from the effective checkout")
    if root is None:
        os.environ.pop("MAVIS_PROJECT_ROOT", None)
    else:
        os.environ["MAVIS_PROJECT_ROOT"] = str(root)
    return root


def launch(
    receipt_path: Path | None, argv: list[str], *, mavis_home: Path | None = None
) -> int:
    bind_project_root(argv)
    if receipt_path is not None:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        with generation_lease(Path(receipt["mavis_home"]), purpose="accepted-main"):
            return _launch_unlocked(receipt_path, argv)
    home = mavis_home or os.environ.get("MAVIS_HOME")
    if home is None:
        raise ValueError("MAVIS_HOME is required for a core launch without a profile")
    with generation_lease(Path(home), purpose="main"):
        return _launch_unlocked(None, argv)


def _preparation_command(
    command: list[str], *, capture_output: bool
) -> subprocess.CompletedProcess[str]:
    """Stop a preparation child before releasing the launcher lease on interrupt."""
    with TrialSignalGuard() as guard:
        child = subprocess.Popen(
            command,
            stdout=subprocess.PIPE if capture_output else subprocess.DEVNULL,
            stderr=subprocess.PIPE if capture_output else None,
            text=True,
            start_new_session=True,
        )
        guard.register(child)
        while True:
            try:
                stdout, stderr = child.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if guard.signal is not None and time.monotonic() - guard.at >= 5:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        if guard.signal is not None:
            raise RuntimeError("launcher was interrupted during model preparation")
        return subprocess.CompletedProcess(command, child.returncode, stdout, stderr)


def _prepared_output(command: list[str]) -> str:
    result = _preparation_command(command, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            (result.stderr or "").strip() or f"preparation exited {result.returncode}"
        )
    return (result.stdout or "").strip()


def managed_launch(args: argparse.Namespace, argv: list[str]) -> int:
    """Hold the host lease through model preparation and the whole core session."""
    if not argv:
        raise ValueError("core binary is required")
    bind_project_root(argv)
    needed = (
        "share_dir", "home", "mavis_home", "base_url", "iris_endpoint",
        "model_dir", "omlx_binary", "gateway_root",
    )
    if any(getattr(args, name) is None for name in needed):
        raise ValueError("managed launch is missing its runtime paths")
    share = args.share_dir.resolve()
    home = args.mavis_home.resolve()
    runtime_home = args.home.resolve()
    prepare = share / "prepare_runtime.py"
    with generation_lease(home, purpose="main-launch"):
        model_command = [
            sys.executable, str(prepare), "--mavis-home", str(home), "--resolve-model",
        ]
        if args.model:
            model_command += ["--model", args.model]
        model = _prepared_output(model_command)
        if not model:
            raise ValueError("model resolution returned no model")
        if (home / "profiles" / "main" / "active.json").exists():
            check_profile_cli_arguments(argv[1:])
        ensure = [
            sys.executable, "-m", "mavis", "runtime", "ensure",
            "--endpoint", args.base_url,
            "--iris-endpoint", args.iris_endpoint,
            "--model", model,
            "--model-dir", str(args.model_dir),
            "--omlx-binary", str(args.omlx_binary),
        ]
        result = _preparation_command(ensure, capture_output=False)
        if result.returncode != 0:
            return result.returncode
        receipt_command = [
            sys.executable, str(prepare),
            "--home", str(runtime_home),
            "--mavis-home", str(home),
            "--print-receipt",
            "--base-url", args.base_url,
            "--persona-template", str(share / "persona.toml"),
            "--instructions-template", str(share / "base-instructions.md"),
            "--gateway-root", str(args.gateway_root),
            "--model-dir", str(args.model_dir),
            "--model", model,
        ]
        if args.gateway_env_file is not None:
            receipt_command += ["--gateway-env-file", str(args.gateway_env_file)]
        receipt = _prepared_output(receipt_command)
        if receipt:
            check_profile_cli_arguments(argv[1:])
        os.environ.update(
            CODEX_HOME=str(runtime_home),
            MAVIS_HOME=str(home),
            LOCAL_CODEX_SHARE_DIR=str(share),
            MAVIS_PRECOMPACT_REQUIRED="1",
            MAVIS_RAW_OUTPUT_REQUIRED="1",
        )
        return _launch_unlocked(Path(receipt) if receipt else None, argv)


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
        source = active.get("accepted_bootstrap") or active.get("accepted_experiment")
        active_pointer = Path(receipt["mavis_home"]) / "experiments/active/main.json"
        if (receipt.get("accepted_source") != ("bootstrap" if active.get("accepted_bootstrap") else "experiment")
                or receipt.get("accepted_source_path") != source["path"]
                or receipt.get("accepted_source_sha256") != source["sha256"]
                or receipt.get("active_experiment_sha256") != digest(active_pointer)):
            raise ValueError("profile launch source changed after preparation")
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
    receipt_path: Path, argv: list[str], *, lease_held: bool = False,
    heartbeat: Callable[[], None] | None = None,
) -> int:
    """Run a frozen E1 arm; process exit alone never establishes trial success."""
    if not lease_held:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        with generation_lease(Path(receipt["mavis_home"]), purpose="e1-trial"):
            return _launch_trial_unlocked(receipt_path, argv, heartbeat=heartbeat)
    return _launch_trial_unlocked(receipt_path, argv, heartbeat=heartbeat)


def _launch_trial_unlocked(receipt_path: Path, argv: list[str], *,
                           heartbeat: Callable[[], None] | None = None) -> int:
    with TrialSignalGuard() as guard:
        return _launch_trial_guarded(receipt_path, argv, guard, heartbeat=heartbeat)


def _launch_trial_guarded(
    receipt_path: Path, argv: list[str], guard: TrialSignalGuard, *,
    heartbeat: Callable[[], None] | None = None,
) -> int:
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
    # The frozen trial checkout owns its raw tool output even when launched elsewhere.
    environment["MAVIS_PROJECT_ROOT"] = str(checkout.resolve(strict=True))
    environment["MAVIS_PROJECT_DIR"] = environment["MAVIS_PROJECT_ROOT"]
    with (
        Path(receipt["stdout_path"]).open("xb") as stdout,
        Path(receipt["stderr_path"]).open("xb") as stderr,
    ):
        if heartbeat is not None:
            heartbeat()
        if guard.signal is not None:
            receipt.update(
                {
                    "state": "terminated",
                    "termination_signal": guard.signal,
                    "observation_status": "inconclusive",
                    "observation_error": "interrupted before core spawn",
                }
            )
            atomic_write(
                receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n"
            )
            return 2
        child = subprocess.Popen(
            launch_argv,
            cwd=checkout,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        guard.register(child)
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
        try:
            while True:
                if heartbeat is not None:
                    heartbeat()
                try:
                    exit_code = child.wait(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    if guard.at is not None and time.monotonic() - guard.at >= 5:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        exit_code = child.wait()
                        break
        except BaseException:
            _stop_process_group(child)
            raise
        terminated_signal = guard.signal
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
    from mavis.e1 import _git, _source_snapshot, _unmanaged_status

    try:
        receipt["resulting_revision"] = _git(checkout, "rev-parse", "HEAD")
        receipt["checkout_dirty"] = bool(_unmanaged_status(checkout))
        snapshot_path = Path(receipt["runtime_home"]) / "post-turn-source.json"
        atomic_write(snapshot_path, json.dumps(_source_snapshot(checkout), sort_keys=True) + "\n")
        receipt["post_turn_source_path"] = str(snapshot_path.resolve())
        receipt["post_turn_source_sha256"] = digest(snapshot_path)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
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
    if guard.signal is not None:
        receipt.update(
            {
                "state": "terminated",
                "termination_signal": guard.signal,
                "observation_status": "inconclusive",
                "observation_error": "E1 wrapper was interrupted",
            }
        )
    atomic_write(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return (
        exit_code
        if (
            exit_code == 0
            and receipt["observation_status"] == "matched_startup_config"
            and "checkout_observation_error" not in receipt
            and guard.signal is None
        )
        else 2
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--check-args", action="store_true")
    parser.add_argument("--managed", action="store_true")
    parser.add_argument("--share-dir", type=Path)
    parser.add_argument("--home", type=Path)
    parser.add_argument("--mavis-home", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--iris-endpoint")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--omlx-binary", type=Path)
    parser.add_argument("--gateway-root", type=Path)
    parser.add_argument("--gateway-env-file", type=Path)
    parser.add_argument("--model")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.check_args:
        check_profile_cli_arguments(command)
        return 0
    if args.managed:
        return managed_launch(args, command)
    return launch(args.receipt, command, mavis_home=args.mavis_home)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"mavis: {error}", file=sys.stderr)
        raise SystemExit(2) from None
