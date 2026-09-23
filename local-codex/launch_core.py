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

from prepare_runtime import accepted_main_profile, atomic_write


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_profile_cli_arguments(arguments: list[str]) -> None:
    """Refuse Codex flags that can change an accepted profile's effective runtime."""
    long_flags = ("--model", "--profile", "--config", "--oss", "--local-provider",
                  "--enable", "--disable")
    for argument in arguments:
        if argument == "--":
            break
        if argument in long_flags or any(argument.startswith(flag + "=") for flag in long_flags):
            raise ValueError(f"{argument} can override an accepted main profile")
        if argument.startswith(("-m", "-p", "-c")) and not argument.startswith("--"):
            raise ValueError(f"{argument} can override an accepted main profile")


def launch(receipt_path: Path | None, argv: list[str]) -> int:
    if not argv:
        raise ValueError("core binary is required")
    receipt = None
    if receipt_path is not None:
        check_profile_cli_arguments(argv[1:])
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("state") != "prepared":
            raise ValueError("profile launch receipt is not prepared")
        for name in ("config", "catalog"):
            if digest(Path(receipt[f"{name}_path"])) != receipt[f"{name}_sha256"]:
                raise ValueError(f"profile {name} changed after preparation")
        if digest(Path(receipt["profile_path"])) != receipt["profile_sha256"]:
            raise ValueError("accepted profile changed after preparation")
        active = accepted_main_profile(Path(receipt["mavis_home"]))
        if active is None or active["_source_sha256"] != receipt["profile_sha256"]:
            raise ValueError("accepted main profile is no longer active")
    child = subprocess.Popen(argv)
    if receipt is not None:
        receipt.update({"state": "spawned", "spawned_at": datetime.now(timezone.utc).isoformat(),
                        "core_binary": str(Path(argv[0]).resolve()), "core_pid": child.pid})
        try:
            atomic_write(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
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
        receipt.update({"state": "exited", "exited_at": datetime.now(timezone.utc).isoformat(),
                        "core_exit_code": return_code})
        atomic_write(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return return_code if return_code >= 0 else 128 - return_code


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
