"""Command-line interface for the Mavis service."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .evidence import run_command
from .objectives import ObjectiveStore
from .runtime import RuntimeConfig, admission, ensure_runtime, endpoint_alive, inventory, owns_running_server, stop_server
from .storage import read_json
from .transcripts import TranscriptArchive


def home_from_env() -> Path:
    return Path(
        os.environ.get(
            "MAVIS_HOME", Path.home() / ".local-codex" / "mavis-service"
        )
    ).expanduser()


def runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        home=home_from_env(),
        endpoint=args.endpoint,
        iris_endpoint=args.iris_endpoint,
        model=args.model,
        model_dir=Path(args.model_dir).expanduser(),
        omlx_binary=Path(args.omlx_binary).expanduser(),
    )


def print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mavis-service")
    subcommands = parser.add_subparsers(dest="command", required=True)
    runtime = subcommands.add_parser("runtime")
    runtime_sub = runtime.add_subparsers(dest="runtime_command", required=True)
    for name in ("ensure", "status", "admission", "stop"):
        command = runtime_sub.add_parser(name)
        command.add_argument("--endpoint", default="http://127.0.0.1:8001/v1")
        command.add_argument("--iris-endpoint", default="http://127.0.0.1:8000/v1")
        command.add_argument("--model", default="Qwen3.8-Flash-Next-Abliterated-MLX-4bit")
        command.add_argument("--model-dir", default="/Users/dustinpainter/models/vlms")
        command.add_argument("--omlx-binary", default="/Users/dustinpainter/.venvs/omlx-dev/bin/omlx")
        if name == "ensure":
            command.add_argument("--no-load", action="store_true")

    objective = subcommands.add_parser("objective")
    objective_sub = objective.add_subparsers(dest="objective_command", required=True)
    create = objective_sub.add_parser("create")
    create.add_argument("path", type=Path)
    show = objective_sub.add_parser("show")
    show.add_argument("objective_id")
    transition = objective_sub.add_parser("transition")
    transition.add_argument("objective_id")
    transition.add_argument("state")
    transition.add_argument("--reason", required=True)

    run = subcommands.add_parser("run")
    run.add_argument("objective_id")
    run.add_argument("--cwd", type=Path, required=True)
    run.add_argument("--timeout", type=float)
    run.add_argument("argv", nargs=argparse.REMAINDER)

    search = subcommands.add_parser("archive-search")
    search.add_argument("conversation_id")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    home = home_from_env()
    if args.command == "runtime":
        config = runtime_config(args)
        if args.runtime_command == "ensure":
            print_json(ensure_runtime(config, load=not args.no_load))
        elif args.runtime_command == "status":
            print_json({"alive": endpoint_alive(config.endpoint), "owned": owns_running_server(config), "models": inventory(config.endpoint) if endpoint_alive(config.endpoint) else []})
        elif args.runtime_command == "admission":
            print_json(admission(config))
        else:
            stop_server(config)
            print_json({"stopped": True})
        return 0
    if args.command == "objective":
        store = ObjectiveStore(home)
        if args.objective_command == "create":
            print_json(store.create(read_json(args.path)))
        elif args.objective_command == "show":
            print_json(store.load(args.objective_id))
        else:
            print_json(store.transition(args.objective_id, args.state, args.reason))
        return 0
    if args.command == "run":
        command = list(args.argv)
        if command and command[0] == "--":
            command = command[1:]
        receipt = run_command(home, args.objective_id, command, args.cwd, timeout=args.timeout)
        print(receipt)
        return 0
    archive = TranscriptArchive(home, args.conversation_id)
    print_json(archive.search(args.query, args.limit))
    return 0
