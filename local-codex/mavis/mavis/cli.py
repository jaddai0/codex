"""Command-line interface for the Mavis service."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .archive_hygiene import ArchiveRetention, storage_pressure
from .evidence import run_command
from .e1 import E1Runner
from .e1_bootstrap import (check_bootstrap_review, create_bootstrap,
                           dispatch_bootstrap_review, import_bootstrap_review_report,
                           prepare_bootstrap_review)
from .e1_bootstrap_gateway import (complete_bootstrap_review, start_bootstrap_verifier,
                                   verify_and_import_bootstrap_review)
from .e1_review import check_review_report
from .e1_review_gateway import (complete_review, start_review_verifier,
                                verify_and_import_review)
from .experiments import ExperimentStore, review_assignment_requirements
from .e0_tasks import prepare_small_repository
from .e2_tasks import prepare_heldout, verify_heldout
from .evaluations import E0_CASES, E0Evaluator
from .maintenance import MaintenanceQueue
from .objectives import ObjectiveStore
from .project_memory import ProjectMemory, KINDS
from .retrieval import ProjectIndex
from .runtime import (
    RuntimeConfig,
    admission,
    ensure_runtime,
    endpoint_alive,
    inventory,
    owns_running_server,
    stop_server,
)
from .storage import read_json
from .transcripts import TranscriptArchive


def home_from_env() -> Path:
    return Path(
        os.environ.get("MAVIS_HOME", Path.home() / ".local-codex" / "mavis-service")
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
        command.add_argument(
            "--model", default="Qwen3.8-Flash-Next-Abliterated-MLX-4bit"
        )
        command.add_argument("--model-dir", default="/Users/dustinpainter/models/vlms")
        command.add_argument(
            "--omlx-binary", default="/Users/dustinpainter/.venvs/omlx-dev/bin/omlx"
        )
        if name == "ensure":
            command.add_argument("--no-load", action="store_true")

    objective = subcommands.add_parser("objective")
    objective_sub = objective.add_subparsers(dest="objective_command", required=True)
    create = objective_sub.add_parser("create")
    create.add_argument("path", type=Path)
    show = objective_sub.add_parser("show")
    show.add_argument("objective_id")
    bind_session = objective_sub.add_parser("bind-session")
    bind_session.add_argument("objective_id")
    bind_session.add_argument("session_id")
    transition = objective_sub.add_parser("transition")
    transition.add_argument("objective_id")
    transition.add_argument("state")
    transition.add_argument("--reason", required=True)
    for name in ("assign", "attach-receipt", "verify"):
        entry = objective_sub.add_parser(name)
        entry.add_argument("objective_id")
        entry.add_argument("path", type=Path)
    gateway_verify = objective_sub.add_parser("gateway-verify")
    gateway_verify.add_argument("objective_id")
    gateway_verify.add_argument("worker_job_id")
    attempt = objective_sub.add_parser("attempt")
    attempt.add_argument("objective_id")
    attempt.add_argument("failure_fingerprint")
    attempt.add_argument("approach")
    attempt.add_argument("--evidence", action="append", default=[])

    run = subcommands.add_parser("run")
    run.add_argument("objective_id")
    run.add_argument("--cwd", type=Path, required=True)
    run.add_argument("--timeout", type=float)
    run.add_argument("--check-id", action="append", default=[])
    run.add_argument("argv", nargs=argparse.REMAINDER)

    search = subcommands.add_parser("archive-search")
    search.add_argument("conversation_id")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--offset", type=int, default=0)

    retention = subcommands.add_parser("archive-retention")
    retention_sub = retention.add_subparsers(dest="retention_command", required=True)
    retention_register = retention_sub.add_parser("register")
    retention_register.add_argument("project_id")
    retention_register.add_argument("--conversation", action="append", required=True)
    retention_register.add_argument("--objective", action="append", default=[])
    retention_close = retention_sub.add_parser("close")
    retention_close.add_argument("project_id")
    retention_close.add_argument("--protect", action="append", default=[])
    retention_compact = retention_sub.add_parser("compact")
    retention_compact.add_argument("project_id")
    retention_restore = retention_sub.add_parser("restore")
    retention_restore.add_argument("project_id")
    retention_reopen = retention_sub.add_parser("reopen")
    retention_reopen.add_argument("project_id")
    storage = subcommands.add_parser("storage-pressure")
    storage.add_argument("--prune-reproducible-cache", action="store_true")

    subcommands.add_parser("pre-compact")

    index = subcommands.add_parser("project-index")
    index.add_argument("--project", type=Path, required=True)
    index_sub = index.add_subparsers(dest="index_command", required=True)
    index_sub.add_parser("refresh")
    index_sub.add_parser("status")
    index_search = index_sub.add_parser("search")
    index_search.add_argument("query")
    index_search.add_argument("--limit", type=int, default=20)
    index_search.add_argument("--offset", type=int, default=0)
    index_symbol = index_sub.add_parser("symbol")
    index_symbol.add_argument("name")
    index_symbol.add_argument("--limit", type=int, default=20)
    index_symbol.add_argument("--offset", type=int, default=0)
    index_dependency = index_sub.add_parser("dependency")
    index_dependency.add_argument("name")
    index_dependency.add_argument("--limit", type=int, default=20)
    index_dependency.add_argument("--offset", type=int, default=0)

    memory = subcommands.add_parser("project-memory")
    memory.add_argument("--project", type=Path, required=True)
    memory_sub = memory.add_subparsers(dest="memory_command", required=True)
    memory_add = memory_sub.add_parser("add")
    memory_add.add_argument("record_id")
    memory_add.add_argument("kind", choices=sorted(KINDS))
    memory_add.add_argument("claim")
    memory_add.add_argument("--source", action="append", required=True)

    evaluate = subcommands.add_parser("eval")
    evaluate_sub = evaluate.add_subparsers(dest="eval_suite", required=True)
    e0 = evaluate_sub.add_parser("e0")
    e0.add_argument("--case", choices=E0_CASES)
    e0.add_argument("--endpoint", default="http://127.0.0.1:8001/v1")
    e0.add_argument("--iris-endpoint", default="http://127.0.0.1:8000/v1")
    e0.add_argument("--model", default="Qwen3.8-Flash-Next-Abliterated-MLX-4bit")
    e0.add_argument("--model-dir", default="/Users/dustinpainter/models/vlms")
    e0.add_argument(
        "--omlx-binary", default="/Users/dustinpainter/.venvs/omlx-dev/bin/omlx"
    )
    evaluate_sub.add_parser("prepare-small")
    e2 = evaluate_sub.add_parser("e2")
    e2_sub = e2.add_subparsers(dest="e2_command", required=True)
    e2_sub.add_parser("prepare-heldout")
    e2_verify = e2_sub.add_parser("verify-heldout")
    e2_verify.add_argument("manifest", type=Path)

    e1 = subcommands.add_parser("e1")
    e1_sub = e1.add_subparsers(dest="e1_command", required=True)
    trial = e1_sub.add_parser("trial")
    trial.add_argument("experiment_id")
    trial.add_argument("arm", choices=("baseline", "candidate"))
    trial.add_argument("case_id")
    trial.add_argument("task", nargs=argparse.REMAINDER)
    seed = e1_sub.add_parser("seed-active")
    seed.add_argument("scope")
    seed.add_argument("config", type=Path)
    bootstrap_prepare = e1_sub.add_parser("bootstrap-review-prepare")
    bootstrap_prepare.add_argument("--owner-provider", required=True)
    bootstrap_prepare.add_argument("--owner-model", required=True)
    bootstrap_prepare.add_argument("--owner-harness", required=True)
    bootstrap_dispatch = e1_sub.add_parser("bootstrap-review-dispatch")
    bootstrap_dispatch.add_argument("--job-id", required=True)
    bootstrap_check = e1_sub.add_parser("bootstrap-review-check")
    bootstrap_check.add_argument("--report", required=True, type=Path)
    e1_sub.add_parser("bootstrap-review-complete")
    e1_sub.add_parser("bootstrap-review-verifier-start")
    e1_sub.add_parser("bootstrap-review-verify")
    e1_sub.add_parser("bootstrap-review-import")
    bootstrap = e1_sub.add_parser("bootstrap")
    bootstrap.add_argument("--review", required=True, type=Path)
    freeze = e1_sub.add_parser("freeze")
    freeze.add_argument("experiment_id")
    freeze.add_argument("--scope", required=True)
    freeze.add_argument(
        "--kind", choices=("prompts", "tool_settings", "retrieval"), required=True
    )
    freeze.add_argument("--candidate", type=Path, required=True)
    freeze.add_argument("--cases", type=Path, required=True)
    freeze.add_argument("--hypothesis", required=True)
    for name in ("prepare", "check"):
        entry = e1_sub.add_parser(name)
        entry.add_argument("experiment_id")
        entry.add_argument("arm", choices=("baseline", "candidate"))
        if name == "check":
            entry.add_argument("case_id")
    coverage = e1_sub.add_parser("coverage")
    coverage.add_argument("experiment_id")
    dispatch = e1_sub.add_parser("dispatch-candidate")
    dispatch.add_argument("experiment_id")
    dispatch.add_argument("case_id")
    dispatch.add_argument("--job-id", required=True)
    dispatch.add_argument("--lane", choices=("minimax", "zcode"), required=True)
    dispatch.add_argument("--model", required=True)
    dispatch.add_argument("--task", required=True)
    compare = e1_sub.add_parser("compare")
    compare.add_argument("experiment_id")
    compare.add_argument("--candidate-job-id", required=True)
    compare.add_argument("--candidate-report", type=Path, required=True)
    native_compare = e1_sub.add_parser("compare-native")
    native_compare.add_argument("experiment_id")
    native_compare.add_argument("case_id")
    review_prepare = e1_sub.add_parser("review-prepare")
    review_prepare.add_argument("experiment_id")
    review_prepare.add_argument("--owner-provider", required=True)
    review_prepare.add_argument("--owner-model", required=True)
    review_prepare.add_argument("--owner-harness", required=True)
    review_dispatch = e1_sub.add_parser("review-dispatch")
    review_dispatch.add_argument("experiment_id")
    review_dispatch.add_argument("--job-id", required=True)
    review_check = e1_sub.add_parser("review-check")
    review_check.add_argument("experiment_id")
    review_check.add_argument("--report", required=True, type=Path)
    for name in ("review-complete", "review-verifier-start", "review-verify"):
        e1_sub.add_parser(name).add_argument("experiment_id")
    review_import = e1_sub.add_parser("review-import")
    review_import.add_argument("experiment_id")
    for name in ("status", "review-requirements", "stage"):
        entry = e1_sub.add_parser(name)
        entry.add_argument("experiment_id")
    review = e1_sub.add_parser("review")
    review.add_argument("experiment_id")
    review.add_argument("receipt", type=Path)
    promote = e1_sub.add_parser("promote")
    promote.add_argument("experiment_id")
    promote.add_argument("--between-objectives", action="store_true")
    rollback = e1_sub.add_parser("rollback")
    rollback.add_argument("experiment_id")
    rollback.add_argument("--reason", required=True)

    maintenance = subcommands.add_parser("maintenance")
    maintenance_sub = maintenance.add_subparsers(
        dest="maintenance_command", required=True
    )
    enqueue = maintenance_sub.add_parser("enqueue")
    enqueue.add_argument(
        "interval", choices=("immediate", "daily", "weekly", "monthly")
    )
    enqueue.add_argument("kind")
    enqueue.add_argument("--payload", default="{}")
    cancel = maintenance_sub.add_parser("cancel")
    cancel.add_argument("job_id")
    cancel.add_argument("--reason", required=True)
    maintenance_list = maintenance_sub.add_parser("list")
    maintenance_list.add_argument("--state")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    home = home_from_env()
    if args.command == "pre-compact":
        payload = json.load(sys.stdin)
        if (
            not isinstance(payload, dict)
            or payload.get("hook_event_name") != "PreCompact"
        ):
            raise ValueError("expected PreCompact hook input")
        session_id = payload.get("session_id")
        path = payload.get("transcript_path")
        codex_home = os.environ.get("CODEX_HOME")
        if (
            not isinstance(session_id, str)
            or not isinstance(path, str)
            or not codex_home
        ):
            raise ValueError(
                "PreCompact needs session_id, transcript_path, and CODEX_HOME"
            )
        source = Path(path).resolve(strict=True)
        if not source.is_relative_to(Path(codex_home).resolve(strict=True)):
            raise ValueError("transcript path is outside the Mavis Codex home")
        with source.open("r", encoding="utf-8") as handle:
            first = json.loads(handle.readline())
        if (
            first.get("type") != "session_meta"
            or first.get("payload", {}).get("id") != session_id
        ):
            raise ValueError("transcript session ID does not match hook input")
        archive = TranscriptArchive(home, session_id)
        segment = archive.import_rollout(source)
        snapshot = ObjectiveStore(home).handoff_snapshot(session_id)
        handoff = snapshot or {
            "goals": [],
            "accepted_decisions": [],
            "completed_requirements": [],
            "current_changes": [],
            "recent_work": [],
            "unresolved_failures": [
                "No objective is bound to this session; consult the archived transcript."
            ],
            "evidence_links": [],
            "unknown_fields": [
                "goals",
                "accepted_decisions",
                "completed_requirements",
                "current_changes",
            ],
        }
        handoff["recent_work"].append(
            {"turn_id": payload.get("turn_id"), "trigger": payload.get("trigger")}
        )
        handoff["evidence_links"].append(str(segment or archive.manifest_path))
        archive.write_handoff(handoff)
        print_json({"continue": True})
        return 0
    if args.command == "archive-retention":
        retention = ArchiveRetention(home)
        if args.retention_command == "register":
            result = retention.register(args.project_id, args.conversation, args.objective)
        elif args.retention_command == "close":
            result = retention.close(args.project_id, protected_paths=args.protect)
        elif args.retention_command == "restore":
            result = retention.restore(args.project_id)
        elif args.retention_command == "reopen":
            result = retention.reopen(args.project_id)
        else:
            result = retention.compact(args.project_id)
        print_json(result)
        return 0
    if args.command == "storage-pressure":
        print_json(storage_pressure(home, prune=args.prune_reproducible_cache))
        return 0
    if args.command == "runtime":
        config = runtime_config(args)
        if args.runtime_command == "ensure":
            print_json(ensure_runtime(config, load=not args.no_load))
        elif args.runtime_command == "status":
            print_json(
                {
                    "alive": endpoint_alive(config.endpoint),
                    "owned": owns_running_server(config),
                    "models": inventory(config.endpoint)
                    if endpoint_alive(config.endpoint)
                    else [],
                }
            )
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
        elif args.objective_command == "assign":
            print_json(store.add_assignment(args.objective_id, read_json(args.path)))
        elif args.objective_command == "bind-session":
            print_json(
                {"binding": str(store.bind_session(args.objective_id, args.session_id))}
            )
        elif args.objective_command == "attach-receipt":
            print_json(store.add_receipt(args.objective_id, args.path))
        elif args.objective_command == "verify":
            print_json(store.add_verification(args.objective_id, read_json(args.path)))
        elif args.objective_command == "gateway-verify":
            print_json(
                store.record_gateway_verification(args.objective_id, args.worker_job_id)
            )
        elif args.objective_command == "attempt":
            print_json(
                store.record_attempt(
                    args.objective_id,
                    args.failure_fingerprint,
                    args.evidence,
                    args.approach,
                )
            )
        else:
            print_json(store.transition(args.objective_id, args.state, args.reason))
        return 0
    if args.command == "run":
        if not args.check_id:
            raise ValueError(
                "run requires at least one --check-id from the objective acceptance checks"
            )
        command = list(args.argv)
        if command and command[0] == "--":
            command = command[1:]
        receipt = run_command(
            home,
            args.objective_id,
            command,
            args.cwd,
            timeout=args.timeout,
            acceptance_check_ids=args.check_id,
        )
        ObjectiveStore(home).add_receipt(args.objective_id, receipt)
        print(receipt)
        return 0
    if args.command == "eval":
        if args.eval_suite == "prepare-small":
            print_json({"manifest": str(prepare_small_repository(home))})
            return 0
        if args.eval_suite == "e2":
            if args.e2_command == "prepare-heldout":
                print_json({"manifest": str(prepare_heldout(home))})
            else:
                print_json(verify_heldout(args.manifest))
            return 0
        result = E0Evaluator(home, runtime_config(args)).run(args.case)
        print_json(result)
        return 0 if result["status"] == "pass" else 1
    if args.command == "e1":
        if args.e1_command == "trial":
            if args.task[:1] != ["--"] or len(args.task) != 2:
                raise ValueError("E1 trial needs exactly one task after --")
            from trial_runtime import run_trial

            print_json(
                {
                    "receipt": str(
                        run_trial(
                            args.experiment_id, args.arm, args.case_id, args.task[1]
                        )
                    )
                }
            )
            return 0
        runner = E1Runner(home)
        store = ExperimentStore(home)
        if args.e1_command == "seed-active":
            result = store.seed_active(args.scope, read_json(args.config))
        elif args.e1_command == "bootstrap-review-prepare":
            result = prepare_bootstrap_review(home, {
                "provider": args.owner_provider, "model": args.owner_model,
                "harness": args.owner_harness,
            })
        elif args.e1_command == "bootstrap-review-dispatch":
            result = dispatch_bootstrap_review(home, args.job_id)
        elif args.e1_command == "bootstrap-review-check":
            result = check_bootstrap_review(home, args.report)
        elif args.e1_command == "bootstrap-review-complete":
            result = complete_bootstrap_review(home)
        elif args.e1_command == "bootstrap-review-verifier-start":
            result = start_bootstrap_verifier(home)
        elif args.e1_command == "bootstrap-review-verify":
            result = verify_and_import_bootstrap_review(home)
        elif args.e1_command == "bootstrap-review-import":
            result = {"review": str(import_bootstrap_review_report(home))}
        elif args.e1_command == "bootstrap":
            result = create_bootstrap(home, args.review)
        elif args.e1_command == "freeze":
            result = runner.freeze(
                args.experiment_id,
                args.scope,
                args.kind,
                args.candidate,
                args.cases,
                args.hypothesis,
            )
        elif args.e1_command == "prepare":
            result = runner.prepare(args.experiment_id, args.arm)
        elif args.e1_command == "check":
            result = runner.check(args.experiment_id, args.arm, args.case_id)
        elif args.e1_command == "coverage":
            result = runner.coverage(args.experiment_id)
        elif args.e1_command == "dispatch-candidate":
            result = runner.dispatch_candidate(
                args.experiment_id,
                args.case_id,
                job_id=args.job_id,
                lane=args.lane,
                model=args.model,
                task=args.task,
            )
        elif args.e1_command == "compare":
            result = runner.compare(
                args.experiment_id, args.candidate_job_id, args.candidate_report
            )
        elif args.e1_command == "compare-native":
            result = runner.compare_native(args.experiment_id, args.case_id)
        elif args.e1_command == "review-prepare":
            result = runner.prepare_review(args.experiment_id, {
                "provider": args.owner_provider, "model": args.owner_model,
                "harness": args.owner_harness,
            })
        elif args.e1_command == "review-dispatch":
            result = runner.dispatch_review(args.experiment_id, args.job_id)
        elif args.e1_command in ("review-check", "review-complete",
                                  "review-verifier-start", "review-verify"):
            record = store.load(args.experiment_id)
            store._check_comparison(record)
            if args.e1_command == "review-check":
                result = check_review_report(home, record, args.report)
            elif args.e1_command == "review-complete":
                result = complete_review(home, record)
            elif args.e1_command == "review-verifier-start":
                result = start_review_verifier(home, record)
            else:
                imported = verify_and_import_review(home, record)
                result = {"gateway": imported,
                          "experiment": store.review(args.experiment_id, Path(imported["review"]))}
        elif args.e1_command == "review-import":
            result = runner.import_review(args.experiment_id)
        elif args.e1_command == "status":
            result = store.load(args.experiment_id)
        elif args.e1_command == "review-requirements":
            record = store.load(args.experiment_id)
            if record["state"] != "compared":
                raise ValueError("review requirements need a completed comparison")
            if record["comparison"]["candidate"].get("e1_native_trial"):
                from .e1_review import _paths

                result = read_json(_paths(home, args.experiment_id)[0])["requirements"]
            else:
                result = review_assignment_requirements(record)
        elif args.e1_command == "review":
            result = store.review(args.experiment_id, args.receipt)
        elif args.e1_command == "promote":
            result = store.promote(args.experiment_id, between_objectives=args.between_objectives)
        elif args.e1_command == "rollback":
            result = store.rollback(args.experiment_id, reason=args.reason)
        else:
            result = store.stage(args.experiment_id)
        print_json(result)
        return 0
    if args.command == "project-index":
        index = ProjectIndex(args.project, home)
        if args.index_command == "refresh":
            print_json(index.refresh())
        elif args.index_command == "status":
            print_json(index.status())
        elif args.index_command == "search":
            print_json(index.search(args.query, args.limit, args.offset))
        elif args.index_command == "symbol":
            print_json(index.symbol(args.name, args.limit, args.offset))
        else:
            print_json(index.dependency(args.name, args.limit, args.offset))
        return 0
    if args.command == "project-memory":
        print_json(
            ProjectMemory(args.project).add(
                args.record_id, args.kind, args.claim, args.source
            )
        )
        return 0
    if args.command == "maintenance":
        queue = MaintenanceQueue(home)
        if args.maintenance_command == "enqueue":
            payload = json.loads(args.payload)
            if not isinstance(payload, dict):
                raise ValueError("maintenance payload must be a JSON object")
            print_json(queue.enqueue(args.interval, args.kind, payload))
        elif args.maintenance_command == "cancel":
            print_json(queue.cancel(args.job_id, reason=args.reason))
        else:
            print_json(queue.list(args.state))
        return 0
    archive = TranscriptArchive(home, args.conversation_id)
    print_json(archive.search(args.query, args.limit, args.offset))
    return 0
