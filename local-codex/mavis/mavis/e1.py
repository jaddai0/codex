"""Frozen E1 cases, native candidate dispatch, and host-recorded paired checks."""

from pathlib import Path
import fcntl
import hashlib
import json
import tomllib
import shutil
import subprocess
from typing import Any

from .evidence import parse_test_output, run_command
from .experiments import ExperimentStore, _digest, candidate_assignment_requirements
from .gateway import harness_assignment_start
from .objectives import _validate_gateway_status
from .package_provenance import package_tree_sha256
from .storage import read_json, require_safe_id, sha256_file, write_json


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=path, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def _clean_revision(checkout: Path) -> str:
    """Bind E1 to committed content; ignored files are disallowed too."""
    revision = _git(checkout, "rev-parse", "HEAD")
    if _git(checkout, "status", "--porcelain", "--untracked-files=all", "--ignored"):
        raise ValueError("E1 checkout must be clean, including ignored files")
    return revision


def _verify_host_receipt(
    path: Path, *, command: list[str], cwd: Path, check_id: str, revision: str
) -> dict[str, Any]:
    receipt = read_json(path)
    raw = receipt.get("raw_output")
    if (
        not isinstance(raw, dict)
        or Path(raw.get("path", "")).resolve() != path.resolve().parent
    ):
        raise ValueError("host receipt raw output path is invalid")
    stdout = (path.parent / "stdout.log").read_bytes()
    stderr = (path.parent / "stderr.log").read_bytes()
    text = (
        stdout.decode("utf-8", errors="replace")
        + "\n"
        + stderr.decode("utf-8", errors="replace")
    )
    status = receipt.get("exit_status")
    timed_out = receipt.get("timed_out")
    if (
        receipt.get("schema_version") != "mavis.evidence-receipt/v1"
        or receipt.get("producer") != "mavis-host-command/v1"
        or receipt.get("command") != command
        or Path(receipt.get("cwd", "")).resolve() != cwd.resolve()
        or receipt.get("acceptance_check_ids") != [check_id]
        or receipt.get("changed_revision") != revision
        or isinstance(status, bool)
        or not isinstance(status, int)
        or not isinstance(timed_out, bool)
        or raw.get("sha256")
        != sha256_file(path.parent / "stdout.log")
        + ":"
        + sha256_file(path.parent / "stderr.log")
        or raw.get("bytes") != len(stdout) + len(stderr)
        or receipt.get("verdict") != parse_test_output(text, status, timed_out)
    ):
        raise ValueError("host receipt is inconsistent with raw output or frozen check")
    return receipt


def _manifest(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if (
        value.get("schema_version") != "mavis.e1-cases/v1"
        or value.get("scoring") != "held_out_pass_fraction/v1"
    ):
        raise ValueError("E1 manifest needs versioned scoring")
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) < 2:
        raise ValueError("E1 needs a regression and held-out cases")
    ids = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("invalid E1 case")
        case_id = require_safe_id(case.get("id"), "case id")
        ids.append(case_id)
        source = Path(case.get("source", "")).resolve(strict=True)
        revision = case.get("revision")
        if (
            source != Path(_git(source, "rev-parse", "--show-toplevel"))
            or not isinstance(revision, str)
            or len(revision) != 40
        ):
            raise ValueError("case needs a repository root and full revision")
        if _git(source, "rev-parse", f"{revision}^{{commit}}") != revision:
            raise ValueError("case revision is unavailable")
        checks = case.get("checks")
        if not isinstance(checks, list) or not checks:
            raise ValueError("case needs acceptance checks")
        check_ids = []
        for check in checks:
            if not isinstance(check, dict):
                raise ValueError("invalid check")
            check_ids.append(require_safe_id(check.get("id"), "check id"))
            argv = check.get("argv")
            timeout = check.get("timeout_seconds")
            if (
                not isinstance(argv, list)
                or not argv
                or not all(isinstance(x, str) and x for x in argv)
            ):
                raise ValueError("check needs argv")
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not 0 < timeout <= 3600
            ):
                raise ValueError("check needs a bounded timeout")
        if len(set(check_ids)) != len(check_ids):
            raise ValueError("duplicate check id")
    if len(set(ids)) != len(ids) or value.get("regression") not in ids:
        raise ValueError("E1 needs unique cases and a regression")
    regression = next(case for case in cases if case["id"] == value["regression"])
    failure = value.get("failure_receipt")
    if (
        not isinstance(failure, dict)
        or not failure.get("path")
        or not failure.get("sha256")
    ):
        raise ValueError("regression needs an earlier failed host receipt")
    failure_path = Path(failure["path"]).resolve(strict=True)
    if sha256_file(failure_path) != failure["sha256"]:
        raise ValueError("original failure receipt changed")
    original = read_json(failure_path)
    matching = [
        check
        for check in regression["checks"]
        if original.get("command") == check["argv"]
        and original.get("acceptance_check_ids") == [check["id"]]
    ]
    if not matching:
        raise ValueError("original failure does not match the regression check")
    original = _verify_host_receipt(
        failure_path,
        command=matching[0]["argv"],
        cwd=Path(regression["source"]),
        check_id=matching[0]["id"],
        revision=regression["revision"],
    )
    if original["verdict"] != "fail" or original["timed_out"]:
        raise ValueError("original failure lacks matched raw host evidence")
    held_out = value.get("held_out")
    if (
        not isinstance(held_out, list)
        or not held_out
        or len(set(held_out)) != len(held_out)
        or set(held_out) != set(ids) - {value["regression"]}
    ):
        raise ValueError("held-out cases must be separate and complete")
    gain = value.get("minimum_gain")
    if (
        isinstance(gain, bool)
        or not isinstance(gain, (int, float))
        or not 0 < gain <= 1
    ):
        raise ValueError("minimum gain must be between zero and one")
    return value


def _argv_hash(argv: list[str]) -> str:
    return hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest()


def _trial_hashes(root: Path, home: Path, record: dict[str, Any], case: dict[str, Any],
                  arm: str, result: dict[str, Any],
                  bootstrap_cache: dict[str, Any] | None = None) -> dict[str, Any]:
    """Recheck the installed host's launch and core observation for one checked case.

    This proves startup identity and the source of the checked checkout. The
    acceptance verdict still comes exclusively from host command receipts.
    """
    case_id = case["id"]
    trial_path = root / "trials" / arm / f"{case_id}.json"
    trial = read_json(trial_path)
    runtime = root / "runtime" / arm / case_id
    checkout = root / "checkouts" / arm / case_id
    task = case.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("native E1 case needs a frozen task")
    core = Path(trial.get("core_binary", ""))
    expected_argv = [str(core.resolve()), "exec", "-C", str(checkout.resolve()), "--", task]
    bindings = {
        "model": trial.get("selected_model"),
        "model_provider": "omlx",
        "model_catalog_json": str((runtime / "omlx-models.json").resolve()),
        "model_instructions_file": str((runtime / "accepted-model-instructions.md").resolve()),
    }
    overrides = [f"{key}={json.dumps(value)}" for key, value in bindings.items()]
    effective_argv = [expected_argv[0], *(item for override in overrides for item in ("-c", override)), *expected_argv[1:]]
    expected = {
        "schema_version": "mavis.e1-trial-launch/v1",
        "state": "exited", "core_exit_code": 0, "termination_signal": None,
        "observation_status": "matched_startup_config",
        "experiment_id": record["experiment_id"], "arm": arm, "case_id": case_id,
        "trial_id": f"{record['experiment_id']}-{arm}-{case_id}",
        "task": task, "task_sha256": hashlib.sha256(task.encode()).hexdigest(),
        "core_argv": expected_argv, "core_argv_sha256": _argv_hash(expected_argv),
        "binding_overrides": overrides, "effective_argv": effective_argv,
        "effective_argv_sha256": _argv_hash(effective_argv),
        "manifest_sha256": record["workload"]["manifest_sha256"],
        "snapshot_path": record[arm]["path"], "snapshot_sha256": record[arm]["sha256"],
        "checkout": str(checkout.resolve()), "starting_revision": case["revision"],
        "resulting_revision": result["checked_revision"], "checkout_dirty": False,
        "runtime_home": str(runtime.resolve()), "mavis_home": str(Path(home).resolve()),
        "core_binary": str(core.resolve()), "core_provenance": "installed-package",
        "model_provider": "omlx",
        "config_path": str((runtime / "config.toml").resolve()),
        "catalog_path": str((runtime / "omlx-models.json").resolve()),
        "instructions_path": str((runtime / "accepted-model-instructions.md").resolve()),
        "observation_path": str((runtime / "effective-config.json").resolve()),
        "stdout_path": str((runtime / "stdout.log").resolve()),
        "stderr_path": str((runtime / "stderr.log").resolve()),
    }
    mismatched = [key for key, value in expected.items() if trial.get(key) != value]
    if mismatched:
        raise ValueError(f"native E1 trial lost its frozen command or checkout binding: {mismatched}")
    if type(trial.get("core_exit_code")) is not int or type(trial.get("checkout_dirty")) is not bool:
        raise ValueError("native E1 trial has invalid exit or checkout observation")
    if any(key in trial for key in ("checkout_observation_error", "observation_error")):
        raise ValueError("native E1 trial has an inconclusive observation")
    if not isinstance(trial.get("selected_model"), str) or not trial["selected_model"]:
        raise ValueError("native E1 trial lacks an exact model identity")
    paths = {
        "core": core,
        "config": runtime / "config.toml",
        "catalog": runtime / "omlx-models.json",
        "instructions": runtime / "accepted-model-instructions.md",
        "stdout": runtime / "stdout.log", "stderr": runtime / "stderr.log",
        "observation": runtime / "effective-config.json",
    }
    for name, path in paths.items():
        if sha256_file(path) != trial.get(f"{name}_sha256"):
            raise ValueError(f"native E1 trial {name} hash changed")
    if trial.get("instructions_sha256") != trial.get("effective_system_prompt_sha256"):
        raise ValueError("native E1 effective prompt changed")
    snapshot_path = Path(record[arm]["path"])
    if sha256_file(snapshot_path) != record[arm]["sha256"]:
        raise ValueError("native E1 frozen snapshot changed")
    snapshot = read_json(snapshot_path)
    prompt = snapshot.get("prompts", {}).get("system")
    template_path = core.parent / "base-instructions.md"
    if (not isinstance(prompt, str) or snapshot.get("tool_settings") != {}
            or snapshot.get("retrieval") != {} or not template_path.is_file()):
        raise ValueError("native E1 prompt configuration cannot be applied")
    effective = template_path.read_text(encoding="utf-8").rstrip() + "\n\n" + prompt + "\n"
    if (not effective.strip() or paths["instructions"].read_text(encoding="utf-8") != effective):
        raise ValueError("native E1 instruction bytes differ from frozen prompt")
    package_path = Path(trial.get("package_manifest") or "")
    if package_path.resolve() != (core.parent / "install-manifest.json").resolve():
        raise ValueError("native E1 package manifest is outside the installed core")
    package = read_json(package_path)
    launcher = Path(package.get("launcher", ""))
    runtime_source = package_path.parent / "trial_runtime.py"
    installed_inputs = {
        "trial_runtime": runtime_source,
        "launch_core": package_path.parent / "launch_core.py",
        "prepare_runtime": package_path.parent / "prepare_runtime.py",
        "generation_lease": package_path.parent / "generation_lease.py",
        "base_instructions": package_path.parent / "base-instructions.md",
        "persona": package_path.parent / "persona.toml",
    }
    if (trial.get("package_manifest_sha256") != sha256_file(package_path)
            or package.get("schema_version") != "mavis.installed-core/v1"
            or package.get("core_binary") != str(core.resolve())
            or package.get("core_sha256") != trial["core_sha256"]
            or any(package.get(f"{name}_sha256") != sha256_file(path)
                   for name, path in installed_inputs.items())
            or package.get("mavis_package_sha256") != package_tree_sha256(package_path.parent / "mavis")
            or not launcher.is_file()
            or package.get("launcher_sha256") != sha256_file(launcher)):
        raise ValueError("native E1 installed core provenance changed")
    source = trial.get("profile_source", "accepted-main")
    if source == "accepted-main":
        if trial.get("bootstrap_receipt_path") is not None or trial.get("bootstrap_receipt_sha256") is not None:
            raise ValueError("native E1 accepted profile has a bootstrap binding")
        profile_path = Path(trial.get("accepted_profile_path") or "")
        if (not trial.get("accepted_profile_id")
                or trial.get("accepted_profile_sha256") != sha256_file(profile_path)):
            raise ValueError("native E1 accepted profile changed")
        profile = read_json(profile_path)
        if (profile.get("profile_id") != trial["accepted_profile_id"]
                or profile.get("model_identity", {}).get("model_id") != trial["selected_model"]):
            raise ValueError("native E1 model differs from accepted profile")
        pointer = read_json(Path(home) / "profiles" / "main" / "active.json")
        version = pointer.get("version")
        if (type(version) is not int or version < 1
                or profile_path.resolve() != (Path(home) / "profiles" / "main" / f"v{version}.json").resolve()
                or pointer.get("path") != str(profile_path.resolve())
                or profile.get("status") != "active" or profile.get("role") != "main"):
            raise ValueError("native E1 accepted main profile pointer changed")
        profile_hashes = {"profile": sha256_file(profile_path)}
    elif source == "e0-bootstrap":
        if any(trial.get(key) is not None for key in (
            "accepted_profile_id", "accepted_profile_path", "accepted_profile_sha256"
        )):
            raise ValueError("native E1 bootstrap trial contains an accepted-profile claim")
        bootstrap_path = Path(home) / "e1" / "bootstrap" / "main.json"
        if (trial.get("bootstrap_receipt_path") != str(bootstrap_path.resolve())
                or trial.get("bootstrap_receipt_sha256") != sha256_file(bootstrap_path)):
            raise ValueError("native E1 bootstrap receipt changed")
        from .e1_bootstrap import validate_bootstrap

        if bootstrap_cache is None:
            bootstrap = validate_bootstrap(Path(home))
        else:
            if "validated" not in bootstrap_cache:
                bootstrap_cache["validated"] = validate_bootstrap(Path(home))
            bootstrap = bootstrap_cache["validated"]
        config = tomllib.loads(paths["config"].read_text(encoding="utf-8"))
        endpoint = config.get("model_providers", {}).get("omlx", {}).get("base_url")
        if not isinstance(endpoint, str) or trial.get("model_endpoint") != endpoint:
            raise ValueError("native E1 bootstrap model endpoint changed")
        if (bootstrap.get("_source_path") != str(bootstrap_path.resolve())
                or bootstrap.get("_source_sha256") != trial["bootstrap_receipt_sha256"]
                or bootstrap.get("model_identity", {}).get("model_id") != trial["selected_model"]
                or bootstrap.get("baseline") != {
                    "path": record["baseline"]["path"],
                    "sha256": record["baseline"]["sha256"],
                }
                or bootstrap.get("package_manifest") != {
                    "path": str(package_path.resolve()), "sha256": sha256_file(package_path)
                }):
            raise ValueError("native E1 bootstrap differs from frozen model or baseline")
        profile_hashes = {"bootstrap_receipt": sha256_file(bootstrap_path)}
        for name in ("e0_summary", "independent_review", "review_assignment", "baseline"):
            ref = bootstrap[name]
            profile_hashes[f"bootstrap_{name}"] = sha256_file(Path(ref["path"]))
        profile_hashes["bootstrap_model_artifacts"] = bootstrap["model_artifacts"]["files"]
    else:
        raise ValueError("native E1 trial has an unsupported profile source")
    event = read_json(paths["observation"])
    event_expected = {
        "schema_version": "mavis.e1-effective-config/v1", "trial_id": trial["trial_id"],
        "model": trial["selected_model"], "model_provider": "omlx",
        "catalog_sha256": trial["catalog_sha256"],
        "base_instructions_sha256": trial["instructions_sha256"],
    }
    if any(event.get(key) != value for key, value in event_expected.items()):
        raise ValueError("native E1 core effective configuration changed")
    session_id = event.get("session_id")
    transcript_path = Path(trial.get("transcript_path") or "")
    if (not isinstance(session_id, str) or not session_id
            or not transcript_path.is_file()
            or runtime.resolve() not in transcript_path.resolve().parents
            or trial.get("transcript_sha256") != sha256_file(transcript_path)
            or Path(event.get("rollout_path", "")).resolve() != transcript_path.resolve()):
        raise ValueError("native E1 transcript identity changed")
    with transcript_path.open(encoding="utf-8") as stream:
        first = json.loads(stream.readline())
    if (first.get("type") != "session_meta"
            or str(first.get("payload", {}).get("id")) != session_id):
        raise ValueError("native E1 transcript session changed")
    for key, digest_key, path in (
        ("gateway_root", "gateway_launcher_sha256", "bin/mcp-server.sh"),
        ("gateway_env_file", "gateway_env_sha256", None),
    ):
        if trial.get(key):
            target = Path(trial[key]) / path if path else Path(trial[key])
            if sha256_file(target) != trial.get(digest_key):
                raise ValueError("native E1 gateway source changed")
    return {"receipt": sha256_file(trial_path), "transcript": sha256_file(transcript_path),
            "package": sha256_file(package_path), **profile_hashes,
            "base_instructions": sha256_file(template_path), "snapshot": sha256_file(snapshot_path),
            "installed_inputs": {name: sha256_file(path) for name, path in installed_inputs.items()},
            "installed_mavis_package": package_tree_sha256(package_path.parent / "mavis"),
            **{name: sha256_file(path) for name, path in paths.items()}}


def validate_e1_bundle(home: Path, record: dict[str, Any], *, require_trials: bool = False) -> str:
    """Recheck every frozen E1 source and return its content digest."""
    root = Path(home) / "e1" / require_safe_id(record["experiment_id"], "experiment id")
    manifest_path = root / "cases.json"
    if sha256_file(manifest_path) != record["workload"]["manifest_sha256"]:
        raise ValueError("frozen E1 manifest changed")
    manifest = _manifest(manifest_path)
    if manifest["held_out"] != record["evaluation_split"]["held_out"]:
        raise ValueError("frozen E1 split changed")
    failure_path = Path(manifest["failure_receipt"]["path"])
    bundle = {
        "manifest": sha256_file(manifest_path),
        "failure_receipt": sha256_file(failure_path),
        "failure_stdout": sha256_file(failure_path.parent / "stdout.log"),
        "failure_stderr": sha256_file(failure_path.parent / "stderr.log"),
        "cases": {},
    }
    scores = {}
    bootstrap_cache: dict[str, Any] = {}
    for arm in ("baseline", "candidate"):
        bundle["cases"][arm] = {}
        passes = {}
        for case in manifest["cases"]:
            case_id = case["id"]
            result_path = root / "results" / arm / f"{case_id}.json"
            result = read_json(result_path)
            checkout = root / "checkouts" / arm / case_id
            if (
                result.get("schema_version") != "mavis.e1-case-result/v1"
                or result.get("experiment_id") != record["experiment_id"]
                or result.get("arm") != arm
                or result.get("case_id") != case_id
                or result.get("starting_revision") != case["revision"]
                or result.get("configuration_sha256") != record[arm]["sha256"]
                or not isinstance(result.get("checked_revision"), str)
                or len(result.get("checks", [])) != len(case["checks"])
            ):
                raise ValueError("E1 case result lost its frozen binding")
            if _clean_revision(checkout) != result["checked_revision"]:
                raise ValueError("E1 checked checkout changed after host evidence")
            check_hashes = []
            for expected, check in zip(case["checks"], result["checks"], strict=True):
                receipt_path = Path(check["receipt"])
                checkout = root / "checkouts" / arm / case_id
                if check.get("id") != expected["id"] or sha256_file(
                    receipt_path
                ) != check.get("sha256"):
                    raise ValueError("E1 check receipt changed")
                receipt = _verify_host_receipt(
                    receipt_path,
                    command=expected["argv"],
                    cwd=checkout,
                    check_id=expected["id"],
                    revision=result["checked_revision"],
                )
                if (
                    receipt["objective_id"] != record["experiment_id"]
                    or receipt["verdict"] != check["verdict"]
                    or receipt["timed_out"]
                ):
                    raise ValueError("E1 check receipt is incomplete")
                check_hashes.append(
                    {
                        "receipt": sha256_file(receipt_path),
                        "stdout": sha256_file(receipt_path.parent / "stdout.log"),
                        "stderr": sha256_file(receipt_path.parent / "stderr.log"),
                    }
                )
            passed = all(check["verdict"] == "pass" for check in result["checks"])
            if result.get("passed") is not passed:
                raise ValueError("E1 case pass claim changed")
            passes[case_id] = passed
            bundle["cases"][arm][case_id] = {
                "result": sha256_file(result_path),
                "checks": check_hashes,
            }
            if require_trials or (record.get("comparison") and record["comparison"][arm].get("e1_native_trial")):
                bundle["cases"][arm][case_id]["trial"] = _trial_hashes(
                    root, home, record, case, arm, result, bootstrap_cache
                )
        scores[arm] = sum(passes[case_id] for case_id in manifest["held_out"]) / len(
            manifest["held_out"]
        )
        if (
            record.get("comparison")
            and record["comparison"][arm]["target_score"] != scores[arm]
        ):
            raise ValueError("E1 score no longer matches case results")
        if arm == "baseline" and passes[manifest["regression"]]:
            raise ValueError("E1 baseline no longer fails regression")
        if arm == "candidate" and (
            not passes[manifest["regression"]]
            or not all(passes[case_id] for case_id in manifest["held_out"])
        ):
            raise ValueError("E1 candidate no longer passes mandatory cases")
    return _digest(bundle)


class E1Runner:
    def __init__(self, home: Path, gateway_assignment_starter=None, gateway_status_reader=None):
        self.home = Path(home)
        self.store = ExperimentStore(home, gateway_status_reader=gateway_status_reader)
        self.gateway_assignment_starter = gateway_assignment_starter or harness_assignment_start
        self.root = self.home / "e1"

    def _root(self, experiment_id: str) -> Path:
        return self.root / require_safe_id(experiment_id, "experiment id")

    def _frozen(self, experiment_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        record = self.store.load(experiment_id)
        root = self._root(experiment_id)
        path = root / "cases.json"
        if sha256_file(path) != record["workload"]["manifest_sha256"]:
            raise ValueError("frozen E1 manifest changed")
        manifest = _manifest(path)
        if record["evaluation_split"]["held_out"] != manifest["held_out"]:
            raise ValueError("frozen E1 split changed")
        return record, manifest

    def freeze(
        self,
        experiment_id: str,
        scope: str,
        kind: str,
        candidate: Path,
        cases: Path,
        hypothesis: str,
    ) -> dict[str, Any]:
        manifest = _manifest(cases)
        root = self._root(experiment_id)
        if root.exists():
            raise FileExistsError(root)
        root.mkdir(parents=True, mode=0o700)
        frozen = root / "cases.json"
        shutil.copyfile(cases, frozen)
        frozen.chmod(0o600)
        workload = {
            "suite": "E1",
            "manifest_sha256": sha256_file(frozen),
            "scoring": manifest["scoring"],
        }
        split = {
            "held_out": manifest["held_out"],
            "minimum_gain": manifest["minimum_gain"],
        }
        try:
            record = self.store.create(
                experiment_id,
                scope,
                kind,
                read_json(candidate),
                hypothesis=hypothesis,
                workload=workload,
                split=split,
            )
        except BaseException:
            shutil.rmtree(root)
            raise
        write_json(
            root / "coverage.json",
            {
                "schema_version": "mavis.e1-coverage/v1",
                "state": "incomplete",
                "arms": {"baseline": {}, "candidate": {}},
                "held_out": manifest["held_out"],
                "regression": manifest["regression"],
            },
        )
        return {
            "record": record,
            "candidate_requirements": candidate_assignment_requirements(record),
            "coverage": str(root / "coverage.json"),
        }

    def prepare(self, experiment_id: str, arm: str) -> dict[str, Any]:
        if arm not in {"baseline", "candidate"}:
            raise ValueError("invalid arm")
        _, manifest = self._frozen(experiment_id)
        root = self._root(experiment_id)
        prepared = {}
        for case in manifest["cases"]:
            case_id = case["id"]
            target = root / "checkouts" / arm / case_id
            if target.exists():
                raise FileExistsError(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--no-hardlinks",
                    "--quiet",
                    case["source"],
                    str(target),
                ],
                check=True,
            )
            try:
                _git(target, "checkout", "--detach", "--quiet", case["revision"])
                if _git(target, "rev-parse", "HEAD") != case["revision"] or _git(
                    target, "status", "--porcelain"
                ):
                    raise ValueError(
                        "case checkout is not at its clean starting revision"
                    )
            except BaseException:
                shutil.rmtree(target)
                raise
            prepared[case_id] = {
                "path": str(target),
                "starting_revision": case["revision"],
            }
        write_json(root / f"prepared-{arm}.json", prepared)
        return prepared

    def dispatch_candidate(self, experiment_id: str, case_id: str, *, job_id: str,
                           lane: str, model: str, task: str) -> dict[str, Any]:
        """Dispatch one prepared candidate checkout with frozen Mavis ownership."""
        record, manifest = self._frozen(experiment_id)
        if record["state"] != "candidate":
            raise ValueError("experiment is no longer a candidate")
        case = next((item for item in manifest["cases"] if item["id"] == case_id), None)
        if case is None:
            raise ValueError("unknown E1 case")
        require_safe_id(job_id, "candidate job id")
        if lane not in {"minimax", "zcode"} or not isinstance(model, str) or not model.strip():
            raise ValueError("candidate requires a named native coding lane and model")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("candidate task is required")
        root = self._root(experiment_id)
        dispatch_path = root / "dispatch" / f"{case_id}.json"
        if dispatch_path.exists():
            raise FileExistsError(dispatch_path)
        prepared = read_json(root / "prepared-candidate.json")
        checkout = root / "checkouts" / "candidate" / case_id
        entry = prepared.get(case_id)
        if (not isinstance(entry, dict) or Path(entry.get("path", "")).resolve() != checkout.resolve()
                or entry.get("starting_revision") != case["revision"]
                or _git(checkout, "rev-parse", "HEAD") != case["revision"]
                or _git(checkout, "status", "--porcelain")):
            raise ValueError("candidate checkout is not clean at its frozen revision")
        provider, harness = {"minimax": ("minimax", "opencode"),
                             "zcode": ("zai", "zcode")}[lane]
        requirements = candidate_assignment_requirements(record)
        arguments = {
            "job_id": job_id, "task": task, "lane": lane, "model": model,
            "cwd": str(checkout), "starting_revision": case["revision"],
            "owned_paths": [str(checkout)], "allowed_effects": ["edit owned checkout"],
            "required_checks": [check["id"] for check in case["checks"]],
            "context_packet": f"E1 candidate snapshot: {record['candidate']['path']} (sha256 {record['candidate']['sha256']}). Apply this frozen configuration; repair the case without changing acceptance checks.",
            "mavis_objective_id": experiment_id,
            "mavis_requirements": requirements,
            "mavis_owner": {"provider": provider, "model": model, "harness": harness},
        }
        started = self.gateway_assignment_starter(arguments)
        report = Path(started.get("report_path", "")) if isinstance(started, dict) else Path("")
        job_dir = Path(started.get("job_dir", "")) if isinstance(started, dict) else Path("")
        assignment = Path(started.get("assignment_path", "")) if isinstance(started, dict) else Path("")
        if (not isinstance(started, dict) or started.get("success") is not True
                or started.get("started") is not True or started.get("accepted") is not False
                or started.get("job_id") != job_id or not job_dir.is_absolute()
                or not report.is_absolute() or report.parent.resolve() != job_dir.resolve()
                or not assignment.is_absolute() or assignment.parent.resolve() != job_dir.resolve()
                or not assignment.is_file()):
            raise ValueError("native gateway did not start the exact Mavis candidate job")
        saved_assignment = read_json(assignment)
        if any(saved_assignment.get(key) != value for key, value in arguments.items() if key != "job_id"):
            raise ValueError("native gateway assignment differs from frozen candidate task")
        receipt = {"schema_version": "mavis.e1-native-dispatch/v1", "experiment_id": experiment_id,
                   "case_id": case_id, "job_id": job_id, "assignment_digest": _digest(arguments),
                   "candidate_sha256": record["candidate"]["sha256"], "starting_revision": case["revision"],
                   "checkout": str(checkout), "report_path": str(report), "job_dir": str(job_dir),
                   "assignment_path": str(assignment), "assignment_sha256": sha256_file(assignment)}
        write_json(dispatch_path, receipt)
        return receipt

    def compare_native(self, experiment_id: str, case_id: str) -> dict[str, Any]:
        """Compare paired installed trials using host checks, never launch status."""
        root = self._root(experiment_id)
        with (root / ".native-compare.lock").open("a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                return self._compare_native_locked(experiment_id, case_id)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def prepare_review(self, experiment_id: str, owner: dict[str, str]) -> dict[str, Any]:
        """Freeze a separate reviewer's exact installed-trial assignment."""
        from .e1_review import prepare_review

        record = self.store.load(experiment_id)
        self.store._check_comparison(record)
        return prepare_review(self.home, record, owner)

    def dispatch_review(self, experiment_id: str, job_id: str) -> dict[str, Any]:
        from .e1_review import dispatch_review

        record = self.store.load(experiment_id)
        self.store._check_comparison(record)
        return dispatch_review(self.home, record, job_id=job_id,
                               starter=self.gateway_assignment_starter)

    def import_review(self, experiment_id: str) -> dict[str, Any]:
        from .e1_review import import_review_report

        record = self.store.load(experiment_id)
        self.store._check_comparison(record)
        receipt = import_review_report(self.home, record, self.store.gateway_status_reader)
        return self.store.review(experiment_id, receipt)

    def _compare_native_locked(self, experiment_id: str, case_id: str) -> dict[str, Any]:
        record, manifest = self._frozen(experiment_id)
        if record["state"] != "candidate":
            raise ValueError("experiment is no longer a candidate")
        if case_id not in {case["id"] for case in manifest["cases"]}:
            raise ValueError("unknown E1 case")
        if (record["scope"], record["kind"]) != ("main", "prompts"):
            raise ValueError("native E1 trials support main prompt experiments only")
        if self.store.active("main")["configuration"] != record["baseline"]:
            raise ValueError("native E1 active baseline changed")
        root = self._root(experiment_id)
        coverage = self.coverage(experiment_id)
        if coverage["state"] != "complete":
            raise ValueError("native E1 coverage is incomplete")
        # This replays every raw host verdict, checkout revision, installed
        # package hash, launch command, core event, prompt, and transcript.
        bundle_digest = validate_e1_bundle(self.home, record, require_trials=True)
        results = {
            arm: {case["id"]: read_json(root / "results" / arm / f"{case['id']}.json")
                  for case in manifest["cases"]}
            for arm in ("baseline", "candidate")
        }
        trials = {
            arm: {case["id"]: read_json(root / "trials" / arm / f"{case['id']}.json")
                  for case in manifest["cases"]}
            for arm in ("baseline", "candidate")
        }
        common_identity = None
        for case in manifest["cases"]:
            baseline = trials["baseline"][case["id"]]
            candidate = trials["candidate"][case["id"]]
            if any(baseline.get(key) != candidate.get(key) for key in (
                "selected_model", "model_provider", "core_binary", "core_sha256",
                "package_manifest_sha256", "profile_source", "accepted_profile_id", "accepted_profile_sha256",
                "bootstrap_receipt_path", "bootstrap_receipt_sha256",
                "model_endpoint",
                "gateway_launcher_sha256", "gateway_env_sha256",
            )):
                raise ValueError("native E1 arms did not use the same installed model and core")
            if baseline["instructions_sha256"] == candidate["instructions_sha256"]:
                raise ValueError("native E1 arms used the same effective prompt")
            identity = tuple(baseline.get(key) for key in (
                "selected_model", "model_provider", "core_binary", "core_sha256",
                "package_manifest_sha256", "profile_source", "accepted_profile_id", "accepted_profile_sha256",
                "bootstrap_receipt_path", "bootstrap_receipt_sha256",
                "model_endpoint",
            ))
            if common_identity is None:
                common_identity = identity
            elif identity != common_identity:
                raise ValueError("native E1 cases did not use one installed model and core")
            if (results["candidate"][case["id"]]["passed"]
                    and results["candidate"][case["id"]]["checked_revision"] == case["revision"]):
                raise ValueError("native E1 candidate passed without a committed trial change")
        regression = manifest["regression"]
        if results["baseline"][regression]["passed"] or not results["candidate"][regression]["passed"]:
            raise ValueError("native E1 regression must fail baseline and pass candidate")
        held = manifest["held_out"]
        scores = {arm: sum(results[arm][case]["passed"] for case in held) / len(held)
                  for arm in ("baseline", "candidate")}
        if (not all(results["candidate"][case]["passed"] for case in held)
                or scores["candidate"] - scores["baseline"] < manifest["minimum_gain"]):
            raise ValueError("native E1 candidate failed mandatory checks or minimum gain")
        summaries = {arm: root / f"native-{arm}-trial-summary.json"
                     for arm in ("baseline", "candidate")}
        for evidence in summaries.values():
            if evidence.exists():
                raise FileExistsError(evidence)
        created = []
        try:
            for arm, evidence in summaries.items():
                write_json(evidence, {
                    "schema_version": "mavis.e1-native-trial-summary/v1",
                    "experiment_id": experiment_id, "arm": arm,
                    "manifest_sha256": record["workload"]["manifest_sha256"],
                    "snapshot_sha256": record[arm]["sha256"],
                    "bundle_digest": bundle_digest,
                    "case_ids": [case["id"] for case in manifest["cases"]],
                    "trial_receipts": {case["id"]: {
                        "path": str(root / "trials" / arm / f"{case['id']}.json"),
                        "sha256": sha256_file(root / "trials" / arm / f"{case['id']}.json"),
                    } for case in manifest["cases"]},
                    "case_results": {case["id"]: {
                        "path": str(root / "results" / arm / f"{case['id']}.json"),
                        "sha256": sha256_file(root / "results" / arm / f"{case['id']}.json"),
                    } for case in manifest["cases"]},
                    "held_out_score": scores[arm],
                })
                created.append(evidence)
            write_json(root / "coverage.json", coverage)
            paired = {}
            for arm in ("baseline", "candidate"):
                paired[arm] = {
                    "configuration_sha256": record[arm]["sha256"],
                    "workload_digest": _digest(record["workload"]),
                    "case_ids": held,
                    "mandatory_passed": all(results[arm][case]["passed"] for case in held),
                    "target_score": scores[arm], "e1_bundle_digest": bundle_digest,
                    "e1_native_trial": True,
                    "evidence": {"path": str(summaries[arm]), "sha256": sha256_file(summaries[arm])},
                }
            # A trial is not a gateway worker. The reserved ID keeps the current
            # review/stage gateway gate closed pending native trial review wiring.
            paired["candidate"]["candidate_job_id"] = f"e1trial-{experiment_id}"
            return self.store.compare(experiment_id, baseline_result=paired["baseline"],
                                      candidate_result=paired["candidate"])
        except BaseException:
            for evidence in created:
                evidence.unlink(missing_ok=True)
            raise

    def check(self, experiment_id: str, arm: str, case_id: str) -> dict[str, Any]:
        record, manifest = self._frozen(experiment_id)
        if arm not in {"baseline", "candidate"}:
            raise ValueError("invalid arm")
        case = next((case for case in manifest["cases"] if case["id"] == case_id), None)
        if case is None:
            raise ValueError("unknown E1 case")
        root = self._root(experiment_id)
        result_path = root / "results" / arm / f"{case_id}.json"
        if result_path.exists():
            raise FileExistsError(result_path)
        prepared = read_json(root / f"prepared-{arm}.json")
        entry = prepared.get(case_id)
        checkout = root / "checkouts" / arm / case_id
        if (
            not entry
            or Path(entry["path"]).resolve() != checkout.resolve()
            or entry["starting_revision"] != case["revision"]
        ):
            raise ValueError("case lacks a matching prepared checkout")
        checked_revision = _clean_revision(checkout)
        if (
            subprocess.run(
                [
                    "git",
                    "merge-base",
                    "--is-ancestor",
                    case["revision"],
                    checked_revision,
                ],
                cwd=checkout,
                check=False,
            ).returncode
            != 0
        ):
            raise ValueError(
                "case checkout no longer descends from its starting revision"
            )
        receipts = []
        for check in case["checks"]:
            receipt_path = run_command(
                self.home,
                experiment_id,
                check["argv"],
                checkout,
                acceptance_check_ids=[check["id"]],
                timeout=check["timeout_seconds"],
            )
            receipt = read_json(receipt_path)
            receipts.append(
                {
                    "id": check["id"],
                    "receipt": str(receipt_path),
                    "sha256": sha256_file(receipt_path),
                    "verdict": receipt["verdict"],
                }
            )
        if _clean_revision(checkout) != checked_revision:
            raise ValueError("E1 check changed the committed checkout")
        result = {
            "schema_version": "mavis.e1-case-result/v1",
            "experiment_id": experiment_id,
            "arm": arm,
            "case_id": case_id,
            "starting_revision": case["revision"],
            "checked_revision": checked_revision,
            "configuration_sha256": record[arm]["sha256"],
            "checks": receipts,
            "passed": all(item["verdict"] == "pass" for item in receipts),
        }
        write_json(result_path, result)
        return result

    def coverage(self, experiment_id: str) -> dict[str, Any]:
        _, manifest = self._frozen(experiment_id)
        root = self._root(experiment_id)
        arms = {}
        for arm in ("baseline", "candidate"):
            arms[arm] = {}
            for case in manifest["cases"]:
                path = root / "results" / arm / f"{case['id']}.json"
                arms[arm][case["id"]] = "complete" if path.is_file() else "unfinished"
        return {
            "schema_version": "mavis.e1-coverage/v1",
            "state": "complete"
            if all(v == "complete" for arm in arms.values() for v in arm.values())
            else "incomplete",
            "arms": arms,
            "held_out": manifest["held_out"],
            "regression": manifest["regression"],
        }

    def compare(
        self, experiment_id: str, candidate_job_id: str, candidate_report: Path
    ) -> dict[str, Any]:
        record, manifest = self._frozen(experiment_id)
        if record["state"] != "candidate":
            raise ValueError("experiment is no longer a candidate")
        coverage = self.coverage(experiment_id)
        write_json(self._root(experiment_id) / "coverage.json", coverage)
        if coverage["state"] != "complete":
            raise ValueError("E1 coverage is incomplete")
        results = {}
        for arm in ("baseline", "candidate"):
            results[arm] = {}
            for case in manifest["cases"]:
                path = (
                    self._root(experiment_id) / "results" / arm / f"{case['id']}.json"
                )
                result = read_json(path)
                if (
                    result.get("arm") != arm
                    or result.get("case_id") != case["id"]
                    or result.get("starting_revision") != case["revision"]
                    or result.get("configuration_sha256") != record[arm]["sha256"]
                    or not result.get("checked_revision")
                ):
                    raise ValueError("E1 case result lost its frozen binding")
                checkout = self._root(experiment_id) / "checkouts" / arm / case["id"]
                if _clean_revision(checkout) != result["checked_revision"]:
                    raise ValueError("E1 checked checkout changed before comparison")
                for expected, check in zip(
                    case["checks"], result["checks"], strict=True
                ):
                    receipt_path = Path(check["receipt"])
                    receipt = read_json(receipt_path)
                    if (
                        check["id"] != expected["id"]
                        or sha256_file(receipt_path) != check["sha256"]
                        or receipt["verdict"] != check["verdict"]
                        or receipt["command"] != expected["argv"]
                        or receipt["acceptance_check_ids"] != [expected["id"]]
                        or receipt["timed_out"]
                        or receipt["objective_id"] != experiment_id
                        or Path(receipt["cwd"]).resolve() != checkout.resolve()
                        or receipt["changed_revision"] != result["checked_revision"]
                    ):
                        raise ValueError("E1 check receipt changed or is incomplete")
                    output = Path(receipt["raw_output"]["path"])
                    if (
                        sha256_file(output / "stdout.log")
                        + ":"
                        + sha256_file(output / "stderr.log")
                        != receipt["raw_output"]["sha256"]
                    ):
                        raise ValueError("E1 raw output changed")
                if result["passed"] != all(
                    check["verdict"] == "pass" for check in result["checks"]
                ):
                    raise ValueError("E1 case pass claim changed")
                results[arm][case["id"]] = result
        regression = manifest["regression"]
        if (
            results["baseline"][regression]["passed"]
            or not results["candidate"][regression]["passed"]
        ):
            raise ValueError("regression must fail baseline and pass candidate")
        held = manifest["held_out"]
        scores = {
            arm: sum(results[arm][case_id]["passed"] for case_id in held) / len(held)
            for arm in ("baseline", "candidate")
        }
        if (
            not all(results["candidate"][case_id]["passed"] for case_id in held)
            or scores["candidate"] - scores["baseline"] < manifest["minimum_gain"]
        ):
            raise ValueError("candidate failed mandatory checks or minimum gain")
        bundle_digest = validate_e1_bundle(self.home, record)
        report = Path(candidate_report).resolve(strict=True)
        require_safe_id(candidate_job_id, "candidate job id")
        candidate_evidence = self._root(experiment_id) / "candidate-worker-report.json"
        if candidate_evidence.exists():
            raise FileExistsError(candidate_evidence)
        shutil.copyfile(report, candidate_evidence)
        candidate_evidence.chmod(0o600)
        baseline_evidence = self._root(experiment_id) / "baseline-results.json"
        write_json(
            baseline_evidence,
            {
                "schema_version": "mavis.e1-baseline-results/v1",
                "results": results["baseline"],
            },
        )
        paired = {}
        for arm, evidence in (
            ("baseline", baseline_evidence),
            ("candidate", candidate_evidence),
        ):
            paired[arm] = {
                "configuration_sha256": record[arm]["sha256"],
                "workload_digest": _digest(record["workload"]),
                "case_ids": held,
                "mandatory_passed": all(
                    results[arm][case_id]["passed"] for case_id in held
                ),
                "target_score": scores[arm],
                "e1_bundle_digest": bundle_digest,
                "evidence": {"path": str(evidence), "sha256": sha256_file(evidence)},
            }
        paired["candidate"]["candidate_job_id"] = candidate_job_id
        return self.store.compare(
            experiment_id,
            baseline_result=paired["baseline"],
            candidate_result=paired["candidate"],
        )
