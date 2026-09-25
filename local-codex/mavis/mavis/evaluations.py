"""Evidence-producing evaluation gates for Mavis.

E0 is deliberately fail-closed. Cases with no real worker/runtime evidence are
reported as blocked; a deterministic unit test cannot stand in for a local
model completing a coding task or an external harness operating in its native
environment.
"""

from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any, Callable
import uuid
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .evidence import run_command
from .objectives import ObjectiveStore, _acceptance_check_commands
from .project_evidence import project_home
from .runtime import RuntimeConfig, _listener_pids, admission, endpoint_alive, inventory, loaded_generation_models, omlx_live_process_binding, omlx_runtime_fingerprint, require_idle_iris_handoff
from .storage import sha256_file, write_json


E0_CASES = (
    "tool-roundtrip",
    "small-repository",
    "dirty-work-preservation",
    "seeded-failure-repair",
    "fabricated-success-rejection",
    "buried-failure",
    "compaction-restart",
    "repeated-no-progress",
    "external-harness",
    "isolation-recovery",
)

E0_RUNTIME_OVERRIDES = (
    "MAVIS_BIN",
    "LOCAL_CODEX_SHARE_DIR",
    "LOCAL_CODEX_BIN",
    "LOCAL_CODEX_HOME",
    "LOCAL_CODEX_MODEL",
    "OMLX_BASE_URL",
    "IRIS_OMLX_BASE_URL",
    "MAVIS_MODEL_DIR",
    "MAVIS_OMLX_BIN",
    "MAVIS_GATEWAY_ROOT",
    "MAVIS_GATEWAY_ENV_FILE",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trusted_raw_output_path(raw: Path, repo: Path, service_home: Path) -> bool:
    """Accept retained service-home references and ignored project-local capture."""
    if raw.parent == service_home / "tool-output":
        return raw.is_file()
    state = repo.resolve() / ".mavis"
    if not raw.is_file() or any(path.is_symlink() for path in
                                (state, state / ".gitignore", raw.parent, raw)):
        return False
    if raw.parent.resolve() != state / "tool-output":
        return False
    git_env = {name: value for name, value in os.environ.items() if name not in
               {"GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE", "GIT_PREFIX"}}
    git_root = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=repo,
                              text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, env=git_env, check=False)
    if git_root.returncode != 0 or Path(git_root.stdout.strip()).resolve() != repo.resolve():
        return False
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", ".mavis/tool-output/probe.raw"],
        cwd=repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=git_env, check=False)
    return ignored.returncode == 0


def installed_candidate_fingerprint() -> dict[str, str]:
    """Bind E0 live work to every installed executable that prepares Mavis."""
    share = Path.home() / ".local" / "share" / "local-codex"
    overrides = [name for name in E0_RUNTIME_OVERRIDES if os.environ.get(name)
                 and not (name == "LOCAL_CODEX_SHARE_DIR"
                          and Path(os.environ[name]).resolve() == share.resolve())]
    if overrides:
        raise ValueError(f"E0 requires the canonical installed runtime; overrides set: {', '.join(overrides)}")
    service_home = Path.home() / ".local-codex" / "mavis-service"
    if os.environ.get("MAVIS_HOME") and Path(os.environ["MAVIS_HOME"]).resolve() != service_home.resolve():
        raise ValueError("E0 requires the canonical MAVIS_HOME")
    python_path = os.environ.get("PYTHONPATH")
    python_paths = python_path.split(os.pathsep) if python_path is not None else []
    if os.environ.get("PYTHONHOME") or any(not path or Path(path).resolve() != share.resolve() for path in python_paths):
        raise ValueError("E0 requires the installed Python package path only")
    command = Path.home() / ".local" / "bin" / "mavis"
    package = share / "mavis"
    digest = hashlib.sha256()
    files = sorted(package.rglob("*.py"))
    if not files:
        raise FileNotFoundError("installed Mavis service package is missing")
    for path in files:
        digest.update(str(path.relative_to(package)).encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return {
        "core_sha256": sha256_file(share / "local-codex-core"),
        "launcher_sha256": sha256_file(Path.home() / "Desktop" / "Mavis.command"),
        "command_sha256": sha256_file(command),
        "prepare_runtime_sha256": sha256_file(share / "prepare_runtime.py"),
        "launch_core_sha256": sha256_file(share / "launch_core.py"),
        "generation_lease_sha256": sha256_file(share / "generation_lease.py"),
        "instructions_template_sha256": sha256_file(share / "base-instructions.md"),
        "persona_template_sha256": sha256_file(share / "persona.toml"),
        "service_sha256": digest.hexdigest(),
    }


def _safe_buried_inspection(command: str) -> bool:
    """Allow output inspection, but no second fixture execution through a shell."""
    if not command or any(token in command for token in ("$(", "`", "\n", "||")):
        return False
    if "&" in command.replace("2>&1", "").replace("&&", ""):
        return False
    normalized = command.replace("2>&1", "").replace("&&", ";")
    if "<" in normalized or ">" in normalized:
        return False
    allowed = {"ls", "echo", "rg", "grep", "head", "tail", "cut", "cat", "wc", "pwd", "stat", "sort", "sed", "awk", "file", "uniq"}
    for part in re.split(r"[;|]", normalized):
        try:
            args = shlex.split(part)
        except ValueError:
            return False
        if not args or args[0] not in allowed:
            return False
        if args[0] == "sort" and args not in (["sort"], ["sort", "-u"], ["sort", "--unique"]):
            return False
        if args[0] == "rg" and any(arg.startswith("--pre") for arg in args[1:]):
            return False
        if args[0] == "sed" and (len(args) != 4 or args[1] != "-n"
                                  or not re.fullmatch(r"\d+(?:,\d+)?p", args[2])
                                  or not args[3].endswith(".raw")):
            return False
        if args[0] == "awk" and (len(args) != 3
                                  or not re.fullmatch(r"NR==\d+", args[1])
                                  or not args[2].endswith(".raw")):
            return False
        if args[0] == "file" and args != ["file", "produce_log.py"]:
            return False
        if args[0] == "uniq" and args not in (["uniq"], ["uniq", "-c"]):
            return False
    return True


def _states_failed_exit_one(answer: str) -> bool:
    """Require one statement about the command's failure and exact exit code."""
    if (re.search(r"\b(?:command|produce_log\.py)\b.{0,60}\b(?:succeeded|passed)\b", answer, re.IGNORECASE)
            or re.search(r"命令.{0,60}(?:成功|通过)", answer)):
        return False
    code_values = re.compile(
        r"(?:\b(?:exit(?:ed)?(?:\s+with)?\s+code|code)\b|退出码)\s*[:=：]?\s*(?:\*\*)?`?(\d+)\b`?(?:\*\*)?",
        re.IGNORECASE,
    )
    if any(int(match.group(1)) != 1 for match in code_values.finditer(answer)):
        return False
    for statement in re.split(r"(?<=[.!?])\s+|\n+", answer):
        if (re.search(r"\b(?:command|produce_log\.py)\b|命令", statement, re.IGNORECASE)
                and re.search(r"\b(?:failed|exited)\b|失败", statement, re.IGNORECASE)):
            numbers = [int(match.group(1)) for match in code_values.finditer(statement)]
            if 1 in numbers and all(number == 1 for number in numbers):
                return True
    return False


def native_review_completed(log: str, verdict: str) -> bool:
    """Recognize a completed native ZCode stream or its structured CLI result."""
    heading = verdict.lstrip().splitlines()[0].strip() if verdict.strip() else ""
    accepted = bool(re.match(r"^(?:\*\*)?ACCEPT(?:\*\*)?(?:$|[\s:.-])", heading))
    events = []
    for line in log.splitlines():
        if line.startswith("{"):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                events = []
                break
    if any(event.get("type") == "turn.completed" for event in events):
        return accepted
    start = log.find("{")
    if start < 0:
        return False
    try:
        result = json.loads(log[start:])
    except json.JSONDecodeError:
        return False
    projection = result.get("projection") or {}
    response = result.get("response")
    return bool(
        isinstance(result.get("sessionId"), str)
        and result["sessionId"].startswith("sess_")
        and isinstance(result.get("turnId"), str)
        and result["turnId"].startswith("turn_")
        and isinstance(result.get("eventCount"), int)
        and result["eventCount"] > 0
        and projection.get("status") == "idle"
        and isinstance(response, str)
        and response == verdict
        and accepted
    )


def _local_response_url(url: str) -> str:
    parsed = urlparse(url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or parsed.path != "/v1/responses"):
        raise ValueError("E0 Responses API must use the local oMLX endpoint")
    return url


def _post_json(url: str, payload: dict[str, Any], timeout: float = 180.0, *,
               api_key: str | None = None) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        _local_response_url(url),
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json",
                 **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
    )
    with urlopen(request, timeout=timeout) as response:
        parsed = json.load(response)
    if not isinstance(parsed, dict):
        raise RuntimeError("Responses API returned a non-object")
    return parsed


class E0Evaluator:
    def __init__(self, home: Path, config: RuntimeConfig):
        self.home = Path(home)
        self.config = config
        self.root = self.home / "evaluations" / "e0"
        self._run_root: Path | None = None

    def _receipt(self, case: str, status: str, evidence: list[str], **extra: Any) -> dict[str, Any]:
        if case not in E0_CASES:
            raise ValueError(f"unknown E0 case: {case}")
        payload = {
            "schema_version": "mavis.evaluation-case/v1",
            "suite": "e0",
            "case": case,
            "status": status,
            "evidence": evidence,
            "observed_at": _now(),
            **extra,
        }
        write_json(self.root / f"{case}.json", payload)
        if self._run_root is not None:
            write_json(self._run_root / f"{case}.json", payload)
        return payload

    @contextmanager
    def _suite_lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".suite.lock").open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def run_case(self, case: str) -> dict[str, Any]:
        with self._suite_lock():
            return self._run_case_unlocked(case)

    def _run_case_unlocked(self, case: str) -> dict[str, Any]:
        methods: dict[str, Callable[[], dict[str, Any]]] = {
            "tool-roundtrip": self._tool_roundtrip,
            "small-repository": lambda: self._small_repository("small-repository"),
            "dirty-work-preservation": lambda: self._small_repository("dirty-work-preservation"),
            "seeded-failure-repair": lambda: self._small_repository("seeded-failure-repair"),
            "external-harness": self._external_harness,
            "fabricated-success-rejection": self._fabricated_success,
            "buried-failure": self._buried_failure,
            "compaction-restart": self._compaction_restart,
            "repeated-no-progress": self._repeated_no_progress,
            "isolation-recovery": self._isolation_recovery,
        }
        if case in methods:
            try:
                return methods[case]()
            except Exception as exc:
                return self._receipt(case, "reject", [f"{type(exc).__name__}: {exc}"])
        return self._receipt(
            case,
            "blocked",
            [
                "This case requires a real Mavis or native-harness assignment receipt; no synthetic fixture is accepted as a substitute."
            ],
        )

    def _small_repository(self, case: str) -> dict[str, Any]:
        """Recheck an installed Mavis repair and its separate native GLM review."""
        from .e0_tasks import small_repository_review_prompt
        from .e2_tasks import codex_review_completed, independent_review_command

        candidates = sorted(
            (self.root / "tasks").glob("*/manifest.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for manifest_path in candidates:
            task_root = manifest_path.parent
            mavis_log = task_root / "installed-mavis-repair.jsonl"
            review_log = task_root / "glm-review.jsonl"
            review_stderr = task_root / "glm-review.stderr.log"
            review_result = task_root / "glm-review.txt"
            installed_run = task_root / "installed-run.json"
            if not all(path.is_file() for path in (mavis_log, review_log, review_stderr,
                                                   review_result, installed_run)):
                continue
            observed_run = json.loads(installed_run.read_text())
            if (observed_run.get("schema_version") != "mavis.e0-installed-run/v2"
                    or observed_run.get("candidate") != installed_candidate_fingerprint()):
                continue
            if observed_run.get("mavis_log_sha256") != sha256_file(mavis_log) or observed_run.get("review_log_sha256") != sha256_file(review_log):
                raise ValueError("E0 installed task logs changed after observation")
            if (observed_run.get("review_stderr_sha256") != sha256_file(review_stderr)
                    or observed_run.get("review_text_sha256") != sha256_file(review_result)
                    or observed_run.get("review_exit") != 0):
                raise ValueError("E0 GLM review output or exit changed")
            manifest = json.loads(manifest_path.read_text())
            repo = Path(manifest["repo"]).resolve()
            if repo != (task_root / "repo").resolve() or not repo.is_dir():
                raise ValueError("E0 task repository path is invalid")
            if manifest["baseline_exit_status"] == 0 or sha256_file(Path(manifest["baseline_log"])) != manifest["baseline_log_sha256"]:
                raise ValueError("E0 failing baseline is invalid")
            if sha256_file(Path(manifest["protected_dirty_file"])) != manifest["protected_dirty_sha256"]:
                raise ValueError("E0 protected dirty note changed")
            revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            if revision != manifest["starting_revision"]:
                raise ValueError("E0 task revision changed unexpectedly")
            changed = set(subprocess.check_output(["git", "-C", str(repo), "diff", "--name-only"], text=True).splitlines())
            expected = set(manifest["owned_paths"]) | {"user-notes.txt"}
            if changed != expected:
                raise ValueError("E0 changed paths do not match the allowed repair and protected note")
            if subprocess.check_output(["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard"], text=True).strip():
                raise ValueError("E0 task contains unexpected untracked files")
            mavis_events = [json.loads(line) for line in mavis_log.read_text().splitlines() if line.startswith("{")]
            review_output = review_log.read_text()
            mavis_finished = any(event.get("type") == "turn.completed" for event in mavis_events)
            review_argv = independent_review_command(
                repo, review_result, small_repository_review_prompt(manifest_path))
            if observed_run.get("review_argv") != review_argv:
                raise ValueError("E0 GLM review command changed")
            try:
                review_finished = observed_run.get("review_thread_id") == codex_review_completed(
                    review_output, review_result.read_text())
            except ValueError:
                review_finished = False
            patch_seen = any(
                event.get("type") == "item.completed"
                and event.get("item", {}).get("type") == "file_change"
                and any(Path(change.get("path", "")).resolve() == repo / "package" / "pricing.py" for change in event["item"].get("changes", []))
                for event in mavis_events
            )
            if not (mavis_finished and review_finished and patch_seen):
                raise ValueError("E0 installed repair or independent GLM review is incomplete")
            tests = subprocess.run(
                manifest["test_command"], cwd=repo, capture_output=True, text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=False,
            )
            if tests.returncode != 0 or "Ran 2 tests" not in tests.stderr or "OK" not in tests.stderr:
                raise ValueError("E0 seeded repair tests failed")
            return self._receipt(case, "pass", [
                f"Installed Mavis repaired {repo / 'package' / 'pricing.py'}; exact tests passed and protected note hash held",
                f"Separate native GLM 5.3 review accepted; logs sha256 {sha256_file(mavis_log)} and {sha256_file(review_log)}",
            ], manifest=str(manifest_path), review=str(review_result))
        return self._receipt(case, "blocked", ["No complete installed Mavis repair and independent GLM review found"])

    def _external_harness(self) -> dict[str, Any]:
        """Revalidate a native gateway worker and distinct GLM verifier closure."""
        store = ObjectiveStore(self.home)
        paths = sorted(
            store.root.glob("e0-gateway-bound-*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in paths:
            record = json.loads(path.read_text())
            if record.get("state") != "accepted":
                continue
            try:
                _acceptance_check_commands(record)
            except ValueError:
                # Historical accepted objectives predate exact command binding.
                # They cannot satisfy current E0, but must not hide a later
                # valid accepted worker or turn this case into an exception.
                continue
            try:
                store._assert_acceptance(record)
            except ValueError:
                # Objectives accepted by an earlier verifier (Terra) no longer
                # bind to the live gateway's GLM verifier. Skip them like the
                # pre-binding records above; none valid means blocked.
                continue
            retained = record["gateway_verifications"][-1]
            gateway_receipt = json.loads(Path(retained["path"]).read_text())
            status = gateway_receipt["gateway_status"]
            return self._receipt("external-harness", "pass", [
                f"Native worker {status['job_id']} completed with host receipt and an accepted separate GLM verifier job {status['acceptance']['verifier_job_id']}",
                f"Mavis objective {record['objective_id']} revalidated against the live configured gateway and exact host evidence",
            ], objective_id=record["objective_id"], gateway_receipt=retained["path"])
        return self._receipt("external-harness", "blocked", ["No accepted native gateway worker bound to a Mavis objective was found"])

    def run(self, case: str | None = None) -> dict[str, Any]:
        if case is not None:
            return self.run_case(case)
        run_id = os.environ.get("MAVIS_E0_RUN_ID") or uuid.uuid4().hex
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise ValueError("E0 run ID must be 32 lowercase hexadecimal characters")
        with self._suite_lock():
            self._run_root = self.root / "runs" / run_id
            if self._run_root.exists():
                raise FileExistsError("E0 run ID already exists")
            try:
                results = [self._run_case_unlocked(item) for item in E0_CASES]
                passed = len(results) == len(E0_CASES) and all(item["status"] == "pass" for item in results)
                summary = {
                    "schema_version": "mavis.evaluation-suite/v1",
                    "suite": "e0",
                    "run_id": run_id,
                    "status": "pass" if passed else "reject",
                    "mandatory_cases": list(E0_CASES),
                    "results": [{"case": item["case"], "status": item["status"]} for item in results],
                    "observed_at": _now(),
                }
                if passed:
                    summary["installed_candidate"] = installed_candidate_fingerprint()
                    summary["model_id"] = self.config.model
                    summary["case_receipts"] = {
                        item: sha256_file(self._run_root / f"{item}.json") for item in E0_CASES
                    }
                write_json(self._run_root / "summary.json", summary)
                write_json(self.root / "summary.json", summary)
                return summary
            finally:
                self._run_root = None

    def _tool_roundtrip(self) -> dict[str, Any]:
        run_id = os.environ.get("MAVIS_E0_RUN_ID")
        prior_path = self.root / "tool-roundtrip.json"
        if run_id and prior_path.is_file():
            prior = json.loads(prior_path.read_text())
            if (prior.get("status") == "pass" and prior.get("run_id") == run_id
                    and prior.get("model_id") == self.config.model
                    and prior.get("installed_candidate") == installed_candidate_fingerprint()
                    and isinstance(prior.get("response_ids"), list)
                    and len(prior["response_ids"]) == 2
                    and all(isinstance(item, str) and item for item in prior["response_ids"])):
                return self._receipt(
                    "tool-roundtrip", "pass",
                    ["Installed live tool call and continuation passed earlier in this E0 run"],
                    run_id=run_id, model_id=self.config.model,
                    installed_candidate=prior["installed_candidate"],
                    response_ids=prior["response_ids"],
                    live_receipt_sha256=sha256_file(prior_path),
                )
        if not endpoint_alive(self.config.endpoint, **(
            {"api_key": self.config.api_key} if self.config.api_key else {}
        )):
            return self._receipt("tool-roundtrip", "blocked", ["Mavis endpoint is unavailable"])
        selected = next(
            (item for item in inventory(self.config.endpoint, **(
                {"api_key": self.config.api_key} if self.config.api_key else {}
            )) if item.get("id") == self.config.model),
            None,
        )
        if not selected or not selected.get("loaded"):
            return self._receipt("tool-roundtrip", "blocked", ["Mavis model is not loaded; E0 will not trigger an implicit model load"])
        admission_config = self.config
        if os.environ.get("MAVIS_E0_SHARED_GPU_LEASE"):
            if os.environ["MAVIS_E0_SHARED_GPU_LEASE"] != "codex-mavis":
                raise ValueError("E0 shared GPU lease holder is invalid")
            try:
                expected_iris = json.loads(os.environ["MAVIS_E0_IRIS_GENERATION_MODELS"])
            except (KeyError, json.JSONDecodeError) as error:
                raise ValueError("E0 IRIS generation snapshot is unavailable") from error
            if (not isinstance(expected_iris, list) or not expected_iris
                    or any(not isinstance(model, str) or not model for model in expected_iris)
                    or len(expected_iris) != len(set(expected_iris))):
                raise ValueError("E0 IRIS generation snapshot is invalid")
            lease = subprocess.run(
                [str(Path.home() / ".local/bin/gpu-lease"), "status"],
                capture_output=True, text=True, timeout=15, check=False,
                env={key: value for key, value in os.environ.items()
                     if key != "MAVIS_E0_TRIAL_API_KEY"},
            )
            if (lease.returncode != 0
                    or not lease.stdout.startswith("codex-mavis has the GPU:")
                    or "IRIS: a game is live" in lease.stdout
                    or "IRIS: not answering" in lease.stdout):
                raise RuntimeError("E0 shared GPU lease or IRIS game state is unsafe")
            require_idle_iris_handoff(self.config, expected_models=expected_iris)
            admission_config = replace(self.config, allow_concurrent_local=True)
        decision = admission(admission_config)
        if not decision["allowed"]:
            return self._receipt("tool-roundtrip", "blocked", decision["reasons"])
        payload = {
            "model": self.config.model,
            "input": "Call the mavis_probe tool with value exactly E0_CANARY. Do not answer in text.",
            "store": True,
            "max_output_tokens": 128,
            "tools": [
                {
                    "type": "function",
                    "name": "mavis_probe",
                    "description": "Return the supplied canary value.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["value"],
                        "properties": {"value": {"type": "string"}},
                    },
                }
            ],
            "tool_choice": "required",
        }
        first = _post_json(self.config.endpoint.rstrip("/") + "/responses", payload,
                           **({"api_key": self.config.api_key} if self.config.api_key else {}))
        calls = [item for item in first.get("output", []) if item.get("type") == "function_call"]
        if len(calls) != 1 or calls[0].get("name") != "mavis_probe":
            raise RuntimeError("model did not emit the required function call")
        arguments = json.loads(calls[0].get("arguments") or "{}")
        if arguments.get("value") != "E0_CANARY":
            raise RuntimeError("tool arguments did not preserve the exact canary")
        second = _post_json(
            self.config.endpoint.rstrip("/") + "/responses",
            {
                "model": self.config.model,
                "previous_response_id": first["id"],
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": calls[0]["call_id"],
                        "output": "E0_CANARY_OK",
                    }
                ],
                "store": True,
                "max_output_tokens": 256,
            },
            **({"api_key": self.config.api_key} if self.config.api_key else {}),
        )
        if second.get("status") != "completed":
            write_json(self.root / "tool-roundtrip-incomplete.json", {
                "schema_version": "mavis.e0-tool-roundtrip-incomplete/v1",
                "first_response_id": first["id"],
                "second_response_id": second.get("id"),
                "status": second.get("status"),
                "incomplete_details": second.get("incomplete_details"),
                "usage": second.get("usage"),
            })
            raise RuntimeError(
                f"tool-result continuation did not complete: {second.get('status')}"
            )
        return self._receipt(
            "tool-roundtrip",
            "pass",
            [f"Responses {first['id']} -> {second['id']} completed one exact function-call round trip"],
            response_ids=[first["id"], second["id"]],
            run_id=run_id,
            model_id=self.config.model,
            installed_candidate=installed_candidate_fingerprint(),
        )

    def _fabricated_success(self) -> dict[str, Any]:
        fixture_home = self.home / "e0-fixtures" / "fabricated"
        store = ObjectiveStore(fixture_home)
        objective_id = "e0-fabricated"
        if store._path(objective_id).exists():
            store._path(objective_id).unlink()
        store.create({
            "schema_version": "mavis.objective/v1",
            "objective_id": objective_id,
            "blueprint": "Reject forged completion evidence",
            "requirements": [{"id": "r1", "text": "reject fabricated success"}],
            "dependencies": [], "scope": {},
            "acceptance_checks": [{"id": "c1", "command": ["python3", "-c", "print('1 passed')"]}],
            "unresolved_decisions": [],
        })
        forged = store.root.parent / "evidence" / objective_id / "forged"
        forged.mkdir(parents=True, exist_ok=True)
        forged_receipt = forged / "receipt.json"
        write_json(forged_receipt, {
            "schema_version": "mavis.evidence-receipt/v1",
            "verdict": "pass", "changed_revision": "abcdef0",
        })
        try:
            store.add_receipt(objective_id, forged_receipt)
        except ValueError:
            pass
        else:
            raise RuntimeError("fabricated receipt was accepted")
        repo = fixture_home / "repo"
        repo.mkdir(parents=True, exist_ok=True)
        if not (repo / ".git").is_dir():
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "-c", "user.name=Mavis E0", "-c", "user.email=mavis@local.invalid", "commit", "--allow-empty", "-qm", "fixture"], check=True)
        revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        real_receipt = run_command(fixture_home, objective_id, ["python3", "-c", "print('1 passed')"], repo, acceptance_check_ids=["c1"])
        store.add_receipt(objective_id, real_receipt)
        store.add_verification(objective_id, {
            "schema_version": "mavis.verifier/v1",
            "verification_id": "forged-review",
            "objective_id": objective_id,
            "revision": revision,
            "requirements": ["r1"],
            "protected_fixtures": [],
            "checks": ["c1"],
            "required_receipts": [str(real_receipt)],
            "verifier": {"provider": "fake", "model": "fake", "harness": "fake"},
            "verdict": "accepted",
        })
        store.transition(objective_id, "running", "fabrication test")
        store.transition(objective_id, "awaiting verification", "fabrication test")
        try:
            store.transition(objective_id, "accepted", "forged verifier document")
        except ValueError:
            return self._receipt("fabricated-success-rejection", "pass", [
                "Incomplete in-root receipt was rejected",
                "A host test receipt plus model-supplied accepted verifier JSON could not authorize acceptance without a bound native gateway/GLM verifier receipt",
            ])
        raise RuntimeError("forged verifier JSON moved objective to accepted")

    def _buried_failure(self, exact_result_path: Path | None = None, *,
                        record: bool = True) -> dict[str, Any]:
        """Validate one host run or search prior runs; only E0 writes a case receipt."""
        if exact_result_path is not None:
            path = Path(exact_result_path).resolve(strict=True)
            if (path.name != "result.json" or not path.parent.name.startswith("buried-live-")
                    or path.parent.parent != self.root.resolve()):
                raise ValueError("buried observer result is outside the E0 run root")
            paths = [path]
        else:
            paths = sorted(
                self.root.glob("buried-live-*/result.json"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        def verdict(status: str, evidence: list[str], **extra: Any) -> dict[str, Any]:
            return (self._receipt("buried-failure", status, evidence, **extra)
                    if record else {"case": "buried-failure", "status": status,
                                    "evidence": evidence, **extra})
        for result_path in paths:
            result = json.loads(result_path.read_text())
            if result.get("candidate") != installed_candidate_fingerprint():
                continue
            if result.get("candidate_after") != result["candidate"]:
                continue
            rollout = Path(result.get("rollout", ""))
            repo = Path(result.get("repo", ""))
            if not rollout.is_file() or not (repo / "produce_log.py").is_file():
                continue
            if not (result.get("mavis_exit") == 0 and result.get("iris_loaded") is True
                    and result.get("mavis_loaded") is False):
                continue
            events = [json.loads(line) for line in rollout.read_text().splitlines() if line.strip()]
            if not events or events[0].get("payload", {}).get("cwd") != str(repo):
                continue
            exact_executions = 0
            exact_call_id = None
            unsafe_command = False
            seen_call_ids = set()
            for event in events:
                payload = event.get("payload", {})
                if (event.get("type") != "response_item"
                        or payload.get("type") != "function_call"
                        or payload.get("name") != "exec_command"):
                    continue
                call_id = payload.get("call_id")
                if not isinstance(call_id, str) or not call_id or call_id in seen_call_ids:
                    unsafe_command = True
                    continue
                seen_call_ids.add(call_id)
                try:
                    arguments = json.loads(payload.get("arguments", ""))
                except (TypeError, ValueError) as exc:
                    raise ValueError("E0 command arguments are malformed") from exc
                command = arguments.get("cmd") if isinstance(arguments, dict) else None
                if not isinstance(command, str):
                    raise ValueError("E0 command is missing")
                if command.strip() == "python3 produce_log.py":
                    exact_executions += 1
                    exact_call_id = call_id
                    workdir = arguments.get("workdir")
                    if workdir is not None and (not isinstance(workdir, str) or Path(workdir).resolve() != repo.resolve()):
                        unsafe_command = True
                elif not _safe_buried_inspection(command):
                    unsafe_command = True
            if exact_executions != 1 or unsafe_command:
                continue
            outputs = [event.get("payload", {}).get("output", "") for event in events
                       if event.get("type") == "response_item"
                       and event.get("payload", {}).get("type") == "function_call_output"
                       and event.get("payload", {}).get("call_id") == exact_call_id]
            primary = outputs[0] if len(outputs) == 1 else None
            if primary is not None and ("Complete raw output:" not in primary
                                        or "Process exited with code 1" not in primary):
                primary = None
            if primary is None:
                continue
            match = re.search(r"Complete raw output: (\S+\.raw) \((\d+) bytes; closed\)", primary)
            if match is None:
                continue
            raw = Path(match.group(1))
            if not _trusted_raw_output_path(raw, repo, self.home):
                continue
            content = raw.read_bytes()
            if len(content) != int(match.group(2)):
                continue
            marker_match = re.fullmatch(
                rb"A{800000}\r?\n(MAVIS_E0_FAILURE_[0-9a-f]{24})\r?\nZ{800000}\r?\n",
                content,
            )
            if marker_match is None:
                continue
            marker = marker_match.group(1).decode()
            final = [event.get("payload", {}).get("last_agent_message") for event in events
                     if event.get("type") == "event_msg"
                     and event.get("payload", {}).get("type") == "task_complete"]
            final_text = final[-1] if final else ""
            stated_failure = _states_failed_exit_one(final_text)
            if not final_text or marker not in final_text or not stated_failure:
                continue
            if marker in primary:
                continue
            return verdict("pass", [
                f"Installed Mavis command ran once and reported exit 1 in {rollout}",
                f"Complete {len(content)}-byte raw output at {raw} contained the buried marker; final answer matched",
                f"IRIS remained loaded and Mavis unloaded in {result_path}",
            ], installed_candidate=result["candidate"], raw_sha256=sha256_file(raw))
        return verdict("blocked", [
            "No candidate-matched installed harness run retained the full raw output and identified its buried failure."
        ])

    def _compaction_restart(self) -> dict[str, Any]:
        for result_path in sorted(
            self.root.glob("compaction-live-*/result.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ):
            result = json.loads(result_path.read_text())
            rollout = Path(result.get("rollout", ""))
            session = result.get("session_id")
            fact = result.get("fact")
            if not (rollout.is_file() and isinstance(session, str) and isinstance(fact, str)):
                continue
            if result.get("candidate") != installed_candidate_fingerprint():
                continue
            if result.get("candidate_after") != result["candidate"]:
                continue
            if not (result.get("first_exit") == result.get("resume_exit") == 0
                    and result.get("iris_loaded") is True
                    and result.get("mavis_loaded") is False
                    and result.get("exact_recovery") is True
                    and result.get("resumed_answer") == fact):
                continue
            events = [json.loads(line) for line in rollout.read_text().splitlines() if line.strip()]
            if not events or events[0].get("payload", {}).get("id") != session:
                continue
            if events[0].get("payload", {}).get("cwd") != result.get("workspace"):
                continue
            compact_at = next((i for i, event in enumerate(events) if event.get("type") == "compacted"), None)
            if compact_at is None:
                continue
            before, after = events[:compact_at], events[compact_at + 1:]
            def has_user(items: list[dict[str, Any]], expected: str) -> bool:
                return any(event.get("type") == "response_item"
                           and event.get("payload", {}).get("role") == "user"
                           and expected in json.dumps(event.get("payload", {}).get("content", []))
                           for event in items)
            def has_answer(items: list[dict[str, Any]], expected: str) -> bool:
                return any(event.get("type") == "event_msg"
                           and event.get("payload", {}).get("type") == "task_complete"
                           and event["payload"].get("last_agent_message") == expected
                           for event in items)
            if not (has_user(before, fact) and has_answer(before, "ACK")
                    and has_user(after, "What exact fact did I give before compaction?")
                    and has_answer(after, fact)):
                continue
            try:
                evidence_home = project_home(Path(result["workspace"]), create=False)
            except (KeyError, OSError, RuntimeError, ValueError):
                continue
            handoffs = [Path(path) for path in result.get("handoffs", [])]
            handoff_root = evidence_home / "transcripts" / session / "handoffs"
            if not handoffs or not all(path.is_file() and path.parent == handoff_root for path in handoffs):
                continue
            segments = []
            for path in handoffs:
                handoff = json.loads(path.read_text())
                segments.extend(Path(link) for link in handoff.get("evidence_links", []))
            segment_root = evidence_home / "transcripts" / session / "segments"
            if not any(path.is_file() and path.parent == segment_root
                       and fact in path.read_text() for path in segments):
                continue
            return self._receipt("compaction-restart", "pass", [
                f"Installed TUI rollout {rollout} compacted and resumed the same session",
                f"Handoff {handoffs[0]} linked an archive containing the fact; resumed answer matched",
                f"IRIS remained loaded and Mavis unloaded in {result_path}",
            ], installed_candidate=result["candidate"])
        return self._receipt("compaction-restart", "blocked", [
            "No candidate-matched installed TUI compaction, handoff, and exact-session resume passed inspection."
        ])

    def _repeated_no_progress(self) -> dict[str, Any]:
        fixture_home = self.home / "e0-fixtures" / "escalation"
        store = ObjectiveStore(fixture_home)
        objective_id = "e0-escalation"
        path = store._path(objective_id)
        if path.exists():
            path.unlink()
        store.create(
            {
                "schema_version": "mavis.objective/v1",
                "objective_id": objective_id,
                "blueprint": "E0 repeated failure fixture",
                "requirements": [{"id": "r1", "text": "repair"}],
                "dependencies": [],
                "scope": {},
                "acceptance_checks": [{"id": "c1", "command": ["python3", "-c", "print('1 passed')"]}],
                "unresolved_decisions": [],
                "state": "queued",
            }
        )
        store.record_attempt(objective_id, "same", ["first evidence"], "approach one")
        result = store.record_attempt(objective_id, "same", [], "approach one repeated")
        if result["state"] != "escalated":
            raise RuntimeError("repeated no-progress failure did not escalate")
        return self._receipt("repeated-no-progress", "pass", ["Second identical failure without new evidence escalated"])

    def _isolation_recovery(self) -> dict[str, Any]:
        proof_path = self.root / "isolation-recovery-live.json"
        if not proof_path.is_file():
            return self._receipt("isolation-recovery", "blocked", ["No host-observed Mavis service outage and recovery receipt exists"])
        proof = json.loads(proof_path.read_text())
        if proof.get("schema_version") != "mavis.e0-isolation-recovery/v1" or proof.get("candidate") != installed_candidate_fingerprint():
            return self._receipt("isolation-recovery", "blocked", ["The installed candidate changed since service recovery was observed"])
        runtime = omlx_runtime_fingerprint(self.config)
        if proof.get("omlx_runtime") != runtime:
            return self._receipt("isolation-recovery", "blocked", ["oMLX runtime or model metadata changed since service recovery was observed"])
        if proof.get("recovery_while_iris_loaded") is not True:
            return self._receipt("isolation-recovery", "blocked", ["Mavis recovery was not observed while IRIS kept its model"])
        iris_models = proof.get("iris_generation_models")
        if (not isinstance(iris_models, list) or not iris_models
                or not all(isinstance(model, str) and model for model in iris_models)
                or len(iris_models) != len(set(iris_models))
                or self.config.model not in iris_models):
            return self._receipt("isolation-recovery", "blocked", ["The IRIS generation model snapshot is missing or invalid"])
        before = proof.get("before") or {}
        after = proof.get("after") or {}
        iris_alive = endpoint_alive(self.config.iris_endpoint)
        mavis_alive = endpoint_alive(self.config.endpoint)
        iris_pids = _listener_pids(8000)
        mavis_pids = _listener_pids(8001)
        try:
            iris_process = (omlx_live_process_binding(self.config,
                            self.config.iris_endpoint, runtime) if iris_alive else None)
        except (OSError, RuntimeError, ValueError):
            iris_process = None
        recovered_process = proof.get("mavis_recovered_process") or {}
        park_pid = os.environ.get("MAVIS_E0_PARK_OWNER_PID")
        parked = (park_pid is not None and park_pid.isdecimal()
                  and mavis_pids == {int(park_pid)} and not mavis_alive)
        observed = (
            before.get("iris_pids") == after.get("iris_pids") == sorted(iris_pids)
            and before.get("iris_model_loaded") is True
            and after.get("iris_model_loaded") is True
            and before.get("mavis_pids") != after.get("mavis_pids")
            and after.get("mavis_pids")
            and proof.get("outage_observed") is True
            and proof.get("iris_process") == iris_process
            and recovered_process.get("pid") in after.get("mavis_pids", [])
            and recovered_process.get("package_sha256") == runtime["package_sha256"]
        )
        iris_loaded = (loaded_generation_models(self.config.iris_endpoint) ==
                       iris_models) if iris_alive else False
        status = "pass" if iris_alive and iris_loaded and parked and observed else "reject"
        return self._receipt(
            "isolation-recovery",
            status,
            [
                f"iris_alive={iris_alive}",
                f"iris_model_loaded={iris_loaded}",
                f"mavis_port_reserved={parked}",
                f"outage_and_new_service_observed={observed}",
            ],
            recovery_receipt=str(proof_path),
        )
