"""Frozen E1 cases and host-recorded paired checks; native workers remain external."""

from pathlib import Path
import shutil
import subprocess
from typing import Any

from .evidence import run_command
from .experiments import ExperimentStore, _digest, candidate_assignment_requirements
from .storage import read_json, require_safe_id, sha256_file, write_json


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=path, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


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
    output = Path(original.get("raw_output", {}).get("path", ""))
    if (
        original.get("schema_version") != "mavis.evidence-receipt/v1"
        or original.get("producer") != "mavis-host-command/v1"
        or original.get("verdict") != "fail"
        or original.get("changed_revision") != regression["revision"]
        or not original.get("acceptance_check_ids")
        or sha256_file(output / "stdout.log") + ":" + sha256_file(output / "stderr.log")
        != original["raw_output"]["sha256"]
    ):
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


class E1Runner:
    def __init__(self, home: Path):
        self.home = Path(home)
        self.store = ExperimentStore(home)
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
        checked_revision = _git(checkout, "rev-parse", "HEAD")
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
                for expected, check in zip(
                    case["checks"], result["checks"], strict=True
                ):
                    receipt_path = Path(check["receipt"])
                    receipt = read_json(receipt_path)
                    checkout = (
                        self._root(experiment_id) / "checkouts" / arm / case["id"]
                    )
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
                "evidence": {"path": str(evidence), "sha256": sha256_file(evidence)},
            }
        paired["candidate"]["candidate_job_id"] = candidate_job_id
        return self.store.compare(
            experiment_id,
            baseline_result=paired["baseline"],
            candidate_result=paired["candidate"],
        )
