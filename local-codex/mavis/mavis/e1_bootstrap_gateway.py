"""Drive the gateway's real completion and independent GLM verifier acceptance chain."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .e1_bootstrap import _assignment, _dispatch, check_bootstrap_review, import_bootstrap_review_report
from .gateway import (harness_job_complete, harness_job_status, harness_job_verify,
                      harness_verifier_start)
from .objectives import _validate_gateway_status
from .storage import read_json, sha256_file, write_json


def _terminal_status(status: dict[str, Any], job_id: str) -> None:
    receipt = status.get("receipt") if isinstance(status, dict) else None
    if (not isinstance(status, dict) or status.get("job_id") != job_id
            or status.get("state") != "completed" or status.get("exit_code") != 0
            or not isinstance(receipt, dict) or receipt.get("job_id") != job_id
            or receipt.get("exit_code") != 0 or receipt.get("cancelled")):
        raise ValueError(f"E1 bootstrap job {job_id} lacks a successful terminal receipt")


def complete_bootstrap_review(
    home: Path, *, status_reader: Callable[[str], dict[str, Any]] | None = None,
    completer: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Record a worker completion with its deterministic host check output."""
    home = Path(home).resolve()
    assignment_path, assignment, _ = _assignment(home)
    dispatch = _dispatch(home, assignment_path, assignment)
    job_id = dispatch["job_id"]
    status = (status_reader or harness_job_status)(job_id)
    _terminal_status(status, job_id)
    if status.get("accepted") is True:
        raise ValueError("E1 bootstrap reviewer is already accepted")
    report = Path(dispatch["report_path"])
    checked = check_bootstrap_review(home, report)
    output = Path(dispatch["job_dir"]) / "independent-bootstrap-review.json"
    if output.exists():
        if read_json(output) != checked:
            raise ValueError("E1 bootstrap host check output changed")
    else:
        write_json(output, checked)
    checks = {"independent-bootstrap-review": {
        "exit_code": 0, "output_path": str(output), "output_sha256": sha256_file(output),
    }}
    result = (completer or harness_job_complete)(job_id, checks)
    completion = result.get("completion") if isinstance(result, dict) else None
    if (not isinstance(result, dict) or result.get("success") is not True
            or result.get("recorded") is not True or result.get("accepted") is not False
            or result.get("job_id") != job_id or not isinstance(completion, dict)
            or completion.get("job_id") != job_id or completion.get("receipt_exit_code") != 0
            or completion.get("assignment_sha256") != dispatch["gateway_assignment_sha256"]
            or completion.get("report_sha256") != sha256_file(report)
            or completion.get("changed_revision") != assignment["starting_revision"]
            or completion.get("check_results") != checks):
        raise ValueError("E1 bootstrap gateway completion did not bind the worker check")
    record = {"schema_version": "mavis.e1-bootstrap-completion/v1",
              "dispatch_sha256": sha256_file(home / "e1/bootstrap/review-dispatch.json"),
              "job_id": job_id, "report_sha256": sha256_file(report),
              "check_output_sha256": sha256_file(output), "gateway_completion": completion}
    write_json(home / "e1/bootstrap/review-completion.json", record)
    return record


def start_bootstrap_verifier(
    home: Path, *, starter: Callable[[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Start a separate GLM verifier job only after the worker check is recorded."""
    home = Path(home).resolve()
    assignment_path, assignment, _ = _assignment(home)
    dispatch = _dispatch(home, assignment_path, assignment)
    completion_path = home / "e1/bootstrap/review-completion.json"
    completion = read_json(completion_path)
    check_path = Path(dispatch["job_dir"]) / "independent-bootstrap-review.json"
    if (completion.get("schema_version") != "mavis.e1-bootstrap-completion/v1"
            or completion.get("job_id") != dispatch["job_id"]
            or completion.get("dispatch_sha256") != sha256_file(home / "e1/bootstrap/review-dispatch.json")
            or completion.get("report_sha256") != sha256_file(Path(dispatch["report_path"]))
            or completion.get("check_output_sha256") != sha256_file(check_path)
            or completion.get("gateway_completion", {}).get("check_results", {}).get(
                "independent-bootstrap-review", {}).get("output_sha256") != sha256_file(check_path)):
        raise ValueError("E1 bootstrap gateway completion changed")
    path = home / "e1/bootstrap/review-verifier-dispatch.json"
    if path.exists():
        raise FileExistsError("E1 bootstrap GLM verifier was already started")
    result = (starter or harness_verifier_start)(dispatch["job_id"], dispatch["verifier_job_id"])
    if (not isinstance(result, dict) or result.get("success") is not True
            or result.get("started") is not True or result.get("accepted") is not False
            or result.get("job_id") != dispatch["verifier_job_id"]):
        raise ValueError("E1 bootstrap gateway did not start the exact GLM verifier")
    record = {"schema_version": "mavis.e1-bootstrap-verifier-dispatch/v1",
              "job_id": dispatch["job_id"], "verifier_job_id": dispatch["verifier_job_id"],
              "completion_sha256": sha256_file(completion_path)}
    write_json(path, record)
    return record


def verify_and_import_bootstrap_review(
    home: Path, *, status_reader: Callable[[str], dict[str, Any]] | None = None,
    verifier: Callable[[str, str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Require the GLM verifier's terminal verdict and accepted gateway status before import."""
    home = Path(home).resolve()
    assignment_path, assignment, _ = _assignment(home)
    dispatch = _dispatch(home, assignment_path, assignment)
    verifier_dispatch = read_json(home / "e1/bootstrap/review-verifier-dispatch.json")
    if (verifier_dispatch.get("schema_version") != "mavis.e1-bootstrap-verifier-dispatch/v1"
            or verifier_dispatch.get("job_id") != dispatch["job_id"]
            or verifier_dispatch.get("verifier_job_id") != dispatch["verifier_job_id"]
            or verifier_dispatch.get("completion_sha256") != sha256_file(
                home / "e1/bootstrap/review-completion.json")):
        raise ValueError("E1 bootstrap GLM verifier dispatch changed")
    read_status = status_reader or harness_job_status
    _terminal_status(read_status(dispatch["verifier_job_id"]), dispatch["verifier_job_id"])
    report_sha = sha256_file(Path(dispatch["report_path"]))
    result = (verifier or harness_job_verify)(
        dispatch["job_id"], dispatch["verifier_job_id"], report_sha)
    acceptance = result.get("acceptance") if isinstance(result, dict) else None
    if (not isinstance(result, dict) or result.get("success") is not True
            or result.get("recorded") is not True or result.get("job_id") != dispatch["job_id"]
            or not isinstance(acceptance, dict) or acceptance.get("accepted") is not True
            or acceptance.get("verifier_job_id") != dispatch["verifier_job_id"]
            or acceptance.get("evidence_sha256") != report_sha):
        raise ValueError("E1 bootstrap GLM verifier verdict was not accepted by the gateway")
    _validate_gateway_status(read_status(dispatch["job_id"]), dispatch["job_id"])
    path = import_bootstrap_review_report(home, status_reader=read_status)
    return {"review": str(path), "review_sha256": sha256_file(path),
            "verifier_job_id": dispatch["verifier_job_id"]}
