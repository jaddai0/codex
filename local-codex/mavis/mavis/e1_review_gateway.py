"""Complete paired E1 review through a real worker receipt and separate Terra job."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .e1_review import _paths, check_review_report, import_review_report
from .gateway import (harness_job_complete, harness_job_status, harness_job_verify,
                      harness_verifier_start)
from .objectives import _validate_gateway_status
from .storage import read_json, sha256_file, write_json


def _dispatch(home: Path, record: dict[str, Any]) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    assignment_path, _ = _paths(home, record["experiment_id"])
    assignment = read_json(assignment_path)
    dispatch_path = assignment_path.parent / "review-dispatch.json"
    dispatch = read_json(dispatch_path)
    if (dispatch.get("experiment_id") != record["experiment_id"]
            or dispatch.get("packet_sha256") != sha256_file(assignment_path)
            or dispatch.get("verifier_job_id") != f"{dispatch.get('job_id')}-terra"):
        raise ValueError("E1 review dispatch changed")
    return dispatch_path, dispatch, assignment


def _terminal_status(status: dict[str, Any], job_id: str) -> None:
    receipt = status.get("receipt") if isinstance(status, dict) else None
    if (not isinstance(status, dict) or status.get("job_id") != job_id
            or status.get("state") != "completed" or status.get("exit_code") != 0
            or not isinstance(receipt, dict) or receipt.get("job_id") != job_id
            or receipt.get("exit_code") != 0 or receipt.get("cancelled")):
        raise ValueError(f"E1 review job {job_id} lacks a successful terminal receipt")


def complete_review(
    home: Path, record: dict[str, Any], *,
    status_reader: Callable[[str], dict[str, Any]] | None = None,
    completer: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the required host check, then record the gateway worker completion."""
    home = Path(home).resolve()
    dispatch_path, dispatch, assignment = _dispatch(home, record)
    job_id = dispatch["job_id"]
    status = (status_reader or harness_job_status)(job_id)
    _terminal_status(status, job_id)
    if status.get("accepted") is True:
        raise ValueError("E1 reviewer is already accepted")
    report = Path(dispatch["report_path"])
    checked = check_review_report(home, record, report)
    output = Path(dispatch["job_dir"]) / "independent-e1-review.json"
    if output.exists():
        if read_json(output) != checked:
            raise ValueError("E1 review host check output changed")
    else:
        write_json(output, checked)
    checks = {"independent-e1-review": {
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
        raise ValueError("E1 gateway completion did not bind the paired review check")
    result_path = dispatch_path.parent / "review-completion.json"
    saved = {"schema_version": "mavis.e1-review-completion/v1",
             "dispatch_sha256": sha256_file(dispatch_path), "job_id": job_id,
             "report_sha256": sha256_file(report),
             "check_output_sha256": sha256_file(output), "gateway_completion": completion}
    write_json(result_path, saved)
    return saved


def start_review_verifier(
    home: Path, record: dict[str, Any], *,
    starter: Callable[[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Start Terra only after the named host check is bound to completion."""
    home = Path(home).resolve()
    dispatch_path, dispatch, _ = _dispatch(home, record)
    completion_path = dispatch_path.parent / "review-completion.json"
    completion = read_json(completion_path)
    check_path = Path(dispatch["job_dir"]) / "independent-e1-review.json"
    check_review_report(home, record, Path(dispatch["report_path"]))
    gateway_completion = completion.get("gateway_completion")
    checks = gateway_completion.get("check_results") if isinstance(gateway_completion, dict) else None
    named_check = checks.get("independent-e1-review") if isinstance(checks, dict) else None
    if (completion.get("schema_version") != "mavis.e1-review-completion/v1"
            or completion.get("job_id") != dispatch["job_id"]
            or completion.get("dispatch_sha256") != sha256_file(dispatch_path)
            or completion.get("report_sha256") != sha256_file(Path(dispatch["report_path"]))
            or completion.get("check_output_sha256") != sha256_file(check_path)
            or not isinstance(named_check, dict)
            or named_check.get("exit_code") != 0
            or named_check.get("output_path") != str(check_path)
            or named_check.get("output_sha256") != sha256_file(check_path)):
        raise ValueError("E1 review gateway completion changed")
    path = dispatch_path.parent / "review-verifier-dispatch.json"
    if path.exists():
        raise FileExistsError("E1 Terra verifier was already started")
    result = (starter or harness_verifier_start)(dispatch["job_id"], dispatch["verifier_job_id"])
    if (not isinstance(result, dict) or result.get("success") is not True
            or result.get("started") is not True or result.get("accepted") is not False
            or result.get("job_id") != dispatch["verifier_job_id"]):
        raise ValueError("E1 gateway did not start the exact Terra verifier")
    saved = {"schema_version": "mavis.e1-review-verifier-dispatch/v1",
             "job_id": dispatch["job_id"], "verifier_job_id": dispatch["verifier_job_id"],
             "completion_sha256": sha256_file(completion_path)}
    write_json(path, saved)
    return saved


def verify_and_import_review(
    home: Path, record: dict[str, Any], *,
    status_reader: Callable[[str], dict[str, Any]] | None = None,
    verifier: Callable[[str, str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Require Terra's accepted gateway verdict before importing report bytes."""
    home = Path(home).resolve()
    dispatch_path, dispatch, _ = _dispatch(home, record)
    verifier_dispatch = read_json(dispatch_path.parent / "review-verifier-dispatch.json")
    if (verifier_dispatch.get("schema_version") != "mavis.e1-review-verifier-dispatch/v1"
            or verifier_dispatch.get("job_id") != dispatch["job_id"]
            or verifier_dispatch.get("verifier_job_id") != dispatch["verifier_job_id"]
            or verifier_dispatch.get("completion_sha256") != sha256_file(
                dispatch_path.parent / "review-completion.json")):
        raise ValueError("E1 Terra dispatch changed")
    checked = check_review_report(home, record, Path(dispatch["report_path"]))
    check_path = Path(dispatch["job_dir"]) / "independent-e1-review.json"
    if read_json(check_path) != checked:
        raise ValueError("E1 paired review host check changed")
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
        raise ValueError("E1 Terra verdict was not accepted by the gateway")
    _validate_gateway_status(read_status(dispatch["job_id"]), dispatch["job_id"])
    path = import_review_report(home, record, read_status)
    return {"review": str(path), "review_sha256": sha256_file(path),
            "verifier_job_id": dispatch["verifier_job_id"]}
