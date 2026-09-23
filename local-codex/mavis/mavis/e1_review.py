"""Independent review binding for paired, installed E1 trials."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Callable

from .e1 import validate_e1_bundle
from .experiments import _digest
from .objectives import _validate_gateway_status
from .storage import read_json, require_safe_id, sha256_file, write_json


def _paths(home: Path, experiment_id: str) -> tuple[Path, Path]:
    root = Path(home).resolve() / "e1" / require_safe_id(experiment_id, "experiment id")
    receipt = Path(home).resolve() / "verifications" / "experiments" / f"{experiment_id}.json"
    return root / "review-assignment.json", receipt


def _requirements(facts: dict[str, Any]) -> list[str]:
    return [
        f"experiment-comparison:{facts['comparison_digest']}",
        f"e1-installed-bundle:{facts['bundle_digest']}",
        f"e1-model-identity:{_digest(facts['model_identity'])}",
        f"e1-prompts:{_digest(facts['prompts'])}",
        f"e1-held-out:{_digest({key: facts[key] for key in ('held_out', 'minimum_gain', 'summaries')})}",
    ]


def _candidate_workers(root: Path, record: dict[str, Any], case_ids: list[str]) -> list[dict[str, Any]]:
    """Replay native candidate dispatch ownership when a worker authored a case repair."""
    workers = []
    dispatch_root = root / "dispatch"
    for path in sorted(dispatch_root.glob("*.json")):
        dispatch = read_json(path)
        assignment_path = Path(dispatch.get("assignment_path") or "")
        if (path.stem not in case_ids
                or dispatch.get("schema_version") != "mavis.e1-native-dispatch/v1"
                or dispatch.get("experiment_id") != record["experiment_id"]
                or dispatch.get("case_id") != path.stem
                or dispatch.get("candidate_sha256") != record["candidate"]["sha256"]
                or dispatch.get("assignment_sha256") != sha256_file(assignment_path)):
            raise ValueError("E1 candidate worker provenance changed")
        assignment = read_json(assignment_path)
        owner = assignment.get("mavis_owner")
        if (not isinstance(owner, dict) or set(owner) != {"provider", "model", "harness"}
                or assignment.get("mavis_objective_id") != record["experiment_id"]
                or f"experiment-candidate-snapshot:{record['candidate']['sha256']}"
                not in assignment.get("mavis_requirements", [])
                or assignment.get("cwd") != dispatch.get("checkout")):
            raise ValueError("E1 candidate worker assignment changed")
        workers.append({"case_id": path.stem, "job_id": dispatch["job_id"], "owner": owner,
                        "dispatch_sha256": sha256_file(path)})
    return workers


def _gateway_arguments(packet: dict[str, Any], assignment_path: Path, job_id: str) -> dict[str, Any]:
    require_safe_id(job_id, "review job id")
    owner = packet["owner"]
    lane = {("zai", "zcode"): "zcode", ("minimax", "opencode"): "minimax"}.get(
        (owner["provider"], owner["harness"])
    )
    if lane is None:
        raise ValueError("E1 review owner has no independent native harness")
    return {
        "job_id": job_id,
        "task": ("Independently inspect the paired installed E1 trial evidence in the review packet. "
                 "Reject missing, changed, or unconvincing checks. Return a JSON object with schema_version "
                 "mavis.e1-review/v1, experiment_id, verdict, assignment_sha256, facts_digest, "
                 "gateway_worker_job_id, and case_findings. For every case ID, case_findings must "
                 "contain a specific nonempty finding and the exact evidence hashes from the packet's "
                 "case_evidence entry. Inspect the case results, trial receipts, and raw check logs. "
                 "Use the facts_digest in the packet and assignment hash in the context packet. "
                 "Return raw JSON without code fences. The host will import the exact report bytes."),
        "lane": lane, "model": owner["model"], "cwd": packet["cwd"],
        "starting_revision": packet["starting_revision"],
        "owned_paths": packet["owned_paths"],
        "allowed_effects": ["review evidence and report verdict"],
        "required_checks": packet["required_checks"],
        "context_packet": f"Review packet: {assignment_path} (sha256 {sha256_file(assignment_path)}).",
        "mavis_objective_id": packet["objective_id"],
        "mavis_requirements": packet["requirements"], "mavis_owner": owner,
    }


def _facts(home: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Replay the complete installed trial bundle before stating review facts."""
    comparison = record.get("comparison") or {}
    if not all(comparison.get(arm, {}).get("e1_native_trial") is True for arm in ("baseline", "candidate")):
        raise ValueError("E1 review requires two installed native trial arms")
    bundle = validate_e1_bundle(home, record, require_trials=True)
    if any(comparison[arm].get("e1_bundle_digest") != bundle for arm in ("baseline", "candidate")):
        raise ValueError("E1 review trial bundle changed")
    root = Path(home).resolve() / "e1" / record["experiment_id"]
    manifest = read_json(root / "cases.json")
    cases = manifest["cases"]
    trials = {
        arm: {case["id"]: read_json(root / "trials" / arm / f"{case['id']}.json") for case in cases}
        for arm in ("baseline", "candidate")
    }
    first = trials["baseline"][cases[0]["id"]]
    identity_keys = ("selected_model", "model_provider", "core_sha256", "package_manifest_sha256",
                     "profile_source", "accepted_profile_sha256", "bootstrap_receipt_sha256", "model_endpoint")
    identity = {key: first.get(key) for key in identity_keys}
    for case in cases:
        baseline = trials["baseline"][case["id"]]
        candidate = trials["candidate"][case["id"]]
        if any({key: trial.get(key) for key in identity_keys} != identity for trial in (baseline, candidate)):
            raise ValueError("E1 review trials do not share one installed model and package")
        if baseline["instructions_sha256"] == candidate["instructions_sha256"]:
            raise ValueError("E1 review trial prompts were identical")
    if first["profile_source"] == "e0-bootstrap":
        from .e1_bootstrap import validate_bootstrap

        bootstrap = validate_bootstrap(home)
        identity["model_identity"] = bootstrap["model_identity"]
        identity["installed_candidate"] = bootstrap["installed_candidate"]
    else:
        identity["model_identity"] = read_json(Path(first["accepted_profile_path"]))["model_identity"]
        identity["installed_candidate"] = {
            "core_sha256": first["core_sha256"],
            "package_manifest_sha256": first["package_manifest_sha256"],
        }
    prompts = {arm: {case["id"]: trials[arm][case["id"]]["instructions_sha256"] for case in cases}
               for arm in ("baseline", "candidate")}
    case_evidence = {case["id"]: {arm: {
        "result_sha256": sha256_file(root / "results" / arm / f"{case['id']}.json"),
        "trial_sha256": sha256_file(root / "trials" / arm / f"{case['id']}.json"),
    } for arm in ("baseline", "candidate")} for case in cases}
    candidate_workers = _candidate_workers(root, record, [case["id"] for case in cases])
    summaries = {arm: {
        "path": comparison[arm]["evidence"]["path"],
        "sha256": comparison[arm]["evidence"]["sha256"],
        "held_out_score": comparison[arm]["target_score"],
    } for arm in ("baseline", "candidate")}
    return {
        "comparison_digest": comparison["comparison_digest"],
        "bundle_digest": bundle,
        "manifest_sha256": record["workload"]["manifest_sha256"],
        "baseline_snapshot_sha256": record["baseline"]["sha256"],
        "candidate_snapshot_sha256": record["candidate"]["sha256"],
        "model_identity": identity,
        "prompts": prompts,
        "case_evidence": case_evidence,
        "candidate_workers": candidate_workers,
        "case_ids": [case["id"] for case in cases],
        "held_out": manifest["held_out"],
        "minimum_gain": manifest["minimum_gain"],
        "summaries": summaries,
        "review_checkout_revision": read_json(root / "results" / "baseline" / f"{manifest['regression']}.json")["checked_revision"],
    }


def prepare_review(home: Path, record: dict[str, Any], owner: dict[str, str]) -> dict[str, Any]:
    """Freeze the comparison and a separate reviewer's exact assignment."""
    if record.get("state") != "compared":
        raise ValueError("E1 review needs a completed comparison")
    if not isinstance(owner, dict) or set(owner) != {"provider", "model", "harness"} or any(
        not isinstance(value, str) or not value.strip() for value in owner.values()
    ):
        raise ValueError("E1 review needs one exact independent owner")
    if (owner["provider"], owner["harness"]) not in {("zai", "zcode"), ("minimax", "opencode")}:
        raise ValueError("E1 review owner needs an independent native harness")
    assignment_path, receipt_path = _paths(home, record["experiment_id"])
    if assignment_path.exists():
        raise FileExistsError("E1 review assignment already exists")
    facts = _facts(home, record)
    if owner in [worker["owner"] for worker in facts["candidate_workers"]]:
        raise ValueError("E1 reviewer must be independent from candidate workers")
    requirements = _requirements(facts)
    review_checkout = assignment_path.parent / "checkouts" / "baseline" / read_json(assignment_path.parent / "cases.json")["regression"]
    assignment = {
        "schema_version": "mavis.e1-review-assignment/v1",
        "objective_id": record["experiment_id"], "scope": record["scope"],
        "cwd": str(review_checkout.resolve()), "owner": owner,
        "owned_paths": [str(receipt_path)],
        "required_checks": ["independent-e1-review"],
        "starting_revision": facts["review_checkout_revision"],
        "requirements": requirements, "facts": facts, "facts_digest": _digest(facts),
    }
    write_json(assignment_path, assignment)
    return {**assignment, "assignment_path": str(assignment_path),
            "assignment_sha256": sha256_file(assignment_path), "review_receipt_path": str(receipt_path)}


def dispatch_review(home: Path, record: dict[str, Any], *, job_id: str,
                    starter: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    """Start an independent native reviewer against the frozen trial packet."""
    require_safe_id(job_id, "review job id")
    assignment_path, receipt_path = _paths(home, record["experiment_id"])
    packet = read_json(assignment_path)
    facts = _facts(home, record)
    owner = packet.get("owner")
    if (record.get("state") != "compared"
            or not isinstance(owner, dict) or set(owner) != {"provider", "model", "harness"}
            or packet.get("facts") != facts or packet.get("requirements") != _requirements(facts)
            or packet.get("facts_digest") != _digest(facts)
            or packet.get("objective_id") != record["experiment_id"]
            or packet.get("cwd") != str(assignment_path.parent / "checkouts" / "baseline" / read_json(assignment_path.parent / "cases.json")["regression"])
            or packet.get("starting_revision") != facts["review_checkout_revision"]
            or packet.get("owned_paths") != [str(receipt_path)]
            or packet.get("required_checks") != ["independent-e1-review"]):
        raise ValueError("E1 reviewer dispatch lost its exact frozen assignment")
    dispatch_path = assignment_path.parent / "review-dispatch.json"
    if dispatch_path.exists() or receipt_path.exists():
        raise FileExistsError("E1 review was already dispatched or recorded")
    arguments = _gateway_arguments(packet, assignment_path, job_id)
    started = starter(arguments)
    report = Path(started.get("report_path", "")) if isinstance(started, dict) else Path("")
    job_dir = Path(started.get("job_dir", "")) if isinstance(started, dict) else Path("")
    gateway_assignment = Path(started.get("assignment_path", "")) if isinstance(started, dict) else Path("")
    if (not isinstance(started, dict) or started.get("success") is not True
            or started.get("started") is not True or started.get("accepted") is not False
            or started.get("job_id") != job_id or not job_dir.is_absolute()
            or not report.is_absolute() or report.parent.resolve() != job_dir.resolve()
            or not gateway_assignment.is_absolute() or gateway_assignment.parent.resolve() != job_dir.resolve()
            or not gateway_assignment.is_file()):
        raise ValueError("native gateway did not start the exact E1 review job")
    saved = read_json(gateway_assignment)
    if any(saved.get(key) != value for key, value in arguments.items() if key != "job_id"):
        raise ValueError("native gateway reviewer assignment differs from frozen trial packet")
    dispatch = {
        "schema_version": "mavis.e1-review-dispatch/v1",
        "experiment_id": record["experiment_id"], "job_id": job_id,
        "packet_path": str(assignment_path), "packet_sha256": sha256_file(assignment_path),
        "arguments_digest": _digest(arguments),
        "gateway_assignment_path": str(gateway_assignment),
        "gateway_assignment_sha256": sha256_file(gateway_assignment),
        "report_path": str(report), "job_dir": str(job_dir),
    }
    write_json(dispatch_path, dispatch)
    return dispatch


def import_review_report(home: Path, record: dict[str, Any],
                         status_reader: Callable[[str], dict[str, Any]]) -> Path:
    """Keep the gateway's exact JSON report in the separate verification store."""
    assignment_path, receipt_path = _paths(home, record["experiment_id"])
    dispatch = read_json(assignment_path.parent / "review-dispatch.json")
    report = Path(dispatch["report_path"])
    status = status_reader(dispatch["job_id"])
    _validate_gateway_status(status, dispatch["job_id"])
    binding = status.get("mavis_binding")
    if (not isinstance(binding, dict)
            or binding.get("report_sha256") != sha256_file(report)
            or status["acceptance"].get("report_sha256_on_disk") != sha256_file(report)):
        raise ValueError("E1 reviewer report is not the accepted gateway report")
    receipt = read_json(report)
    if receipt.get("gateway_worker_job_id") != dispatch["job_id"]:
        raise ValueError("E1 reviewer report names the wrong gateway job")
    if receipt_path.exists():
        if sha256_file(receipt_path) != sha256_file(report):
            raise FileExistsError("E1 review receipt differs from the accepted gateway report")
        validate_review(home, record, receipt_path, status_reader)
        return receipt_path
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(report, receipt_path)
        validate_review(home, record, receipt_path, status_reader)
    except BaseException:
        receipt_path.unlink(missing_ok=True)
        raise
    return receipt_path


def validate_review(home: Path, record: dict[str, Any], receipt_path: Path,
                    status_reader: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    """Require a different accepted gateway worker and Terra verifier for exact facts."""
    assignment_path, canonical_receipt = _paths(home, record["experiment_id"])
    if Path(receipt_path).resolve() != canonical_receipt.resolve():
        raise ValueError("E1 review receipt must be in the separate verification store")
    assignment = read_json(assignment_path)
    dispatch = read_json(assignment_path.parent / "review-dispatch.json")
    facts = _facts(home, record)
    expected_requirements = _requirements(facts)
    review_checkout = assignment_path.parent / "checkouts" / "baseline" / read_json(assignment_path.parent / "cases.json")["regression"]
    if (assignment.get("schema_version") != "mavis.e1-review-assignment/v1"
            or assignment.get("objective_id") != record["experiment_id"]
            or assignment.get("scope") != record["scope"]
            or assignment.get("cwd") != str(review_checkout.resolve())
            or assignment.get("owned_paths") != [str(canonical_receipt)]
            or assignment.get("required_checks") != ["independent-e1-review"]
            or assignment.get("starting_revision") != facts["review_checkout_revision"]
            or assignment.get("requirements") != expected_requirements
            or assignment.get("facts") != facts
            or assignment.get("facts_digest") != _digest(facts)):
        raise ValueError("E1 review assignment differs from installed trial evidence")
    owner = assignment.get("owner")
    if not isinstance(owner, dict) or set(owner) != {"provider", "model", "harness"} or any(
        not isinstance(value, str) or not value.strip() for value in owner.values()
    ):
        raise ValueError("E1 review assignment has no exact owner")
    if owner in [worker["owner"] for worker in facts["candidate_workers"]]:
        raise ValueError("E1 reviewer matches a candidate worker")
    gateway_assignment_path = Path(dispatch.get("gateway_assignment_path") or "")
    gateway_assignment = read_json(gateway_assignment_path)
    expected_gateway = _gateway_arguments(assignment, assignment_path, dispatch.get("job_id"))
    if (dispatch.get("schema_version") != "mavis.e1-review-dispatch/v1"
            or dispatch.get("experiment_id") != record["experiment_id"]
            or dispatch.get("packet_path") != str(assignment_path)
            or dispatch.get("packet_sha256") != sha256_file(assignment_path)
            or dispatch.get("gateway_assignment_sha256") != sha256_file(gateway_assignment_path)
            or dispatch.get("arguments_digest") != _digest(expected_gateway)
            or any(gateway_assignment.get(key) != value for key, value in expected_gateway.items() if key != "job_id")
            or Path(dispatch.get("report_path") or "").parent.resolve() != Path(dispatch.get("job_dir") or "").resolve()
            or gateway_assignment_path.parent.resolve() != Path(dispatch.get("job_dir") or "").resolve()):
        raise ValueError("E1 review gateway dispatch differs from frozen assignment")
    receipt = read_json(canonical_receipt)
    if sha256_file(Path(dispatch["report_path"])) != sha256_file(canonical_receipt):
        raise ValueError("E1 review receipt differs from gateway report bytes")
    if (receipt.get("schema_version") != "mavis.e1-review/v1"
            or receipt.get("experiment_id") != record["experiment_id"]
            or receipt.get("verdict") not in {"accepted", "rejected"}
            or receipt.get("assignment_sha256") != sha256_file(assignment_path)
            or receipt.get("facts_digest") != assignment["facts_digest"]):
        raise ValueError("E1 review receipt does not match frozen evidence")
    findings = receipt.get("case_findings")
    if (not isinstance(findings, dict) or set(findings) != set(facts["case_ids"])):
        raise ValueError("E1 review lacks case-specific findings")
    for case_id, finding in findings.items():
        if (not isinstance(finding, dict) or set(finding) != {"finding", "evidence"}
                or not isinstance(finding["finding"], str)
                or not finding["finding"].strip()
                or len(finding["finding"]) > 4000
                or finding["evidence"] != facts["case_evidence"][case_id]):
            raise ValueError("E1 review finding lacks exact case evidence")
    worker_job_id = require_safe_id(receipt.get("gateway_worker_job_id"), "review worker job id")
    if worker_job_id != dispatch.get("job_id"):
        raise ValueError("E1 review worker differs from dispatched gateway job")
    status = status_reader(worker_job_id)
    _validate_gateway_status(status, worker_job_id)
    binding = status.get("mavis_binding")
    acceptance = status["acceptance"]
    verifier_job_id = acceptance["verifier_job_id"]
    if len({worker_job_id, verifier_job_id, record["comparison"]["candidate"]["candidate_job_id"]}) != 3:
        raise ValueError("E1 review worker and verifier must be independent")
    expected_binding = {
        "schema_version": "model-gateway-mavis-objective-binding/v1",
        "objective_id": assignment["objective_id"],
        "cwd": assignment["cwd"], "owner": owner,
        "starting_revision": assignment["starting_revision"],
        "changed_revision": assignment["starting_revision"],
        "owned_paths": assignment["owned_paths"],
        "requirements": assignment["requirements"],
        "required_checks": assignment["required_checks"],
        "assignment_sha256": sha256_file(gateway_assignment_path),
        "report_sha256": sha256_file(canonical_receipt),
    }
    if (not isinstance(binding, dict)
            or any(binding.get(key) != value for key, value in expected_binding.items())
            or acceptance.get("target_sha256") != binding.get("target_sha256")
            or acceptance.get("report_sha256_on_disk") != sha256_file(canonical_receipt)):
        raise ValueError("E1 review lacks exact independent gateway acceptance")
    return {"receipt": receipt, "status": status, "assignment_sha256": sha256_file(assignment_path)}
