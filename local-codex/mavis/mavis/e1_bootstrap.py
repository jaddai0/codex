"""Trial-only first-profile binding. This never activates a main profile."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .evaluations import E0_CASES, installed_candidate_fingerprint
from .gateway import harness_job_status
from .objectives import _validate_gateway_status
from .storage import read_json, sha256_file, write_json


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _package_manifest_path() -> Path:
    return Path.home() / ".local/share/local-codex/install-manifest.json"


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _checked_ref(value: Any, expected: Path, label: str) -> Path:
    if not isinstance(value, dict) or value.get("path") != str(expected.resolve()):
        raise ValueError(f"E1 bootstrap {label} path is not canonical")
    if not expected.is_file() or value.get("sha256") != sha256_file(expected):
        raise ValueError(f"E1 bootstrap {label} changed or is missing")
    return expected


def _summary(home: Path) -> tuple[Path, dict[str, Any]]:
    root = home / "evaluations/e0"
    path = root / "summary.json"
    summary = read_json(path)
    if (summary.get("schema_version") != "mavis.evaluation-suite/v1"
            or summary.get("suite") != "e0" or summary.get("status") != "pass"
            or summary.get("mandatory_cases") != list(E0_CASES)
            or summary.get("results") != [{"case": case, "status": "pass"} for case in E0_CASES]
            or summary.get("installed_candidate") != installed_candidate_fingerprint()
            or not isinstance(summary.get("model_id"), str) or not summary["model_id"]):
        raise ValueError("E1 bootstrap requires a current installed E0 all-pass summary")
    receipts = summary.get("case_receipts")
    if not isinstance(receipts, dict) or set(receipts) != set(E0_CASES):
        raise ValueError("E1 bootstrap E0 case receipts are incomplete")
    for case in E0_CASES:
        receipt_path = root / f"{case}.json"
        if receipts[case] != sha256_file(receipt_path):
            raise ValueError("E1 bootstrap E0 case receipt changed")
        receipt = read_json(receipt_path)
        if (receipt.get("schema_version") != "mavis.evaluation-case/v1"
                or receipt.get("suite") != "e0" or receipt.get("case") != case
                or receipt.get("status") != "pass"
                or not receipt.get("evidence")):
            raise ValueError("E1 bootstrap E0 case receipt is not passing evidence")
    return path, summary


def _model_identity(identity: Any, model_id: str) -> None:
    if not isinstance(identity, dict) or identity.get("model_id") != model_id:
        raise ValueError("E1 bootstrap model differs from installed E0")
    for key in ("architecture", "weights_fingerprint", "tokenizer_fingerprint",
                "chat_template_fingerprint", "quantization"):
        if not isinstance(identity.get(key), str) or not identity[key]:
            raise ValueError(f"E1 bootstrap model_identity.{key} is missing")


def _baseline(home: Path) -> tuple[Path, dict[str, Any]]:
    from .experiments import ExperimentStore

    store = ExperimentStore(home)
    active = store.active("main")
    if active.get("experiment_id") is not None:
        raise ValueError("E1 bootstrap requires an unpromoted seed baseline")
    ref = active["configuration"]
    snapshot = store._read_snapshot(ref)
    if (set(snapshot) != {"prompts", "tool_settings", "retrieval"}
            or not isinstance(snapshot["prompts"], dict)
            or set(snapshot["prompts"]) != {"system"}
            or not isinstance(snapshot["prompts"]["system"], str)
            or snapshot["tool_settings"] != {} or snapshot["retrieval"] != {}):
        raise ValueError("E1 bootstrap baseline has unsupported settings")
    return Path(ref["path"]), snapshot


def _review(home: Path, receipt: dict[str, Any], status_reader: Callable[[str], dict[str, Any]]) -> None:
    path = _checked_ref(receipt.get("independent_review"),
                        home / "verifications/e1-bootstrap/main.json", "independent review")
    review = read_json(path)
    if (review.get("schema_version") != "mavis.e1-bootstrap-review/v1"
            or review.get("verdict") != "accepted"
            or review.get("e0_summary_sha256") != receipt["e0_summary"]["sha256"]
            or review.get("baseline_sha256") != receipt["baseline"]["sha256"]
            or review.get("package_manifest_sha256") != receipt["package_manifest"]["sha256"]
            or review.get("installed_candidate_digest") != _digest(receipt["installed_candidate"])
            or review.get("model_identity_digest") != _digest(receipt["model_identity"])):
        raise ValueError("E1 bootstrap independent review does not match frozen evidence")
    job_id = review.get("gateway_worker_job_id")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("E1 bootstrap independent review has no gateway job")
    status = status_reader(job_id)
    _validate_gateway_status(status, job_id)
    acceptance = status["acceptance"]
    binding = status.get("mavis_binding")
    if (acceptance.get("verifier_job_id") != review.get("verifier_job_id")
            or not isinstance(binding, dict)
            or binding.get("objective_id") != "e1-bootstrap-main"
            or binding.get("report_sha256") != sha256_file(path)
            or status.get("accepted") is not True):
        raise ValueError("E1 bootstrap review lacks exact independent gateway acceptance")


def validate_bootstrap(
    home: Path, *, status_reader: Callable[[str], dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Recheck every retained first-profile dependency without activating it."""
    home = Path(home).resolve()
    if (home / "profiles/main/active.json").exists():
        raise ValueError("E1 bootstrap cannot override an accepted main profile")
    path = home / "e1/bootstrap/main.json"
    receipt = read_json(path)
    summary_path, summary = _summary(home)
    baseline_path, baseline = _baseline(home)
    package_path = _package_manifest_path()
    if (receipt.get("schema_version") != "mavis.e1-bootstrap/v1"
            or receipt.get("scope") != "main"
            or receipt.get("installed_candidate") != summary["installed_candidate"]):
        raise ValueError("E1 bootstrap receipt is invalid or stale")
    _checked_ref(receipt.get("e0_summary"), summary_path, "E0 summary")
    _checked_ref(receipt.get("baseline"), baseline_path, "baseline")
    _checked_ref(receipt.get("package_manifest"), package_path, "package manifest")
    _model_identity(receipt.get("model_identity"), summary["model_id"])
    package = read_json(package_path)
    if (package.get("schema_version") != "mavis.installed-core/v1"
            or package.get("core_sha256") != receipt["installed_candidate"]["core_sha256"]):
        raise ValueError("E1 bootstrap installed package differs from E0")
    _review(home, receipt, status_reader or harness_job_status)
    return {**receipt, "prompts": baseline["prompts"],
            "_source_path": str(path), "_source_sha256": sha256_file(path)}


def create_bootstrap(
    home: Path, model_identity: dict[str, Any], review_path: Path,
    *, status_reader: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Record a one-time trial binding after a separate gateway review exists."""
    home = Path(home).resolve()
    path = home / "e1/bootstrap/main.json"
    if path.exists() or (home / "profiles/main/active.json").exists():
        raise FileExistsError("E1 bootstrap already exists or main profile is active")
    summary_path, summary = _summary(home)
    baseline_path, _ = _baseline(home)
    package_path = _package_manifest_path()
    _model_identity(model_identity, summary["model_id"])
    if Path(review_path).resolve() != (home / "verifications/e1-bootstrap/main.json").resolve():
        raise ValueError("E1 bootstrap review must be retained at its canonical path")
    receipt = {"schema_version": "mavis.e1-bootstrap/v1", "scope": "main",
               "e0_summary": _ref(summary_path), "installed_candidate": summary["installed_candidate"],
               "model_identity": model_identity, "baseline": _ref(baseline_path),
               "package_manifest": _ref(package_path), "independent_review": _ref(Path(review_path)),
               "created_at": datetime.now(timezone.utc).isoformat()}
    _review(home, receipt, status_reader or harness_job_status)
    write_json(path, receipt)
    return validate_bootstrap(home, status_reader=status_reader)
