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
from .runtime import inventory
from .storage import read_json, sha256_file, write_json


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _package_manifest_path() -> Path:
    return Path.home() / ".local/share/local-codex/install-manifest.json"


def _model_root_path() -> Path:
    return Path("/Users/dustinpainter/models/vlms")


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


def _model_artifacts(model_id: str, model_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash every file in one bounded, symlink-free inventory model directory."""
    root = _model_root_path()
    if root.is_symlink() or not root.is_dir() or model_path.is_symlink():
        raise ValueError("E1 bootstrap model directory is missing or symlinked")
    root = root.resolve(strict=True)
    path = model_path.resolve(strict=True)
    if path.parent != root or not path.is_dir():
        raise ValueError("E1 bootstrap inventory model_path is outside bounded model root")
    entries = []
    total_bytes = 0
    for entry in path.rglob("*"):
        if len(entries) >= 256 or entry.is_symlink():
            raise ValueError("E1 bootstrap model directory exceeds bound or contains a symlink")
        relative = entry.relative_to(path)
        if len(relative.parts) > 2 or not (entry.is_file() or entry.is_dir()):
            raise ValueError("E1 bootstrap model directory contains an unsupported entry")
        if entry.is_file():
            total_bytes += entry.stat().st_size
            if total_bytes > 512 * 1024**3:
                raise ValueError("E1 bootstrap model directory exceeds byte bound")
        entries.append(entry)
    files = {entry.relative_to(path).as_posix(): sha256_file(entry)
             for entry in sorted(entries) if entry.is_file()}
    required = {"config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}
    weights = {name: digest for name, digest in files.items() if name.endswith(".safetensors")}
    if not required <= files.keys() or not weights:
        raise ValueError("E1 bootstrap model lacks config, tokenizer, template, or weight shards")
    config = read_json(path / "config.json")
    architecture = config.get("architectures")
    quantization = config.get("quantization") or config.get("quantization_config")
    if (not isinstance(architecture, list) or len(architecture) != 1
            or not isinstance(architecture[0], str) or not architecture[0]
            or not isinstance(quantization, dict) or not quantization):
        raise ValueError("E1 bootstrap model config lacks exact architecture or quantization")
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        index = read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or set(weight_map.values()) != set(weights):
            raise ValueError("E1 bootstrap weight index differs from shard files")
    identity = {
        "model_id": model_id,
        "architecture": architecture[0],
        "weights_fingerprint": _digest(weights),
        "tokenizer_fingerprint": _digest({name: files[name] for name in ("tokenizer.json", "tokenizer_config.json")}),
        "chat_template_fingerprint": files["chat_template.jinja"],
        "quantization": _digest(quantization),
    }
    return identity, {"model_path": str(path), "model_root": str(root), "files": files}


def _inventory_model(records: Any, model_id: str) -> Path:
    if not isinstance(records, list):
        raise ValueError("E1 bootstrap oMLX inventory is invalid")
    matches = [row for row in records if isinstance(row, dict) and row.get("id") == model_id]
    if len(matches) != 1 or not isinstance(matches[0].get("model_path"), str):
        raise ValueError("E1 bootstrap oMLX inventory has no unique model_path")
    return Path(matches[0]["model_path"])


def _baseline(home: Path) -> tuple[Path, dict[str, Any]]:
    from .experiments import ExperimentStore

    store = ExperimentStore(home)
    # Comparison review may already hold this store's nonreentrant flock.
    active = read_json(store._active_path("main"))
    if (active.get("schema_version") != "mavis.experiment-active/v1"
            or active.get("scope") != "main"
            or active.get("experiment_id") is not None):
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


def _requirements(assignment: dict[str, Any]) -> list[str]:
    return [
        f"e0-summary:{assignment['e0_summary']['sha256']}",
        f"baseline:{assignment['baseline']['sha256']}",
        f"installed-candidate:{_digest(assignment['installed_candidate'])}",
        f"package-manifest:{assignment['package_manifest']['sha256']}",
        f"model-artifacts:{_digest(assignment['model_artifacts'])}",
    ]


def _assignment(home: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = home / "e1/bootstrap/review-assignment.json"
    assignment = read_json(path)
    summary_path, summary = _summary(home)
    baseline_path, baseline = _baseline(home)
    package_path = _package_manifest_path()
    if (assignment.get("schema_version") != "mavis.e1-bootstrap-assignment/v1"
            or assignment.get("objective_id") != "e1-bootstrap-main"
            or assignment.get("scope") != "main"
            or assignment.get("cwd") != str(path.parent.resolve())
            or assignment.get("owned_paths") != [str((home / "verifications/e1-bootstrap/main.json").resolve())]
            or assignment.get("required_checks") != ["independent-bootstrap-review"]
            or assignment.get("installed_candidate") != summary["installed_candidate"]):
        raise ValueError("E1 bootstrap review assignment is invalid or stale")
    owner = assignment.get("owner")
    if (not isinstance(owner, dict) or set(owner) != {"provider", "model", "harness"}
            or any(not isinstance(value, str) or not value for value in owner.values())):
        raise ValueError("E1 bootstrap review assignment has no exact owner")
    _checked_ref(assignment.get("e0_summary"), summary_path, "E0 summary")
    _checked_ref(assignment.get("baseline"), baseline_path, "baseline")
    _checked_ref(assignment.get("package_manifest"), package_path, "package manifest")
    model_inventory = assignment.get("model_inventory")
    model_path = model_inventory.get("model_path") if isinstance(model_inventory, dict) else None
    if (not isinstance(model_path, str)
            or model_inventory.get("model_id") != summary["model_id"]):
        raise ValueError("E1 bootstrap review assignment has no inventory model")
    identity, artifacts = _model_artifacts(summary["model_id"], Path(model_path))
    if assignment.get("model_identity") != identity or assignment.get("model_artifacts") != artifacts:
        raise ValueError("E1 bootstrap model artifact bytes changed")
    package = read_json(package_path)
    if (package.get("schema_version") != "mavis.installed-core/v1"
            or package.get("core_sha256") != summary["installed_candidate"]["core_sha256"]):
        raise ValueError("E1 bootstrap installed package differs from E0")
    if (assignment.get("requirements") != _requirements(assignment)
            or assignment.get("starting_revision") != _digest(assignment["installed_candidate"])):
        raise ValueError("E1 bootstrap review assignment requirements changed")
    return path, assignment, baseline


def prepare_bootstrap_review(
    home: Path, owner: dict[str, str], *, inventory_reader: Callable[[str], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Freeze exact E0, installed model bytes, and owner before dispatching review."""
    home = Path(home).resolve()
    path = home / "e1/bootstrap/review-assignment.json"
    if path.exists() or (home / "profiles/main/active.json").exists():
        raise FileExistsError("E1 bootstrap review assignment exists or main profile is active")
    if (not isinstance(owner, dict) or set(owner) != {"provider", "model", "harness"}
            or any(not isinstance(value, str) or not value for value in owner.values())):
        raise ValueError("E1 bootstrap review assignment has no exact owner")
    summary_path, summary = _summary(home)
    baseline_path, _ = _baseline(home)
    package_path = _package_manifest_path()
    records = (inventory_reader or inventory)("http://127.0.0.1:8001/v1")
    model_path = _inventory_model(records, summary["model_id"])
    identity, artifacts = _model_artifacts(summary["model_id"], model_path)
    assignment = {
        "schema_version": "mavis.e1-bootstrap-assignment/v1",
        "objective_id": "e1-bootstrap-main", "scope": "main",
        "cwd": str(path.parent.resolve()), "owner": owner,
        "owned_paths": [str((home / "verifications/e1-bootstrap/main.json").resolve())],
        "required_checks": ["independent-bootstrap-review"],
        "e0_summary": _ref(summary_path), "installed_candidate": summary["installed_candidate"],
        "baseline": _ref(baseline_path), "package_manifest": _ref(package_path),
        "model_inventory": {"model_id": summary["model_id"], "model_path": artifacts["model_path"]},
        "model_identity": identity, "model_artifacts": artifacts,
        "starting_revision": _digest(summary["installed_candidate"]),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    assignment["requirements"] = _requirements(assignment)
    write_json(path, assignment)
    return assignment


def _review(home: Path, receipt: dict[str, Any], status_reader: Callable[[str], dict[str, Any]],
            assignment_path: Path, assignment: dict[str, Any]) -> None:
    path = _checked_ref(receipt.get("independent_review"),
                        home / "verifications/e1-bootstrap/main.json", "independent review")
    _checked_ref(receipt.get("review_assignment"), assignment_path, "review assignment")
    review = read_json(path)
    if (review.get("schema_version") != "mavis.e1-bootstrap-review/v1"
            or review.get("verdict") != "accepted"
            or review.get("e0_summary_sha256") != receipt["e0_summary"]["sha256"]
            or review.get("baseline_sha256") != receipt["baseline"]["sha256"]
            or review.get("package_manifest_sha256") != receipt["package_manifest"]["sha256"]
            or review.get("installed_candidate_digest") != _digest(receipt["installed_candidate"])
            or review.get("model_identity_digest") != _digest(receipt["model_identity"])):
        raise ValueError("E1 bootstrap independent review does not match frozen evidence")
    if review.get("assignment_sha256") != sha256_file(assignment_path):
        raise ValueError("E1 bootstrap independent review changed assignment")
    job_id = review.get("gateway_worker_job_id")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("E1 bootstrap independent review has no gateway job")
    status = status_reader(job_id)
    _validate_gateway_status(status, job_id)
    acceptance = status["acceptance"]
    binding = status.get("mavis_binding")
    expected_binding = {
        "schema_version": "mavis.e1-bootstrap-gateway-binding/v1",
        "objective_id": assignment["objective_id"], "scope": assignment["scope"],
        "cwd": assignment["cwd"], "owner": assignment["owner"],
        "starting_revision": assignment["starting_revision"],
        "changed_revision": assignment["starting_revision"],
        "owned_paths": assignment["owned_paths"],
        "requirements": assignment["requirements"],
        "required_checks": assignment["required_checks"],
        "assignment_sha256": sha256_file(assignment_path),
        "report_sha256": sha256_file(path),
        "target_sha256": sha256_file(assignment_path),
    }
    if (acceptance.get("verifier_job_id") != review.get("verifier_job_id")
            or acceptance.get("report_sha256_on_disk") != sha256_file(path)
            or acceptance.get("target_sha256") != sha256_file(assignment_path)
            or binding != expected_binding
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
    assignment_path, assignment, baseline = _assignment(home)
    if (receipt.get("schema_version") != "mavis.e1-bootstrap/v1"
            or receipt.get("scope") != "main"
            or receipt.get("installed_candidate") != assignment["installed_candidate"]
            or receipt.get("model_identity") != assignment["model_identity"]
            or receipt.get("model_artifacts") != assignment["model_artifacts"]):
        raise ValueError("E1 bootstrap receipt is invalid or stale")
    for name in ("e0_summary", "baseline", "package_manifest"):
        if receipt.get(name) != assignment[name]:
            raise ValueError(f"E1 bootstrap {name} differs from frozen assignment")
    _review(home, receipt, status_reader or harness_job_status, assignment_path, assignment)
    return {**receipt, "prompts": baseline["prompts"],
            "_source_path": str(path), "_source_sha256": sha256_file(path)}


def create_bootstrap(
    home: Path, review_path: Path,
    *, status_reader: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Record a one-time trial binding after a separate gateway review exists."""
    home = Path(home).resolve()
    path = home / "e1/bootstrap/main.json"
    if path.exists() or (home / "profiles/main/active.json").exists():
        raise FileExistsError("E1 bootstrap already exists or main profile is active")
    assignment_path, assignment, baseline = _assignment(home)
    if Path(review_path).resolve() != (home / "verifications/e1-bootstrap/main.json").resolve():
        raise ValueError("E1 bootstrap review must be retained at its canonical path")
    receipt = {"schema_version": "mavis.e1-bootstrap/v1", "scope": "main",
               "e0_summary": assignment["e0_summary"],
               "installed_candidate": assignment["installed_candidate"],
               "model_identity": assignment["model_identity"],
               "model_artifacts": assignment["model_artifacts"],
               "baseline": assignment["baseline"],
               "package_manifest": assignment["package_manifest"],
               "review_assignment": _ref(assignment_path),
               "independent_review": _ref(Path(review_path)),
               "created_at": datetime.now(timezone.utc).isoformat()}
    _review(home, receipt, status_reader or harness_job_status, assignment_path, assignment)
    write_json(path, receipt)
    return {**receipt, "prompts": baseline["prompts"],
            "_source_path": str(path), "_source_sha256": sha256_file(path)}
