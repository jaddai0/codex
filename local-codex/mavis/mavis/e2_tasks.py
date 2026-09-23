"""Held-out cross-package E2 fixture and fail-closed host evidence checks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

from .evaluations import installed_candidate_fingerprint, native_review_completed
from .e1_bootstrap import _summary as current_e0_summary
from .runtime import RuntimeConfig
from .storage import sha256_file, write_json


CATALOG = "catalog/price.py"
CHECKOUT = "checkout/invoice.py"
DIRTY = "user-notes.txt"
PRIVATE = "private-draft.txt"
BLUEPRINT = "BLUEPRINT.md"
CATALOG_TEST = ["python3", "-m", "unittest", "tests.test_catalog", "-q"]
FULL_TEST = ["python3", "-m", "unittest", "discover", "-s", "tests", "-q"]


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _check(repo: Path, command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=repo, capture_output=True, text=True,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=False)


def prepare_heldout(home: Path) -> Path:
    """Create a fresh failing task; the worker receives only the fixture repo."""
    task = (Path(home) / "evaluations" / "e2" / "tasks" / uuid.uuid4().hex).resolve()
    repo = task / "repo"
    for name in ("catalog", "checkout", "tests"):
        (repo / name).mkdir(parents=True)
        (repo / name / "__init__.py").write_text("", encoding="utf-8")
    (repo / BLUEPRINT).write_text(
        "# Invoice correction blueprint\n\n"
        "Catalog computes discounted item subtotal in integer cents. Reject negative "
        "quantities, prices, and discounts. A discount cannot exceed the item subtotal.\n\n"
        "Checkout computes the invoice total: discounted item subtotal plus tax on "
        "that subtotal plus shipping. Shipping is not taxable. The tax rate is a "
        "whole percentage from 0 to 100 and tax rounds down to a cent.\n\n"
        "Fix both packages. Keep this blueprint, tests, and local user files intact. "
        "Run the focused catalog check before expanding to checkout, then the "
        "complete test suite. Do not commit this fixture.\n",
        encoding="utf-8",
    )
    (repo / CATALOG).write_text(
        "def discounted_subtotal(unit_cents: int, quantity: int, discount_cents: int) -> int:\n"
        "    if min(unit_cents, quantity, discount_cents) < 0:\n"
        "        raise ValueError('amounts cannot be negative')\n"
        "    return unit_cents * quantity + discount_cents\n",
        encoding="utf-8",
    )
    (repo / CHECKOUT).write_text(
        "from catalog.price import discounted_subtotal\n\n"
        "def invoice_total(unit_cents: int, quantity: int, discount_cents: int, "
        "tax_percent: int, shipping_cents: int) -> int:\n"
        "    subtotal = discounted_subtotal(unit_cents, quantity, discount_cents)\n"
        "    if shipping_cents < 0 or not 0 <= tax_percent <= 100:\n"
        "        raise ValueError('invalid shipping or tax')\n"
        "    return subtotal + shipping_cents + ((subtotal + shipping_cents) * tax_percent // 100)\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_catalog.py").write_text(
        "import unittest\nfrom catalog.price import discounted_subtotal\n\n"
        "class CatalogTests(unittest.TestCase):\n"
        "    def test_discount_reduces_subtotal(self):\n"
        "        self.assertEqual(discounted_subtotal(1000, 2, 300), 1700)\n"
        "    def test_discount_cannot_exceed_subtotal(self):\n"
        "        with self.assertRaises(ValueError):\n"
        "            discounted_subtotal(100, 1, 101)\n"
        "    def test_invalid_amounts(self):\n"
        "        with self.assertRaises(ValueError):\n"
        "            discounted_subtotal(100, -1, 0)\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_checkout.py").write_text(
        "import unittest\nfrom checkout.invoice import invoice_total\n\n"
        "class CheckoutTests(unittest.TestCase):\n"
        "    def test_cross_package_invoice(self):\n"
        "        self.assertEqual(invoice_total(1000, 2, 300, 10, 500), 2370)\n"
        "    def test_shipping_is_not_taxed(self):\n"
        "        self.assertEqual(invoice_total(1000, 1, 0, 10, 500), 1600)\n"
        "    def test_invalid_tax(self):\n"
        "        with self.assertRaises(ValueError):\n"
        "            invoice_total(100, 1, 0, 101, 0)\n",
        encoding="utf-8",
    )
    (repo / DIRTY).write_text("Keep my working notes.\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", BLUEPRINT, "catalog", "checkout", "tests", DIRTY], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Mavis E2",
                    "-c", "user.email=mavis@local.invalid", "commit", "-qm",
                    "Seed held-out invoice failure"], check=True)
    revision = _git(repo, "rev-parse", "HEAD")
    (repo / DIRTY).write_text("Keep my working notes.\nPrivate edit: orchid-47.\n", encoding="utf-8")
    (repo / PRIVATE).write_text("Untracked private draft: violet-29.\n", encoding="utf-8")
    baseline = _check(repo, FULL_TEST)
    if baseline.returncode == 0 or "Ran 6 tests" not in baseline.stderr or "FAILED" not in baseline.stderr:
        raise RuntimeError("E2 fixture does not have the expected six-test failing baseline")
    baseline_path = task / "baseline-test.log"
    baseline_path.write_text(baseline.stdout + baseline.stderr, encoding="utf-8")
    manifest = task / "manifest.json"
    write_json(manifest, {
        "schema_version": "mavis.e2-heldout/v1",
        "repo": str(repo.resolve()), "starting_revision": revision,
        "blueprint_sha256": sha256_file(repo / BLUEPRINT),
        "dirty_sha256": sha256_file(repo / DIRTY),
        "private_sha256": sha256_file(repo / PRIVATE),
        "baseline_exit_status": baseline.returncode,
        "baseline_sha256": sha256_file(baseline_path),
        "catalog_test": CATALOG_TEST, "full_test": FULL_TEST,
        "owned_paths": [CATALOG, CHECKOUT],
    })
    return manifest


def fixture_state(manifest_path: Path, *, stage: str) -> dict[str, str]:
    """Validate immutable inputs and the exact allowed two-package worktree."""
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "mavis.e2-heldout/v1":
        raise ValueError("E2 manifest schema is invalid")
    repo = Path(manifest["repo"]).resolve(strict=True)
    if repo != manifest_path.parent / "repo" or _git(repo, "rev-parse", "HEAD") != manifest["starting_revision"]:
        raise ValueError("E2 repository or starting revision changed")
    for name, key in ((BLUEPRINT, "blueprint_sha256"), (DIRTY, "dirty_sha256"),
                      (PRIVATE, "private_sha256")):
        if sha256_file(repo / name) != manifest[key]:
            raise ValueError(f"E2 protected file changed: {name}")
    baseline = manifest_path.parent / "baseline-test.log"
    if manifest["baseline_exit_status"] == 0 or sha256_file(baseline) != manifest["baseline_sha256"]:
        raise ValueError("E2 failing baseline changed")
    if manifest["catalog_test"] != CATALOG_TEST or manifest["full_test"] != FULL_TEST or manifest["owned_paths"] != [CATALOG, CHECKOUT]:
        raise ValueError("E2 acceptance commands or ownership changed")
    changed = set(_git(repo, "diff", "--name-only").splitlines())
    untracked = set(_git(repo, "ls-files", "--others", "--exclude-standard").splitlines())
    if untracked != {PRIVATE}:
        raise ValueError("E2 untracked files changed")
    expected = {DIRTY} if stage == "baseline" else ({DIRTY, CATALOG} if stage == "catalog" else {DIRTY, CATALOG, CHECKOUT})
    if stage not in {"baseline", "catalog", "complete"} or changed != expected:
        raise ValueError(f"E2 changed paths do not match {stage}: {sorted(changed)}")
    if stage != "baseline" and sha256_file(repo / CATALOG) == _committed_hash(repo, CATALOG):
        raise ValueError("E2 catalog was not repaired")
    if stage == "complete" and sha256_file(repo / CHECKOUT) == _committed_hash(repo, CHECKOUT):
        raise ValueError("E2 checkout was not repaired")
    return {"repo": str(repo), "revision": manifest["starting_revision"],
            "catalog_sha256": sha256_file(repo / CATALOG),
            "checkout_sha256": sha256_file(repo / CHECKOUT)}


def _committed_hash(repo: Path, name: str) -> str:
    original = subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:{name}"])
    return hashlib.sha256(original).hexdigest()


def run_host_check(task: Path, name: str, command: list[str]) -> Path:
    """Retain raw host output instead of trusting the worker's test claim."""
    if name not in {"catalog", "complete"}:
        raise ValueError("unknown E2 host check")
    expected = CATALOG_TEST if name == "catalog" else FULL_TEST
    if command != expected:
        raise ValueError("E2 host check command changed")
    task = Path(task).resolve(strict=True)
    repo = task / "repo"
    output = _check(repo, command)
    root = task / "host-checks" / name
    root.mkdir(parents=True, exist_ok=False)
    stdout = root / "stdout.log"
    stderr = root / "stderr.log"
    stdout.write_text(output.stdout, encoding="utf-8")
    stderr.write_text(output.stderr, encoding="utf-8")
    receipt = root / "receipt.json"
    write_json(receipt, {
        "schema_version": "mavis.e2-host-check/v1", "name": name,
        "command": command, "cwd": str(repo.resolve()),
        "starting_revision": _git(repo, "rev-parse", "HEAD"),
        "exit_status": output.returncode,
        "stdout_sha256": sha256_file(stdout), "stderr_sha256": sha256_file(stderr),
        "stdout_bytes": stdout.stat().st_size, "stderr_bytes": stderr.stat().st_size,
    })
    return receipt


def _verify_host_check(receipt_path: Path, *, task: Path, name: str, revision: str) -> None:
    root = task / "host-checks" / name
    if receipt_path.resolve() != (root / "receipt.json").resolve():
        raise ValueError("E2 host receipt path changed")
    receipt = json.loads(receipt_path.read_text())
    stdout, stderr = root / "stdout.log", root / "stderr.log"
    command = CATALOG_TEST if name == "catalog" else FULL_TEST
    if (receipt.get("schema_version") != "mavis.e2-host-check/v1"
            or receipt.get("name") != name or receipt.get("command") != command
            or receipt.get("cwd") != str((task / "repo").resolve())
            or receipt.get("starting_revision") != revision
            or receipt.get("exit_status") != 0
            or receipt.get("stdout_sha256") != sha256_file(stdout)
            or receipt.get("stderr_sha256") != sha256_file(stderr)
            or receipt.get("stdout_bytes") != stdout.stat().st_size
            or receipt.get("stderr_bytes") != stderr.stat().st_size):
        raise ValueError(f"E2 {name} host check is incomplete or changed")
    summary = stderr.read_text(encoding="utf-8", errors="replace")
    count = 3 if name == "catalog" else 6
    if f"Ran {count} tests" not in summary or "\nOK" not in summary or "FAILED" in summary:
        raise ValueError(f"E2 {name} host check did not pass the frozen suite")


def _rollout_records(path: Path) -> tuple[bytes, list[dict]]:
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        raise ValueError("E2 rollout has an incomplete final record")
    try:
        records = [json.loads(line) for line in data.splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("E2 rollout contains malformed JSONL") from exc
    if not records or any(not isinstance(row, dict) for row in records):
        raise ValueError("E2 rollout contains no valid records")
    return data, records


def verify_heldout(manifest_path: Path) -> dict[str, object]:
    """Recheck an installed, restarted, independently reviewed task."""
    manifest_path = Path(manifest_path).resolve(strict=True)
    task = manifest_path.parent
    result = json.loads((task / "result.json").read_text())
    manifest = json.loads(manifest_path.read_text())
    if result.get("schema_version") != "mavis.e2-installed-observation/v1" or result.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("E2 installed observation is missing or changed")
    observer = Path(result.get("observer_path", "")).resolve(strict=True)
    if observer.name != "observe_heldout_e2.py" or result.get("observer_sha256") != sha256_file(observer):
        raise ValueError("E2 observer source changed")
    if result.get("error") or result.get("verification_error") or result.get("iris_restore_error") or result.get("mavis_unload_error"):
        raise ValueError("E2 observer recorded a runtime or restoration failure")
    candidate = installed_candidate_fingerprint()
    if result.get("candidate") != candidate or result.get("candidate_after") != candidate:
        raise ValueError("E2 observation belongs to a different installed candidate")
    e0_path, e0 = current_e0_summary(Path.home() / ".local-codex" / "mavis-service")
    if (result.get("e0_summary_sha256") != sha256_file(e0_path)
            or result.get("model_id") != e0["model_id"]
            or result["model_id"] != RuntimeConfig(home=e0_path.parent.parent.parent).model):
        raise ValueError("E2 model or E0 prerequisite changed")
    final_state = fixture_state(manifest_path, stage="complete")
    if result.get("final_state") != final_state:
        raise ValueError("E2 final repository state changed")
    stage1 = result.get("stage1_state")
    if not isinstance(stage1, dict) or stage1.get("repo") != final_state["repo"] or stage1.get("revision") != final_state["revision"] or stage1.get("catalog_sha256") != final_state["catalog_sha256"]:
        raise ValueError("E2 catalog checkpoint was not preserved")
    if stage1.get("checkout_sha256") != _committed_hash(Path(final_state["repo"]), CHECKOUT):
        raise ValueError("E2 checkpoint did not isolate the catalog slice")
    _verify_host_check(Path(result["catalog_check"]), task=task, name="catalog", revision=manifest["starting_revision"])
    _verify_host_check(Path(result["complete_check"]), task=task, name="complete", revision=manifest["starting_revision"])
    if (result.get("first_exit") != 0 or result.get("resume_exit") != 0
            or not isinstance(result.get("first_pid"), int)
            or not isinstance(result.get("resume_pid"), int)
            or result["first_pid"] == result["resume_pid"]
            or result.get("iris_loaded") is not True or result.get("mavis_loaded") is not False):
        raise ValueError("E2 installed process restart or model handoff did not complete")
    rollout = Path(result["rollout"]).resolve(strict=True)
    data, records = _rollout_records(rollout)
    prefix_bytes = result.get("first_rollout_bytes")
    prefix_count = result.get("first_rollout_events")
    if (not isinstance(prefix_bytes, int) or prefix_bytes <= 0 or prefix_bytes >= len(data)
            or not isinstance(prefix_count, int) or prefix_count <= 0 or prefix_count >= len(records)
            or hashlib.sha256(data[:prefix_bytes]).hexdigest() != result.get("first_rollout_sha256")
            or len(data[:prefix_bytes].splitlines()) != prefix_count):
        raise ValueError("E2 first-session rollout prefix changed")
    first, later = records[:prefix_count], records[prefix_count:]
    if (first[0].get("type") != "session_meta"
            or first[0].get("payload", {}).get("cwd") != final_state["repo"]
            or first[0].get("payload", {}).get("id") != result.get("session_id")
            or not any(row.get("type") == "compacted" for row in first)
            or not any(row.get("type") == "event_msg" and row.get("payload", {}).get("type") == "task_complete" for row in first)
            or not any(row.get("type") == "event_msg" and row.get("payload", {}).get("type") == "task_complete" for row in later)):
        raise ValueError("E2 compaction or resumed task completion is unproved")
    review_log = task / "terra-review.jsonl"
    review_text = task / "terra-review.txt"
    if (result.get("review_exit") != 0
            or result.get("review_log_sha256") != sha256_file(review_log)
            or result.get("review_text_sha256") != sha256_file(review_text)
            or not native_review_completed(review_log.read_text(), review_text.read_text())):
        raise ValueError("E2 independent native review did not accept")
    return {"schema_version": "mavis.e2-verification/v1", "status": "pass",
            "manifest": str(manifest_path), "session_id": result["session_id"],
            "installed_candidate": candidate,
            "evidence": {"result_sha256": sha256_file(task / "result.json"),
                         "rollout_sha256": sha256_file(rollout),
                         "catalog_check_sha256": sha256_file(Path(result["catalog_check"])),
                         "complete_check_sha256": sha256_file(Path(result["complete_check"])),
                         "review_log_sha256": sha256_file(review_log)}}
